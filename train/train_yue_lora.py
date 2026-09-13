#!/usr/bin/env python3
"""粤语 SFT v2：Qwen3-8B + LoRA。

针对 v1 的四个缺陷逐条修：
  1. v1 没有验证集 → 这里 eval_strategy=steps + 早停 + load_best_model_at_end
  2. v1 只挂 q_proj/v_proj、r=8 → 这里默认挂满七个线性层、r=32
  3. v1 fp32 训 + fp32 落盘 → 这里 bf16
  4. v1 大概率整段算 loss → 这里只对 assistant 段算 loss（prompt 部分置 -100）
"""
import argparse, json, math, os, random, sys
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback, set_seed)
from peft import LoraConfig, get_peft_model

ALL_LINEAR = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

class ChatSFT(Dataset):
    """jsonl，每行 {"messages":[{user},{assistant}]}；只对 assistant 段算 loss。"""
    def __init__(self, path, tok, max_len, limit=0):
        self.rows, self.tok, self.max_len = [], tok, max_len
        for i, line in enumerate(open(path, encoding="utf-8")):
            if limit and i >= limit: break
            try: d = json.loads(line)
            except Exception: continue
            m = d.get("messages") or []
            if len(m) < 2: continue
            self.rows.append((m[0]["content"], m[1]["content"]))
        self.n_trunc = 0

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        user, asst = self.rows[i]
        prompt = self.tok.apply_chat_template([{"role":"user","content":user}],
                                              tokenize=False, add_generation_prompt=True)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(asst + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = p_ids + a_ids
        labels = [-100]*len(p_ids) + a_ids[:]
        if len(ids) > self.max_len:
            ids, labels = ids[:self.max_len], labels[:self.max_len]
            self.n_trunc += 1
        return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default=",".join(ALL_LINEAR))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true", help="训完把 LoRA 合并成完整权重")
    a = ap.parse_args()
    set_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = dict(device_map=None, trust_remote_code=True)
    try:    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, **kw)
    except TypeError: model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, **kw)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=a.targets.split(","))
    model = get_peft_model(model, lcfg)
    # LoRA 参数保持 fp32：纯 bf16 训适配器容易不稳，v1 就是训飞的
    n_up = 0
    for n, p_ in model.named_parameters():
        if p_.requires_grad and p_.dtype != torch.float32:
            p_.data = p_.data.float(); n_up += 1
    print(f"[dtype] 基座 bf16，{n_up} 个可训练张量升到 fp32", flush=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable={trainable/1e6:.1f}M / total={total/1e9:.2f}B = {100*trainable/total:.3f}%", flush=True)

    tr = ChatSFT(a.train, tok, a.max_len, a.limit_train)
    ev = ChatSFT(a.dev, tok, a.max_len)
    print(f"[data] train={len(tr)} dev={len(ev)}", flush=True)

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.accum, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_ratio=a.warmup_ratio, weight_decay=0.01,
        bf16=True, logging_steps=25, save_strategy="steps", save_steps=a.eval_steps,
        eval_strategy="steps", eval_steps=a.eval_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=[], seed=a.seed, gradient_checkpointing=True,
        dataloader_num_workers=4, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=ev,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)])
    trainer.train()
    trainer.save_model(os.path.join(a.out, "adapter-best"))
    hist = trainer.state.log_history
    json.dump({"log_history": hist, "best_metric": trainer.state.best_metric,
               "best_ckpt": trainer.state.best_model_checkpoint,
               "n_truncated_train": tr.n_trunc, "args": vars(a)},
              open(os.path.join(a.out, "train_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[best] eval_loss =", trainer.state.best_metric, flush=True)

    # 只在 rank 0 上合并落盘：多卡时每个 rank 都写同一个目录会把 safetensors 写坏
    is_main = int(os.environ.get("RANK", "0")) == 0
    if a.merge and is_main:
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        mdir = os.path.join(a.out, "merged")
        merged.save_pretrained(mdir, safe_serialization=True)
        tok.save_pretrained(mdir)
        print("[merged] ->", mdir, flush=True)

if __name__ == "__main__":
    main()
