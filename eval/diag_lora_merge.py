#!/usr/bin/env python3
"""诊断 v1 生成退化：是合并权重坏了，还是 LoRA 本身就这样。

三条对照：base / base+adapter(未合并) / v1(已合并)，同样的贪心解码；
外加合并权重与基座的逐模块相对差异 ||ΔW||/||W||。
r=8、alpha=16 的 LoRA 合并后相对差异应该在 1e-3~1e-2 量级；
如果某些层大出几个数量级，就是 QLoRA 在 4bit 基座上训、却合并回全精度基座的典型症状。
"""
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

PROMPTS = ["香港有咩好玩嘅地方？","點樣由旺角去中環最快？","你可唔可以解釋下咩係強積金？",
           "介紹下香港嘅天氣。","1+1 等於幾多？"]

def gen(tag, model, tok, dev, n=100):
    out = []
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role":"user","content":p}], tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
        r = tok.decode(g[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)
        out.append({"prompt": p, "response": r})
        print(f"[{tag}] {p} -> {r[:120]}", flush=True)
    return out

def weight_delta(base_dir, merged_dir, topk=12):
    def load(d):
        import glob, os
        w = {}
        for f in glob.glob(os.path.join(d, "*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys(): w[k] = sf.get_tensor(k)
        return w
    B, M = load(base_dir), load(merged_dir)
    rows = []
    for k in B:
        if k not in M or B[k].shape != M[k].shape: continue
        b = B[k].float(); m = M[k].float()
        nb = b.norm().item()
        if nb == 0: continue
        rows.append((k, (m-b).norm().item()/nb))
    rows.sort(key=lambda x: -x[1])
    return {"n_compared": len(rows), "top": rows[:topk], "all": rows,
            "median_rel_delta": sorted(r[1] for r in rows)[len(rows)//2] if rows else None,
            "n_identical": sum(1 for _, v in rows if v == 0.0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--merged", required=True)
    ap.add_argument("--adapter", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}

    print("=== 权重差异 ===", flush=True)
    res["weight_delta"] = weight_delta(a.base, a.merged)
    print(json.dumps(res["weight_delta"], ensure_ascii=False, indent=1)[:900], flush=True)

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    def load(p):
        try:    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, device_map=dev)
        except TypeError: return AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map=dev)

    print("=== base ===", flush=True)
    mb = load(a.base); res["base"] = gen("base", mb, tok, dev)
    if a.adapter:
        print("=== base + adapter ===", flush=True)
        try:
            from peft import PeftModel
            mp = PeftModel.from_pretrained(mb, a.adapter)
            res["base_plus_adapter"] = gen("adapter", mp, tok, dev)
        except Exception as e:
            res["base_plus_adapter_error"] = repr(e)[:300]; print("adapter 加载失败:", e, flush=True)
    del mb; torch.cuda.empty_cache()

    print("=== merged (v1) ===", flush=True)
    mm = load(a.merged); res["merged_v1"] = gen("merged", mm, tok, dev)

    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("WROTE", a.out)

if __name__ == "__main__":
    main()
