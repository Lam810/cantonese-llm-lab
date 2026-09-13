#!/usr/bin/env python3
"""把训练好的 LoRA adapter 合并进基座，单进程落盘（bf16，不量化）。

独立成脚本的原因：多卡训练结束后在 4 个 rank 上同时 merge_and_unload + save_pretrained
会并发写同一批 safetensors。合并本身是单卡几分钟的事，分开做更稳。
"""
import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    try:    m = AutoModelForCausalLM.from_pretrained(a.base, dtype=dt, device_map="cpu", trust_remote_code=True)
    except TypeError: m = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dt, device_map="cpu", trust_remote_code=True)
    m = PeftModel.from_pretrained(m, a.adapter)
    m = m.merge_and_unload()
    m.config.use_cache = True
    os.makedirs(a.out, exist_ok=True)
    m.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)

    # 体检：合并量级是否合理（v1 就是这里炸的）
    import glob
    from safetensors import safe_open
    def load(d):
        w = {}
        for f in glob.glob(os.path.join(d, "*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys(): w[k] = sf.get_tensor(k)
        return w
    B, M = load(a.base), load(a.out)
    rows = []
    for k in B:
        if k in M and B[k].shape == M[k].shape:
            nb = B[k].float().norm().item()
            if nb: rows.append((k, (M[k].float()-B[k].float()).norm().item()/nb))
    rows.sort(key=lambda x: -x[1])
    rep = {"n_compared": len(rows), "n_changed": sum(1 for _, v in rows if v > 0),
           "max_rel_delta": rows[0][1] if rows else None, "top10": rows[:10]}
    json.dump(rep, open(os.path.join(a.out, "merge_check.json"), "w"), indent=1)
    print("合并完成 ->", a.out)
    print("改动张量 %d/%d，最大相对差 %.4f" % (rep["n_changed"], rep["n_compared"], rep["max_rel_delta"] or 0))
    print("（r=32/alpha=64 的 LoRA 正常应在 0.01–0.10；接近 1.0 说明训飞了）")

if __name__ == "__main__":
    main()
