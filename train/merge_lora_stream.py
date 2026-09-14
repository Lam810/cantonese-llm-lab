#!/usr/bin/env python3
"""低内存流式合并 LoRA：逐分片处理，峰值内存 ≈ 一个分片 + adapter。

写这个脚本的原因：本机只有 15GB 内存，8B 的 base(16GB) + merged(16GB) 同时驻留放不下。
safetensors 按分片读写，一次只碰一个分片，峰值约 4~5GB。
合并完自动做 ‖ΔW‖/‖W‖ 体检，用来和集群上那份合并产物对齐（应为 252/399、最大 0.0782）。
"""
import argparse, glob, json, os, re, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def load_adapter(path):
    """返回 {base_tensor_name: (A, B)} 与 scaling。"""
    cfg = json.load(open(os.path.join(path, "adapter_config.json"), encoding="utf-8"))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / r
    f = os.path.join(path, "adapter_model.safetensors")
    pairs = {}
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            m = re.match(r"base_model\.model\.(.+)\.lora_([AB])(?:\.default)?\.weight$", k)
            if not m: continue
            base_name, which = m.group(1) + ".weight", m.group(2)
            pairs.setdefault(base_name, {})[which] = sf.get_tensor(k)
    out = {k: (v["A"], v["B"]) for k, v in pairs.items() if "A" in v and "B" in v}
    return out, scaling, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    os.makedirs(a.out, exist_ok=True)

    deltas, scaling, cfg = load_adapter(a.adapter)
    print(f"adapter: {len(deltas)} 个目标张量, r={cfg['r']} alpha={cfg['lora_alpha']} scaling={scaling}")

    shards = sorted(glob.glob(os.path.join(a.base, "*.safetensors")))
    print(f"基座分片 {len(shards)} 个")
    applied, report = 0, []
    for sh in shards:
        name = os.path.basename(sh)
        tensors = {}
        with safe_open(sh, framework="pt") as sf:
            meta = sf.metadata() or {}
            for k in sf.keys():
                t = sf.get_tensor(k)
                if k in deltas:
                    A, B = deltas[k]
                    # 用 fp32 算增量再落回目标精度，避免 bf16 累加误差
                    dW = (B.float() @ A.float()) * scaling
                    w32 = t.float() + dW
                    rel = dW.norm().item() / max(t.float().norm().item(), 1e-12)
                    report.append((k, rel))
                    t = w32.to(dt); applied += 1
                    del A, B, dW, w32
                else:
                    t = t.to(dt) if t.is_floating_point() else t
                tensors[k] = t.contiguous()
        save_file(tensors, os.path.join(a.out, name), metadata={"format": "pt"})
        print(f"  {name}: {len(tensors)} 张量，本片改动 {sum(1 for k in tensors if k in deltas)}")
        del tensors

    # 非权重文件照抄（分词器用基座的，LoRA 不动分词器）
    for f in ("model.safetensors.index.json", "config.json", "generation_config.json",
              "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "added_tokens.json", "special_tokens_map.json"):
        src = os.path.join(a.base, f)
        if os.path.exists(src): shutil.copy2(src, os.path.join(a.out, f))

    report.sort(key=lambda x: -x[1])
    chk = {"n_applied": applied, "n_adapter_targets": len(deltas),
           "max_rel_delta": report[0][1] if report else None,
           "median_rel_delta": sorted(r[1] for r in report)[len(report)//2] if report else None,
           "top10": report[:10], "scaling": scaling, "dtype": a.dtype}
    json.dump(chk, open(os.path.join(a.out, "merge_check.json"), "w"), indent=1)
    print(f"\n合并完成 -> {a.out}")
    print(f"改动张量 {applied}/{len(deltas)}（应等于 adapter 目标数），最大相对差 {chk['max_rel_delta']:.4f}")
    print("（r=32/alpha=64 正常 0.01–0.10；集群那份是 0.0782，对得上即等价）")

if __name__ == "__main__":
    main()
