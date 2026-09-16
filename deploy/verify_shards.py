#!/usr/bin/env python3
"""核对分片后的张量与原单文件逐字节相同。

分片是为了绕过 HF git-lfs 的 5GB 单文件上限，但「切了一刀」就必须验——
这和验 LoRA 合并产物是同一个纪律：本地重做的产物和原始产物只是**假定**等价。
"""
import argparse, json, math, os, random
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--sharded", required=True)
ap.add_argument("--n-sample", type=int, default=6)
a = ap.parse_args()

idx = json.load(open(os.path.join(a.sharded, "model.safetensors.index.json")))
wm = idx["weight_map"]
with safe_open(a.src, framework="pt") as f:
    keys = list(f.keys())
    print("原文件张量 %d | 索引张量 %d | 集合一致: %s"
          % (len(keys), len(wm), set(keys) == set(wm)))
    if set(keys) != set(wm):
        print("  只在原文件:", sorted(set(keys) - set(wm))[:5])
        print("  只在索引  :", sorted(set(wm) - set(keys))[:5])
    sizes = {k: math.prod(f.get_slice(k).get_shape()) for k in keys}
    picks = [max(sizes, key=sizes.get), min(sizes, key=sizes.get)]
    picks += [k for k in keys if "embed_tokens" in k or "lm_head" in k][:2]
    rng = random.Random(42)
    picks += rng.sample(keys, max(0, a.n_sample - len(picks)))
    picks = list(dict.fromkeys(picks))
    handles, ok = {}, 0
    for k in picks:
        sh = wm[k]
        if sh not in handles:
            handles[sh] = safe_open(os.path.join(a.sharded, sh), framework="pt")
        x, y = f.get_tensor(k), handles[sh].get_tensor(k)
        same = (x.shape == y.shape) and (x.dtype == y.dtype) and torch.equal(x, y)
        ok += int(same)
        print("  %s %-52s %s %s -> %s" % ("OK " if same else "BAD", k,
                                          tuple(x.shape), x.dtype, sh))
    print("抽检 %d/%d 逐字节相同" % (ok, len(picks)))
    tot_src = os.path.getsize(a.src)
    tot_sh = sum(os.path.getsize(os.path.join(a.sharded, s)) for s in set(wm.values()))
    print("字节数 原 %d / 分片合计 %d / 差 %+d（header 开销差异，张量内容不受影响）"
          % (tot_src, tot_sh, tot_sh - tot_src))
    raise SystemExit(0 if ok == len(picks) and set(keys) == set(wm) else 1)
