#!/usr/bin/env python3
"""把单个大 safetensors 切成 <5GB 的分片 + index.json。

为什么需要：HF 的 git-lfs 上传路径对单文件有 5 GB 上限
（"You need to configure your repository to enable upload of files > 5GB"）。
绕它要用 `hf lfs-enable-largefiles` 注册一个 multipart 传输代理，那需要 Python 版
的 hf CLI；分片是更干净的做法，而且 HF 的惯例本来就是 ≤5 GB 一片、能局部下载。

流式逐张量写，峰值内存 ≈ 最大单张量，不是整个模型。
"""
import argparse, json, os, shutil
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="单文件 model.safetensors")
ap.add_argument("--out", required=True, help="输出目录")
ap.add_argument("--max-gb", type=float, default=4.0)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
LIMIT = int(a.max_gb * (1 << 30))

with safe_open(a.src, framework="pt") as f:
    keys = list(f.keys())
    sizes = {}
    for k in keys:
        sl = f.get_slice(k)
        n = 1
        for d in sl.get_shape(): n *= d
        # dtype 字符串形如 'BF16' / 'F32' / 'I8'
        bits = int("".join(c for c in sl.get_dtype() if c.isdigit()) or 16)
        sizes[k] = n * bits // 8

    # 按原顺序贪心装箱，保持层序（便于顺序加载）
    groups, cur, cur_sz = [], [], 0
    for k in keys:
        if cur and cur_sz + sizes[k] > LIMIT:
            groups.append(cur); cur, cur_sz = [], 0
        cur.append(k); cur_sz += sizes[k]
    if cur: groups.append(cur)

    n_sh = len(groups)
    weight_map, total = {}, 0
    print(f"张量 {len(keys)} 个，切成 {n_sh} 片（上限 {a.max_gb} GB/片）", flush=True)
    for i, g in enumerate(groups, 1):
        name = f"model-{i:05d}-of-{n_sh:05d}.safetensors"
        tensors = {k: f.get_tensor(k) for k in g}
        save_file(tensors, os.path.join(a.out, name), metadata={"format": "pt"})
        del tensors
        sz = os.path.getsize(os.path.join(a.out, name))
        total += sz
        for k in g: weight_map[k] = name
        print(f"  {name}  {len(g):>3} 张量  {sz/2**30:.2f} GiB", flush=True)

json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
src_dir = os.path.dirname(os.path.abspath(a.src))
for fn in os.listdir(src_dir):
    if fn == os.path.basename(a.src) or fn.endswith(".safetensors"): continue
    shutil.copy2(os.path.join(src_dir, fn), os.path.join(a.out, fn))
    print("  ← 附带", fn)
print(f"合计 {total} 字节 ({total/2**30:.2f} GiB)；原文件 {os.path.getsize(a.src)} 字节")
print("注意：分片后总字节与原文件会有几百字节出入（header 对齐与 padding 差异），张量内容逐字节不变——用 verify_shards.py 核对。")
