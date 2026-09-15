#!/usr/bin/env python3
"""把 results/rivals/*.json 汇成一张可直接抄进报告的表。"""
import json, glob, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/data/user/zlin810/yue-lab/results/rivals"

def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict): return None
        d = d.get(k)
    return d

def f(v, spec="{:.4f}", dash="  —  "):
    return dash if v is None else spec.format(v)

rows = {}
for p in sorted(glob.glob(os.path.join(D, "*.json"))):
    if os.path.basename(p).startswith(("mcnemar", "compare")): continue
    d = json.load(open(p, encoding="utf-8"))
    rows[d.get("name", os.path.basename(p))] = d

nothink = {k[:-8]: v for k, v in rows.items() if k.endswith("_nothink")}
think   = {k[:-6]: v for k, v in rows.items() if k.endswith("_think")}

print("\n=== 口径 A：关思维链（主表）===")
h = "%-24s %9s %9s %10s %9s %8s %9s %9s %9s" % (
    "模型", "HKMMLU", "macro", "95%CI宽", "纯度117", "退化", "中位tok", "PPL留出", "PPL维基")
print(h); print("-" * len(h))
for n in sorted(nothink):
    d = nothink[n]
    ci = g(d, "hkmmlu", "ci95")
    ciw = (ci["hi"] - ci["lo"]) * 100 if ci else None
    print("%-24s %9s %9s %10s %9s %8s %9s %9s %9s" % (
        n, f(g(d,"hkmmlu","overall_acc")), f(g(d,"hkmmlu","macro_acc")),
        f(ciw, "±{:.2f}pp"),
        f(g(d,"purity","full","mean_yue_ratio")),
        f(g(d,"purity","full","degenerate_rate"), "{:.3f}"),
        f(g(d,"purity","full","median_out_tokens"), "{:.0f}"),
        f(g(d,"ppl","heldout","ppl"), "{:.2f}"), f(g(d,"ppl","wiki","ppl"), "{:.2f}")))

print("\n=== 口径 B：开思维链（HKMMLU 为生成式，n≈660）===")
h2 = "%-24s %9s %10s %9s %8s %10s %11s %9s" % (
    "模型", "HKMMLU", "解析失败", "纯度117", "退化", "中位tok", "中位think", "出think率")
print(h2); print("-" * len(h2))
for n in sorted(think):
    d = think[n]
    print("%-24s %9s %10s %9s %8s %10s %11s %9s" % (
        n, f(g(d,"hkmmlu_gen","overall_acc")),
        f(g(d,"hkmmlu_gen","unparsed_rate"), "{:.3f}"),
        f(g(d,"purity","full","mean_yue_ratio")),
        f(g(d,"purity","full","degenerate_rate"), "{:.3f}"),
        f(g(d,"purity","full","median_out_tokens"), "{:.0f}"),
        f(g(d,"purity","median_think_tokens"), "{:.0f}"),
        f(g(d,"purity","think_rate"), "{:.2f}")))

print("\n=== 与已发布口径的对照（前 30 条提问，关思维链）===")
h3 = "%-24s %9s %8s %9s" % ("模型", "纯度30", "退化30", "中位长度")
print(h3); print("-" * len(h3))
for n in sorted(nothink):
    d = nothink[n]
    print("%-24s %9s %8s %9s" % (
        n, f(g(d,"purity","first30","mean_yue_ratio")),
        f(g(d,"purity","first30","degenerate_rate"), "{:.3f}"),
        f(g(d,"purity","first30","median_len"), "{:.0f}")))

json.dump({"nothink": {k: {kk: vv for kk, vv in v.items() if kk != "purity"}
                       for k, v in nothink.items()}},
          open(os.path.join(D, "rivals_summary.json"), "w"), ensure_ascii=False, indent=1)
print("\n汇总已写入", os.path.join(D, "rivals_summary.json"))
