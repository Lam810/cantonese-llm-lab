#!/usr/bin/env python3
"""把官方口径的结果汇成两张表：HKMMLU 选择题、HKMMLU 翻译。"""
import json, glob, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else "."

def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict): return None
        d = d.get(k)
    return d
def f(v, s="{:.4f}"):
    return "   —  " if v is None else s.format(v)

mc, tr = {}, {}
for p in sorted(glob.glob(os.path.join(D, "*.json"))):
    b = os.path.basename(p)
    if b.startswith(("mcnemar", "compare", "summary")): continue
    try: d = json.load(open(p, encoding="utf-8"))
    except Exception: continue
    n = d.get("name", b)
    if "mc" in d: mc[n] = d
    if "trans" in d: tr[n] = d

print("\n=== HKMMLU 选择题 · 官方口径（zero-shot prompting，贪心，全量）===")
h = "%-24s %9s %9s %10s %10s %9s %9s %11s" % (
    "模型", "准确率", "macro", "解析失败", "平均输出tok", "hk_子集", "hkdse_子集", "题数")
print(h); print("-" * len(h))
for n, d in sorted(mc.items(), key=lambda x: -(g(x[1], "mc", "overall_acc") or 0)):
    m = d["mc"]
    print("%-24s %9s %9s %10s %10s %9s %9s %11s" % (
        n, f(m.get("overall_acc")), f(m.get("macro_acc")),
        f(m.get("unparsed_rate"), "{:.4f}"), f(m.get("mean_out_tokens"), "{:.1f}"),
        f(m.get("hk_subset_acc")), f(m.get("hkdse_subset_acc")), m.get("n")))

print("\n=== HKMMLU 翻译 · 官方任务（各 2000 条，seed=42）===")
h2 = "%-24s %20s %20s %11s" % ("模型", "粤→普 chrF / BLEU", "普→粤 chrF / BLEU", "空输出")
print(h2); print("-" * len(h2))
def key(x):
    t = x[1]["trans"]
    return -((g(t, "man2can", "chrF") or 0) + (g(t, "can2man", "chrF") or 0))
for n, d in sorted(tr.items(), key=key):
    t = d["trans"]
    c2m = "%s / %s" % (f(g(t,"can2man","chrF"), "{:.2f}"), f(g(t,"can2man","BLEU_zh"), "{:.2f}"))
    m2c = "%s / %s" % (f(g(t,"man2can","chrF"), "{:.2f}"), f(g(t,"man2can","BLEU_zh"), "{:.2f}"))
    empt = (g(t,"can2man","empty_output") or 0) + (g(t,"man2can","empty_output") or 0)
    print("%-24s %20s %20s %11d" % (n, c2m, m2c, empt))

print("\n注：普→粤（man2can）那一列才是直接量「能不能产出粤语」的，")
print("    选择题量不到这一点。chrF 是字符级，中文上比 BLEU 更可靠。")
