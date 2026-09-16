#!/usr/bin/env python3
"""发布闸门：纯度 >=0.85 且 退化 <=0.10 且 HKMMLU >=0.55。

为什么要有这道闸门：v1 那次没评测就发，结果是 93.3% 生成退化的模型。
「服务起得来」和「能发布」是两件事——静态 W8A8 那版 serve rc=0、吞吐还更高，
但输出是 token 沙拉。
用法：gate_check.py <purity.json> <mc.json>
"""
import json, sys

def g(p, *ks):
    d = json.load(open(p, encoding="utf-8"))
    for k in ks:
        if not isinstance(d, dict): return None
        d = d.get(k)
    return d

pj, mj = sys.argv[1], sys.argv[2]
pur = g(pj, "purity", "full", "mean_yue_ratio")
deg = g(pj, "purity", "full", "degenerate_rate")
acc = g(mj, "mc", "overall_acc")
unp = g(mj, "mc", "unparsed_rate")
otk = g(mj, "mc", "mean_out_tokens")
if None in (pur, deg, acc):
    print("结果不全，无法判定：", dict(纯度=pur, 退化=deg, HKMMLU=acc)); sys.exit(1)
row = lambda n, v, op, th, ok: print("  %-12s %.4f   闸门 %s%.2f   %s" % (n, v, op, th, "通过" if ok else "不通过"))
row("书面粤语纯度", pur, ">=", 0.85, pur >= 0.85)
row("退化率", deg, "<=", 0.10, deg <= 0.10)
row("HKMMLU", acc, ">=", 0.55, acc >= 0.55)
print("  解析失败率   %.4f   平均输出 %.1f tok" % (unp or 0, otk or 0))
ok = pur >= 0.85 and deg <= 0.10 and acc >= 0.55
print("  ==>", "过闸门，可以发布" if ok else "未过闸门，不发布")
sys.exit(0 if ok else 2)
