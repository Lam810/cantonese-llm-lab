#!/usr/bin/env python3
"""发布闸门。

为什么要有这道闸门：v1 那次没评测就发，结果是 93.3% 生成退化的模型。
「服务起得来」和「能发布」是两件事——静态 W8A8 那版 serve rc=0、吞吐还更高
（511 vs 312 tok/s），但输出是 token 沙拉：它从不吐 EOS、一路生成到上限，
所以吞吐反而更高。只看 tok/s 会把报废的模型报成更好的。

## 这道闸门的阈值，我定错过两次

**第一次：跨口径搬绝对阈值。** 原来这里写死 `HKMMLU >= 0.55`。那个 0.55 是从
**logprob 口径**倒推的（bf16 基线 0.6094）。换成官方生成口径后，同一个 bf16
模型只有 0.5627——阈值没动，尺子换了，于是一个只掉 1.5pp 的变体被判「不通过」。
**绝对阈值不能跨口径搬。**

**第二次：拿独立比例的标准误当配对用。** 改成「相对同口径对照」之后，我用两个
准确率各自的标准误（0.87pp）判断 −1.48pp「在噪声内」。错了——两个模型答的是
同一批题，必须配对。配对消掉题目难度的方差，标准误降到 0.65pp，那 −1.48pp 是
统计上真实的退化（McNemar 精确 p=0.025）。

## 现在的规则：非劣效检验，不是显著性检验

显著性不能当容差门槛：**n 越大越容易显著**。3300 题能测出 1pp 的差别，全量
26,368 题能测出 0.5pp——如果规则是「不显著才发」，那数据越多越发不出去，荒谬。

容差判断的正确形式是非劣效：先定可接受的损失上限 δ，再要求**损失的 95% 置信
上界不超过 δ**。这个判据不随 n 变严，只随 n 变准。

δ = 2.0pp，依据是既有先例而不是为这次结果挑的：已发布的 int4 变体相对同口径
bf16 的损失是 1.85pp，那就是这个项目已经对外认可过的容差。

纯度和退化率仍用绝对阈值——它们量的是「模型有没有被打坏」，不是精度权衡，
而且 117 条的分辨率只有 1/117=0.0085，撑不起细粒度检验。

用法：
  gate_check.py <purity.json> <mc.json>
  gate_check.py <purity.json> <mc.json> --control <对照_mc.json> [--delta 0.02]
"""
import argparse, json, math, sys

DELTA_DEFAULT = 0.02        # 非劣效容差 2.0pp，依据是已发布 int4 的 1.85pp
PURITY_MIN = 0.85
DEGEN_MAX = 0.10


def g(p, *ks):
    d = json.load(open(p, encoding="utf-8"))
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def flat_preds(path):
    """把按 config 存的紧凑串摊平，config 名排序，两份结果才能逐题配对。"""
    d = json.load(open(path, encoding="utf-8"))["mc"]
    gold, pred = [], []
    for cfg in sorted(d["preds"]):
        v = d["preds"][cfg]
        if len(v["gold"]) != len(v["pred"]):
            raise SystemExit(f"{path}:{cfg} gold/pred 长度不一致")
        gold += list(v["gold"])
        pred += list(v["pred"])
    return gold, pred


def binom_two_sided(b, c):
    """McNemar 精确检验，必须在对数空间算——n 上千时 comb(n,i)*0.5**n 会 OverflowError。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    ln2 = math.log(2.0)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * ln2
            for i in range(k + 1)]
    mx = max(logs)
    lse = mx + math.log(sum(math.exp(x - mx) for x in logs))
    return min(1.0, 2 * math.exp(lse)) if lse > -700 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("purity_json")
    ap.add_argument("mc_json")
    ap.add_argument("--control", default="",
                    help="同口径对照的 mc.json（一般是 bf16）。给了就做非劣效检验；"
                         "不给就退回绝对阈值，而绝对阈值只在同一口径内有意义")
    ap.add_argument("--delta", type=float, default=DELTA_DEFAULT,
                    help="非劣效容差，默认 0.02（2.0pp，依据已发布 int4 的 1.85pp）")
    a = ap.parse_args()

    pur = g(a.purity_json, "purity", "full", "mean_yue_ratio")
    deg = g(a.purity_json, "purity", "full", "degenerate_rate")
    acc = g(a.mc_json, "mc", "overall_acc")
    unp = g(a.mc_json, "mc", "unparsed_rate")
    otk = g(a.mc_json, "mc", "mean_out_tokens")
    if None in (pur, deg, acc):
        print("结果不全，无法判定：", dict(纯度=pur, 退化=deg, HKMMLU=acc))
        sys.exit(1)

    def row(name, val, op, th, ok):
        print("  %-12s %.4f   闸门 %s%.2f   %s" % (name, val, op, th, "通过" if ok else "不通过"))

    ok_pur = pur >= PURITY_MIN
    ok_deg = deg <= DEGEN_MAX
    row("书面粤语纯度", pur, ">=", PURITY_MIN, ok_pur)
    row("退化率", deg, "<=", DEGEN_MAX, ok_deg)
    print("  解析失败率   %.4f   平均输出 %.1f tok" % (unp or 0, otk or 0))

    if not a.control:
        ok_acc = acc >= 0.55
        row("HKMMLU", acc, ">=", 0.55, ok_acc)
        print("  ⚠️ 没给 --control：这个 0.55 是 logprob 口径的遗留阈值，跨口径不能用")
    else:
        gc_, pc = flat_preds(a.control)
        gv, pv = flat_preds(a.mc_json)
        if gc_ != gv:
            raise SystemExit("对照和变体的标准答案序列不一致，配不上对，判不了")
        n = len(gc_)
        a_ctl = sum(x == y for x, y in zip(gc_, pc)) / n
        a_var = sum(x == y for x, y in zip(gv, pv)) / n
        b = sum(1 for gg, x, y in zip(gc_, pc, pv) if y == gg and x != gg)   # 变体赢
        c = sum(1 for gg, x, y in zip(gc_, pc, pv) if x == gg and y != gg)   # 对照赢
        d = a_var - a_ctl
        se = math.sqrt(b + c) / n          # 配对差的标准误
        lo, hi = d - 1.96 * se, d + 1.96 * se
        p = binom_two_sided(b, c)
        worst = -lo                        # 损失的 95% 置信上界
        ok_acc = worst <= a.delta

        print()
        print("  HKMMLU 非劣效检验（同口径对照，逐题配对，n=%d）" % n)
        print("    对照 %.4f   变体 %.4f   Δ %+.4f" % (a_ctl, a_var, d))
        print("    配对 变体赢 %d / 对照赢 %d   McNemar 精确 p=%.4f" % (b, c, p))
        print("    95%%CI [%+.2f, %+.2f]pp   损失置信上界 %.2fpp   容差 δ=%.2fpp   %s"
              % (lo * 100, hi * 100, worst * 100, a.delta * 100,
                 "通过" if ok_acc else "不通过"))
        if p < 0.05 and ok_acc:
            print("    注：差异统计显著但在容差内。显著≠不可接受——n 越大越容易显著，")
            print("        所以这里用非劣效判据，不用显著性判据。")

    ok = ok_pur and ok_deg and ok_acc
    print("  ==>", "过闸门，可以发布" if ok else "未过闸门，不发布")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
