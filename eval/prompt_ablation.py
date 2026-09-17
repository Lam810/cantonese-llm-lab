#!/usr/bin/env python3
"""提示词消融：strict vs scaffold，逐模型、逐题配对。

背景。我们早期那张把自己排第一的 HKMMLU 表，用的提示词末尾多一句
「請喺最後一行寫「答案：X」」。换成严格零样本（只要字母）之后我们掉到第 4。
两种提示词的差值在 ours 身上是 +8.3pp，在最强对手身上只有 +1.5pp。

只有两个数字说明不了问题——可能脚手架帮了所有人，只是帮我们更多；也可能
是别的东西变了。所以这里把提示词做成**唯一变量**（同题、同题量、同解析器、
同采样参数、同 max_tokens）全量重跑，然后：

  1) 每个模型自己 strict vs scaffold 逐题配对（McNemar 精确检验）
  2) 把「脚手架值多少分」在模型之间排序，看是不是我们吃得最多
  3) 拆开来看：涨的分里有多少只是「原来解析不出来」

第 3 条是关键。如果增量主要来自 unparsed 变成可解析，那它量的是格式服从，
不是知识；如果 unparsed 早就是 0 还在涨，那是推理空间带来的真实增益。
"""
import json, math, os, sys


def binom_two_sided(b, c):
    """McNemar 精确检验。必须在对数空间算。

    直接 sum(math.comb(n,i)) * 0.5**n 在 n 上千时会炸：组合数是天文数字的
    大整数，而 0.5**n 已经下溢成 0.0，相乘直接 OverflowError。
    """
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


def load(path):
    """把按 config 存的紧凑串摊平成一条题目序列，顺序与 CSV 一致。"""
    d = json.load(open(path, encoding="utf-8"))["mc"]
    gold, pred = [], []
    for cfg in sorted(d["preds"]):
        v = d["preds"][cfg]
        g, p = v["gold"], v["pred"]
        if len(g) != len(p):
            raise SystemExit(f"{path}:{cfg} gold/pred 长度不一致 {len(g)}/{len(p)}")
        gold += list(g)
        pred += list(p)
    return d, gold, pred


def main():
    if len(sys.argv) != 3:
        raise SystemExit("用法: prompt_ablation.py <strict目录> <scaffold目录>")
    a_dir, b_dir = sys.argv[1], sys.argv[2]

    names = sorted(f[:-len("_mc.json")] for f in os.listdir(a_dir) if f.endswith("_mc.json"))
    rows = []
    for n in names:
        pa, pb = os.path.join(a_dir, f"{n}_mc.json"), os.path.join(b_dir, f"{n}_mc.json")
        if not os.path.exists(pb):
            print(f"· {n}: scaffold 侧没有结果，跳过")
            continue
        da, ga, sa = load(pa)
        db, gb, sb = load(pb)
        if ga != gb:
            print(f"!! {n}: 两轮的标准答案序列不一致，配对不成立，跳过")
            continue

        n_all = len(ga)
        acc_a = sum(x == y for x, y in zip(ga, sa)) / n_all
        acc_b = sum(x == y for x, y in zip(gb, sb)) / n_all
        u_a = sa.count("?") / n_all
        u_b = sb.count("?") / n_all

        # b = scaffold 对 / strict 错；c = 反过来
        b = sum(1 for g, x, y in zip(ga, sa, sb) if y == g and x != g)
        c = sum(1 for g, x, y in zip(ga, sa, sb) if x == g and y != g)
        p = binom_two_sided(b, c)

        # 增量里有多少是「strict 下根本没解析出来」的题贡献的
        from_unparsed = sum(1 for g, x, y in zip(ga, sa, sb) if x == "?" and y == g)
        share = from_unparsed / b if b else float("nan")

        rows.append(dict(name=n, n=n_all, acc_a=acc_a, acc_b=acc_b, d=acc_b - acc_a,
                         u_a=u_a, u_b=u_b, b=b, c=c, p=p,
                         from_unparsed=from_unparsed, share=share))

    rows.sort(key=lambda r: -r["d"])
    print()
    print("=== 提示词消融：strict（只要字母） → scaffold（多一句「請喺最後一行寫「答案：X」」）===")
    print(f"{'模型':24s} {'strict':>8s} {'scaffold':>9s} {'脚手架值':>9s} "
          f"{'未解析 s→c':>12s} {'翻对/翻错':>11s} {'p':>10s} {'增量中来自未解析':>16s}")
    print("-" * 110)
    for r in rows:
        share = "—" if r["b"] == 0 else f"{r['share']:.0%}"
        print(f"{r['name']:24s} {r['acc_a']:8.4f} {r['acc_b']:9.4f} {r['d']:+9.4f} "
              f"{r['u_a']:6.4f}→{r['u_b']:5.4f} {r['b']:5d}/{r['c']:5d} "
              f"{r['p']:10.2e} {share:>16s}")
    print()
    print("脚手架值 = scaffold 准确率 − strict 准确率（同题配对，唯一变量是提示词末尾那句话）。")
    print("「增量中来自未解析」= 在 scaffold 下答对、而 strict 下连字母都没解析出来的题，")
    print("占全部「翻对」题的比例。这个比例高 = 涨的分主要是格式服从，不是知识。")

    if rows:
        top = rows[0]
        print()
        print(f"吃脚手架最多的是 {top['name']}（{top['d']:+.4f}），"
              f"最少的是 {rows[-1]['name']}（{rows[-1]['d']:+.4f}）。")


if __name__ == "__main__":
    main()
