#!/usr/bin/env python3
"""提示词 2×2：语言（标准书面中文 / 书面粤语）× 格式脚手架（无 / 有）。

                     标准书面中文主体      粤语主体
    无脚手架         strict（主表）        yue_strict
    有脚手架         scaffold              legacy（= 旧表用的那个）

## 为什么要做满这个格子

早先那张把本项目排第一的 HKMMLU 表报 0.6227，官方严格零样本口径只有 0.5394，
差 8.33pp。已经查清两项：

  · scaffold 那句「請喺最後一行寫「答案：X」」  +3.00pp（全量配对实测）
  · 「每科前 10 题」子集偏差                    +0.91pp

还剩约 4.4pp 没交代。旧提示词的**主体本身也不一样，而且整段是书面粤语写的**
（「以下係一道香港知識選擇題，請答 A、B、C 或 D」），这一项必须单独量。

关键是：**给粤语专化的模型喂粤语提示词，可能是真实增益，不是作弊。** 如果
粤语提示词对所有模型都有帮助，那它是个中立的提示词选择；如果只帮我们，那它
和格式脚手架一样，是把自己的短板藏起来。不做满 2×2 就分不清这两种情况。

## 读法

  语言主效应   = (yue_strict − strict) 和 (legacy − scaffold) 的平均
  脚手架主效应 = (scaffold − strict) 和 (legacy − yue_strict) 的平均
  交互         = (legacy − yue_strict) − (scaffold − strict)

所有比较都是同题配对（McNemar 精确检验），因为四臂答的是同一批 26,368 题。
"""
import argparse, json, math, os


def binom_two_sided(b, c):
    """McNemar 精确检验，对数空间——n 上千时 comb(n,i)*0.5**n 会 OverflowError。"""
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


def flat(path):
    d = json.load(open(path, encoding="utf-8"))["mc"]
    gold, pred = [], []
    for cfg in sorted(d["preds"]):
        v = d["preds"][cfg]
        if len(v["gold"]) != len(v["pred"]):
            raise SystemExit(f"{path}:{cfg} gold/pred 长度不一致")
        gold += list(v["gold"])
        pred += list(v["pred"])
    return gold, pred, d


def paired(gold, pa, pb):
    """返回 (acc_a, acc_b, delta, se, p, b, c)。b = B 对 A 错。"""
    n = len(gold)
    aa = sum(x == y for x, y in zip(gold, pa)) / n
    ab = sum(x == y for x, y in zip(gold, pb)) / n
    b = sum(1 for g, x, y in zip(gold, pa, pb) if y == g and x != g)
    c = sum(1 for g, x, y in zip(gold, pa, pb) if x == g and y != g)
    return aa, ab, ab - aa, math.sqrt(b + c) / n, binom_two_sided(b, c), b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", required=True)
    ap.add_argument("--scaffold", required=True)
    ap.add_argument("--legacy", required=True)
    ap.add_argument("--yue-strict", required=True)
    a = ap.parse_args()
    dirs = {"strict": a.strict, "scaffold": a.scaffold,
            "legacy": a.legacy, "yue_strict": a.yue_strict}

    names = sorted(f[:-len("_mc.json")] for f in os.listdir(a.strict)
                   if f.endswith("_mc.json"))
    rows = []
    for n in names:
        paths = {k: os.path.join(d, f"{n}_mc.json") for k, d in dirs.items()}
        miss = [k for k, p in paths.items() if not os.path.exists(p)]
        if miss:
            print(f"· {n}: 缺 {miss}，跳过")
            continue
        got = {k: flat(p) for k, p in paths.items()}
        golds = {k: v[0] for k, v in got.items()}
        if len({tuple(g) for g in golds.values()}) != 1:
            print(f"!! {n}: 四臂的标准答案序列不一致，配对不成立，跳过")
            continue
        gold = golds["strict"]
        P = {k: v[1] for k, v in got.items()}
        D = {k: v[2] for k, v in got.items()}
        acc = {k: sum(x == y for x, y in zip(gold, P[k])) / len(gold) for k in dirs}
        unp = {k: P[k].count("?") / len(gold) for k in dirs}

        # 两个语言对比（分别在无/有脚手架下）
        _, _, d_lang_0, se_l0, p_l0, *_ = paired(gold, P["strict"], P["yue_strict"])
        _, _, d_lang_1, se_l1, p_l1, *_ = paired(gold, P["scaffold"], P["legacy"])
        # 两个脚手架对比（分别在中文/粤语提示词下）
        _, _, d_sc_0, se_s0, p_s0, *_ = paired(gold, P["strict"], P["scaffold"])
        _, _, d_sc_1, se_s1, p_s1, *_ = paired(gold, P["yue_strict"], P["legacy"])

        rows.append(dict(name=n, n=len(gold), acc=acc, unp=unp,
                         lang=(d_lang_0 + d_lang_1) / 2, scaf=(d_sc_0 + d_sc_1) / 2,
                         inter=d_sc_1 - d_sc_0,
                         d_lang_0=d_lang_0, p_l0=p_l0, d_lang_1=d_lang_1, p_l1=p_l1,
                         d_sc_0=d_sc_0, p_s0=p_s0, d_sc_1=d_sc_1, p_s1=p_s1,
                         tok={k: D[k]["mean_out_tokens"] for k in dirs}))

    if not rows:
        raise SystemExit("四臂都齐的模型一个都没有")

    print()
    print(f"=== 四臂准确率（全量 {rows[0]['n']:,} 题，同题配对）===")
    print(f"{'模型':24s} {'strict':>8s} {'scaffold':>9s} {'yue_strict':>11s} {'legacy':>8s} "
          f"{'最好的臂':>12s}")
    print("-" * 82)
    for r in sorted(rows, key=lambda x: -x["acc"]["strict"]):
        best = max(r["acc"], key=lambda k: r["acc"][k])
        print(f"{r['name']:24s} {r['acc']['strict']:8.4f} {r['acc']['scaffold']:9.4f} "
              f"{r['acc']['yue_strict']:11.4f} {r['acc']['legacy']:8.4f} "
              f"{best:>12s}")

    print()
    print("=== 主效应与交互（正 = 该因素提高准确率）===")
    print(f"{'模型':24s} {'语言(粤>中)':>12s} {'脚手架(有>无)':>14s} {'交互':>9s}   "
          f"{'语言@无脚手架':>14s} {'语言@有脚手架':>14s}")
    print("-" * 100)
    for r in sorted(rows, key=lambda x: -x["lang"]):
        print(f"{r['name']:24s} {r['lang']*100:+12.2f} {r['scaf']*100:+14.2f} "
              f"{r['inter']*100:+9.2f}   "
              f"{r['d_lang_0']*100:+8.2f}(p={r['p_l0']:.1e}) "
              f"{r['d_lang_1']*100:+8.2f}(p={r['p_l1']:.1e})")

    print()
    print("=== 解析失败率：四臂 ===")
    print(f"{'模型':24s} {'strict':>8s} {'scaffold':>9s} {'yue_strict':>11s} {'legacy':>8s}")
    print("-" * 66)
    for r in sorted(rows, key=lambda x: -x["unp"]["strict"]):
        print(f"{r['name']:24s} {r['unp']['strict']:8.4f} {r['unp']['scaffold']:9.4f} "
              f"{r['unp']['yue_strict']:11.4f} {r['unp']['legacy']:8.4f}")

    print()
    print("判读要点：")
    print("  · 语言主效应如果**只对本项目为正**，那粤语提示词和格式脚手架一样，")
    print("    是在给自己加别人没有的助力，主表不能用它。")
    print("  · 如果对**所有模型**都为正，那它是一个中立且更合适的提示词选择，")
    print("    可以作为并列口径公布——但必须四臂全公开，不能只报对自己有利的那一臂。")
    print("  · 交互项大 = 两个因素不可加，不能把 8.33pp 简单拆成几项之和。")


if __name__ == "__main__":
    main()
