#!/usr/bin/env python3
"""两个模型在同一份 HKMMLU 上的配对比较。

为什么不用「两个独立比例的 z 检验」：所有模型答的是同一批题，对错是**配对**的。
独立检验会把题目难度的方差算进去，白白损失功效——同样 3300 题，配对检验能分辨
1pp 级别的差异，独立检验分辨不了。

用法：
  compare_models.py A.json B.json [C.json ...]        # 两两对比，第一个作基准
"""
import json, math, sys, random

def flat(d, key="hkmmlu"):
    """把逐题预测摊平成 {题目id: 对/错}。id 用 cfg#序号，跨文件对得上。"""
    h = d.get(key) or {}
    preds = h.get("preds") or {}
    out = {}
    for cfg, v in preds.items():
        for i, (g, p) in enumerate(zip(v["gold"], v["pred"])):
            out[f"{cfg}#{i}"] = int(g == p)
    return out

def binom_two_sided(b, c):
    """精确二项检验：在 b+c 个「两边不一致」的题里，b 个偏向 A 是否偏离 50/50。

    必须在对数空间算。直接 sum(math.comb(n,i)) * 0.5**n 在 n 上千时会炸：
    组合数是天文数字的大整数，而 0.5**n 已经下溢成 0.0，相乘直接 OverflowError。
    （这是自测造出 n≈1500 的配对时抓到的。）"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    ln2 = math.log(2.0)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * ln2
            for i in range(k + 1)]
    mx = max(logs)
    lse = mx + math.log(sum(math.exp(x - mx) for x in logs))
    return min(1.0, 2 * math.exp(lse)) if lse > -700 else 0.0

def mcnemar(a, b):
    keys = sorted(set(a) & set(b))
    if not keys: return None
    n01 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)   # A 对 B 错
    n10 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)   # A 错 B 对
    acc_a = sum(a[k] for k in keys) / len(keys)
    acc_b = sum(b[k] for k in keys) / len(keys)
    chi2 = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10) if (n01 + n10) else 0.0
    # 配对差的 bootstrap 区间
    rng = random.Random(0); diffs = [a[k] - b[k] for k in keys]; N = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(N)] for _ in range(N)) / N for _ in range(2000))
    return {"n_paired": len(keys), "acc_a": acc_a, "acc_b": acc_b,
            "delta": acc_a - acc_b,
            "delta_ci95": [boots[50], boots[1949]],
            "A对B错": n01, "A错B对": n10,
            "chi2_cc": chi2, "p_exact": binom_two_sided(n01, n10)}

def main():
    files = sys.argv[1:]
    if len(files) < 2: sys.exit(__doc__)
    ds = [(f, json.load(open(f, encoding="utf-8"))) for f in files]
    base_f, base_d = ds[0]
    base = flat(base_d)
    if not base: sys.exit(f"{base_f} 里没有逐题预测（preds）——要用新版 eval_yue.py 重跑")
    print(f"基准：{base_d.get('name')}   n={len(base)}   ({base_f})")
    hdr = "%-32s %9s %20s %8s %10s %10s  %s" % (
        "对照", "Δacc", "95%CI", "配对n", "A胜/A负", "p(精确)", "判定")
    print(hdr); print("-" * len(hdr))
    out = {}
    for f, d in ds[1:]:
        o = flat(d)
        name = d.get("name", "?")
        if not o:
            print("%-32s  (没有逐题预测，跳过)" % name); continue
        r = mcnemar(base, o)
        if r is None:
            print("%-32s  (题目对不上)" % name); continue
        out[name] = r
        ci = "[%+.2f,%+.2f]" % (r["delta_ci95"][0]*100, r["delta_ci95"][1]*100)
        wl = "%d/%d" % (r["A对B错"], r["A错B对"])
        sig = "显著" if r["p_exact"] < 0.05 else "不显著"
        print("%-32s %+8.2fpp %20s %8d %10s %10.2e  %s" % (
            name, r["delta"]*100, ci, r["n_paired"], wl, r["p_exact"], sig))
    print()
    print("Δacc 为正 = 基准更强。p 用精确二项检验（McNemar 的精确版），")
    print("只看两边答得不一致的题；配对能把题目难度的方差消掉。")
    json.dump(out, open("compare_models_out.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
