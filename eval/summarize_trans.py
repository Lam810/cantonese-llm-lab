#!/usr/bin/env python3
"""汇总官方翻译任务：raw chrF 和繁简归一后的 chrF 并排。

为什么要两列。官方 Can2Man 的参考译文是**简体**中文，而几个粤语模型（包括我们的）
输出**繁体**。chrF 是字符级指标，繁简不同会让语域已经转对的句子拿到接近 0 的分。
实测：

    src 鐳射影碟喺香港同澳門地區仍然係一種常見嘅稱呼。
    ref 镭射影碟在香港和澳门地区仍然是一种常见的称呼。
    hyp 鐳射影碟在香港和澳門地區仍然是一種常見的稱呼。

喺→在、同→和、係→是、嘅→的 全部转对了，只差字形。

  chrF      照做不误。提示词明写「簡體中文」，输出繁体就是没照做，该罚。
  chrF_t2s  hyp 和 ref 都归到简体再算，只量语域转换能力。

两者的差 = 字形合规值多少分。**不要只报对自己有利的那一列。**

归一用 zh-hans（纯字形）而不是 zh-cn（还会换词汇）：换词汇会把真实的用词差异
一起抹掉。注意 `嘅`/`唔`/`喺` 这些粤语虚词没有简体形式，归一后仍然在，所以
归一后的 chrF 依然会罚语域错误——这正是我们要的。
"""
import json, os, sys

DIRS = [("can2man", "粤→普"), ("man2can", "普→粤")]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("用法: summarize_trans.py <结果目录>")
    root = sys.argv[1]
    rows = []
    for f in sorted(os.listdir(root)):
        if not f.endswith("_trans.json"):
            continue
        d = json.load(open(os.path.join(root, f), encoding="utf-8"))["trans"]
        rows.append((f[:-len("_trans.json")], d))
    if not rows:
        raise SystemExit(f"{root} 里没有 *_trans.json")

    missing = [n for n, d in rows
               if any(not d[t].get("zhconv_available", False) for t, _ in DIRS if t in d)]
    if missing:
        print(f"⚠️ 这些模型跑的时候 zhconv 没装上，chrF_t2s 等于 raw，不可用：{missing}")

    for tag, label in DIRS:
        sub = [(n, d[tag]) for n, d in rows if tag in d]
        if not sub:
            continue
        sub.sort(key=lambda x: -x[1]["chrF_t2s"])
        print()
        print(f"=== {label}（{tag}，各 {sub[0][1]['n']} 条，seed=42）===")
        print(f"{'模型':24s} {'chrF':>7s} {'chrF_t2s':>9s} {'字形分差':>9s} "
              f"{'繁体输出率':>10s} {'BLEU':>7s} {'BLEU_t2s':>9s} {'空输出':>6s} {'平均tok':>8s}")
        print("-" * 104)
        for n, v in sub:
            print(f"{n:24s} {v['chrF']:7.2f} {v['chrF_t2s']:9.2f} "
                  f"{v['chrF_t2s'] - v['chrF']:+9.2f} {v.get('trad_rate', 0):10.3f} "
                  f"{v['BLEU_zh']:7.2f} {v['BLEU_zh_t2s']:9.2f} "
                  f"{v['empty_output']:6d} {v['mean_out_tokens']:8.1f}")

    print()
    print("字形分差 = chrF_t2s − chrF。这一列大 = 该模型的失分里有很大一块只是繁简，")
    print("不是不会转换语域。繁体输出率是「输出里含繁体字的句子占比」。")
    print()
    print("读法提醒：普→粤 方向参考译文本身是繁体，归一到简体对两边同时生效，")
    print("所以那一列的字形分差应该很小——如果不小，说明有模型在该出繁体时出了简体。")


if __name__ == "__main__":
    main()
