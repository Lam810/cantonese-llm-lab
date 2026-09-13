#!/usr/bin/env python3
"""重建粤语 SFT 数据集：多源合并 → 脚本归一 → 去重 → 固定 train/dev/test 切分。

v1 最大的缺陷是没有留出集，所以这里先切分再训练，切分种子固定、写进 stats。
粤语程度只做度量不做过滤（过滤阈值最容易把最该研究的样本丢掉），
要做消融再用 --min-yue-ratio。
"""
import argparse, glob, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict

YUE = ["嘅","係","唔","咗","喺","佢","哋","畀","乜","嘢","睇","攞","嚟","咁","啲","嗰","冇","咩","喇","啦","噉"]
ZH  = ["的","是","不","了","在","他","们","們","给","給","看","来","來","些","那","没","沒","吗","嗎","什","么","麼","这","這","吧"]

def markers(t):
    y = sum(t.count(m) for m in YUE); z = sum(t.count(m) for m in ZH)
    return y, z, (y/(y+z) if (y+z) else None)

def norm_ws(t):
    return re.sub(r"[ \t]+", " ", (t or "").strip())

# ---------- 各数据源读取 ----------
def read_jsonl(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def load_parquet_dir(d):
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True)):
        rows.append(pd.read_parquet(f))
    if not rows:
        import pandas as pd
        for f in sorted(glob.glob(os.path.join(d, "**", "*.json*"), recursive=True)):
            if "/.cache/" in f: continue
            if f.endswith(".jsonl"):
                rows.append(pd.DataFrame(read_jsonl(f)))
            elif f.endswith(".json"):
                try:                       # 可能是 JSON 数组
                    obj = json.load(open(f, encoding="utf-8"))
                    if isinstance(obj, dict): obj = obj.get("data") or obj.get("rows") or []
                    if obj: rows.append(pd.DataFrame(obj))
                except Exception:          # 也可能是每行一条
                    r = read_jsonl(f)
                    if r: rows.append(pd.DataFrame(r))
    if not rows: return []
    import pandas as pd
    return pd.concat(rows, ignore_index=True).to_dict("records")

def src_zeteng(path):
    for r in read_jsonl(path):
        yield {"instruction": r.get("instruction",""), "input": r.get("input",""),
               "output": r.get("output","")}

def src_alpaca(d):
    for r in load_parquet_dir(d):
        yield {"instruction": str(r.get("instruction") or ""), "input": str(r.get("input") or ""),
               "output": str(r.get("output") or "")}

def src_parallel(d, cap):
    rows = load_parquet_dir(d)
    random.Random(42).shuffle(rows)
    n = 0
    for r in rows:
        t = r.get("translation")
        if isinstance(t, dict):
            yue, zh = t.get("yue"), t.get("zh")
        else:
            yue, zh = r.get("yue"), r.get("zh")
        if not yue or not zh: continue
        if n >= cap: break
        n += 1
        if n % 2:
            yield {"instruction": "請將以下句子改寫成自然嘅香港口語粵語。", "input": str(zh), "output": str(yue)}
        else:
            yield {"instruction": "請將以下粵語句子改寫成書面中文。", "input": str(yue), "output": str(zh)}

SOURCES = {
  "zeteng-cantonese-llm-data": ("jsonl", "cantonese_finetune_alpaca_v2.jsonl", None),
  "indiejoseph-cantonese-cot": ("alpaca", "indiejoseph_cantonese-cot", 30000),
  "stvlynn-cantonese-dialogue": ("alpaca", "stvlynn_Cantonese-Dialogue", None),
  "raptorkwok-parallel": ("parallel", "raptorkwok_cantonese-chinese-parallel-corpus-base", 12000),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeteng-jsonl", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-yue-ratio", type=float, default=None, help="默认不过滤；只用于消融")
    ap.add_argument("--no-opencc", action="store_true")
    a = ap.parse_args()

    cc = None
    if not a.no_opencc:
        try:
            import opencc; cc = opencc.OpenCC("s2hk")      # 简体 → 香港繁体（只归一字形）
        except Exception as e:
            print("opencc 不可用，跳过脚本归一:", e, file=sys.stderr)

    records, stats = [], Counter()
    def add(source, it):
        for r in it:
            stats[f"{source}/raw"] += 1
            ins, inp, out = (norm_ws(r["instruction"]), norm_ws(r["input"]), norm_ws(r["output"]))
            if not ins or len(out) < 4: stats[f"{source}/drop_short"] += 1; continue
            if len(out) > 4000: stats[f"{source}/drop_long"] += 1; continue
            changed = False
            if cc:
                ins2, inp2, out2 = cc.convert(ins), cc.convert(inp), cc.convert(out)
                changed = (ins2, inp2, out2) != (ins, inp, out)
                ins, inp, out = ins2, inp2, out2
            y, z, ratio = markers(out)
            records.append({"source": source, "instruction": ins, "input": inp, "output": out,
                            "yue_markers": y, "zh_markers": z, "yue_ratio": ratio,
                            "script_normalized": changed})
            stats[f"{source}/kept"] += 1
            if changed: stats[f"{source}/script_normalized"] += 1

    add("zeteng-cantonese-llm-data", src_zeteng(a.zeteng_jsonl))
    add("indiejoseph-cantonese-cot", src_alpaca(os.path.join(a.raw_dir, "indiejoseph_cantonese-cot")))
    add("stvlynn-cantonese-dialogue", src_alpaca(os.path.join(a.raw_dir, "stvlynn_Cantonese-Dialogue")))
    add("raptorkwok-parallel", src_parallel(os.path.join(a.raw_dir, "raptorkwok_cantonese-chinese-parallel-corpus-base"), 12000))

    # cot 封顶，避免单一来源压过其它
    cap = {"indiejoseph-cantonese-cot": 30000}
    by_src = defaultdict(list)
    for r in records: by_src[r["source"]].append(r)
    rng = random.Random(a.seed)
    capped = []
    for s, rs in by_src.items():
        if s in cap and len(rs) > cap[s]:
            rng.shuffle(rs); stats[f"{s}/capped_out"] = len(rs) - cap[s]; rs = rs[:cap[s]]
        capped.extend(rs)
    records = capped

    # 去重：先整条去重，再按 instruction 去重（同题不同答只留一条）
    seen_full, seen_ins, dedup = set(), set(), []
    for r in records:
        hf = hashlib.md5((r["instruction"]+"\x00"+r["input"]+"\x00"+r["output"]).encode()).hexdigest()
        hi = hashlib.md5((r["instruction"]+"\x00"+r["input"]).encode()).hexdigest()
        if hf in seen_full: stats["dedup/exact"] += 1; continue
        if hi in seen_ins:  stats["dedup/same_instruction"] += 1; continue
        seen_full.add(hf); seen_ins.add(hi); dedup.append(r)
    records = dedup

    if a.min_yue_ratio is not None:
        before = len(records)
        records = [r for r in records if (r["yue_ratio"] or 0) >= a.min_yue_ratio]
        stats["filter/min_yue_ratio_dropped"] = before - len(records)

    # 按来源分层切分（先切再训，杜绝泄漏）
    rng = random.Random(a.seed)
    by_src = defaultdict(list)
    for r in records: by_src[r["source"]].append(r)
    test, dev, train = [], [], []
    total = len(records)
    for s, rs in by_src.items():
        rng.shuffle(rs)
        nt = max(1, round(a.n_test * len(rs) / total))
        nd = max(1, round(a.n_dev  * len(rs) / total))
        test += rs[:nt]; dev += rs[nt:nt+nd]; train += rs[nt+nd:]
    for split in (train, dev, test): rng.shuffle(split)

    os.makedirs(a.out_dir, exist_ok=True)
    def dump(name, rows):
        with open(os.path.join(a.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
        with open(os.path.join(a.out_dir, f"{name}.chat.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                user = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
                f.write(json.dumps({"messages":[{"role":"user","content":user},
                                                {"role":"assistant","content":r["output"]}]},
                                   ensure_ascii=False)+"\n")
    dump("train", train); dump("dev", dev); dump("test", test)

    ratios = [r["yue_ratio"] for r in records if r["yue_ratio"] is not None]
    ratios.sort()
    summary = {
      "seed": a.seed, "n_total": len(records),
      "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
      "per_source": {s: len(v) for s, v in by_src.items()},
      "yue_ratio": {"mean": sum(ratios)/len(ratios) if ratios else None,
                    "p25": ratios[len(ratios)//4] if ratios else None,
                    "median": ratios[len(ratios)//2] if ratios else None,
                    "p75": ratios[3*len(ratios)//4] if ratios else None,
                    "n_with_markers": len(ratios)},
      "counters": dict(stats), "opencc": "s2hk" if cc else None,
    }
    json.dump(summary, open(os.path.join(a.out_dir, "stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1)[:2000])

if __name__ == "__main__":
    main()
