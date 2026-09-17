#!/usr/bin/env python3
"""HKMMLU 官方口径评测：zero-shot prompting（生成），用 vLLM 跑。

为什么要另写一个：`eval_yue.py` 的主表用的是「比较 A/B/C/D 四个字母 token 的 logprob」，
那个口径绕开了指令遵循能力这个混淆变量，但**和官方榜单不可直接比**。
官方 HKMMLU 的标准是 zero-shot **prompting**（见数据集 README：
"Model performance on multi-choice tasks in HKMMLU using zero-shot prompting"）。
这个脚本按官方口径来，好把数字放到同一把尺子上。

两个任务：
  mc     选择题，全量（我们这份拷贝 26,368 题，官方 README 写 26,698，版本差异）
  trans  官方翻译任务 Can2Man / Man2Can —— 直接量粤语产出能力，选择题量不到

用法：
  eval_hkmmlu_official.py --model <路径> --name <名字> --task mc    --hkmmlu-dir <dir> --out x.json
  eval_hkmmlu_official.py --model <路径> --name <名字> --task trans --trans-dir <dir>  --out y.json
"""
import argparse, csv, glob, json, os, re, sys, time

# ---------- 答案解析 ----------
# 不能用 \b：中文字在 Unicode 下算 \w，「答案是A。」里 A 左边没有词边界
_ANS_RE = re.compile(r"(?:答案|answer|選|选)\s*(?:係|是|为|為)?\s*[:：]?\s*[（(\[]?\s*([ABCD])",
                     re.IGNORECASE)
_BARE_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")

def strip_think(t):
    """思维链模型会先输出 <think>…</think>，答案在那之后。"""
    if "</think>" in t:
        return t.split("</think>")[-1]
    return t

def parse_choice(text):
    body = strip_think(text)
    for rx in (_ANS_RE, _BARE_RE):
        m = list(rx.finditer(body))
        if m: return m[-1].group(1).upper()
    if body is not text:                      # think 没闭合，退回整段
        for rx in (_ANS_RE, _BARE_RE):
            m = list(rx.finditer(text))
            if m: return m[-1].group(1).upper()
    return None

# ---------- 繁简归一 ----------
# 为什么必须做：官方 can2man 的参考译文是**简体**，而我们的模型输出**繁体**。
# chrF 是字符级的，繁简不同会让每个共享字都判成不匹配。实际样例：
#   src 鐳射影碟喺香港同澳門地區仍然係一種常見嘅稱呼。
#   ref 镭射影碟在香港和澳门地区仍然是一种常见的称呼。
#   hyp 鐳射影碟在香港和澳門地區仍然是一種常見的稱呼。
# 语域转换全对（喺→在 同→和 係→是 嘅→的），只差字形，raw chrF 却接近 0。
# 所以两个都报：
#   chrF        照做不误——提示词明写了「簡體中文」，输出繁体就是没照做
#   chrF_t2s    hyp 和 ref 都归到简体再算，只量语域转换
# 两者的差 = 字形合规带来的分差，把「不会理解粤语」和「没按要求出简体」分开。
# 用 zh-hans（纯字形）而不是 zh-cn（还会换词汇），换词汇会把真实的用词差异抹掉。
try:
    from zhconv import convert as _zhconv
    def to_simp(t): return _zhconv(t, "zh-hans")
    HAS_ZHCONV = True
except Exception:
    def to_simp(t): return t
    HAS_ZHCONV = False

# ---------- 翻译输出清理 ----------
# 只做最小、对所有模型对称的清理。过度清理会不均匀地抬高某些模型的分。
_PREFIX_RE = re.compile(r"^\s*(?:譯文|译文|翻譯|翻译|粵語|粤语|普通話|普通话|答案|Translation)"
                        r"\s*[:：]\s*", re.IGNORECASE)

def clean_translation(text):
    t = strip_think(text).strip()
    t = _PREFIX_RE.sub("", t)
    t = t.split("\n\n")[0].strip()            # 取第一段，丢掉后面的解释
    t = t.strip("「」\"'“” ")
    return t

# ---------- 提示词 ----------
# 两种提示词，差别只有最后那一句。这是一个**单一变量的消融**，不是两种口径。
#
# strict   = 严格零样本，要一个字母、立刻给。这是我们对官方 "zero-shot
#            prompting" 的严格读法，主表用它。
# scaffold = 多一句「請喺最後一行寫「答案：X」」。这句话同时做了两件事：
#            (1) 给解析器一个锚点；(2) 允许模型先写推理再作答。
#            我们早期那张表用的就是它——**而它对我们的加成远大于对手**
#            （ours +8.3pp，最强对手 CantoneseLLMChat 只有 +1.5pp），因为
#            对手本来就 0 解析失败、不需要脚手架。把它当成"另一种口径"
#            报出去，等于用自己加的提示把自己的指令遵循短板盖住，所以这里
#            降级成可开关的消融项，主表只用 strict。
_MC_BODY = ("以下是一道單項選擇題，請直接回答選項字母。\n\n"
            "{q}\n"
            "A. {a}\n"
            "B. {b}\n"
            "C. {c}\n"
            "D. {d}\n\n")

# legacy = 完整复刻 `eval_yue.py:241-243` 那个旧提示词。它和 strict 差三处，
#          不止末尾那句脚手架：
#            (1) 整段是**书面粤语**（「以下係」「請答」），strict 是标准书面中文
#            (2) 题目前面加了「題目：」
#            (3) 明确列出「A、B、C 或 D」
#          (1) 尤其要单独量：给粤语专化的模型喂粤语提示词，本身可能就是一个
#          真实的增益来源，而不是作弊。把它和 scaffold 分开，才知道旧那个
#          0.6227 里各占多少。
_MC_BODY_YUE = ("以下係一道香港知識選擇題，請答 A、B、C 或 D。\n\n"
                "題目：{q}\n"
                "A. {a}\n"
                "B. {b}\n"
                "C. {c}\n"
                "D. {d}\n\n")

MC_PROMPTS = {
    "strict":   _MC_BODY + "答案：",
    "scaffold": _MC_BODY + "請喺最後一行寫「答案：X」。",
    "legacy":   _MC_BODY_YUE + "請喺最後一行寫「答案：X」。",
    # 只换语言、不加脚手架：把「粤语提示词」这一个变量单独摘出来
    "yue_strict": _MC_BODY_YUE + "答案：",
}
MC_PROMPT = MC_PROMPTS["strict"]        # 兼容旧调用

TRANS_PROMPT = {
    "can2man": "請將以下粵語句子翻譯成普通話（簡體中文）。只輸出譯文，不要加任何解釋。\n\n{s}",
    "man2can": "請將以下普通話句子翻譯成粵語（繁體中文）。只輸出譯文，不要加任何解釋。\n\n{s}",
}

def load_mc(data_dir, limit_per_cfg=0, style="strict"):
    tmpl = MC_PROMPTS[style]
    files = sorted(glob.glob(os.path.join(data_dir, "**", "test", "*.csv"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*test*.csv"), recursive=True))
    items = []
    letters = ["A", "B", "C", "D"]
    for f in files:
        cfg = os.path.splitext(os.path.basename(f))[0]
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if limit_per_cfg: rows = rows[:limit_per_cfg]
        for r in rows:
            q = (r.get("Question") or r.get("question") or "").strip()
            opts = [(r.get(L) or "").strip() for L in letters]
            gold = (r.get("Answer") or r.get("answer") or "").strip().upper()[:1]
            if not q or gold not in letters: continue
            items.append({"cfg": cfg, "gold": gold,
                          "prompt": tmpl.format(q=q, a=opts[0], b=opts[1],
                                                c=opts[2], d=opts[3])})
    return items

def load_trans(trans_dir, n=0):
    out = {}
    for tag in ("can2man", "man2can"):
        p = os.path.join(trans_dir, f"{tag}_2000.jsonl")
        if not os.path.exists(p): continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        if n: rows = rows[:n]
        out[tag] = [{"prompt": TRANS_PROMPT[tag].format(s=r["input"]),
                     "ref": r["output"], "src": r["input"]} for r in rows]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--task", choices=["mc", "trans", "purity"], required=True)
    ap.add_argument("--quantization", default="",
                    help="传给 vLLM 的量化方法。昇腾原生格式要传 ascend——"
                         "注意 config.json 里**不能**有 quantization_config，"
                         "否则 vLLM 拿它当 quant_description、不再读 "
                         "quant_model_description.json，查 embed_tokens 直接 KeyError")
    ap.add_argument("--hkmmlu-dir", default=""); ap.add_argument("--trans-dir", default="")
    ap.add_argument("--limit-per-cfg", type=int, default=0)
    ap.add_argument("--mc-style",
                    choices=["strict", "scaffold", "legacy", "yue_strict"],
                    default="strict",
                    help="选择题提示词。strict=主表口径（只要字母）；scaffold=多一句「請喺最後一行寫「答案：X」」，"
                         "用来消融「格式脚手架值多少分」——它不是另一种口径，别拿它报战绩")
    ap.add_argument("--trans-n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="思维链模型要留够，否则会在 think 块中间截断、答案根本没出来")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--purity-limit", type=int, default=0)
    ap.add_argument("--purity-max-new", type=int, default=128)
    ap.add_argument("--save-hyps", action="store_true",
                    help="把全部译文写成 <out>.<方向>.hyps.jsonl，省得为了换个打分口径重跑模型")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    kw = dict(model=a.model, tensor_parallel_size=1, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_util, trust_remote_code=True)
    if a.quantization: kw["quantization"] = a.quantization
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)   # 贪心，可复算
    _tk = llm.get_tokenizer()
    tok_enc = lambda s: _tk.encode(s, add_special_tokens=False)

    res = {"name": a.name, "model": a.model, "task": a.task,
           "protocol": "HKMMLU official: zero-shot prompting (generation), greedy",
           "mc_style": a.mc_style,
           "mc_prompt_template": MC_PROMPTS[a.mc_style],
           "trans_prompt_template": TRANS_PROMPT,
           "max_tokens": a.max_tokens}
    t0 = time.time()

    if a.task == "mc":
        items = load_mc(a.hkmmlu_dir, a.limit_per_cfg, a.mc_style)
        print(f"[{a.name}] MC 题数 {len(items)}", flush=True)
        outs = llm.chat([[{"role": "user", "content": it["prompt"]}] for it in items], sp)
        per_cfg, preds = {}, {}
        n_all = n_ok = n_unparsed = tok_sum = 0
        for it, o in zip(items, outs):
            txt = o.outputs[0].text
            tok_sum += len(o.outputs[0].token_ids)
            pred = parse_choice(txt)
            c = per_cfg.setdefault(it["cfg"], {"c": 0, "n": 0, "u": 0})
            c["n"] += 1; n_all += 1
            if pred is None: c["u"] += 1; n_unparsed += 1
            if pred == it["gold"]: c["c"] += 1; n_ok += 1
            d = preds.setdefault(it["cfg"], {"gold": "", "pred": ""})
            d["gold"] += it["gold"]; d["pred"] += (pred or "?")
        for k, v in per_cfg.items(): v["acc"] = v["c"] / v["n"]
        hk = {k: v for k, v in per_cfg.items() if k.startswith("hk_")}
        dse = {k: v for k, v in per_cfg.items() if k.startswith("hkdse_")}
        agg = lambda d: (sum(v["acc"] * v["n"] for v in d.values()) / sum(v["n"] for v in d.values())
                         if d else None)
        res["mc"] = {"overall_acc": n_ok / n_all, "n": n_all,
                     "macro_acc": sum(v["acc"] for v in per_cfg.values()) / len(per_cfg),
                     "unparsed": n_unparsed, "unparsed_rate": n_unparsed / n_all,
                     "mean_out_tokens": tok_sum / n_all,
                     "hk_subset_acc": agg(hk), "hkdse_subset_acc": agg(dse),
                     "random_baseline": 0.25, "per_config": per_cfg, "preds": preds}
        m = res["mc"]
        print(f"[{a.name}] MC acc={m['overall_acc']:.4f} macro={m['macro_acc']:.4f} "
              f"解析失败={m['unparsed_rate']:.4f} 平均输出tok={m['mean_out_tokens']:.1f}", flush=True)

    elif a.task == "purity":
        # 提问集与标记词表从 eval_yue 导入，不另抄一份——抄一份就不可比了
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from eval_yue import PURITY_PROMPTS, YUE_MARKERS, ZH_MARKERS, count_markers
        prompts = PURITY_PROMPTS[:a.purity_limit] if a.purity_limit else PURITY_PROMPTS
        sp2 = SamplingParams(temperature=0.0, max_tokens=a.purity_max_new)
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp2)
        recs = []
        for p, o in zip(prompts, outs):
            raw = o.outputs[0].text
            body = strip_think(raw)
            y, z = count_markers(body, YUE_MARKERS), count_markers(body, ZH_MARKERS)
            recs.append({"prompt": p, "response": body[:400], "yue": y, "zh": z,
                         "ratio": (y / (y + z)) if (y + z) else None,
                         "len": len(body), "out_tokens": len(o.outputs[0].token_ids),
                         "think_tokens": (len(tok_enc(raw.split("</think>")[0]))
                                          if "</think>" in raw else 0),
                         "degenerate": bool(re.search(r"(.{6,})\1{2,}", body))})
        def blk(sub):
            v = [r for r in sub if r["ratio"] is not None]
            med = lambda k: sorted(r[k] for r in sub)[len(sub) // 2] if sub else None
            return {"n_prompts": len(sub),
                    "mean_yue_ratio": (sum(r["ratio"] for r in v) / len(v)) if v else None,
                    "median_len": med("len"), "median_out_tokens": med("out_tokens"),
                    "median_think_tokens": med("think_tokens"),
                    "degenerate_rate": (sum(r["degenerate"] for r in sub) / len(sub)) if sub else None}
        res["purity"] = {"first30": blk(recs[:30]), "full": blk(recs), "samples": recs[:8]}
        f30, fl = res["purity"]["first30"], res["purity"]["full"]
        print(f"[{a.name}] 纯度(全{fl['n_prompts']}条)={fl['mean_yue_ratio']} "
              f"退化={fl['degenerate_rate']} 中位tok={fl['median_out_tokens']} | "
              f"前30条纯度={f30['mean_yue_ratio']}", flush=True)

    elif a.task == "trans":
        import sacrebleu
        data = load_trans(a.trans_dir, a.trans_n)
        res["trans"] = {}
        for tag, items in data.items():
            outs = llm.chat([[{"role": "user", "content": it["prompt"]}] for it in items], sp)
            hyps, refs, raws, tok_sum = [], [], [], 0
            for it, o in zip(items, outs):
                raw = o.outputs[0].text
                tok_sum += len(o.outputs[0].token_ids)
                hyps.append(clean_translation(raw)); refs.append(it["ref"]); raws.append(raw)
            chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score
            # 繁简归一后再算一遍：只量语域转换，不罚字形
            hs, rs = [to_simp(h) for h in hyps], [to_simp(r) for r in refs]
            chrf_t2s = sacrebleu.corpus_chrf(hs, [rs]).score
            bleu_t2s = sacrebleu.corpus_bleu(hs, [rs], tokenize="zh").score
            # 输出里有多少条带繁体字（归一后发生了变化）= 没按「簡體中文」这条指令做
            trad = sum(1 for h, x in zip(hyps, hs) if h != x)
            empty = sum(1 for h in hyps if not h)
            res["trans"][tag] = {
                "n": len(hyps), "chrF": chrf, "BLEU_zh": bleu,
                "chrF_t2s": chrf_t2s, "BLEU_zh_t2s": bleu_t2s,
                "zhconv_available": HAS_ZHCONV,
                "trad_output": trad, "trad_rate": trad / len(hyps),
                "empty_output": empty, "empty_rate": empty / len(hyps),
                "mean_out_tokens": tok_sum / len(hyps),
                "samples": [{"src": items[i]["src"], "ref": refs[i], "hyp": hyps[i],
                             "raw": raws[i][:300]} for i in range(min(5, len(hyps)))]}
            # 全部译文单独落盘：再分析时不必重跑模型
            if a.save_hyps:
                hp = a.out.replace(".json", f".{tag}.hyps.jsonl")
                with open(hp, "w", encoding="utf-8") as fh:
                    for i in range(len(hyps)):
                        fh.write(json.dumps({"src": items[i]["src"], "ref": refs[i],
                                             "hyp": hyps[i]}, ensure_ascii=False) + "\n")
                print(f"[{a.name}] 译文已存 {hp}", flush=True)
            print(f"[{a.name}] {tag} chrF={chrf:.2f} BLEU={bleu:.2f} "
                  f"空输出={empty} 平均tok={tok_sum/len(hyps):.1f}", flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("WROTE", a.out, f"({res['elapsed_s']}s)", flush=True)

if __name__ == "__main__":
    main()
