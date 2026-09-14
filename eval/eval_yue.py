#!/usr/bin/env python3
"""粤语模型评测：HKMMLU 选择题 / 书面粤语纯度 / 分布内外困惑度。

三个任务都不依赖 LLM 判官，全部是可复算的客观量。
用法：eval_yue.py --model <path> --name <tag> --tasks hkmmlu,purity,ppl --out results/<tag>.json
"""
import argparse, csv, glob, json, math, os, re, sys, time
import torch
try:                      # 昇腾 NPU：import 之后 torch.npu 才存在
    import torch_npu      # noqa: F401
    _HAS_NPU = torch.npu.is_available()
except Exception:
    _HAS_NPU = False
from transformers import AutoModelForCausalLM, AutoTokenizer

def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if _HAS_NPU: return "npu"
    return "cpu"

# ---------- 书面粤语 / 书面普通话 标记词 ----------
YUE_MARKERS = ["嘅","係","唔","咗","喺","佢","哋","畀","乜","嘢","睇","攞","嚟","咁","啲","嗰","邊個","乜嘢","點解","冇","梗係","咩","喇","啦","噉","唔係","係唔係","可以","譬如"][:24]
ZH_MARKERS  = ["的","是","不","了","在","他","们","們","给","給","看","来","來","些","那","没","沒","吗","嗎","什","么","麼","这","這","吧"]
# 只保留互斥性强的（"可以/譬如"两词粤普通用，剔掉）
YUE_MARKERS = ["嘅","係","唔","咗","喺","佢","哋","畀","乜","嘢","睇","攞","嚟","咁","啲","嗰","冇","咩","喇","啦","噉"]

PURITY_PROMPTS = [
 "香港有咩好玩嘅地方？","點樣由旺角去中環最快？","你可唔可以解釋下咩係強積金？",
 "推薦幾間抵食嘅茶餐廳啦。","八達通冇錢點算好？","香港嘅公屋申請資格係點？",
 "介紹下香港嘅天氣，夏天熱唔熱？","我想學煮餸，有咩簡單嘅餸式推薦？",
 "點解香港樓價咁貴？","香港嘅巴士同小巴有咩分別？","中秋節香港人通常做咩？",
 "我部手機好慢，有咩方法可以快返？","打工仔請年假有咩要注意？",
 "香港有咩大學比較出名？","點樣開銀行戶口？","去街市買餸有咩貼士？",
 "我想去日本旅行，有咩準備要做？","香港嘅急症室收費係幾多？",
 "點樣可以慳返啲電費？","你識唔識講笑話？講個嚟聽下。",
 "我今日心情唔好，可以點樣令自己開心啲？","香港嘅小學派位制度係點運作？",
 "邊度可以睇到香港靚嘅夜景？","咩叫做強制檢測？","做運動有咩好處？",
 "點樣同上司講加人工？","香港有咩傳統小食？","我想養隻貓，要注意咩？",
 "點樣分辨網上嘅假新聞？","坐飛機前有咩要檢查？",
]

def load_model(path, dtype):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dev = pick_device()
    try:                    # transformers >= 5 用 dtype=
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, device_map=dev, trust_remote_code=True)
    except TypeError:       # transformers 4.x 用 torch_dtype=
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map=dev, trust_remote_code=True)
    model.eval()
    return tok, model

def chat_prompt(tok, user, no_think=False):
    """no_think=True 时关掉 Qwen3 的思维链。

    为什么必须有这个开关：选择题是比较第一个位置上 A/B/C/D 的 logprob，而开着思维链的
    模型第一个 token 是 <think>，于是量到的是「它想先想一想」而不是它的知识。
    原版 Qwen3-8B 在这个混淆下只有 0.2888（接近随机），关掉之后才是它真实的水平。
    """
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": user}]
        if no_think:
            try:
                return tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                # 模板不认 enable_thinking：手动补一个空的思维块把它关掉
                base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                return base + "<think>\n\n</think>\n\n"
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return user + "\n"

# ---------------- 任务 1：HKMMLU ----------------
def task_hkmmlu(tok, model, data_dir, limit_per_cfg, device, no_think=True):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "test", "*.csv"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*test*.csv"), recursive=True))
    if not files:
        return {"error": f"no test csv under {data_dir}"}
    letters = ["A","B","C","D"]
    # 每个选项字母的首 token id（对 Qwen 系是单 token；用 strip 前缀兼容）
    letter_ids = []
    for L in letters:
        ids = tok.encode(L, add_special_tokens=False)
        letter_ids.append(ids[0])
    per_cfg, n_all, n_correct = {}, 0, 0
    for f in files:
        cfg = os.path.splitext(os.path.basename(f))[0]
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if limit_per_cfg: rows = rows[:limit_per_cfg]
        c = t = 0
        for r in rows:
            q = (r.get("Question") or r.get("question") or "").strip()
            opts = [(r.get(L) or "").strip() for L in letters]
            gold = (r.get("Answer") or r.get("answer") or "").strip().upper()[:1]
            if not q or gold not in letters: continue
            body = (f"以下係一道香港知識選擇題，請只答 A、B、C 或 D。\n\n題目：{q}\n"
                    + "\n".join(f"{L}. {o}" for L, o in zip(letters, opts)) + "\n\n答案：")
            ids = tok(chat_prompt(tok, body, no_think=no_think), return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**ids).logits[0, -1]
            pred = letters[int(torch.tensor([logits[i] for i in letter_ids]).argmax())]
            c += int(pred == gold); t += 1
        if t:
            per_cfg[cfg] = {"acc": c/t, "n": t}
            n_all += t; n_correct += c
    hk = {k: v for k, v in per_cfg.items() if k.startswith("hk_")}
    dse = {k: v for k, v in per_cfg.items() if k.startswith("hkdse_")}
    agg = lambda d: (sum(v["acc"]*v["n"] for v in d.values())/sum(v["n"] for v in d.values())
                     if d else None)
    return {"overall_acc": n_correct/n_all if n_all else None, "n": n_all, "no_think": no_think,
            "hk_subset_acc": agg(hk), "hkdse_subset_acc": agg(dse),
            "random_baseline": 0.25, "per_config": per_cfg}

# ---------------- 任务 2：书面粤语纯度 ----------------
def count_markers(text, markers):
    return sum(text.count(m) for m in markers)

def task_purity(tok, model, device, max_new_tokens=128, no_think=False):
    outs = []
    for p in PURITY_PROMPTS:
        ids = tok(chat_prompt(tok, p, no_think=no_think), return_tensors="pt").to(device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        resp = tok.decode(g[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)
        y, z = count_markers(resp, YUE_MARKERS), count_markers(resp, ZH_MARKERS)
        # 退化检测：末尾 40 字里有没有 8 字以上的重复块
        tail = resp[-60:]
        degen = bool(re.search(r"(.{6,})\1{2,}", resp))
        outs.append({"prompt": p, "response": resp, "yue": y, "zh": z,
                     "ratio": y/(y+z) if (y+z) else None, "len": len(resp), "degenerate": degen})
    valid = [o for o in outs if o["ratio"] is not None]
    return {"no_think": no_think,
            "mean_yue_ratio": sum(o["ratio"] for o in valid)/len(valid) if valid else None,
            "median_len": sorted(o["len"] for o in outs)[len(outs)//2],
            "degenerate_rate": sum(o["degenerate"] for o in outs)/len(outs),
            "n_prompts": len(outs), "samples": outs}

# ---------------- 任务 3：困惑度（分布内 / 分布外）----------------
def corpus_nll(tok, model, texts, device, max_len=512):
    tot_nll = tot_tok = tot_chr = 0
    for t in texts:
        if not t.strip(): continue
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        if ids["input_ids"].shape[-1] < 2: continue
        with torch.no_grad():
            out = model(**ids, labels=ids["input_ids"])
        n = ids["input_ids"].shape[-1] - 1
        tot_nll += out.loss.item()*n; tot_tok += n; tot_chr += len(t)
    if not tot_tok: return None
    return {"ppl": math.exp(tot_nll/tot_tok),
            "bits_per_char": tot_nll/math.log(2)/tot_chr,   # 跨分词器可比
            "n_tokens": tot_tok, "n_chars": tot_chr}

def task_ppl(tok, model, device, corpora):
    return {name: corpus_nll(tok, model, texts, device) for name, texts in corpora.items()}

def load_corpora(spec, limit):
    out = {}
    for item in spec.split(","):
        if not item: continue
        name, path = item.split("=", 1)
        texts = []
        if path.endswith(".jsonl"):
            for i, line in enumerate(open(path, encoding="utf-8")):
                if i >= limit: break
                try: d = json.loads(line)
                except Exception: continue
                if "messages" in d:
                    texts.append("\n".join(m.get("content","") for m in d["messages"]))
                elif "text" in d: texts.append(d["text"])
                else: texts.append(" ".join(str(v) for v in d.values() if isinstance(v,str)))
        else:
            texts = [l.strip() for l in open(path, encoding="utf-8")][:limit]
        out[name] = texts
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", default="hkmmlu,purity,ppl")
    ap.add_argument("--hkmmlu-dir", default="$LAB_ROOT/data/HKMMLU")
    ap.add_argument("--limit-per-cfg", type=int, default=0)
    ap.add_argument("--purity-no-think", action="store_true",
                    help="生成任务也关掉思维链。对照基座时建议开，否则基座满屏 <think> 的普通话推理，"
                         "粤语纯度量到的是它的思考过程而不是它的回答")
    ap.add_argument("--hkmmlu-think", action="store_true",
                    help="选择题保留思维链（默认关掉；开着会把 <think> 当成答案位）")
    ap.add_argument("--corpora", default="")
    ap.add_argument("--corpus-limit", type=int, default=300)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    t0 = time.time()
    tok, model = load_model(a.model, dtype)
    device = next(model.parameters()).device
    res = {"name": a.name, "model": a.model, "dtype": a.dtype, "device": str(device),
           "n_params": sum(p.numel() for p in model.parameters())}
    tasks = a.tasks.split(",")
    if "hkmmlu" in tasks:
        res["hkmmlu"] = task_hkmmlu(tok, model, a.hkmmlu_dir, a.limit_per_cfg, device,
                                    no_think=not a.hkmmlu_think)
        print(f"[{a.name}] hkmmlu overall={res['hkmmlu'].get('overall_acc')}", flush=True)
    if "purity" in tasks:
        res["purity"] = task_purity(tok, model, device, no_think=a.purity_no_think)
        print(f"[{a.name}] yue_ratio={res['purity']['mean_yue_ratio']}", flush=True)
    if "ppl" in tasks and a.corpora:
        res["ppl"] = task_ppl(tok, model, device, load_corpora(a.corpora, a.corpus_limit))
        print(f"[{a.name}] ppl={ {k:(v or {}).get('ppl') for k,v in res['ppl'].items()} }", flush=True)
    res["elapsed_s"] = round(time.time()-t0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("WROTE", a.out, f"({res['elapsed_s']}s)")

if __name__ == "__main__":
    main()
