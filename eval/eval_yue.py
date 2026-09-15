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
 # ---- 以下 90 條為 2026-09 擴充：退化率在 30 條上分辨率只有 1/30=0.033，
 # ---- 差一兩條就是一格，沒法做變體之間的比較；擴到 120 條才有意義。
 # 交通
 "西鐵同東鐵有咩唔同？","過大海搭船定搭巴士好？","點樣由機場返市區最平？",
 "紅隧塞車嘅話有咩替代路線？","電車坐一程要幾錢？","的士紅色綠色藍色有咩分別？",
 # 飲食
 "點樣煲一煲靚湯？","蛋撻同葡撻邊樣好食啲？","茶餐廳嘅絲襪奶茶點解咁香？",
 "打邊爐要準備啲咩？","有咩宵夜推薦？","素食者喺香港食咩好？",
 # 住屋
 "劏房同套房有咩分別？","租樓要注意啲咩條款？","裝修大約要預幾多錢？",
 "業主唔肯退按金點算？","居屋同公屋有咩分別？",
 # 理財
 "點樣開始儲錢？","強積金可唔可以自己揀基金？","信用卡簽賬有咩陷阱？",
 "香港嘅薪俸稅係點計？","買保險要留意咩？","定期存款同債券邊樣穩陣啲？",
 # 醫療
 "感冒同流感點分？","公立醫院專科排期要幾耐？","打疫苗前要注意咩？",
 "腰痠背痛有咩方法紓緩？","點樣改善失眠？","體檢一般要驗邊啲項目？",
 # 教育
 "DSE 選科有咩策略？","副學士值唔值得讀？","小朋友幾多歲開始學英文好？",
 "點樣揀補習社？","大學宿舍申請難唔難？",
 # 工作
 "面試要準備啲咩？","辭職要提前幾耐通知？","打工仔放病假有冇糧出？",
 "點樣寫一份好嘅 CV？","公司要求成日超時工作可以點？","轉行要考慮啲咩？",
 # 科技
 "點樣防止手機中毒？","Wi-Fi 好慢可以點解決？","雲端備份邊個好用？",
 "電腦開機好慢點算？","點樣設定一個安全嘅密碼？","AI 會唔會搶走我份工？",
 # 節慶
 "農曆新年有咩習俗？","端午節點解要食糉？","盂蘭節係咩嚟？",
 "冬至同新年邊個大？","天后誕有咩活動？","香港有咩本地節慶值得睇？",
 # 情感人際
 "同朋友嗌交咗點樣和好？","覺得工作壓力好大點算？","點樣同父母溝通？",
 "第一次約會去邊好？","朋友借錢唔還可以點？","點樣拒絕人而唔傷感情？",
 # 旅遊
 "台灣自由行點樣安排？","去泰國要唔要簽證？","窮遊有咩慳錢方法？",
 "帶老人家去旅行要注意咩？","香港離島邊個最值得去？","行山新手行邊條路線好？",
 # 法律行政
 "點樣申請身份證？","結婚登記要準備咩文件？","被解僱可以點追討？",
 "交通違例罰款點交？","遺囑要點寫先至有效？",
 # 娛樂
 "點樣揀一部好睇嘅電影？","學樂器邊樣最易入門？","養魚要注意咩？",
 "點樣影一張靚相？","打機打得太耐有咩壞處？","點樣養成睇書嘅習慣？",
 # 知識解釋
 "點解天空係藍色嘅？","通脹係咩意思？","咩叫做碳中和？",
 "地震點解會發生？","區塊鏈簡單嚟講係咩？","點解要有時區？",
 "水點解會結冰？","咩係複利效應？","疫苗嘅原理係點？",
 "點解飛機可以飛得起身？","基因改造食物安唔安全？","咩係量子電腦？",
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
    per_cfg, n_all, n_correct, preds = {}, 0, 0, {}
    for f in files:
        cfg = os.path.splitext(os.path.basename(f))[0]
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if limit_per_cfg: rows = rows[:limit_per_cfg]
        c = t = 0; golds, preds_s = [], []
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
            golds.append(gold); preds_s.append(pred)
            c += int(pred == gold); t += 1
        if t:
            per_cfg[cfg] = {"acc": c/t, "n": t}
            preds[cfg] = {"gold": "".join(golds), "pred": "".join(preds_s)}
            n_all += t; n_correct += c
    hk = {k: v for k, v in per_cfg.items() if k.startswith("hk_")}
    dse = {k: v for k, v in per_cfg.items() if k.startswith("hkdse_")}
    agg = lambda d: (sum(v["acc"]*v["n"] for v in d.values())/sum(v["n"] for v in d.values())
                     if d else None)
    macro = (sum(v["acc"] for v in per_cfg.values())/len(per_cfg)) if per_cfg else None
    return {"overall_acc": n_correct/n_all if n_all else None, "n": n_all, "no_think": no_think,
            "macro_acc": macro,                       # config 不等长，micro 会被大 config 拉走
            "ci95": bootstrap_ci(per_cfg, preds),
            "hk_subset_acc": agg(hk), "hkdse_subset_acc": agg(dse),
            "random_baseline": 0.25, "per_config": per_cfg,
            "preds": preds}                           # 逐题预测：配对检验的唯一依据



def bootstrap_ci(per_cfg, preds, n_boot=2000, seed=0):
    """对逐题对错做 bootstrap。不是把 acc 当正态——config 之间差异很大，
    正态近似在这里会低估不确定性。"""
    import random
    flat = []
    for cfg, d in preds.items():
        flat.extend(int(g == p) for g, p in zip(d["gold"], d["pred"]))
    if not flat: return None
    rng = random.Random(seed); N = len(flat)
    accs = []
    for _ in range(n_boot):
        s = sum(flat[rng.randrange(N)] for _ in range(N))
        accs.append(s / N)
    accs.sort()
    return {"lo": accs[int(0.025*n_boot)], "hi": accs[int(0.975*n_boot)], "n_boot": n_boot}

# ---------------- 任务 1b：生成式 HKMMLU（给 Thinking 模型的公平口径）----------------
# 首 token logprob 那套对开思维链的模型是错的——第一个 token 是 <think>，量到的是
# 「它想先想一想」。要让思维链模型发挥，必须让它把推理生成完，再从末尾解析答案。
_ANS_RE = re.compile(r"(?:答案|answer|選|选)\s*(?:係|是|为|為)?\s*[:：]?\s*[（(\[]?\s*([ABCD])",
                     re.IGNORECASE)
# 不能用 \b：中文字在 Unicode 下算 \w，「答案是A。」里 A 左边没有词边界
_BARE_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")

def parse_choice(text):
    """从生成文本里解析 A/B/C/D。解析不出返回 None——不要静默算错，那会把
    「答不出格式」和「答错了」混成一个数字。"""
    body = text.split("</think>")[-1] if "</think>" in text else text
    for rx in (_ANS_RE, _BARE_RE):
        m = list(rx.finditer(body))
        if m: return m[-1].group(1).upper()
    if body is not text:                       # think 没闭合时退回整段
        for rx in (_ANS_RE, _BARE_RE):
            m = list(rx.finditer(text))
            if m: return m[-1].group(1).upper()
    return None

def _gen_batch(tok, model, prompts, device, max_new_tokens):
    old_side = tok.padding_side
    tok.padding_side = "left"                  # 批量生成必须左填充，否则续写位置错
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    plen = enc["input_ids"].shape[-1]
    tok.padding_side = old_side
    return [tok.decode(row[plen:], skip_special_tokens=False) for row in g]

def task_hkmmlu_gen(tok, model, data_dir, limit_per_cfg, device,
                    no_think=False, max_new_tokens=512, batch=8):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "test", "*.csv"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*test*.csv"), recursive=True))
    if not files: return {"error": f"no test csv under {data_dir}"}
    letters = ["A","B","C","D"]
    per_cfg, n_all, n_correct, n_unparsed, tok_total = {}, 0, 0, 0, 0
    for f in files:
        cfg = os.path.splitext(os.path.basename(f))[0]
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        if limit_per_cfg: rows = rows[:limit_per_cfg]
        items = []
        for r in rows:
            q = (r.get("Question") or r.get("question") or "").strip()
            opts = [(r.get(L) or "").strip() for L in letters]
            gold = (r.get("Answer") or r.get("answer") or "").strip().upper()[:1]
            if not q or gold not in letters: continue
            body = (f"以下係一道香港知識選擇題，請答 A、B、C 或 D。\n\n題目：{q}\n"
                    + "\n".join(f"{L}. {o}" for L, o in zip(letters, opts))
                    + "\n\n請喺最後一行寫「答案：X」。")
            items.append((chat_prompt(tok, body, no_think=no_think), gold))
        c = t = u = 0
        for i in range(0, len(items), batch):
            chunk = items[i:i+batch]
            outs = _gen_batch(tok, model, [x[0] for x in chunk], device, max_new_tokens)
            for (_, gold), o in zip(chunk, outs):
                tok_total += len(tok.encode(o, add_special_tokens=False))
                pred = parse_choice(o)
                if pred is None: u += 1
                c += int(pred == gold); t += 1
        if t:
            per_cfg[cfg] = {"acc": c/t, "n": t, "unparsed": u}
            n_all += t; n_correct += c; n_unparsed += u
    hk  = {k: v for k, v in per_cfg.items() if k.startswith("hk_")}
    dse = {k: v for k, v in per_cfg.items() if k.startswith("hkdse_")}
    agg = lambda d: (sum(v["acc"]*v["n"] for v in d.values())/sum(v["n"] for v in d.values())
                     if d else None)
    return {"mode": "gen", "overall_acc": n_correct/n_all if n_all else None, "n": n_all,
            "no_think": no_think, "n_unparsed": n_unparsed,
            "unparsed_rate": n_unparsed/n_all if n_all else None,
            "mean_out_tokens": tok_total/n_all if n_all else None,
            "hk_subset_acc": agg(hk), "hkdse_subset_acc": agg(dse),
            "random_baseline": 0.25, "per_config": per_cfg}

# ---------------- 任务 2：书面粤语纯度 ----------------
def count_markers(text, markers):
    return sum(text.count(m) for m in markers)

def task_purity(tok, model, device, max_new_tokens=128, no_think=False, limit=0):
    outs = []
    for p in (PURITY_PROMPTS[:limit] if limit else PURITY_PROMPTS):
        ids = tok(chat_prompt(tok, p, no_think=no_think), return_tensors="pt").to(device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        out_ids = g[0][ids["input_ids"].shape[-1]:]
        raw = tok.decode(out_ids, skip_special_tokens=False)
        n_tok = int(out_ids.shape[-1])
        # 思维链的开销要单独记：在昇腾 462 tok/s 下，烧 500 个 token 想一想是实打实的延迟。
        # 纯度只量正文——think 块里常是普通话/英文，混进去会污染这个指标。
        if "</think>" in raw:
            think_txt, body_txt = raw.split("</think>", 1)
            n_think = len(tok.encode(think_txt, add_special_tokens=False))
        else:
            body_txt, n_think = raw, 0
        resp = tok.decode(tok.encode(body_txt, add_special_tokens=False),
                          skip_special_tokens=True).strip()
        y, z = count_markers(resp, YUE_MARKERS), count_markers(resp, ZH_MARKERS)
        # 退化检测：末尾 40 字里有没有 8 字以上的重复块
        tail = resp[-60:]
        degen = bool(re.search(r"(.{6,})\1{2,}", resp))
        outs.append({"prompt": p, "response": resp, "yue": y, "zh": z,
                     "ratio": y/(y+z) if (y+z) else None, "len": len(resp),
                     "out_tokens": n_tok, "think_tokens": n_think, "degenerate": degen})
    valid = [o for o in outs if o["ratio"] is not None]
    med = lambda k: sorted(o[k] for o in outs)[len(outs)//2]
    def block(sub):
        v = [o for o in sub if o["ratio"] is not None]
        m = lambda k: sorted(o[k] for o in sub)[len(sub)//2] if sub else None
        return {"n_prompts": len(sub),
                "mean_yue_ratio": sum(o["ratio"] for o in v)/len(v) if v else None,
                "median_len": m("len"), "median_out_tokens": m("out_tokens"),
                "degenerate_rate": sum(o["degenerate"] for o in sub)/len(sub) if sub else None}
    return {"no_think": no_think,
            "first30": block(outs[:30]),   # 与 2026-09 已发布的 30 条口径逐项可比
            "full": block(outs),
            "mean_yue_ratio": sum(o["ratio"] for o in valid)/len(valid) if valid else None,
            "median_len": med("len"),
            "median_out_tokens": med("out_tokens"),
            "median_think_tokens": med("think_tokens"),
            "think_rate": sum(o["think_tokens"] > 0 for o in outs)/len(outs),
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
    ap.add_argument("--purity-limit", type=int, default=0,
                    help="只跑前 N 条提问。0=全部（117 条）。传 30 可复现已发布口径")
    ap.add_argument("--purity-max-new", type=int, default=128,
                    help="生成长度上限。默认 128 与已发布口径一致；开思维链时必须调大，"
                         "否则会在 think 块中间截断，量到的是半截思考过程")
    ap.add_argument("--hkmmlu-gen", action="store_true",
                    help="选择题改成生成式：让模型把思维链生成完，再从末尾解析答案。"
                         "对 Thinking 模型这是唯一公平的口径")
    ap.add_argument("--hkmmlu-gen-limit", type=int, default=10,
                    help="生成式口径下每个 config 取几题（贵，默认 10 → n≈660，标准误约 1.7pp）")
    ap.add_argument("--gen-max-new", type=int, default=512)
    ap.add_argument("--gen-batch", type=int, default=8)
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
        print(f"[{a.name}] hkmmlu(logprob) overall={res['hkmmlu'].get('overall_acc')}", flush=True)
    if "hkmmlu_gen" in tasks or a.hkmmlu_gen:
        res["hkmmlu_gen"] = task_hkmmlu_gen(tok, model, a.hkmmlu_dir, a.hkmmlu_gen_limit, device,
                                            no_think=not a.hkmmlu_think,
                                            max_new_tokens=a.gen_max_new, batch=a.gen_batch)
        g = res["hkmmlu_gen"]
        print(f"[{a.name}] hkmmlu(gen) overall={g.get('overall_acc')} "
              f"解析不出={g.get('unparsed_rate')} 平均输出 token={g.get('mean_out_tokens')}", flush=True)
    if "purity" in tasks:
        res["purity"] = task_purity(tok, model, device, max_new_tokens=a.purity_max_new,
                                    no_think=a.purity_no_think, limit=a.purity_limit)
        pu = res["purity"]
        print(f"[{a.name}] yue_ratio={pu['mean_yue_ratio']} 退化={pu['degenerate_rate']} "
              f"中位输出token={pu['median_out_tokens']} 中位think token={pu['median_think_tokens']} "
              f"出现think比例={pu['think_rate']}", flush=True)
    if "ppl" in tasks and a.corpora:
        res["ppl"] = task_ppl(tok, model, device, load_corpora(a.corpora, a.corpus_limit))
        print(f"[{a.name}] ppl={ {k:(v or {}).get('ppl') for k,v in res['ppl'].items()} }", flush=True)
    res["elapsed_s"] = round(time.time()-t0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("WROTE", a.out, f"({res['elapsed_s']}s)")

if __name__ == "__main__":
    main()
