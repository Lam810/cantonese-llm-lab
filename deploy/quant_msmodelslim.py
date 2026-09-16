#!/usr/bin/env python3
"""昇腾原生量化：msmodelslim（ModelSlim）。

为什么要用它而不是 llmcompressor：`vllm-ascend` 只要检测到 NPU 就无条件把量化方法
劫持成它自己的（`override_quantization_method` 无条件返回 ASCEND），而它只读
`quant_model_description.json`（ModelSlim 格式，每个权重前缀都要有一条记录）。
llmcompressor 出的 compressed-tensors 写在 `config.json` 的 `quantization_config` 里，
没有这个文件，直接 KeyError。补描述文件也不行——张量布局都不一样
（compressed-tensors 是 weight+weight_scale，昇腾要 deq_scale/quant_bias）。

而且 msmodelslim 认 NPU（`dev_type='npu'`），不像 llmcompressor 只能跑 CPU
（8B + 256 条校准在 CPU 上花了 6 小时 43 分）。

⚠️ 预期：`vllm_ascend/quantization/` 支持的四种方案（W8A8 / W8A8_DYNAMIC /
W4A8_DYNAMIC / W4A4_FLATQUANT_DYNAMIC）**全部都要量化激活**，没有 weight-only 选项。
而实测激活量化是最伤的那一刀：int4（权重-only）HKMMLU 只掉 1.85pp，
llmcompressor 的 W8A8（权重+激活）掉 6.67pp、低于基座。
所以这里大概率也是负结果——跑它的价值是把这条路走完并留下证据。
"""
import argparse, json, os, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--calib-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scheme", choices=["w8a8", "w8a8_dynamic"], default="w8a8_dynamic")
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--dev-type", default="npu")
    ap.add_argument("--anti-method", default="m2",
                    help="anti-outlier 方法。m2 是昇腾文档里的常用项")
    ap.add_argument("--no-anti", action="store_true",
                    help="跳过 anti-outlier。质量会差，但 anti 那步挂了还能出个产物")
    ap.add_argument("--disable-level", default="L0",
                    help="L0=不自动回退；L1..L5=把最差的 N 层回退成浮点")
    a = ap.parse_args()

    import torch
    if a.dev_type == "npu":
        import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
    from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    # msmodelslim 的常规用法：fp32 装 CPU，逐层挪去 NPU 做校准
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32,
                                                 trust_remote_code=True).eval()
    if a.dev_type == "npu":
        model = model.npu()          # fp32 8B ≈ 32GB，910C 单 die 64GB 装得下
    else:
        model = model.cpu()

    # 校准集：训练完全没用过的 test 切分，粤语。和之前 llmcompressor 那次同一批，
    # 好让两个工具的结果可比。量化工具默认拿英文语料校准，那会把粤语语域量化掉。
    texts = []
    for line in open(a.calib_jsonl, encoding="utf-8"):
        if len(texts) >= a.n_calib: break
        try: d = json.loads(line)
        except Exception: continue
        if "messages" in d:
            try:
                texts.append(tok.apply_chat_template(d["messages"], tokenize=False,
                                                     add_generation_prompt=False))
                continue
            except Exception:
                texts.append("\n".join(m.get("content", "") for m in d["messages"]))
        else:
            texts.append(" ".join(str(v) for v in d.values() if isinstance(v, str)))
    print(f"校准集 {len(texts)} 条粤语样本，来自 {a.calib_jsonl}", flush=True)

    # check_calib_data 要求：list of list，元素必须是 torch.Tensor，
    # 且 model(*(calib_data[0])) 能跑通 —— HF CausalLM 的位置参数是 (input_ids, attention_mask)
    # dev_type='npu' 时 Calibrator 会把模型挪到 npu:0，校准张量必须跟着走，
    # 否则第一步 embedding 就报 "found at least two devices, npu:0 and cpu"
    to_dev = (lambda x: x.npu()) if a.dev_type == "npu" else (lambda x: x)
    calib = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=a.max_len)
        calib.append([to_dev(enc["input_ids"]), to_dev(enc["attention_mask"])])

    dyn = a.scheme == "w8a8_dynamic"

    # anti-outlier 必须作为**单独一步**跑，不能靠 QuantConfig(do_smooth=True)。
    # do_smooth=True 走的是 DAG 自动识别网络结构（get_llm_network_pattern_auto），
    # 在 Qwen3 上它返回空的 attn 列表，于是 split_module_name_get_info 里
    # module_names[0] 直接 IndexError。长度校验还过得了（四个列表等长），
    # 内容是空的 —— 这种"校验通过但内容为空"最难查。
    # 正规做法是显式告诉它 norm 层的类名，绕开自动识别。
    if not a.no_anti:
        norm_cls = type(model.model.layers[0].input_layernorm).__name__
        print(f"anti-outlier: method={a.anti_method} norm_class_name={norm_cls}", flush=True)
        anti_cfg = AntiOutlierConfig(w_bit=8, a_bit=8, anti_method=a.anti_method,
                                     dev_type=a.dev_type, dev_id=0, w_sym=True)
        anti = AntiOutlier(model, calib_data=calib, cfg=anti_cfg, norm_class_name=norm_cls)
        t_anti = time.time()
        anti.process()
        print(f"  anti-outlier 完成，用时 {(time.time()-t_anti)/60:.1f} 分", flush=True)

    cfg = QuantConfig(
        w_bit=8, a_bit=8,
        mm_tensor=False,        # per-channel 而不是 per-tensor
        w_sym=True,
        dev_type=a.dev_type, dev_id=0,
        act_method=3,
        do_smooth=False,        # 上面已经单独做过 anti-outlier，这里必须关
        use_sigma=False,
        is_dynamic=dyn,         # True -> W8A8_DYNAMIC（激活按 token 动态定标，通常更温和）
        disable_last_linear=True,   # lm_head 不量化
    )
    print(f"scheme={a.scheme} is_dynamic={dyn} dev_type={a.dev_type} "
          f"disable_level={a.disable_level} anti={not a.no_anti}", flush=True)

    t0 = time.time()
    cal = Calibrator(model, cfg, calib_data=calib, disable_level=a.disable_level)
    cal.run()
    os.makedirs(a.out, exist_ok=True)
    cal.save(a.out, save_type=["safe_tensor"])
    tok.save_pretrained(a.out)
    json.dump({"tool": "msmodelslim", "scheme": a.scheme, "is_dynamic": dyn,
               "anti_outlier": (None if a.no_anti else a.anti_method),
               "n_calib": len(texts), "max_len": a.max_len,
               "dev_type": a.dev_type, "disable_level": a.disable_level,
               "calib_source": "v2-clean test split (Cantonese, unseen in training)"},
              open(os.path.join(a.out, "quant_info.json"), "w"), indent=1, ensure_ascii=False)
    print(f"✓ {a.scheme} -> {a.out}  用时 {(time.time()-t0)/60:.1f} 分", flush=True)
    print("落盘文件:", sorted(os.listdir(a.out)), flush=True)

if __name__ == "__main__":
    main()
