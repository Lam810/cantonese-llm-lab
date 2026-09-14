# 数据授权：为什么合并语料不能再分发

重建版用了四个公开来源。**上传前查一遍授权,发现只有两个能再分发。**

| 来源 | 进入重建版 | 授权 | 能否再分发 |
|---|---|---|---|
| `Zeteng/cantonese-llm-data` | 17,257 | `cc-by-nc-4.0` | ✅ 可以,限非商业 |
| `raptorkwok/cantonese-chinese-parallel-corpus-base` | 9,950 | `cc0-1.0` | ✅ 可以 |
| `stvlynn/Cantonese-Dialogue` | 14,064 | **`agpl-3.0`** | ❌ 传染性 copyleft,与 CC-BY-NC 不相容 |
| `indiejoseph/cantonese-cot` | 29,931 | **无任何授权声明** | ❌ 没有再分发权 |
| `jed351/cantonese-wikipedia`（只用于分布外困惑度,不进 SFT） | — | **无声明** | ❌ |

不能再分发的那两个占重建版的 **62%**。把 71,202 条打包成一份新数据集,
无论挂哪个 license 都站不住:混了一份无授权的和一份 AGPL 的,而 AGPL 与 CC-BY-NC 互不相容。

## 所以怎么做

1. **只发授权干净的子集**(27,207 条 = 自有 + CC0),整体按 CC-BY-NC-4.0。
2. **完整版靠脚本复现**,不靠分发数据。`data/build_yue_sft.py` 的 `--only-sources`
   控制收哪几个来源:
   ```bash
   # 授权干净子集
   python data/build_yue_sft.py ... --only-sources zeteng-cantonese-llm-data,raptorkwok-parallel
   # 完整版（自己从原始来源下载后本地构建，不再分发）
   python data/build_yue_sft.py ...
   ```
   确定性已验证:在两台不同机器上重跑,各来源条数、去重数、切分大小逐项一致
   (`seed=42`;`yue_ratio` 均值只在浮点末两位有差,是求和顺序不同)。

## 对模型权重的连带影响

训练数据里含 **CC-BY-NC-4.0** 的部分,所以微调权重**不能标 Apache-2.0**,
即使基座(Qwen3)是 Apache-2.0。老实的标法是 **cc-by-nc-4.0**,并在 card 里列出全部数据来源。

"训练是否构成数据的再分发"在法律上仍有争议,这里取保守做法:
**取所有输入里最严格的那个**。对一个有资助编号的研究项目,保守比省事重要。

## 一条可以直接抄的检查

上传任何合并语料之前,先跑一遍:

```bash
for d in <各来源 repo id>; do
  printf "%-50s " "$d"
  curl -s "https://huggingface.co/api/datasets/$d" | python3 -c "
import json,sys
d=json.load(sys.stdin); cd=d.get('cardData') or {}
lic=cd.get('license') or [t.split(':')[1] for t in d.get('tags',[]) if t.startswith('license:')] or 'UNKNOWN'
print('license =', lic)"
done
```

**`UNKNOWN` 要当成"禁止再分发",不是"随便用"。** HuggingFace 上没有 license 字段的数据集
比想象的多——本例四个来源里就有两个。
