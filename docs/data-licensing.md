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

---

## 后续（2026-09-16）：那个"只能发子集"的遗憾是不成立的

写这份文档的时候，隐含假设是「公开的 27,207 条是个不得已的阉割版，完整的 71,202 条更好」。
**实测下来这个假设是错的。**

只用公开的那 27,207 条训练，与完整语料同配方、同 seed、唯一变量是数据：

| | 训练数据 | HKMMLU（全量 26,368 题） | 书面粤语纯度（117 条） |
|---|---|---|---|
| **只用公开子集** | 26,007 | **0.6138** | **0.9119** |
| 完整语料 | 68,202 | 0.6019 | 0.8975 |

配对检验（McNemar 精确版）**+1.19pp，p = 6.0e-7，显著**，95% 置信区间不重叠。
**那 62% 不能再分发的数据不仅没帮上忙，而且在拖后腿。**

而且这不是「训得少所以没过拟合」：把子集训到和完整语料一样的 8,525 步，结果**更差**
（HKMMLU 0.6067、纯度 0.8615）。

细节、双向数据泄漏的处理、以及哪一列不能用来比，见
[`../results/ablation_clean_data.md`](../results/ablation_clean_data.md)。

**所以这份文档的结论要改一个方向：** 授权约束在这里没有造成能力损失。
公开出去的那份数据，配上公开的构建脚本和训练脚本，**足以让任何人从头复现一个同级别的模型**
——这一点比「完整语料」更有价值。数据卡里不应该写"这是可分发的子集，完整版更好"，
那个说法是错的。

**仍然没做的：** 逐个来源的留一消融。现在只知道那两个来源合起来是负贡献，
不知道是 `cantonese-cot` 还是 `Cantonese-Dialogue`，还是两个都是。
