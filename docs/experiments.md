# 实验结果记录

> 训练环境：4×RTX 3090，Python 3.11.16 / torch 2.11.0+cu128 / torchvision 0.26.0
> 数据：`archive.zip`（Kaggle *Fruit and Vegetable Image Recognition*，2.13 GB，3825 张）
> 代码：本仓库 `src/` + `scripts/`，划分只认 `data/splits*.json`

## 1. 数据与清洗

| 项目 | 10 类（E01/E05） | 11 类窄口径 other（E02） | 11 类宽口径 other（E03/E04） |
| --- | --- | --- | --- |
| splits 文件 | `data/splits.json` | `data/splits_other.json` | `data/splits_other_broad.json` |
| 原始水果图 | 1033 | 1033 | 1033 |
| pHash 去重后 | 787（丢弃 246，23.8%） | 787 | 787 |
| other 来源 | — | 17 个蔬菜目录 × 26（442） | 26 个目录 × 17（含番茄/黄瓜/甜椒/辣椒/茄子/柠檬） |
| 去重后总量 | 787 | 1172 | 1203 |
| train / val / test | 551 / 118 / 118 | 820 / 176 / 176 | 843 / 180 / 180 |
| 训练集 other 占比 | — | 32.8% | 32.4% |

**去重是关键**：官方 `train/`、`validation/`、`test/` 三个目录里有同一张图重复出现（pHash 汉明距离 0），
合并后必须先去重再重划分，否则 test 会直接泄漏进 train（238 张/1033 张，23%）。

跨划分泄漏复查：两份 splits 都是"未发现近重复"。

## 2. OOD 测试集

`data/ood`：225 张，9 类 —— tomato / cucumber / bell pepper / capsicum / paprika /
chilli pepper / jalepeno / eggplant / lemon（数据集中没有 avocado，自动跳过）。

- 窄口径 other 训练时这些类别**完全没见过**；
- 宽口径 other 训练时它们**部分进入 other**，OOD 只用 manifest 里未被占用的图片
  （`scripts/02_build_ood.py --exclude-from data/manifest_other_broad.csv`），保证不重叠。

## 3. 实验矩阵与结果

| run | 数据 | head | 冻结策略 | test acc | 95% CI | macro-F1 | top-3 | ECE(校准后) | OOD→other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E01 | 10 类 | linear | 全程冻结 | 0.9576 | 0.9046–0.9818 | 0.9572 | 0.9746 | 0.042 | 无 other 类（MSP 拒识 24.0%） |
| E02 | 11 类窄口径 | linear | 全程冻结 | 0.8920 | 0.8375–0.9298 | 0.8872 | 0.9773 | 0.049 | 28.9% |
| E03 | 11 类宽口径 | linear | 全程冻结 | 0.8667 | 0.8093–0.9087 | 0.8680 | 0.9778 | 0.075 | 72.0% |
| **E04** | **11 类宽口径** | linear | **解冻 features[-3:] 15 轮** | **0.9056** | 0.8540–0.9402 | **0.8990** | **0.9833** | 0.045 | **81.3%** |
| E05 | 10 类 | linear | 解冻 features[-3:] 15 轮 | 0.9576 | 0.9046–0.9818 | 0.9572 | 0.9746 | 0.042 | 无 other 类（MSP 拒识 24.0%） |

> `ECE(校准后)` 为 **test** 口径（取自 `docs/E0*_test.json`）；val 上的校准前后对比见 `ckpt/*_calib.json`（E04：0.1764 → 0.0359）。

### 3b. 只看水果类（把 other 从测试集里摘掉，口径才可比）

测试集里 118 张水果图对所有模型都是同一批（`seed=42`、同一 `split_class`）：

| run | 水果类准确率 | 说明 |
| --- | --- | --- |
| E01 / E05 | **0.9576**（113/118） | 10 类模型，不学 other |
| E02 | 0.9237（109/118） | 加窄口径 other，掉 3.4 个点 |
| E03 | 0.9068（107/118） | 加宽口径 other，掉 5.1 个点 |
| E04 | 0.9153（108/118） | 宽口径 + 解冻，把 1.7 个点捞回来 |

> `OOD→other` 是上线真正生效的指标：陌生图被判成 `other` 的比例。10 类模型没有 other 类，
> 只能靠 MSP 阈值拒识（E01/E05 在 FRR≤5% 时只拒掉 24–28% 的 OOD）。

## 4. 结论

1. **10 类基线很强**：test acc 95.8%、macro-F1 95.7%，只有 787 张数据（冻结 backbone + 线性头）。
2. **窄口径 other 几乎不起作用**：把 17 个根茎类蔬菜当负类训练后，只有 29% 的番茄/柠檬/黄瓜
   被判成 other，其余 71% 被硬判成 apple(62)/orange(28)/banana(25) —— one-vs-rest 的边界没被教到。
3. **宽口径 other 才能压住开集误判**：把蔬果难分的类别也放进 other（OOD 只用不重叠图片），
   OOD→other 从 29% → 72%；再加解冻 features[-3:] 到 **81.3%**，水果准确率也从 90.7% 回到 91.5%。
4. **解冻对 10 类无收益**（E05 与 E01 指标完全一致：val macro-F1 0.9354 两次都提前早停），
   但对 11 类有明显收益（E03→E04：acc +3.9、OOD→other +9.3）。小数据下解冻要谨慎，`lr_bb=1e-4` 已经不低。
5. **温度校准有效**：ECE 从 0.20–0.22 压到 0.04–0.08，T≈0.43–0.52（小数据微调模型的典型过自信）。
6. **阈值不能手填**：10 类模型 MSP 阈值标出来是 0.52（不是常见 0.6+），手填 0.6 等于永不拒识。

## 5. 上线推理实测（E04，`python -m src.predict`）

```
tomato_0001.jpg: 识别 -> 其他 / 非目标(other) 0.981 | 橙子(orange) 0.007  [这不是我认识的水果（支持苹果/香蕉/葡萄/猕猴桃/芒果/橙子/梨/菠萝/石榴/西瓜）]
tomato_0003.jpg: 拒识 -> 其他 / 非目标(other) 0.359 | 苹果(apple) 0.339  [未识别为支持的水果，请换一张更清晰、主体更完整的照片]
lemon_0001.png : 识别 -> 橙子(orange) 0.964 | 猕猴桃(kiwi) 0.010  [可能是橙子]      ← 残留误判
apple_00000.jpg: 识别 -> 苹果(apple) 0.990 | 石榴(pomegranate) 0.007  [可能是苹果]
selfcheck: 5/5 命中
```

三态文案都跑通了：**拒识**（置信度低于标定阈值）、**other**（"这不是我认识的水果"）、**命中**（"可能是苹果"）。

**残留问题**：柠檬仍常被判成橙子（形状颜色都接近）。可选处理：
- 把 lemon 也放进 other 训练（它的 OOD 测试价值会让给别的类别）；
- 或做**级联门控**：E01 的 10 类模型 + 一个二分类 fruit/non-fruit 门控模型，先判"是不是水果"再细分类。

## 6. 取舍建议

| 场景 | 推荐配置 | 理由 |
| --- | --- | --- |
| 只识别 10 种水果、用户基本只拍水果 | E01（10 类） | 水果准确率最高（95.8%）；误拍其它东西靠 MSP 阈值兜底 |
| 用户可能拍任何东西，宁可说"不认识" | **E04（宽口径 other + 解冻）** | 81.3% 的陌生图被正确归为 other，水果准确率 91.5% |

## 7. 复现命令

```bash
# 在仓库根目录运行，并激活训练环境
PY=python
SRC=data/raw/fruit-veg

# E01 + E02（10 类 / 窄口径 11 类）+ 校准 + 指标
SRC=$SRC PY=$PY bash scripts/03_run_experiments.sh

# E03/E04（宽口径 11 类，含蔬果难分；OOD 用不重叠图片）+ 重标定
SRC=$SRC PY=$PY bash scripts/04_broad_other_pipeline.sh

# 第二阶段：解冻 features[-3:]
PY=$PY $PY -m src.train --tag E04_broad_unfreeze3 --split data/splits_other_broad.json \
    --epochs 30 --unfreeze-last 3 --unfreeze-epochs 15 --weighted-sampler
```

全部指标落在 `experiments/runs.csv`（训练写前半段，`src/calibrate.py` 回填 OOD/ECE 列），
混淆矩阵在 `docs/curves/cm_*.png`，逐类 P/R/F1 在 `docs/E0*_test.json`。

## 8. 日常物体复测与识别范围判断

原 E04 的「其他」主要由蔬菜样本训练。补充测试引入 Caltech 101 的日常物体、动物和交通工具照片。每类最多取 10 张，并按物体类别分到训练、验证或测试集。测试集的 17 个物体类别未用于训练或选择二分类阈值。原有水果和蔬菜数据按 §7 的宽口径流程重建划分。 [Caltech 101 数据说明](https://data.caltech.edu/records/mzrjq-6wc02)

冻结 E04 的特征提取和水果分类权重，在同一组池化特征上训练一个线性二分类头，区分支持的水果与其他照片。阈值在验证集上选择，保留 E04 已正确识别并接纳的全部验证水果。上线时只有 E04 和二分类头均接纳，页面才显示水果结果。

| 测试输入 | 样本数 | E04 | 加入识别范围判断 |
| --- | ---: | ---: | ---: |
| Caltech 101 日常物体，误判为水果 | 170 | 37 | 12 |
| Kaggle 其他类，误判为支持的水果 | 62 | 6 | 3 |
| Kaggle 支持的水果，正确识别并接纳 | 118 | 107 | 106 |

例如，测试集中的直升机照片 `helicopter/image_0013.jpg` 原来被判为香蕉，得分约 91.9%。加入二分类头后，网页显示「未识别到支持的水果」。本次复测中仍有 12 张日常物体被误判，另有 1 张原来能正确识别的水果被拒绝。这组数据仅代表所测类别。

重建 §7 的数据划分后，在仓库根目录运行下面的命令。发布用的 `model/fruitscope.pth` 可作为输入，脚本直接复用 E04 的特征和类别输出，重新拟合二分类头。

```bash
python scripts/train_scope_filter.py \
    --checkpoint model/fruitscope.pth \
    --split data/splits_other_broad.json \
    --caltech /path/to/caltech-101.zip \
    --output ckpt/fruitscope-with-scope-filter.pth \
    --report ckpt/scope-report.json
```
