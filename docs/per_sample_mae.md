# DeepPBS per-sample PWM MAE 说明

## 1. 当前改动解决了什么

原始 DeepPBS 的 `mae` 是 column-level average：

```text
所有样本的有效 PWM columns 拼在一起 -> 统一求平均
```

这样会导致对齐窗口更长的样本在最终 MAE 中权重更大。

现在改成 per-sample MAE：

```text
每个样本先单独计算 MAE -> 再对所有样本求平均
```

这里的“样本”指输入列表里的一个条目，也就是 `id.txt` / `train*.txt` / `valid*.txt` 里的一行，对应一个 PyG `Data` 对象。

## 2. 现在的 MAE 逻辑

对一个样本，先在有效对齐窗口内计算每个 PWM column 的误差：

```text
column_error = sum_bases(abs(pred - true))
```

然后对该样本自己的有效 columns 求平均：

```text
sample_mae = mean_valid_columns(column_error)
```

最后对所有样本等权平均：

```text
dataset_mae = mean_samples(sample_mae)
```

因此现在的总 MAE 不再由 column 数量加权，而是每个样本权重相同。

## 3. strand 怎么处理

DeepPBS 内部会同时处理同一段双链 DNA 的两个方向：

```text
strand0: 原方向
strand1: 反向互补方向
```

如果结构里有 8 个 base-pair positions，模型内部会输出：

```text
strand0: 8 x 4
strand1: 8 x 4
拼接后: 16 x 4
```

这不是 16 个独立 motif 位点，而是同一个 8 bp 区域的两个方向表示。

per-sample MAE 里，一个样本的两个 strand 会在各自 mask 后合并，再算这个样本的 MAE：

```text
sample_valid_columns = valid_columns(strand0) + valid_columns(strand1)
sample_mae = mae(sample_valid_columns)
```

## 4. 和 PWM trim / alignment 的关系

这个改动只改变“样本之间如何平均”，不改变 DeepPBS 原本的 PWM alignment 逻辑。

DeepPBS 仍然会：

```text
真实 PWM 去掉两端低信息量 columns
-> 与结构 DNA 序列做 ungapped alignment
-> 只在 pwm_mask / dna_mask 标记的对齐窗口内计算 loss 和 MAE
```

所以现在的指标更准确地说是：

```text
aligned-window per-sample MAE
```

它不是完整 motif MAE。结构未覆盖或未对齐的 motif 区域仍然不会进入评估。

## 5. 代码位置

主要改动在：

```text
deeppbs/nn/evaluator.py
```

核心函数：

```python
_samplewise_mae(...)
```

触发位置：

```python
elif metric == "mae":
    metric_values[metric].extend(_samplewise_mae(...))
```

`deeppbs/nn/metrics/metrics.py` 里的 `mae()` 没有改，它仍然定义“给定一组 columns 时如何计算 MAE”。现在改变的是调用粒度：从所有 columns 一起算，变成每个样本单独算。

## 6. 如何训练

正常训练即可，不需要额外参数。只要 metrics 里包含 `mae`，训练和验证日志中的 `mae` 就会使用新的 per-sample 逻辑。

单个 fold：

```bash
cd run

python -W ignore driver.py \
  ./folds/train0.txt \
  ./folds/valid0.txt \
  -c config.json \
  --balance unmasked \
  --eval_every 1 \
  --single_gpu \
  --run_name fold0_per_sample_mae
```

五折训练：

```bash
cd run

for i in 0 1 2 3 4
do
  python -W ignore driver.py \
    ./folds/train${i}.txt \
    ./folds/valid${i}.txt \
    -c config.json \
    --balance unmasked \
    --eval_every 1 \
    --single_gpu \
    --run_name fold${i}_per_sample_mae
done
```

说明：

```text
--single_gpu 不是算法必需项，但用于确认 MAE 逻辑时更稳妥，batch 边界更直接。
```

训练结果看：

```text
run/output/<run_name>/run.log
run/output/<run_name>/Model_metrics.json
```

## 7. 如何在 benchmark 上测试

benchmark 列表通常是：

```text
run/folds/id.txt
```

如果使用已有 checkpoint 做 benchmark，注意第一性原则：

```text
评测时的数据 scaler 必须和该 checkpoint 训练时一致。
```

当前 `driver.py` 会用第一个参数的 train list 重新构建 scaler，然后加载 checkpoint。因此测试某个 fold 的 checkpoint 时，第一个参数应该用该 checkpoint 对应的训练列表。

示例：用 fold0 的模型评测 benchmark：

```bash
cd run

python -W ignore driver.py \
  ./folds/train0.txt \
  ./folds/id.txt \
  -c config.json \
  --balance unmasked \
  --epochs 0 \
  --eval_every 1 \
  --single_gpu \
  --load ./output/<fold0_run_name>/Model.best.tar \
  --run_name benchmark_fold0_per_sample_mae
```

结果位置：

```text
run/output/benchmark_fold0_per_sample_mae/run.log
run/output/benchmark_fold0_per_sample_mae/predictions/
```

`run.log` 中每个 benchmark 样本的 `mae` 已经是该样本自己的 MAE。要得到整个 benchmark 的 mean MAE，应对这些样本级 MAE 再求平均。

## 8. 汇报时建议怎么描述

可以写成：

```text
We changed DeepPBS PWM MAE from column-level averaging to per-sample averaging.
For each input complex, MAE is first computed over its valid aligned PWM window
from both DNA directions, and the final benchmark MAE is then averaged across
complexes. This removes the previous bias where samples with longer aligned
windows contributed more to the final MAE.
```

中文：

```text
我们将 DeepPBS 的 PWM MAE 从 column-level average 改为 per-sample average。
现在每个复合物样本会先在自己的有效对齐窗口内计算 MAE，再对所有样本等权平均。
这样避免了长对齐窗口样本在最终 MAE 中权重更大的问题。
```

同时需要说明：

```text
该指标仍然是 aligned-window MAE，不是完整 motif MAE。
```
