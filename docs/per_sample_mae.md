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
  --output_path ./output \
  --run_name fold0_per_sample_mae
```

五折训练：

```bash
cd run

mkdir -p logs

for i in 0 1 2 3 4
do
  echo "===== fold ${i} start $(date) =====" | tee logs/fold${i}_per_sample_mae.log

  PYTHONUNBUFFERED=1 python -u -W ignore driver.py \
    ./folds/train${i}.txt \
    ./folds/valid${i}.txt \
    -c config.json \
    --balance unmasked \
    --eval_every 1 \
    --single_gpu \
    --output_path ./output \
    --run_name fold${i}_per_sample_mae \
    2>&1 | tee -a logs/fold${i}_per_sample_mae.log

  echo "===== fold ${i} end $(date) =====" | tee -a logs/fold${i}_per_sample_mae.log
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

原始 DeepPBS 的预测风格不是只用单个 checkpoint，而是 5 个模型做 ensemble。

```text
5 个 fold 模型 -> 分别预测 -> softmax 输出求平均 -> 合并正反两个方向
```

代码位置是：

```text
run/predict.py
```

以 DeepPBS 主模型为例，5 个 checkpoint 名字来自：

```text
run/plot_scripts/txts/DeepPBS.txt
```

每个模型会使用自己训练时保存的 scaler：

```text
run/output/<run_name>/scaler.pkl
```

然后加载对应 checkpoint：

```text
run/output/<run_name>/Model.best.tar
```

ensemble 的核心逻辑是：

```python
outputs.append(torch.softmax(model(batch), dim=1))
output = reduce(lambda x, y: x + y, outputs)
output = output / len(models)
```

DeepPBS 还会把正反两个方向合并成最终 PWM：

```python
idx = output.shape[0] // 2
output = (output[:idx, :] + np.flip(output[idx:, :])) / 2
```

也就是说，如果要保持和原始 DeepPBS 一致的 benchmark 风格，应该用：

```text
5-model ensemble prediction + per-sample MAE evaluation
```

不是只跑某一个 fold 的单模型 MAE。

当前仓库里的 `run/predict.py` 主要负责生成 ensemble 预测文件，不直接计算 benchmark MAE。因此新增了：

```text
run/benchmark_per_sample_mae.py
```

这个脚本复用 `run/predict.py` 的 5-model ensemble 逻辑，然后对每个 id 样本计算 per-sample MAE。

这样最贴近 DeepPBS 原始 inference 流程，也能避免 scaler/checkpoint 混用。

运行命令：

```bash
cd run

python -W ignore benchmark_per_sample_mae.py \
  ./folds/id.txt \
  -c config.json \
  --model_list ./plot_scripts/txts/DeepPBS.txt \
  --output ./output/benchmark_per_sample_mae.csv
```

如果用的是其他模型分支，替换 `--model_list`：

```text
DeepPBS:                ./plot_scripts/txts/DeepPBS.txt
DeepPBSwithDNAseqInfo:  ./plot_scripts/txts/DeepPBSwithDNAseqInfo.txt
BaseReadout:            ./plot_scripts/txts/BaseReadout.txt
ShapeReadout:           ./plot_scripts/txts/ShapeReadout.txt
```

输出文件：

```text
run/output/benchmark_per_sample_mae.csv
```

包含字段：

```text
id
mae
aligned_len
pwm_len
dna_len
pwm_coverage
dna_coverage
```

终端会同时打印：

```text
num_samples
mean_mae
output
```

需要保留的逻辑包括：

```text
1. 从 run/plot_scripts/txts/DeepPBS.txt 读取 5 个 run name
2. 每个 run 加载自己的 scaler.pkl
3. 每个 run 加载自己的 Model.best.tar
4. 对同一个 benchmark 样本得到 5 份 softmax prediction
5. 对 5 份 prediction 求平均
6. 将 strand0 和 strand1 反向翻回后平均成最终 PWM
7. 对每个 benchmark 样本单独计算 MAE
8. 对所有样本 MAE 求平均
```

正式 benchmark 应保持这个 5-model ensemble 流程。
