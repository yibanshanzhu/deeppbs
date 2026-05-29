# Contact-aware PWM Alignment Idea

## 核心判断

当前 DeepPBS 的 PWM alignment 本质是：

| 输入 | 当前作用 |
|---|---|
| PWM | 提供 TF motif 偏好 |
| DNA sequence | 提供候选窗口 |
| IC-weighted PCC | 给 PWM/window 打分 |
| protein-DNA contact | 只在 alignment 之后统计，不参与选窗口 |

这会导致一个问题：最高分 window 可能只是 motif-like sequence，但不在 protein-DNA interface 上。

我们要验证的定义是：

```text
bound motif = PWM-matched DNA window ∩ protein-DNA interface
```

## 最短路径方案

把 alignment 逻辑从：

```text
argmax PWM_score(window)
```

改成：

```text
argmax PWM_score(window)
where window overlaps protein-DNA contact mask
```

第一版只使用硬约束：

| 条件 | 规则 |
|---|---|
| contact mask | alignment 前先计算每个 DNA base 是否接触 protein |
| window 资格 | `contact_count(window) > 0` |
| window 排序 | 仍然用现有 IC-weighted PCC |
| tie-break | PWM 分数相同再比较 `contact_count` |

## 代码入口

| 文件 | 当前问题 | 需要调整 |
|---|---|---|
| `run/process_co_crystal.py` | 先 alignment，后统计 contact | 在 `computeYAndMask` 前计算 contact mask |
| `deeppbs/count_contacts.py` | 只能统计已选 `dna_mask` 的 contact count | 新增按 DNA base 返回 bool/count mask 的函数 |
| `deeppbs/compute_Y_and_mask.py` | 不知道 contact 信息 | 接收 contact mask，并传给 PWM alignment |
| `deeppbs/align_PWM_seq.py` | 遍历所有窗口，只按 PWM 分数选最高 | 跳过无 contact 的候选 window |

## 代码实现步骤

### 1. 生成 per-base contact mask

在 `deeppbs/count_contacts.py` 新增函数，例如：

```text
computeContactMask(protein, pdb, V_dna, cutoff=5.0)
```

输出：

| 返回值 | shape | 含义 |
|---|---|---|
| `contact_mask` | `(N,)` | 每个 DNA base 是否接触 protein |
| `contact_counts` | `(N,)` | 每个 DNA base 周围 protein atom contact 数 |

计算逻辑：

```text
对每个 DNA base 的所有 CG 点搜索 cutoff 内 protein atom
只要该 base 任一 CG 点有 contact，则 contact_mask[i] = True
```

`countContacts` 保留，用于统计最终 aligned region 的 contact count。

### 2. alignment 接收 contact 约束

修改 `deeppbs/align_PWM_seq.py`：

| 函数 | 调整 |
|---|---|
| `ungappedAlign` | 增加可选 `seq_contact_mask=None` |
| `alignPWMSeq` | 增加可选 `contact_mask=None` |

窗口筛选规则：

```text
如果 contact_mask is not None:
    只允许 contact_mask[seq_start : seq_start + k].sum() > 0 的 window
```

tie-break 规则：

```text
先比较 PWM score
PWM score 相同再比较 contact_count
```

如果 `contact_mask is None`，行为保持当前 PWM-only alignment。

### 3. 双链 contact mask 对齐

修改 `deeppbs/compute_Y_and_mask.py`：

```text
computeYAndMask(pwm, dna_seq, contact_mask=None)
```

传参规则：

| strand | `dna_seq` | contact mask |
|---|---|---|
| 正链 | `dna_seq[0,:,:]` | `contact_mask` |
| 反链 | `dna_seq[1,:,:]` | `np.flip(contact_mask)` |

原因：当前代码对反链的 `dna_mask` 使用 `np.flip`，contact 约束必须使用同一坐标方向。

### 4. 数据处理入口前移 contact

修改 `run/process_co_crystal.py`：

当前顺序：

```text
makeDNACG -> computeYAndMask -> makeProteinGraph -> countContacts
```

改成：

```text
makeDNACG -> computeContactMask -> computeYAndMask -> makeProteinGraph -> countContacts
```

即在：

```text
Y_pwm, pwm_mask, dna_mask, aln_score = computeYAndMask(pwm, dna_seq)
```

之前生成 `contact_mask`，然后改为：

```text
Y_pwm, pwm_mask, dna_mask, aln_score = computeYAndMask(pwm, dna_seq, contact_mask)
```

### 5. 保留可追踪输出

保存 `.npz` 时增加：

| 字段 | 用途 |
|---|---|
| `contact_mask` | 复查每个 base 是否在 interface |
| `contact_counts` | 分析 contact 强度 |

原字段 `contacts` 继续表示最终 aligned region 的 contact count。

## 重处理和重训顺序

必须先完成代码修改，再重处理数据；否则新训练仍然使用旧的 PWM-only label。

| 步骤 | 命令/动作 | 产物 |
|---|---|---|
| 1 | 切到 `contact-aware-alignment` | 新逻辑分支 |
| 2 | 实现并提交上述代码 | contact-aware label 生成逻辑 |
| 3 | 用原始 co-crystal PDB/PWM list 重跑 `run/process_co_crystal.py` | 新 `.npz` 数据目录 |
| 4 | 复制 `run/config.json`，只改 `data_dir` 和 `output_path` | 新训练配置 |
| 5 | 先跑 fold0 | 快速验证代码和数据 |
| 6 | 再跑 5-fold | 完整结果 |
| 7 | 对比 baseline 和 contact-aware metrics | 判断是否提升 |

建议新数据目录不要覆盖原数据：

```text
~/deeppbs_contact_aware_exp/assembly
```

建议新输出目录：

```text
~/deeppbs_contact_aware_exp/output
```

## 全链路逻辑

1. `makeDNACG` 生成 `V_dna`、`dna_seq`。
2. 用 protein 原子和 `V_dna` 计算 `contact_mask`。
3. `computeYAndMask(pwm, dna_seq, contact_mask)` 做 contact-aware alignment。
4. `alignPWMSeq` 只允许满足 `contact_count > 0` 的 window 参与竞争。
5. 输出的 `dna_mask` 天然应该落在 interface 附近。
6. 原有 `contacts` 继续统计最终选中窗口的 contact count，用于质控和数据过滤。

## 第一版验收

| 验证项 | 预期 |
|---|---|
| 原 alignment 结果 | 作为 baseline 保留用于对比实验 |
| 新 alignment 结果 | 选中窗口至少有一个 DNA base 接触 protein |
| `contacts` | 新结果中应显著减少 0-contact 或低-contact aligned region |
| PWM score | 可能下降，这是合理代价，因为目标从 motif-like window 改成 bound motif |

## 不做的事

| 不做 | 原因 |
|---|---|
| 不把 contact 加成软分数 | 第一版目标是验证定义，不引入额外权重超参 |
| 不做兜底回退到 PWM-only | 会把无 contact 的 motif-like window 混回来，偏离 bound motif 定义 |
| 不同时重构训练流程 | 先改数据生成的 label 逻辑，验证 alignment 是否更符合 interface |
