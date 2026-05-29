# Contact-aware Alignment 重新处理、训练、测试流程

## 目标

用新的 contact-aware PWM alignment 重新生成训练 `.npz`，再用同一模型配置重训，和原始 DeepPBS baseline 做对比。

核心原则：

| 原则 | 说明 |
|---|---|
| 不覆盖原数据 | 新 `.npz` 单独放目录 |
| 不改模型结构 | 只验证 label/alignment 策略变化 |
| 尽量复用原 folds | 保证和 baseline 可比 |
| 不做 PWM-only 回退 | 无 contact window 的样本应跳过 |

## 0. 拉取新代码

在服务器上：

```bash
cd ~/DeepPBS
git switch contact-aware-alignment
git pull --ff-only
```

确认当前 commit 至少包含：

```bash
git log --oneline -3
```

应看到：

```text
735b0ee feat: add contact-aware PWM alignment
```

## 1. 准备实验目录

不要覆盖原始 DeepPBS 数据。

```bash
export EXP_ROOT=$HOME/deeppbs_contact_aware_exp
export NPZ_DIR=$EXP_ROOT/assembly
export OUT_DIR=$EXP_ROOT/output
export FOLD_DIR=$EXP_ROOT/folds

mkdir -p "$NPZ_DIR" "$OUT_DIR" "$FOLD_DIR"
```

## 2. 确认原始输入

需要两类原始输入：

| 输入 | 作用 |
|---|---|
| co-crystal PDB/mmCIF 文件目录 | `process_co_crystal.py` 读取结构 |
| PDB/PWM 对应表 | 每行 `pdb_file,pwm_id` |

例子：

```text
1abc.pdb,MA0001.1.jaspar
2xyz.cif,TF_NAME.H11MO.0.A
```

设置路径：

```bash
export PDB_DIR=/path/to/original_pdb_or_cif_dir
export INPUT_LIST=/path/to/original_cocrystal_pwm_list.csv
```

先检查：

```bash
test -d "$PDB_DIR" && echo "PDB_DIR ok"
test -f "$INPUT_LIST" && echo "INPUT_LIST ok"
head "$INPUT_LIST"
```

## 3. 写新的处理配置

在仓库根目录执行：

```bash
cd ~/DeepPBS

cat > run/process_config_contact_aware.json <<EOF
{
  "PDB_FILES_PATH": "$PDB_DIR",
  "FEATURE_DATA_PATH": "$NPZ_DIR"
}
EOF
```

检查：

```bash
cat run/process_config_contact_aware.json
```

## 4. 先处理小样本验证

先不要直接全量跑。取前 5 个样本：

```bash
head -5 "$INPUT_LIST" > "$EXP_ROOT/input_smoke.csv"
python -W ignore run/process_co_crystal.py "$EXP_ROOT/input_smoke.csv" run/process_config_contact_aware.json
```

检查新 `.npz` 是否包含 contact-aware 字段：

```bash
python - <<'PY'
import glob
import numpy as np
import os

npz_dir = os.environ["NPZ_DIR"]
files = sorted(glob.glob(npz_dir + "/*.npz"))
print("npz_count", len(files))
assert files, "no npz generated"

for f in files[:5]:
    z = np.load(f, allow_pickle=True)
    keys = set(z.files)
    required = {"contact_mask", "contact_counts", "dna_mask", "pwm_mask", "contacts", "aln_score"}
    missing = required - keys
    assert not missing, (f, missing)

    contact_mask = z["contact_mask"].astype(bool)
    dna_mask = z["dna_mask"][0].astype(bool)
    overlap = int((contact_mask & dna_mask).sum())
    print(os.path.basename(f), "contacts", int(z["contacts"][0]), "overlap_bases", overlap, "aln_score", float(z["aln_score"][0]))
    assert overlap > 0, f"{f} aligned region has no contact overlap"
PY
```

如果这里失败，先不要全量重跑。

## 5. 全量重新处理数据

确认 smoke test 通过后，清空 smoke 产物或换新目录：

```bash
rm -f "$NPZ_DIR"/*.npz
```

全量处理：

```bash
python -W ignore run/process_co_crystal.py "$INPUT_LIST" run/process_config_contact_aware.json \
  > "$EXP_ROOT/process_contact_aware.log" 2>&1
```

查看处理结果：

```bash
grep -E "ERROR|CONTACT COUNT" "$EXP_ROOT/process_contact_aware.log" | tail -50
find "$NPZ_DIR" -name "*.npz" | wc -l
```

## 6. 生成可用 folds

优先复用原始 folds，保证和 baseline 可比。但 contact-aware alignment 可能跳过部分样本，所以要过滤掉新目录里不存在的 `.npz`。

```bash
python - <<'PY'
import os
from pathlib import Path

repo = Path.home() / "DeepPBS"
npz_dir = Path(os.environ["NPZ_DIR"])
fold_dir = Path(os.environ["FOLD_DIR"])
fold_dir.mkdir(parents=True, exist_ok=True)

available = {p.name for p in npz_dir.glob("*.npz")}

for split in ["train", "valid"]:
    for i in range(5):
        src = repo / "run" / "folds" / f"{split}{i}.txt"
        dst = fold_dir / f"{split}{i}.txt"
        rows = [x.strip() for x in src.read_text().splitlines() if x.strip()]
        kept = [x for x in rows if x in available]
        missing = [x for x in rows if x not in available]
        dst.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"{split}{i}: kept={len(kept)} missing={len(missing)}")
PY
```

如果 missing 很多，需要先看处理日志，确认是不是 PDB 路径、PWM id 或 contact 过严导致。

## 7. 写训练配置

从原配置复制，只改 `data_dir` 和 `output_path`：

```bash
cd ~/DeepPBS/run
cp config.json config_contact_aware.json
```

编辑 `config_contact_aware.json`：

```json
"data_dir": "/home/dangqi/deeppbs_contact_aware_exp/assembly",
"output_path": "/home/dangqi/deeppbs_contact_aware_exp/output"
```

确认：

```bash
grep -E '"data_dir"|"output_path"|"epochs"|"batch_size"|"condition"' config_contact_aware.json
```

## 8. 先训练 fold0

先跑单折验证数据和训练流程：

```bash
cd ~/DeepPBS/run

python -W ignore driver.py "$FOLD_DIR/train0.txt" "$FOLD_DIR/valid0.txt" \
  -c config_contact_aware.json \
  --balance unmasked \
  --eval_every 1 \
  --run_name contact_aware_fold0
```

检查输出：

```bash
ls -lh "$OUT_DIR/contact_aware_fold0"
tail -80 "$OUT_DIR/contact_aware_fold0/run.log"
```

## 9. 训练 5-fold

单折没问题后跑 5 折。

如果直接后台跑：

```bash
cd ~/DeepPBS/run

for i in 0 1 2 3 4
do
  python -W ignore driver.py "$FOLD_DIR/train${i}.txt" "$FOLD_DIR/valid${i}.txt" \
    -c config_contact_aware.json \
    --balance unmasked \
    --eval_every 1 \
    --run_name contact_aware_fold${i} \
    > "$OUT_DIR/contact_aware_fold${i}.stdout" 2>&1 &
done
wait
```

如果用集群调度，按服务器资源管理方式把上面每个 fold 拆成一个 job。

## 10. 基本测试和质控

### 10.1 label contact overlap

确认训练数据中所有 aligned region 都有 contact overlap：

```bash
python - <<'PY'
import glob
import os
import numpy as np

bad = []
for f in glob.glob(os.environ["NPZ_DIR"] + "/*.npz"):
    z = np.load(f, allow_pickle=True)
    contact_mask = z["contact_mask"].astype(bool)
    dna_mask = z["dna_mask"][0].astype(bool)
    if (contact_mask & dna_mask).sum() == 0:
        bad.append(os.path.basename(f))

print("bad_overlap_count", len(bad))
print(bad[:20])
assert not bad
PY
```

### 10.2 训练输出完整性

```bash
for i in 0 1 2 3 4
do
  test -f "$OUT_DIR/contact_aware_fold${i}/Model.best.tar" && echo "fold$i best ok"
  test -f "$OUT_DIR/contact_aware_fold${i}/Model_metrics.json" && echo "fold$i metrics ok"
done
```

### 10.3 汇总 metrics

```bash
python - <<'PY'
import glob
import json
import os
from statistics import mean

out_dir = os.environ["OUT_DIR"]
rows = []
for f in sorted(glob.glob(out_dir + "/contact_aware_fold*/Model_metrics.json")):
    with open(f) as fh:
        m = json.load(fh)

    best_epoch = m.get("best_epoch")
    if best_epoch is None:
        idx = -1
    else:
        idx = m["epochs"].index(best_epoch)

    validation = m["validation"]
    row = {
        "run": os.path.basename(os.path.dirname(f)),
        "best_epoch": best_epoch,
        "mae": validation["mae"][idx],
        "ic_weighted_pcc": validation["ic_weighted_pcc"][idx],
        "auroc": validation["auroc"][idx],
        "loss": validation["loss"][idx],
    }
    rows.append(row)
    print(row)

for key in ["mae", "ic_weighted_pcc", "auroc", "loss"]:
    vals = [r[key] for r in rows]
    print("mean", key, mean(vals))
PY
```

如果 metrics JSON 结构和预期不同，直接看每个 fold 的 `run.log`：

```bash
grep -E "validation|mae|IC|auroc|best" "$OUT_DIR"/contact_aware_fold*/run.log | tail -100
```

## 11. 和 baseline 对比

至少比较这些：

| 对比项 | baseline | contact-aware |
|---|---|---|
| `.npz` 数量 | 原数据目录 | `$NPZ_DIR` |
| 平均 `contacts` | 原 `.npz` | 新 `.npz` |
| 平均 `aln_score` | 原 `.npz` | 新 `.npz` |
| validation MAE | 原训练输出 | 新训练输出 |
| validation PCC/IC 指标 | 原训练输出 | 新训练输出 |

先做 `.npz` 级别对比：

```bash
export BASELINE_NPZ_DIR=/path/to/original/deeppbs/assembly

python - <<'PY'
import glob
import os
import numpy as np
from statistics import mean

def collect(npz_dir):
    rows = []
    for f in glob.glob(npz_dir + "/*.npz"):
        z = np.load(f, allow_pickle=True)
        row = {
            "name": os.path.basename(f),
            "contacts": float(z["contacts"][0]) if "contacts" in z.files else None,
            "aln_score": float(z["aln_score"][0]) if "aln_score" in z.files else None,
            "align_len": int(z["pwm_mask"][0].sum()) if "pwm_mask" in z.files else None,
        }
        if "contact_mask" in z.files:
            row["contact_overlap"] = int((z["contact_mask"].astype(bool) & z["dna_mask"][0].astype(bool)).sum())
        rows.append(row)
    return rows

baseline = collect(os.environ["BASELINE_NPZ_DIR"])
new = collect(os.environ["NPZ_DIR"])

print("baseline_count", len(baseline))
print("contact_aware_count", len(new))

for label, rows in [("baseline", baseline), ("contact_aware", new)]:
    print(label)
    for key in ["contacts", "aln_score", "align_len", "contact_overlap"]:
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        if vals:
            print(" ", key, "mean", mean(vals), "min", min(vals), "max", max(vals))
PY
```

注意：

| 现象 | 解释 |
|---|---|
| 新 `aln_score` 下降 | 正常，因为不再允许无 contact 的高 PWM window |
| 新 `.npz` 数量减少 | 可能正常，无 contact window 的样本被跳过 |
| `contacts` 增加 | 预期现象 |
| validation 变好 | 支持新 alignment 策略 |
| validation 变差 | 需要进一步看跳过样本数量、contact 定义和 label 长度分布 |

## 12. 最小结论标准

只有同时满足下面条件，才算这轮实验有效：

| 条件 | 必须满足 |
|---|---|
| 新 `.npz` 包含 `contact_mask/contact_counts` | 是 |
| 每个训练样本 aligned region 与 contact mask 有 overlap | 是 |
| 训练 folds 与 baseline 尽量一致 | 是 |
| 至少 fold0 成功跑完 | 是 |
| 有 baseline vs contact-aware 指标对比 | 是 |
