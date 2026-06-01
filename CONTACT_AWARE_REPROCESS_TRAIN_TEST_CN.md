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
export RAW_CIF=$EXP_ROOT/raw_cif
export PDB_DIR=$EXP_ROOT/pdb_chain_cif
export NPZ_DIR=$EXP_ROOT/assembly
export OUT_DIR=$EXP_ROOT/output
export FOLD_DIR=$EXP_ROOT/folds

mkdir -p "$RAW_CIF" "$PDB_DIR" "$NPZ_DIR" "$OUT_DIR" "$FOLD_DIR"
```

## 2. 从原 DeepPBS 数据还原输入

本地没有完整保存 656 个 co-crystal 原始 PDB/mmCIF。现有数据位置：

```bash
export ORIG_DEEPPBS_DATA=$HOME/deeppbs_repro/deeppbs_data/deeppbsmar24
```

主要文件：

| 路径 | 作用 |
|---|---|
| `$ORIG_DEEPPBS_DATA/run/folds/` | 原 DeepPBS fold / 样本列表 |
| `$ORIG_DEEPPBS_DATA/data/assembly2024/` | 原 DeepPBS 已处理好的全量 `.npz` |
| `$ORIG_DEEPPBS_DATA/run/process/pdb/` | demo/少量结构文件，不是全量 co-crystal |
| `$ORIG_DEEPPBS_DATA/run/process/process_config.json` | 原处理配置 |

注意：

```text
run/process/pdb/ 只有少量 demo 结构，不是完整原始结构目录。
完整样本名在 run/folds/*.txt。
原始 PDB/mmCIF 需要按 pdb_id 从 RCSB 重新下载。
```

检查原始 fold：

```bash
cd "$ORIG_DEEPPBS_DATA"
ls run/folds/
head run/folds/train0.txt
head run/folds/valid0.txt
```

样本名示例：

```text
7jsl_J_MA0760.1.jaspar.npz
```

拆解为：

| 字段 | 值 |
|---|---|
| `pdb_id` | `7jsl` |
| `protein_chain` | `J` |
| `pwm_id` | `MA0760.1.jaspar` |

### 2.1 从 folds 生成处理输入表

DeepPBS 的 `process_co_crystal.py` 输入表需要每行：

```text
pdb_file,pwm_id
```

我们用 `pdb_chain.cif` 作为 `pdb_file`，这样原代码里的 chain 解析逻辑仍然可用。

```bash
cd "$ORIG_DEEPPBS_DATA"

python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["ORIG_DEEPPBS_DATA"])
out = Path(os.environ["EXP_ROOT"])
samples = set()

for f in (root / "run/folds").glob("*.txt"):
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stem = line[:-4] if line.endswith(".npz") else line
        pdb_id, chain, pwm_id = stem.split("_", 2)
        samples.add((pdb_id.lower(), chain, pwm_id))

(out / "contact_aware_input.csv").write_text(
    "\n".join(f"{pdb}_{chain}.cif,{pwm}" for pdb, chain, pwm in sorted(samples)) + "\n"
)
(out / "pdb_ids.txt").write_text(
    "\n".join(sorted({pdb for pdb, _, _ in samples})) + "\n"
)

print("samples", len(samples))
print("pdb_ids", len({pdb for pdb, _, _ in samples}))
PY
```

设置输入表：

```bash
export INPUT_LIST=$EXP_ROOT/contact_aware_input.csv
```

检查：

```bash
test -f "$INPUT_LIST" && echo "INPUT_LIST ok"
head "$INPUT_LIST"
wc -l "$INPUT_LIST" "$EXP_ROOT/pdb_ids.txt"
```

### 2.2 从 RCSB 下载原始 CIF

```bash
cd "$RAW_CIF"

while read pdb; do
  test -f "${pdb}.cif" || curl -L -f "https://files.rcsb.org/download/${pdb^^}.cif" -o "${pdb}.cif"
done < "$EXP_ROOT/pdb_ids.txt"
```

检查：

```bash
find "$RAW_CIF" -name "*.cif" | wc -l
```

### 2.3 创建 `pdb_chain.cif` 软链接

`process_co_crystal.py` 会从文件名解析 chain：

```text
7jsl_J.cif -> chain J
```

所以需要把同一个 `7jsl.cif` 链接成不同 chain 名。

```bash
python - <<'PY'
import os
from pathlib import Path

exp = Path(os.environ["EXP_ROOT"])
raw = Path(os.environ["RAW_CIF"])
pdb_dir = Path(os.environ["PDB_DIR"])

for line in (exp / "contact_aware_input.csv").read_text().splitlines():
    pdb_file, _ = line.split(",", 1)
    pdb_id = pdb_file.split("_")[0]
    src = raw / f"{pdb_id}.cif"
    dst = pdb_dir / pdb_file
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

print("linked", len(list(pdb_dir.glob("*.cif"))))
PY
```

最终确认：

```bash
test -d "$PDB_DIR" && echo "PDB_DIR ok"
test -f "$INPUT_LIST" && echo "INPUT_LIST ok"
find "$PDB_DIR" -name "*.cif" | wc -l
head "$INPUT_LIST"
```

到这里，真实输入是：

| 变量 | 值 |
|---|---|
| `PDB_DIR` | `$EXP_ROOT/pdb_chain_cif` |
| `INPUT_LIST` | `$EXP_ROOT/contact_aware_input.csv` |


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

## 4. 加载 DeepPBS 预处理依赖

DeepPBS 的 DNA 预处理会调用 `x3dna-dssr`。如果没有把仓库里的 `dependencies/bin` 加到 `PATH`，`processDNA()` 会失败，并且早期脚本可能表现为没有生成任何 `.npz`。

在每次新 shell / 新任务脚本里都执行：

```bash
cd ~/DeepPBS

export PATH="$PWD/dependencies/bin:$PATH"
export X3DNA="$PWD/x3dna-v2.3-linux-64bit/x3dna-v2.3"
```

确认：

```bash
which x3dna-dssr
x3dna-dssr --help | head
```

期望看到：

```text
~/DeepPBS/dependencies/bin/x3dna-dssr
```

如果 `which x3dna-dssr` 没有输出：

```bash
ls -lh dependencies/bin/x3dna-dssr
chmod +x dependencies/bin/x3dna-dssr
export PATH="$PWD/dependencies/bin:$PATH"
```

## 5. 先处理小样本验证

先不要直接全量跑。取前 5 个样本：

```bash
head -5 "$INPUT_LIST" > "$EXP_ROOT/input_smoke.csv"

python -W ignore run/process_co_crystal.py \
  "$EXP_ROOT/input_smoke.csv" \
  run/process_config_contact_aware.json \
  2>&1 | tee "$EXP_ROOT/smoke_process.log"
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

如果 `npz_count` 是 0，先看日志：

```bash
grep -E "PROCESSING|ERROR|CONTACT COUNT|Traceback" "$EXP_ROOT/smoke_process.log"
```

常见问题：

| 现象 | 原因 | 处理 |
|---|---|---|
| `FileNotFoundError: x3dna-dssr` | 没加载 `dependencies/bin` | 回到第 4 步设置 `PATH` |
| 没有任何 `.npz` | 样本被异常跳过 | 看 `smoke_process.log` 的 `ERROR/Traceback` |
| `structure load error` | CIF 软链接或下载失败 | 检查 `$PDB_DIR` 和 `$RAW_CIF` |
| `contact-aware alignment error` | 该样本没有 contact-overlap window | 样本会被跳过，继续统计全量影响 |

## 6. 全量重新处理数据

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

当前实验记录：

| 项 | 结果 |
|---|---|
| 新 `.npz` 数量 | `439` |
| 正常日志 | 大量 `CONTACT COUNT` |
| 主要跳过原因 | `ERROR: helix count problem 2` |

解释：

```text
helix count problem 2 表示该结构检测到不止一个 DNA helix，被 DeepPBS 原始处理逻辑跳过。
```

## 7. 生成可用 folds

优先复用原始 folds，保证和 baseline 可比。但 contact-aware alignment 可能跳过部分样本，所以要过滤掉新目录里不存在的 `.npz`。

```bash
python - <<'PY'
import os
from pathlib import Path

orig = Path(os.environ["ORIG_DEEPPBS_DATA"])
npz_dir = Path(os.environ["NPZ_DIR"])
fold_dir = Path(os.environ["FOLD_DIR"])
fold_dir.mkdir(parents=True, exist_ok=True)

available = {p.name for p in npz_dir.glob("*.npz")}

for split in ["train", "valid"]:
    for i in range(5):
        src = orig / "run" / "folds" / f"{split}{i}.txt"
        dst = fold_dir / f"{split}{i}.txt"
        rows = [x.strip() for x in src.read_text().splitlines() if x.strip()]
        kept = [x for x in rows if x in available]
        missing = [x for x in rows if x not in available]
        dst.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"{split}{i}: kept={len(kept)} missing={len(missing)}")
PY
```

如果 missing 很多，需要先看处理日志，确认是不是 PDB 路径、PWM id 或 contact 过严导致。

当前实验 fold 过滤结果：

| fold | train kept | train missing | valid kept | valid missing |
|---|---:|---:|---:|---:|
| 0 | 276 | 143 | 65 | 39 |
| 1 | 270 | 149 | 71 | 33 |
| 2 | 269 | 150 | 72 | 32 |
| 3 | 276 | 143 | 65 | 39 |
| 4 | 276 | 143 | 65 | 39 |

## 8. 检查 label-contact overlap

训练前必须确认每个新 label 的 aligned region 都与 contact mask 有交集：

```bash
python - <<'PY'
import glob
import os
import numpy as np

bad = []
for f in glob.glob(os.environ["NPZ_DIR"] + "/*.npz"):
    z = np.load(f, allow_pickle=True)
    overlap = (z["contact_mask"].astype(bool) & z["dna_mask"][0].astype(bool)).sum()
    if overlap == 0:
        bad.append(os.path.basename(f))

print("bad_overlap_count", len(bad))
print(bad[:20])
assert not bad
PY
```

如果 `bad_overlap_count` 不是 0，先不要训练。

## 9. 写训练配置

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

## 10. 先训练 fold0

先跑单折验证数据和训练流程。

重要经验：

| 经验 | 原因 |
|---|---|
| 加 `--single_gpu` | PyG `DataParallel` 在 2 GPU 下可能报 `RuntimeError: Could not infer dtype of NoneType` |
| 用 `nohup` 写日志 | 训练时间长，断开 shell 不影响进程 |

不要直接用默认多 GPU。当前可用命令：

```bash
cd ~/DeepPBS/run

nohup python -W ignore driver.py "$FOLD_DIR/train0.txt" "$FOLD_DIR/valid0.txt" \
  -c config_contact_aware.json \
  --balance unmasked \
  --eval_every 1 \
  --single_gpu \
  --run_name contact_aware_fold0_single_gpu \
  > "$OUT_DIR/contact_aware_fold0_single_gpu.nohup.log" 2>&1 &
```

检查启动日志：

```bash
tail -f "$OUT_DIR/contact_aware_fold0_single_gpu.nohup.log"
```

期望看到：

```text
INFO:    Running model on device cuda:0.
INFO:    Beginning Training (50 epochs)
```

检查完成输出：

```bash
ls -lh "$OUT_DIR/contact_aware_fold0_single_gpu"
tail -80 "$OUT_DIR/contact_aware_fold0_single_gpu/run.log"
```

当前 fold0 结果：

| 项 | 结果 |
|---|---|
| 训练状态 | `Training Successfully Ended` |
| best epoch | `46` |
| best tracked metric | `0.715` |
| 模型文件 | `Model.best.tar` 已生成 |
| metrics 文件 | `Model_metrics.json` 已生成 |
| 验证预测 | `validation_set_predictions.npz` 已生成 |

## 11. 训练 5-fold

单折没问题后跑 5 折。

继续保留 `--single_gpu`，避免 PyG `DataParallel` 报错。fold0 已经跑完时，可以直接并发跑剩余 fold1-fold4：

```bash
cd ~/DeepPBS/run

for i in 1 2 3 4
do
  nohup python -W ignore driver.py "$FOLD_DIR/train${i}.txt" "$FOLD_DIR/valid${i}.txt" \
    -c config_contact_aware.json \
    --balance unmasked \
    --eval_every 1 \
    --single_gpu \
    --run_name contact_aware_fold${i}_single_gpu \
    > "$OUT_DIR/contact_aware_fold${i}_single_gpu.nohup.log" 2>&1 &
done
```

如果需要从头重跑 5 折，把循环改成 `for i in 0 1 2 3 4`。

## 12. 基本测试和质控

### 12.1 label contact overlap

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

### 12.2 训练输出完整性

```bash
for i in 0 1 2 3 4
do
  d="$OUT_DIR/contact_aware_fold${i}_single_gpu"
  echo "===== fold$i ====="
  grep -q "Training Successfully Ended" "$d/run.log" && echo "train ok" || echo "train failed"
  grep -E "Writing best state|Best tracked metric" "$d/run.log" | tail -2
  test -f "$d/Model.best.tar" && echo "best model ok"
  test -f "$d/Model_metrics.json" && echo "metrics ok"
  test -f "$d/validation_set_predictions.npz" && echo "valid pred ok"
done
```

当前 5-fold 完整性检查结果：

| fold | 状态 | best epoch | best tracked metric |
|---|---|---:|---:|
| 0 | `train ok` | 46 | 0.715 |
| 1 | `train ok` | 12 | 0.665 |
| 2 | `train ok` | 50 | 0.659 |
| 3 | `train ok` | 47 | 0.671 |
| 4 | `train ok` | 37 | 0.637 |

### 12.3 汇总 metrics

```bash
python - <<'PY'
import glob, json, os
from statistics import mean

rows = []
for f in sorted(glob.glob(os.environ["OUT_DIR"] + "/contact_aware_fold*_single_gpu/Model_metrics.json")):
    m = json.load(open(f))
    epochs = m["epochs"]
    best_epoch = m.get("best_epoch", epochs[-1])
    idx = epochs.index(best_epoch) if best_epoch in epochs else -1
    val = m["validation"]

    def get(k):
        xs = val.get(k)
        return xs[idx] if xs else None

    row = {
        "run": os.path.basename(os.path.dirname(f)),
        "best_epoch": best_epoch,
        "auroc": get("auroc"),
        "mae": get("mae"),
        "ic_weighted_pcc": get("ic_weighted_pcc"),
        "pearsonr": get("pearsonr"),
        "spearmanr": get("spearmanr"),
        "loss": get("loss"),
    }
    rows.append(row)

cols = ["run", "best_epoch", "auroc", "mae", "ic_weighted_pcc", "pearsonr", "spearmanr", "loss"]
print("\t".join(cols))
for r in rows:
    print("\t".join("" if r[c] is None else str(round(r[c], 4)) if isinstance(r[c], float) else str(r[c]) for c in cols))

print("\nMEAN")
for k in cols[2:]:
    vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
    if vals:
        print(k, round(mean(vals), 4))
PY
```

当前 contact-aware 5-fold validation 结果：

| fold | best epoch | auroc | mae | ic_weighted_pcc | pearsonr | spearmanr | loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 46 | 0.8259 | 0.7149 | 0.4321 | 0.5308 | 0.4100 | 0.7149 |
| 1 | 12 | 0.8344 | 0.6653 | 0.4577 | 0.5539 | 0.4234 | 0.6653 |
| 2 | 50 | 0.8587 | 0.6589 | 0.4649 | 0.5748 | 0.4417 | 0.6589 |
| 3 | 47 | 0.8368 | 0.6709 | 0.4729 | 0.5832 | 0.4658 | 0.6709 |
| 4 | 37 | 0.8581 | 0.6365 | 0.4720 | 0.6067 | 0.4719 | 0.6365 |
| mean | - | 0.8428 | 0.6693 | 0.4599 | 0.5699 | 0.4426 | 0.6693 |

如果 metrics JSON 结构和预期不同，直接看每个 fold 的 `run.log`：

```bash
grep -E "validation|mae|IC|auroc|best" "$OUT_DIR"/contact_aware_fold*_single_gpu/run.log | tail -100
```

## 13. 和 baseline 对比

下一步必须先做严格 internal baseline，再考虑外部 benchmark。

严格 baseline 的定义：

| 变量 | baseline_filtered | contact-aware |
|---|---|---|
| fold 列表 | `$FOLD_DIR/train*.txt` / `$FOLD_DIR/valid*.txt` | 同左 |
| 训练配置 | 同一套超参数 | 同左 |
| 模型代码 | 同一份代码 | 同左 |
| `.npz` 数据目录 | 原 DeepPBS `assembly2024` | 新 `$NPZ_DIR` |
| 唯一差异 | 原 PWM-only alignment label | contact-aware alignment label |

这个对照回答的是：只改变 alignment 策略，结果是否变好。

### 13.1 训练 baseline_filtered

先建 baseline 配置：

```bash
cd ~/DeepPBS/run

cp config_contact_aware.json config_baseline_filtered.json

python - <<'PY'
import json
from pathlib import Path

p = Path("config_baseline_filtered.json")
c = json.loads(p.read_text())
c["data_dir"] = "/home/dangqi/deeppbs_repro/deeppbs_data/deeppbsmar24/data/assembly2024"
c["output_path"] = "/home/dangqi/deeppbs_contact_aware_exp/output"
p.write_text(json.dumps(c, indent=2) + "\n")
PY
```

确认配置：

```bash
grep -E '"data_dir"|"output_path"|"best_state_metric"|"best_state_metric_goal"' config_baseline_filtered.json
```

跑同一批 filtered folds：

```bash
for i in 0 1 2 3 4
do
  nohup python -W ignore driver.py "$FOLD_DIR/train${i}.txt" "$FOLD_DIR/valid${i}.txt" \
    -c config_baseline_filtered.json \
    --balance unmasked \
    --eval_every 1 \
    --single_gpu \
    --run_name baseline_filtered_fold${i}_single_gpu \
    > "$OUT_DIR/baseline_filtered_fold${i}_single_gpu.nohup.log" 2>&1 &
done
```

检查进度：

```bash
jobs
tail -f "$OUT_DIR/baseline_filtered_fold0_single_gpu.nohup.log"
```

### 13.2 检查 baseline_filtered 输出

```bash
for i in 0 1 2 3 4
do
  d="$OUT_DIR/baseline_filtered_fold${i}_single_gpu"
  echo "===== baseline fold$i ====="
  grep -q "Training Successfully Ended" "$d/run.log" && echo "train ok" || echo "train failed"
  grep -E "Writing best state|Best tracked metric" "$d/run.log" | tail -2
  test -f "$d/Model.best.tar" && echo "best model ok"
  test -f "$d/Model_metrics.json" && echo "metrics ok"
  test -f "$d/validation_set_predictions.npz" && echo "valid pred ok"
done
```

### 13.3 汇总 baseline vs contact-aware

```bash
python - <<'PY'
import glob, json, os
from statistics import mean

out_dir = os.environ["OUT_DIR"]

def collect(pattern, label):
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, pattern, "Model_metrics.json"))):
        m = json.load(open(f))
        epochs = m["epochs"]
        best_epoch = m.get("best_epoch", epochs[-1])
        idx = epochs.index(best_epoch) if best_epoch in epochs else -1
        val = m["validation"]

        def get(k):
            xs = val.get(k)
            return xs[idx] if xs else None

        rows.append({
            "group": label,
            "run": os.path.basename(os.path.dirname(f)),
            "best_epoch": best_epoch,
            "auroc": get("auroc"),
            "mae": get("mae"),
            "ic_weighted_pcc": get("ic_weighted_pcc"),
            "pearsonr": get("pearsonr"),
            "spearmanr": get("spearmanr"),
            "loss": get("loss"),
        })
    return rows

rows = []
rows += collect("baseline_filtered_fold*_single_gpu", "baseline_filtered")
rows += collect("contact_aware_fold*_single_gpu", "contact_aware")

cols = ["group", "run", "best_epoch", "auroc", "mae", "ic_weighted_pcc", "pearsonr", "spearmanr", "loss"]
print("\t".join(cols))
for r in rows:
    print("\t".join("" if r[c] is None else str(round(r[c], 4)) if isinstance(r[c], float) else str(r[c]) for c in cols))

print("\nMEAN")
for group in ["baseline_filtered", "contact_aware"]:
    sub = [r for r in rows if r["group"] == group]
    print(group)
    for k in cols[3:]:
        vals = [r[k] for r in sub if isinstance(r[k], (int, float))]
        if vals:
            print(" ", k, round(mean(vals), 4))
PY
```

判定逻辑：

| 结果 | 结论 |
|---|---|
| contact-aware 明显优于 baseline_filtered | alignment 策略有正向信号，可以继续跑外部 benchmark |
| 两者接近 | 需要看 benchmark，不能只靠 internal 指标判断 |
| contact-aware 明显差于 baseline_filtered | 当前 hard contact-aware alignment 基本判负，先不要扩展实验 |

### 13.4 `.npz` 级别对比

训练指标之外，至少比较这些：

| 对比项 | baseline | contact-aware |
|---|---|---|
| `.npz` 数量 | 原数据目录 | `$NPZ_DIR` |
| 平均 `contacts` | 原 `.npz` | 新 `.npz` |
| 平均 `aln_score` | 原 `.npz` | 新 `.npz` |
| validation MAE | 原训练输出 | 新训练输出 |
| validation PCC/IC 指标 | 原训练输出 | 新训练输出 |

`.npz` 级别对比命令：

```bash
export BASELINE_NPZ_DIR=$ORIG_DEEPPBS_DATA/data/assembly2024

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

## 14. 最小结论标准

只有同时满足下面条件，才算这轮实验有效：

| 条件 | 必须满足 |
|---|---|
| 新 `.npz` 包含 `contact_mask/contact_counts` | 是 |
| 每个训练样本 aligned region 与 contact mask 有 overlap | 是 |
| 训练 folds 与 baseline 尽量一致 | 是 |
| 至少 fold0 成功跑完 | 是 |
| 有 baseline vs contact-aware 指标对比 | 是 |
