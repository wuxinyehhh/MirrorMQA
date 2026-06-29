#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"



# 8卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SWIFT_HOME="${SWIFT_HOME:-$REPO_ROOT/ms-swift}"
export PYTHONPATH="$SWIFT_HOME:${PYTHONPATH:-}"

export XDG_CACHE_HOME=$SWIFT_HOME/.cache
export MODELSCOPE_CACHE=$SWIFT_HOME/.cache/modelscope
export HF_HOME=$SWIFT_HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export SWIFT_PYTHON="${SWIFT_PYTHON:-python}"


# 确保目录存在
mkdir -p "$MODELSCOPE_CACHE/hub/_github" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# DeepSpeed: avoid fused_adam build on old toolchains
export DS_BUILD_OPS=0
export DS_USE_DEFAULT_ADAM=1

"$SWIFT_PYTHON" - <<'PY'
import sys
import transformers

def parse_ver(v: str):
    parts = []
    for tok in v.replace("+", ".").replace("-", ".").split("."):
        if tok.isdigit():
            parts.append(int(tok))
        else:
            break
    return tuple(parts + [0] * (3 - len(parts)))

ver = parse_ver(transformers.__version__)
if not (parse_ver("4.36.0") <= ver < parse_ver("4.49.0")):
    print(f"ERROR: transformers=={transformers.__version__} is unsupported. "
          "Please install: pip install \"transformers>=4.36,<4.49\" -U")
    sys.exit(1)

try:
    import soundfile  # noqa: F401
except Exception:
    print("ERROR: soundfile not installed. Please install: pip install soundfile -U")
    sys.exit(1)
PY

OUTPUT_BASE="$REPO_ROOT/output/Phi4"

"$SWIFT_PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port=29501 \
  $SWIFT_HOME/swift/cli/sft.py \
  --model $REPO_ROOT/models/Phi4 \
  --model_type phi4_multimodal \
  --split_dataset_ratio 0.0 \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --custom_dataset_info $REPO_ROOT/Dataset/dataset_info.json \
  --dataset images_zhouyan \
  --val_dataset images_zhouyan_val \
  --gradient_accumulation_steps 8 \
  --eval_steps 20 \
  --save_steps 40 \
  --save_total_limit 5 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 8 \
  --deepspeed zero3 \
  --model_name swift-bot \
  --model_author swift

# Post-process: generate train/eval loss plot and copy into each checkpoint-* directory
RUN_DIR="$(ls -1dt "$OUTPUT_BASE"/v* 2>/dev/null | head -n1)"
if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/logging.jsonl" ]; then
  RUN_DIR="$RUN_DIR" "$SWIFT_PYTHON" - <<'PY'
import json
from pathlib import Path
import os

run_dir = Path(os.environ.get("RUN_DIR", ""))
log_file = run_dir / "logging.jsonl"
if not log_file.exists():
    raise SystemExit(0)

steps_train, loss_train = [], []
steps_eval, loss_eval = [], []
with log_file.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        step = obj.get("global_step") or obj.get("step")
        if "loss" in obj and step is not None:
            loss_train.append(obj["loss"])
            steps_train.append(step)
        if "eval_loss" in obj and step is not None:
            loss_eval.append(obj["eval_loss"])
            steps_eval.append(step)

if not steps_train and not steps_eval:
    raise SystemExit(0)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"[WARN] matplotlib not available, skip plot: {e}")
    raise SystemExit(0)

plt.figure(figsize=(8, 5))
if steps_train:
    plt.plot(steps_train, loss_train, label="train_loss")
if steps_eval:
    plt.plot(steps_eval, loss_eval, label="eval_loss")
plt.xlabel("global_step")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
out_path = run_dir / "loss_curve.png"
plt.savefig(out_path, dpi=150)
print(f"[INFO] saved {out_path}")
PY

  if [ -f "$RUN_DIR/loss_curve.png" ]; then
    LAST_CKPT="$(ls -1dt "$RUN_DIR"/checkpoint-* 2>/dev/null | head -n1)"
    if [ -n "$LAST_CKPT" ] && [ -d "$LAST_CKPT" ]; then
      cp -f "$RUN_DIR/loss_curve.png" "$LAST_CKPT"/
    fi
  fi
fi
