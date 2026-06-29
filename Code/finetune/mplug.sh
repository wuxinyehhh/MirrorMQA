#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 单卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SWIFT_HOME="${SWIFT_HOME:-$REPO_ROOT/ms-swift}"
export PYTHONPATH="$SWIFT_HOME:${PYTHONPATH:-}"

export SWIFT_PYTHON="${SWIFT_PYTHON:-python}"

# Cache layout:
# - Put assets (modelscope/hf hub weights, datasets, dynamic modules) under ms-swift/.cache by default.
: "${SWIFT_CACHE_ROOT:=$SWIFT_HOME/.cache}"
CACHE_ROOT_SHARED="$SWIFT_CACHE_ROOT"

# Allow override if you want a separate local cache root.
: "${SWIFT_CACHE_ROOT_LOCAL:=$CACHE_ROOT_SHARED}"
CACHE_ROOT_LOCAL="$SWIFT_CACHE_ROOT_LOCAL"

export XDG_CACHE_HOME="$CACHE_ROOT_SHARED"
export MODELSCOPE_CACHE="$CACHE_ROOT_SHARED/modelscope"
export HF_HOME="$CACHE_ROOT_SHARED/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# transformers 新版更推荐 HF_HOME；这里 TRANSFORMERS_CACHE 仍可保留，但会有 FutureWarning
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_MODULES_CACHE="$CACHE_ROOT_LOCAL/huggingface/modules"

# 确保目录存在
mkdir -p "$MODELSCOPE_CACHE/hub/_github" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_MODULES_CACHE"

# DeepSpeed: avoid fused_adam build on old toolchains
export DS_BUILD_OPS=0
export DS_USE_DEFAULT_ADAM=1
# Avoid mmap SIGBUS on some network filesystems
export SAFETENSORS_NO_MMAP=1

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
if not (parse_ver("4.36.0") <= ver):
    print(f"ERROR: transformers=={transformers.__version__} is unsupported. "
          "Please install: pip install \"transformers>=4.36.0\" -U")
    sys.exit(1)

for pkg, pip_name in [("soundfile", "soundfile"), ("decord", "decord"), ("icecream", "icecream")]:
    try:
        __import__(pkg)
    except Exception:
        print(f"ERROR: {pkg} not installed. Please install: pip install {pip_name} -U")
        sys.exit(1)
PY

OUTPUT_BASE="$REPO_ROOT/output/mPLUG_Owl3_7B"
MODEL_TYPE="${SWIFT_MODEL_TYPE:-mplug_owl3}"

# Prefer local-disk weights copy to avoid SIGBUS on some network filesystems.
MODEL_DIR_SHARED="$REPO_ROOT/models/mPLUG_Owl3_7B"
MODEL_DIR_LOCAL="$REPO_ROOT/models_local/mPLUG_Owl3_7B"

has_weights() {
  local d="$1"
  [ -f "$d/model.safetensors" ] \
    || [ -f "$d/model.safetensors.index.json" ] \
    || [ -f "$d/pytorch_model.bin" ] \
    || [ -f "$d/pytorch_model.bin.index.json" ] \
    || ls "$d"/model-*-of-*.safetensors >/dev/null 2>&1 \
    || ls "$d"/pytorch_model-*-of-*.bin >/dev/null 2>&1
}

if [ -n "${MODEL_DIR_OVERRIDE:-}" ]; then
  MODEL_DIR="$MODEL_DIR_OVERRIDE"
elif has_weights "$MODEL_DIR_LOCAL"; then
  MODEL_DIR="$MODEL_DIR_LOCAL"
else
  MODEL_DIR="$MODEL_DIR_SHARED"
fi

if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: MODEL_DIR is invalid: $MODEL_DIR" >&2
  exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "ERROR: $MODEL_DIR/config.json not found. Please provide a valid local model dir." >&2
  exit 1
fi

if [ "${MPLUG_OWL3_AUTO_DOWNLOAD:-0}" = "1" ] && ! has_weights "$MODEL_DIR"; then
  : "${MPLUG_OWL3_MODEL_ID:=iic/mPLUG-Owl3-7B-240728}"
  export MPLUG_OWL3_MODEL_ID
  echo "INFO: Downloading weights via ModelScope: $MPLUG_OWL3_MODEL_ID" >&2
  DOWNLOADED_DIR="$(
  "$SWIFT_PYTHON" - <<'PY'
import os
import sys

model_id = os.environ.get("MPLUG_OWL3_MODEL_ID")
cache_dir = os.environ.get("MODELSCOPE_CACHE")
if not model_id:
    print("ERROR: MPLUG_OWL3_MODEL_ID is empty", file=sys.stderr)
    sys.exit(1)
if not cache_dir:
    print("ERROR: MODELSCOPE_CACHE is empty", file=sys.stderr)
    sys.exit(1)

try:
    from modelscope import snapshot_download
except Exception as e:
    print(f"ERROR: modelscope import failed: {e}", file=sys.stderr)
    sys.exit(1)

local_dir = snapshot_download(model_id, cache_dir=cache_dir)
print(local_dir)
PY
)"
  if [ -z "$DOWNLOADED_DIR" ] || [ ! -d "$DOWNLOADED_DIR" ]; then
    echo "ERROR: ModelScope download failed, got: $DOWNLOADED_DIR" >&2
    exit 1
  fi
  MODEL_DIR="$DOWNLOADED_DIR"
fi

if ! has_weights "$MODEL_DIR"; then
  echo "ERROR: No model weights found in: $MODEL_DIR" >&2
  echo "Expected one of:" >&2
  echo "  - model.safetensors" >&2
  echo "  - model.safetensors.index.json + model-*.safetensors shards" >&2
  echo "  - pytorch_model.bin (or sharded pytorch_model-*.bin)" >&2
  echo "" >&2
  echo "Your directory currently looks like a 'code/tokenizer only' snapshot; you must download/copy weights first." >&2
  echo "" >&2
  echo "Option A (recommended): copy the weights into $MODEL_DIR_SHARED (shared FS), then re-run." >&2
  echo "If you downloaded to another path, you can point the script to it:" >&2
  echo "  export MODEL_DIR_OVERRIDE=\"$REPO_ROOT/models/mPLUG_Owl3_7B\"" >&2
  echo "  bash $0" >&2
  echo "Option B: auto-download from ModelScope cache (set SWIFT_CACHE_ROOT to a large persistent path):" >&2
  echo "  export SWIFT_CACHE_ROOT=\"$SWIFT_HOME/.cache\"  # or another big shared dir" >&2
  echo "  export MPLUG_OWL3_MODEL_ID=\"iic/mPLUG-Owl3-7B-240728\"" >&2
  echo "  export MPLUG_OWL3_AUTO_DOWNLOAD=1" >&2
  echo "  bash $0" >&2
  exit 1
fi

# ✅ 修复点：不要用 ${MODEL_DIR@Q}，改为通过环境变量传入 Python（兼容所有 bash）
export MODEL_DIR
"$SWIFT_PYTHON" - <<'PY'
import json
import os
import sys
import transformers

model_dir = os.environ.get("MODEL_DIR")
if not model_dir:
    print("ERROR: MODEL_DIR is not set")
    sys.exit(1)

config_path = os.path.join(model_dir, "config.json")
required = None
try:
    with open(config_path, "r", encoding="utf-8") as f:
        required = json.load(f).get("transformers_version")
except Exception as e:
    print(f"WARNING: failed to read {config_path}: {e}")

installed = transformers.__version__.split("+", 1)[0]
if required and installed != required:
    print(f"ERROR: model requires transformers=={required} (from config.json), but current env is transformers=={installed}.")
    print(f'Fix: pip install --index-url https://pypi.org/simple "transformers=={required}" "tokenizers<0.20,>=0.19" -U')
    sys.exit(1)

try:
    import peft  # noqa: F401
except Exception as e:
    print(f"ERROR: peft import failed: {e}")
    print('Fix: pip install --index-url https://pypi.org/simple "peft==0.11.1" -U')
    sys.exit(1)
PY

"$SWIFT_PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port=29501 \
  $SWIFT_HOME/swift/cli/sft.py \
  --model "$MODEL_DIR" \
  --model_type "$MODEL_TYPE" \
  --output_dir $OUTPUT_BASE \
  --split_dataset_ratio 0.0 \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 5e-6 \
  --custom_dataset_info $REPO_ROOT/Dataset/dataset_info.json \
  --dataset images_zhouyan \
  --val_dataset images_zhouyan_val \
  --gradient_accumulation_steps 8 \
  --eval_steps 20 \
  --save_steps 40 \
  --save_total_limit 5 \
  --logging_steps 10 \
  --warmup_ratio 0.1 \
  --label_smoothing_factor 0.02 \
  --lora_dropout 0.05 \
  --num_train_epochs 1 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 8 \
  --deepspeed zero3 \
  --model_name swift-bot \
  --model_author swift
