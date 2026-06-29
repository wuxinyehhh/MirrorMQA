#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"



# 单卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SWIFT_HOME="${SWIFT_HOME:-$REPO_ROOT/ms-swift}"
export PYTHONPATH="$SWIFT_HOME:${PYTHONPATH:-}"

export XDG_CACHE_HOME=$SWIFT_HOME/.cache
export MODELSCOPE_CACHE=$SWIFT_HOME/.cache/modelscope
export HF_HOME=$SWIFT_HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export SWIFT_PYTHON="${SWIFT_PYTHON:-python}"
export SWIFT_LOCAL_REPO_PATH="$SWIFT_HOME/.cache/DeepSeek-VL2"


# 确保目录存在
mkdir -p "$MODELSCOPE_CACHE/hub/_github" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

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
if not (ver < parse_ver("4.42.0")):
    print(f"ERROR: transformers=={transformers.__version__} is unsupported. "
          "Please install: pip install \"transformers<4.42.0\" -U")
    sys.exit(1)

try:
    import soundfile  # noqa: F401
except Exception:
    print("ERROR: soundfile not installed. Please install: pip install soundfile -U")
    sys.exit(1)
PY

OUTPUT_BASE="$REPO_ROOT/output/deepseek-vl2"

# Prefer local-disk copy to avoid SIGBUS on some network filesystems. Fall back to shared path if not present.
MODEL_DIR_SHARED="$REPO_ROOT/models/deepseek-vl2"
MODEL_DIR_LOCAL="$REPO_ROOT/models_local/deepseek-vl2"
if [ -f "$MODEL_DIR_LOCAL/model.safetensors.index.json" ] \
  && [ -f "$MODEL_DIR_LOCAL/model-00001-of-000008.safetensors" ] \
  && [ -f "$MODEL_DIR_LOCAL/model-00002-of-000008.safetensors" ] \
  && [ -f "$MODEL_DIR_LOCAL/model-00003-of-000008.safetensors" ] \
  && [ -f "$MODEL_DIR_LOCAL/model-00004-of-000008.safetensors" ] \
  && [ -f "$MODEL_DIR_LOCAL/model-00005-of-000008.safetensors" ] \
  && [ -f "$MODEL_DIR_LOCAL/model-00006-of-000008.safetensors" ] \
  && [ -f "$MODEL_DIR_LOCAL/model-00007-of-000008.safetensors" ] \
  && [ -f "$MODEL_DIR_LOCAL/model-00008-of-000008.safetensors" ]; then
  MODEL_DIR="$MODEL_DIR_LOCAL"
else
  MODEL_DIR="$MODEL_DIR_SHARED"
fi

if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: MODEL_DIR is invalid. Checked: $MODEL_DIR_LOCAL and $MODEL_DIR_SHARED" >&2
  exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "ERROR: $MODEL_DIR/config.json not found. Please provide a valid local model dir or set MODEL_DIR_SHARED/MODEL_DIR_LOCAL." >&2
  exit 1
fi

"$SWIFT_PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port=29501 \
  $SWIFT_HOME/swift/cli/sft.py \
  --model "$MODEL_DIR" \
  --model_type deepseek_vl2 \
  --output_dir $OUTPUT_BASE \
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
