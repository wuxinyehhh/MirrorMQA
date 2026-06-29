import json
import os

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR as RESULT_DIR_CFG, ROLE_PROMPT

BASE_MODEL_PATH = "models/LLaVA-OneVision-1.5-8B-Instruct"
LORA_PATH = "outputs/lora/LLaVA_OV/v1-20260402-131704/checkpoint-210"

if isinstance(RESULT_DIR_CFG, dict):
    RESULT_DIR = RESULT_DIR_CFG.get("lora", RESULT_DIR_CFG.get("base", "./results"))
else:
    RESULT_DIR = RESULT_DIR_CFG

os.makedirs(RESULT_DIR, exist_ok=True)

run_idx = 1
ckpt_name = os.path.basename(LORA_PATH.rstrip("/"))
result_file = f"llava_{ckpt_name}_run{run_idx}.jsonl"


def patch_transformers_compat() -> None:
    try:
        from transformers import cache_utils

        if not hasattr(cache_utils, "SlidingWindowCache") and hasattr(cache_utils, "DynamicCache"):
            class SlidingWindowCache(cache_utils.DynamicCache):
                pass

            cache_utils.SlidingWindowCache = SlidingWindowCache
            print("[WARN] transformers.cache_utils.SlidingWindowCache missing; patched lightweight compatibility class.")
    except Exception:
        pass

    try:
        from transformers import modeling_rope_utils

        rope_init_functions = modeling_rope_utils.ROPE_INIT_FUNCTIONS
        if "default" not in rope_init_functions and len(rope_init_functions) > 0:
            def _default_rope_init(config=None, device=None, seq_len=None, layer_type=None):
                import torch as _torch

                if config is None:
                    raise ValueError("`config` is required for default RoPE init patch.")
                head_dim = getattr(config, "head_dim", None)
                if head_dim is None:
                    hidden_size = getattr(config, "hidden_size", None)
                    num_attention_heads = getattr(config, "num_attention_heads", None)
                    if hidden_size is None or num_attention_heads is None:
                        raise ValueError("Cannot infer `head_dim` from config for default RoPE init patch.")
                    head_dim = hidden_size // num_attention_heads
                rope_theta = float(getattr(config, "rope_theta", 10000.0))
                inv_freq = 1.0 / (
                    rope_theta
                    ** (_torch.arange(0, int(head_dim), 2, dtype=_torch.float, device=device) / float(head_dim))
                )
                return inv_freq, 1.0

            rope_init_functions["default"] = _default_rope_init
            print("[WARN] ROPE_INIT_FUNCTIONS['default'] missing; patched custom default implementation.")
    except Exception:
        pass


def get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


patch_transformers_compat()

adapter_config_path = os.path.join(LORA_PATH, "adapter_config.json")
if not os.path.exists(adapter_config_path):
    raise FileNotFoundError(
        f"[ERROR] LoRA checkpoint seems invalid: {LORA_PATH}\n"
        f"adapter_config.json not found."
    )

print(f"[INFO] Loading processor from: {BASE_MODEL_PATH}")
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

if hasattr(processor, "tokenizer") and getattr(processor.tokenizer, "pad_token_id", None) is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
pad_token_id = processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None

device = "cuda" if torch.cuda.is_available() else "cpu"
config = AutoConfig.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if getattr(config, "pad_token_id", None) is None:
    config.pad_token_id = pad_token_id
if hasattr(config, "text_config") and getattr(config.text_config, "pad_token_id", None) is None:
    config.text_config.pad_token_id = config.pad_token_id

model_kwargs = {
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}
if device == "cuda":
    model_kwargs["torch_dtype"] = torch.bfloat16
    model_kwargs["device_map"] = "auto"
else:
    model_kwargs["torch_dtype"] = torch.float32

print(f"[INFO] Loading base model from: {BASE_MODEL_PATH}")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, config=config, **model_kwargs)
if device != "cuda":
    base_model.to(device)

print(f"[INFO] Loading LoRA adapter from: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

if getattr(model.config, "pad_token_id", None) is None and getattr(processor, "tokenizer", None) is not None:
    model.config.pad_token_id = processor.tokenizer.pad_token_id
if hasattr(model.config, "text_config") and getattr(model.config.text_config, "pad_token_id", None) is None:
    model.config.text_config.pad_token_id = model.config.pad_token_id


@torch.inference_mode()
def chat(entry):
    question = entry["question"]
    image_name = entry["image"]
    options = entry["options"]
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    image = Image.open(image_filepath).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_filepath},
                {
                    "type": "text",
                    "text": (
                        ROLE_PROMPT
                        + "Input: Image: "
                        + f"\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(get_model_device(model))
    if "mm_token_type_ids" in inputs:
        inputs.pop("mm_token_type_ids")

    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    trimmed_output = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)]
    res = processor.batch_decode(
        trimmed_output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return res


output_path = os.path.join(RESULT_DIR, result_file)
print(f"[INFO] Writing results to: {output_path}")

with open(DATA_PATH, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as fout:
    lines = f.readlines()
    for line in tqdm(lines, total=len(lines), desc="Processing entries"):
        entry = json.loads(line)
        pred = chat(entry)
        if not pred:
            pred = "--"

        result_json = {
            "image": entry["image"],
            "question": entry["question"],
            "options": entry["options"],
            "pred": pred,
            "gold": entry["answer"],
        }
        fout.write(json.dumps(result_json, ensure_ascii=False) + "\n")

print("[INFO] Done.")
