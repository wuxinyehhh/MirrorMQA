import re
import torch
import transformers
from transformers import AutoModelForCausalLM


def patch_transformers_for_peft():
    cache_utils = getattr(transformers, "cache_utils", None)
    if cache_utils is None:
        return

    for attr in ("Cache", "DynamicCache", "EncoderDecoderCache"):
        if not hasattr(transformers, attr) and hasattr(cache_utils, attr):
            setattr(transformers, attr, getattr(cache_utils, attr))

    if not hasattr(transformers, "EncoderDecoderCache") and hasattr(transformers, "DynamicCache"):
        class EncoderDecoderCache(transformers.DynamicCache):
            pass

        transformers.EncoderDecoderCache = EncoderDecoderCache


patch_transformers_for_peft()

from peft import PeftModel

from janus.models import VLChatProcessor, MultiModalityCausalLM
from janus.utils.io import load_pil_images

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR as RESULT_DIR_CFG, ROLE_PROMPT
import os
import json
from tqdm import tqdm

BASE_MODEL_PATH = "models/Janus-Pro-7B"
LORA_PATH = "outputs/lora/Janus-Pro-7B/v1-20260402-081824/checkpoint-315"

if isinstance(RESULT_DIR_CFG, dict):
    RESULT_DIR = RESULT_DIR_CFG.get("lora", RESULT_DIR_CFG.get("base", "./results"))
else:
    RESULT_DIR = RESULT_DIR_CFG
os.makedirs(RESULT_DIR, exist_ok=True)
run_idx = 0
ckpt_name = os.path.basename(LORA_PATH.rstrip("/"))
result_file = f"janus_{ckpt_name}_run{run_idx}.jsonl"

adapter_config_path = os.path.join(LORA_PATH, "adapter_config.json")
if not os.path.exists(adapter_config_path):
    raise FileNotFoundError(
        f"[ERROR] LoRA checkpoint seems invalid: {LORA_PATH}\n"
        f"adapter_config.json not found."
    )

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(BASE_MODEL_PATH)
tokenizer = vl_chat_processor.tokenizer

base_vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
base_vl_gpt = base_vl_gpt.to(torch.bfloat16).cuda().eval()
vl_gpt = PeftModel.from_pretrained(base_vl_gpt, LORA_PATH).eval()


def get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


OPTION_TOKENS = None

def get_option_tokens():
    global OPTION_TOKENS
    if OPTION_TOKENS is None:
        OPTION_TOKENS = {
            opt: tokenizer.encode(opt, add_special_tokens=False)[0]
            for opt in ["A", "B", "C", "D"]
        }
    return OPTION_TOKENS


@torch.inference_mode()
def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    prompt = (
        ROLE_PROMPT +
        f"Image: <image_placeholder>\nQuestion: {question}\nOptions: {'; '.join(options)}\n"
        f"Answer:"
    )
    conversation = [
        {
            "role": "User",
            "content": prompt,
            "images": [image_filepath]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(get_model_device(vl_gpt))

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
    )

    # 取最后一个位置的 logit，比较 A/B/C/D 哪个最高
    logits = outputs.logits[0, -1, :]
    option_tokens = get_option_tokens()
    best = max(option_tokens, key=lambda x: logits[option_tokens[x]].item())
    return best


def extract_answer(raw_output):
    if not raw_output or raw_output == "--":
        return "--"
    match = re.search(r'\b([ABCD])\b', raw_output)
    return match.group(1) if match else "--"


def load_valid_results(output_path):
    if not os.path.exists(output_path):
        return [], set()
    valid_lines, processed = [], set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if extract_answer(entry.get("pred", "")) != "--":
                    valid_lines.append(line + "\n")
                    processed.add(entry["image"])
            except Exception:
                continue
    return valid_lines, processed


output_path = os.path.join(RESULT_DIR, result_file)
valid_lines, processed_images = load_valid_results(output_path)
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(valid_lines)
print(f"[INFO] 保留 {len(valid_lines)} 条有效结果，跳过 {len(processed_images)} 张图片")

with open(DATA_PATH, 'r', encoding="utf-8") as f, open(output_path, "a", encoding="utf-8") as fout:
    lines = f.readlines()
    for line in tqdm(lines, total=len(lines), desc="Processing entries"):
        entry = json.loads(line)
        if entry["image"] in processed_images:
            continue
        try:
            raw_output = chat(entry)
            if not raw_output:
                raw_output = "--"
        except Exception as e:
            raw_output = "--"
            print(f"Error on {entry['image']}: {e}")
        result_json = {
            "image": entry["image"],
            "question": entry["question"],
            "options": entry["options"],
            "raw_output": raw_output,
            "pred": extract_answer(raw_output),
            "gold": entry["answer"],
        }
        fout.write(json.dumps(result_json, ensure_ascii=False) + "\n")
        fout.flush()
