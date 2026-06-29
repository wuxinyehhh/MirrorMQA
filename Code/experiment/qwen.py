import os
import json
import re

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR as RESULT_DIR_CFG, ROLE_PROMPT

BASE_MODEL_PATH = "models/Qwen3-VL-32B-Instruct"
LORA_PATH = "outputs/lora/Qwen3-VL-32B-Instruct/v0-20260402-013814/checkpoint-50"

if isinstance(RESULT_DIR_CFG, dict):
    RESULT_DIR = RESULT_DIR_CFG.get("lora", RESULT_DIR_CFG.get("base", "./results"))
else:
    RESULT_DIR = RESULT_DIR_CFG
os.makedirs(RESULT_DIR, exist_ok=True)

# run_idx = 0
# ckpt_name = os.path.basename(LORA_PATH.rstrip("/"))
result_file = f"qwen3_32b_lora.jsonl"

adapter_config_path = os.path.join(LORA_PATH, "adapter_config.json")
if not os.path.exists(adapter_config_path):
    raise FileNotFoundError(
        f"[ERROR] LoRA checkpoint seems invalid: {LORA_PATH}\n"
        f"adapter_config.json not found."
    )


def get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device

# 先加载基座模型，再挂载 LoRA adapter。
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, LORA_PATH).eval()

processor = AutoProcessor.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ROLE_PROMPT + "Input: Image: "},
                {
                    "type": "image",
                    "image": image_filepath,
                },
                {
                    "type": "text",
                    "text": f"\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # ✅ 自动放到模型设备（更安全）
    model_device = get_model_device(model)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()


def extract_answer(raw_output):
    """从原始输出中提取A/B/C/D答案"""
    if not raw_output or raw_output == "--":
        return "--"

    # 尝试匹配 A、B、C、D（可能带有数字，如 B.2）
    match = re.search(r'\b([ABCD])(?:\.\d+)?', raw_output)
    if match:
        return match.group(1)

    return "--"


def filter_valid_results(result_path):
    """删除pred为'--'的无效行，返回有效行列表"""
    if not os.path.exists(result_path):
        return []

    valid_lines = []
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('pred') != '--':
                    valid_lines.append(line)
            except:
                continue

    return valid_lines


def get_processed_images(valid_lines):
    """从有效结果中获取已处理的图片列表"""
    processed = set()
    for line in valid_lines:
        try:
            entry = json.loads(line)
            processed.add(entry['image'])
        except:
            continue
    return processed


# 在开始推理前，询问用户是否保留之前的有效结果
result_path = os.path.join(RESULT_DIR, result_file)
print("\n选择运行模式:")
print("1 - 保留之前的有效结果，只处理未完成的样本")
print("0 - 清空结果，重新处理所有样本")

while True:
    try:
        choice = int(input("请输入选择 (0 或 1): ").strip())
        if choice in [0, 1]:
            break
        else:
            print("无效输入，请输入 0 或 1")
    except ValueError:
        print("无效输入，请输入 0 或 1")

# 根据选择处理结果文件
if choice == 1:
    # 保留有效结果
    valid_lines = filter_valid_results(result_path)
    processed_images = get_processed_images(valid_lines)
    print(f"保留了 {len(valid_lines)} 条有效结果，已处理 {len(processed_images)} 张图片")

    # 先写入已有的有效结果
    with open(result_path, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)

    file_mode = 'a'  # 追加模式
else:
    # 清空结果
    processed_images = set()
    print("清空之前的结果，重新开始")
    file_mode = 'w'  # 覆盖模式

with open(DATA_PATH, 'r', encoding="utf-8") as f, \
     open(result_path, file_mode, encoding="utf-8") as fout:

    lines = f.readlines()

    for line in tqdm(lines, total=len(lines), desc="Processing entries"):
        entry = json.loads(line)

        # 如果选择保留结果，跳过已处理的图片
        if choice == 1 and entry['image'] in processed_images:
            continue

        try:
            raw_output = chat(entry)
        except Exception as e:
            print("Error:", e)
            raw_output = "--"

        if len(raw_output) == 0:
            raw_output = '--'

        # 从原始输出中提取A/B/C/D
        pred = extract_answer(raw_output)

        result_json = {
            "image": entry['image'],
            "question": entry['question'],
            "options": entry['options'],
            "raw_output": raw_output,  # 保存原始输出
            "pred": pred,  # 只保留A/B/C/D
            "gold": entry['answer']
        }

        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
        fout.flush()
