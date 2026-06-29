import os

# 放在 import torch 之前更稳
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
from tqdm import tqdm
from PIL import Image
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText
from config import DATA_PATH, IMAGE_DIR, RESULT_DIR as RESULT_DIR_CFG, ROLE_PROMPT

BASE_MODEL_PATH = "models/InternVL3_5-38B-HF"
LORA_PATH = "outputs/lora/InternVL3_5-38B-HF/v0-20260402-033151/checkpoint-315"

if isinstance(RESULT_DIR_CFG, dict):
    RESULT_DIR = RESULT_DIR_CFG.get("lora", RESULT_DIR_CFG.get("base", "./results"))
else:
    RESULT_DIR = RESULT_DIR_CFG
os.makedirs(RESULT_DIR, exist_ok=True)
run_idx = 0
ckpt_name = os.path.basename(LORA_PATH.rstrip("/"))
result_file = f"internvl35_38b_{ckpt_name}_run{run_idx}.jsonl"

adapter_config_path = os.path.join(LORA_PATH, "adapter_config.json")
if not os.path.exists(adapter_config_path):
    raise FileNotFoundError(
        f"[ERROR] LoRA checkpoint seems invalid: {LORA_PATH}\n"
        f"adapter_config.json not found."
    )

def build_max_memory():
    if not torch.cuda.is_available():
        return {"cpu": "120GiB"}

    reserve_gib = int(os.environ.get("INTERNVL_GPU_RESERVE_GIB", "10"))
    min_budget_gib = int(os.environ.get("INTERNVL_MIN_GPU_BUDGET_GIB", "20"))
    max_memory = {"cpu": os.environ.get("INTERNVL_CPU_MAX_MEMORY", "120GiB")}
    usable_gpu_count = 0

    # 注意：设置了 CUDA_VISIBLE_DEVICES=4,5,6,7 后，当前进程内可见卡会重新编号成 0,1,2,3。
    for idx in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        free_gib = free_bytes / (1024 ** 3)
        budget_gib = max(0, int(free_gib) - reserve_gib)
        gpu_name = torch.cuda.get_device_name(idx)

        print(
            f"[INFO] visible cuda:{idx} ({gpu_name}) free={free_gib:.1f}GiB "
            f"reserve={reserve_gib}GiB budget={budget_gib}GiB"
        )

        if budget_gib >= min_budget_gib:
            max_memory[idx] = f"{budget_gib}GiB"
            usable_gpu_count += 1

    if usable_gpu_count == 0:
        raise RuntimeError(
            "[ERROR] No usable visible GPU after applying reserve budget. "
            "Lower INTERNVL_GPU_RESERVE_GIB or choose less busy GPUs."
        )

    return max_memory


max_memory = build_max_memory()
print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"[INFO] max_memory={max_memory}")

processor = AutoProcessor.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
)

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_PATH,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="balanced_low_0",
    max_memory=max_memory,
).eval()
model = PeftModel.from_pretrained(base_model, LORA_PATH).eval()

# 看看是否真的拆到了两张卡
print("hf_device_map =", getattr(model, "hf_device_map", None))


def get_input_device(model) -> torch.device:
    """
    对于 device_map='auto' 分片模型：
    输入通常送到 device map 里的第一个 CUDA 设备即可。
    """
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map is None and hasattr(model, "base_model"):
        hf_device_map = getattr(model.base_model, "hf_device_map", None)
    if hf_device_map is None and hasattr(model, "model"):
        hf_device_map = getattr(model.model, "hf_device_map", None)

    if hf_device_map:
        for _, dev in hf_device_map.items():
            # 可能是 int / str / torch.device
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
            if isinstance(dev, torch.device) and dev.type == "cuda":
                return dev

    # 兜底
    return next(model.parameters()).device


INPUT_DEVICE = get_input_device(model)
print("input_device =", INPUT_DEVICE)


def extract_option(text: str) -> str:
    if not text:
        return "--"

    text = text.strip()

    patterns = [
        r'(?i)\banswer\s*[:：]?\s*([ABCD])\b',
        r'(?i)\boption\s*[:：]?\s*([ABCD])\b',
        r'(?i)\bthe answer is\s*([ABCD])\b',
        r'(?i)\bchoose\s*([ABCD])\b',
        r'(?i)\bselect\s*([ABCD])\b',
        r'(?i)[选答案项为是：:\s]*([ABCD])\b',
        r'\b([ABCD])\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    return "--"


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
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        ROLE_PROMPT
                        + f"Question: {question}\n"
                        + f"Options: {'; '.join(options)}\n"
                        + "Please answer with the option only."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    # 关键：不要用 model.device
    # 分片模型没有单一“完整模型所在卡”，输入送到首个 CUDA 设备即可
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if k == "pixel_values":
                inputs[k] = v.to(device=INPUT_DEVICE, dtype=torch.bfloat16)
            else:
                inputs[k] = v.to(INPUT_DEVICE)

    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    output_text = processor.decode(
        generate_ids[0, input_len:],
        skip_special_tokens=True,
    ).strip()

    pred = extract_option(output_text)
    return output_text, pred


with open(DATA_PATH, "r", encoding="utf-8") as f, \
     open(os.path.join(RESULT_DIR, result_file), "w+", encoding="utf-8") as fout:
    lines = f.readlines()
    for line in tqdm(lines, total=len(lines), desc="Processing entries"):
        entry = json.loads(line)
        model_output, pred = chat(entry)

        result_json = {
            "image": entry["image"],
            "question": entry["question"],
            "options": entry["options"],
            "model_output": model_output,
            "pred": pred,
            "gold": entry["answer"],
        }
        fout.write(json.dumps(result_json, ensure_ascii=False) + "\n")
# import os
# import re
# import json
# from tqdm import tqdm
# from PIL import Image
# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT

# model_path = "models/InternVL3_5-38B-HF"

# RESULT_DIR = RESULT_DIR["base"]
# os.makedirs(RESULT_DIR, exist_ok=True)
# run_idx = 0
# result_file = f"intern_38b_sft_run{run_idx}.jsonl"

# # HF 格式：用 AutoProcessor + AutoModelForImageTextToText
# processor = AutoProcessor.from_pretrained(
#     model_path,
#     trust_remote_code=True,
# )

# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map="auto",
# ).eval()


# def extract_option(text: str) -> str:
#     """
#     从模型输出中提取 A/B/C/D 作为最终 pred
#     优先匹配更常见的表达：
#     - A
#     - Answer: A
#     - The answer is B
#     - 选A / 选项C
#     """
#     if not text:
#         return "--"

#     text = text.strip()

#     patterns = [
#         r'(?i)\banswer\s*[:：]?\s*([ABCD])\b',
#         r'(?i)\boption\s*[:：]?\s*([ABCD])\b',
#         r'(?i)\bthe answer is\s*([ABCD])\b',
#         r'(?i)\bchoose\s*([ABCD])\b',
#         r'(?i)\bselect\s*([ABCD])\b',
#         r'(?i)[选答案项为是：:\s]*([ABCD])\b',
#         r'\b([ABCD])\b',
#     ]

#     for pattern in patterns:
#         match = re.search(pattern, text)
#         if match:
#             return match.group(1).upper()

#     return "--"


# def chat(entry):
#     question = entry["question"]
#     image_name = entry["image"]
#     options = entry["options"]
#     image_filepath = os.path.join(IMAGE_DIR, image_name)

#     image = Image.open(image_filepath).convert("RGB")

#     # 用 HF chat template 组织消息
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image"},
#                 {
#                     "type": "text",
#                     "text": (
#                         ROLE_PROMPT
#                         + f"Question: {question}\n"
#                         + f"Options: {'; '.join(options)}\n"
#                         + "Please answer with the option only."
#                     ),
#                 },
#             ],
#         }
#     ]

#     prompt = processor.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=False,
#     )

#     inputs = processor(
#         images=image,
#         text=prompt,
#         return_tensors="pt",
#     )

#     # 放到模型所在设备
#     for k, v in inputs.items():
#         if torch.is_tensor(v):
#             inputs[k] = v.to(model.device)

#     if "pixel_values" in inputs:
#         inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

#     with torch.no_grad():
#         generate_ids = model.generate(
#             **inputs,
#             max_new_tokens=20,
#             do_sample=False,
#         )

#     # 只解码新生成部分
#     output_text = processor.decode(
#         generate_ids[0, inputs["input_ids"].shape[1]:],
#         skip_special_tokens=True,
#     ).strip()

#     pred = extract_option(output_text)

#     return output_text, pred


# with open(DATA_PATH, "r", encoding="utf-8") as f, \
#      open(os.path.join(RESULT_DIR, result_file), "w+", encoding="utf-8") as fout:
#     lines = f.readlines()
#     for line in tqdm(lines, total=len(lines), desc="Processing entries"):
#         entry = json.loads(line)
#         model_output, pred = chat(entry)

#         result_json = {
#             "image": entry["image"],
#             "question": entry["question"],
#             "options": entry["options"],
#             "model_output": model_output,   # 新增：模型原始输出
#             "pred": pred,                   # 从原始输出中提取的 A/B/C/D
#             "gold": entry["answer"],
#         }
#         fout.write(json.dumps(result_json, ensure_ascii=False) + "\n")
