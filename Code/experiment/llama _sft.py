import os
import json
import re

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT


# =========================
# 基本配置
# =========================
base_model_path = "models/Llama-4-Scout-17B-16E-Instruct"
adapter_path = "outputs/lora/Llama-4-Scout-17B-16E-Instruct/v10-20260328-214838/checkpoint-400"

RESULT_DIR = RESULT_DIR["base"]
os.makedirs(RESULT_DIR, exist_ok=True)

run_idx = 0
result_file = f"llama_sft_400_run{run_idx}.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# =========================
# 加载模型与 processor
# =========================
processor = AutoProcessor.from_pretrained(base_model_path)

if torch.cuda.is_available():
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        dtype=DTYPE,
        device_map="auto",
    )
else:
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        dtype=DTYPE,
    )
    base_model.to(DEVICE)

model = PeftModel.from_pretrained(base_model, adapter_path)

model.eval()


# =========================
# 从完整输出中提取 A/B/C/D
# =========================
def extract_option(raw_text, options):
    """
    从模型完整输出中尽量提取 A/B/C/D
    提取优先级：
    1. 单独出现的 A/B/C/D
    2. 出现 A.1 / B.2 / C.3 / D.4
    3. 出现 option A / answer is B / correct answer: C
    4. 若出现完整选项文本，也映射回 A/B/C/D
    """
    if raw_text is None:
        return "--"

    text = raw_text.strip()
    if not text:
        return "--"

    # 统一大写，方便匹配
    upper_text = text.upper()

    # 1) 整体就是单个字母
    m = re.fullmatch(r"\s*([ABCD])\s*[\.\:\)\-]?\s*", upper_text)
    if m:
        return m.group(1)

    # 2) 开头就是 A / B / C / D
    m = re.match(r"^\s*([ABCD])\s*[\.\:\)\-]?", upper_text)
    if m:
        return m.group(1)

    # 3) answer / option / correct answer 之类
    patterns = [
        r"CORRECT ANSWER\s*[:：]?\s*([ABCD])\b",
        r"ANSWER\s*[:：]?\s*([ABCD])\b",
        r"OPTION\s*([ABCD])\b",
        r"CHOOSE\s*([ABCD])\b",
        r"\b([ABCD])\s*\.\s*[1234]\b",
        r"\b([ABCD])\s*[)\].:-]\s*[1234]?\b",
    ]
    for p in patterns:
        m = re.search(p, upper_text)
        if m:
            return m.group(1)

    # 4) 若输出里含完整选项文本，则映射回字母
    # options 例如 ["A.1", "B.2", "C.3", "D.4"]
    option_map = {}
    for opt in options:
        opt = opt.strip()
        if len(opt) >= 1 and opt[0].upper() in "ABCD":
            option_map[opt[0].upper()] = opt.upper()

    for letter, opt_text in option_map.items():
        if opt_text in upper_text:
            return letter

    return "--"


# =========================
# 单条推理
# =========================
def chat(entry):
    question = entry["question"]
    image_name = entry["image"]
    options = entry["options"]
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    image = Image.open(image_filepath).convert("RGB")

    # 强约束 prompt：只允许输出 A/B/C/D，不允许解释
    prompt = (
        f"{ROLE_PROMPT}\n"
        f"You are solving a multiple-choice visual reasoning problem.\n"
        f"Question: {question}\n"
        f"Options: {'; '.join(options)}\n\n"
        f"Rules:\n"
        f"1. Return ONLY one uppercase letter: A, B, C, or D.\n"
        f"2. Do NOT output the option text.\n"
        f"3. Do NOT output any explanation, reasoning, steps, or extra words.\n"
        f"4. Do NOT output punctuation.\n\n"
        f"Final answer:"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = {
        k: v.to(model_device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=8,   # 既然只要 A/B/C/D，就不需要太长
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output[:, input_len:]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    pred = extract_option(raw_output, options)

    return pred, raw_output


# =========================
# 读取已有结果：保留有效行，删除无效行
# =========================
def load_valid_results(output_path):
    if not os.path.exists(output_path):
        return [], set()

    valid_lines = []
    processed_images = set()

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                continue

            pred = item.get("pred", "--")
            image_name = item.get("image")

            if pred != "--" and image_name:
                valid_lines.append(json.dumps(item, ensure_ascii=False) + "\n")
                processed_images.add(image_name)

    return valid_lines, processed_images


def main():
    output_path = os.path.join(RESULT_DIR, result_file)
    print(f"[INFO] saving to: {output_path}")

    valid_lines, processed_images = load_valid_results(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(valid_lines)

    print(f"[INFO] 保留 {len(valid_lines)} 条有效结果")
    print(f"[INFO] 删除其余无效结果，并跳过已处理 {len(processed_images)} 张图片")

    with open(DATA_PATH, "r", encoding="utf-8") as f, open(output_path, "a", encoding="utf-8") as fout:
        lines = f.readlines()

        for idx, line in enumerate(tqdm(lines, total=len(lines), desc="Processing entries"), start=1):
            entry = json.loads(line)

            if entry["image"] in processed_images:
                continue

            try:
                pred, raw_output = chat(entry)
            except Exception as e:
                raw_output = f"ERROR: {type(e).__name__}: {str(e)}"
                pred = "--"

            result_json = {
                "image": entry["image"],
                "question": entry["question"],
                "options": entry["options"],
                "pred": pred,
                "raw_output": raw_output,
                "gold": entry["answer"],
            }

            fout.write(json.dumps(result_json, ensure_ascii=False) + "\n")
            fout.flush()

            print(
                json.dumps(
                    {
                        "idx": idx,
                        "image": entry["image"],
                        "pred": pred,
                        "raw_output": raw_output,
                        "gold": entry["answer"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print("[INFO] done.")


if __name__ == "__main__":
    main()
