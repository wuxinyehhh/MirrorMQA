import os
import json
import re

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT


model_path = "models/Phi-4"

RESULT_DIR = RESULT_DIR["base"]
os.makedirs(RESULT_DIR, exist_ok=True)

run_idx = 0
result_file = f"phi_run{run_idx}.jsonl"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda" if torch.cuda.is_available() else None,
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="eager",
).eval()

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    num_crops=4,
)

print(f"[INFO] model type: {type(model)}")
print(f"[INFO] processor type: {type(processor)}")


@torch.inference_mode()
def infer_one(entry):
    question = entry["question"]
    image_name = entry["image"]
    options = entry["options"]
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_filepath):
        raise FileNotFoundError(f"Image not found: {image_filepath}")

    image = Image.open(image_filepath).convert("RGB")

    images = [image]
    placeholder = "<|image_1|>"

    prompt = (
        ROLE_PROMPT
        + f"Input: Image: {placeholder}\n"
          f"Question: {question}, Options: {'; '.join(options)}.\n"
          f"Output:"
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    prompt_text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(prompt_text, images, return_tensors="pt").to(DEVICE)

    generation_args = {
        "max_new_tokens": 20,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args,
    )

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response.strip()


def extract_answer(raw_output):
    if not raw_output or raw_output == "--":
        return "--"

    text = str(raw_output).strip()

    match = re.search(r"Answer\s*[:：]?\s*([ABCD])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"(?:option|choice|答案|选项)\s*[:：]?\s*([ABCD])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return "--"


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
    print(f"[INFO] 跳过已处理 {len(processed_images)} 张图片")

    with open(DATA_PATH, "r", encoding="utf-8") as f, open(output_path, "a", encoding="utf-8") as fout:
        lines = f.readlines()

        for line in tqdm(lines, total=len(lines), desc="Processing entries"):
            entry = json.loads(line)

            if entry["image"] in processed_images:
                continue

            try:
                raw_output = infer_one(entry)
                if not raw_output:
                    raw_output = "--"
            except Exception as e:
                raw_output = "--"
                print(f"[WARN] {entry['image']} failed: {e}")

            pred = extract_answer(raw_output)

            result_json = {
                "image": entry["image"],
                "question": entry["question"],
                "options": entry["options"],
                "raw_output": raw_output,
                "pred": pred,
                "gold": entry["answer"],
            }

            fout.write(json.dumps(result_json, ensure_ascii=False) + "\n")
            fout.flush()

    print("[INFO] done.")


if __name__ == "__main__":
    main()