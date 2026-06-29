import os
import re
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT


model_path = "models/deepseek-vl2"
RESULT_DIR = RESULT_DIR["base"]
os.makedirs(RESULT_DIR, exist_ok=True)

run_idx = 0
result_file = f"deepseek_run{run_idx}.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def extract_choice(text):
    text = text.strip().upper()

    # 优先匹配固定格式：Answer: 1 / Answer: 2 ...
    m = re.search(r"ANSWER\s*[:：]\s*([1234])", text)
    if m:
        return {"1": "A", "2": "B", "3": "C", "4": "D"}[m.group(1)]

    # 兼容模型只输出单个数字
    if text in {"1", "2", "3", "4"}:
        return {"1": "A", "2": "B", "3": "C", "4": "D"}[text]

    # 兼容最后一行就是 1 / 2 / 3 / 4
    lines = [line.strip().upper() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        if last_line in {"1", "2", "3", "4"}:
            return {"1": "A", "2": "B", "3": "C", "4": "D"}[last_line]
        m = re.fullmatch(r"(?:ANSWER\s*[:：]\s*)?([1234])\.?", last_line)
        if m:
            return {"1": "A", "2": "B", "3": "C", "4": "D"}[m.group(1)]

    return "--"


vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
)

if torch.cuda.is_available():
    vl_gpt = vl_gpt.to(DTYPE).cuda().eval()
else:
    vl_gpt = vl_gpt.to(DTYPE).to(DEVICE).eval()


def chat(entry):
    question = entry["question"]
    image_name = entry["image"]
    image_filepath = os.path.join(IMAGE_DIR, image_name)

    prompt = (
        ROLE_PROMPT
        + "You are solving a visual reasoning multiple-choice problem.\n"
          "The image contains one question figure and four candidate option figures labeled 1, 2, 3, and 4.\n"
          "Find which candidate is the correct mirror image of the question figure.\n"
          "Carefully compare the question figure with all four options before answering.\n"
          "Give the final answer on the last line only, using exactly this format:\n"
          "Answer: <number>\n"
          "where <number> must be one of 1, 2, 3, or 4.\n\n"
          f"Question: {question}\n"
    )

    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": [image_filepath]
        },
        {
            "role": "<|Assistant|>",
            "content": ""
        }
    ]

    pil_images = load_pil_images(conversation)

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    with torch.no_grad():
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=32,
            use_cache=True,
            return_dict_in_generate=False
        )

    full_ids = outputs[0].cpu().tolist()
    raw_full = tokenizer.decode(full_ids, skip_special_tokens=True).strip()

    print("IMAGE:", image_name)
    print("LEN(outputs[0]):", len(full_ids))
    print("ATTN_LEN:", prepare_inputs.attention_mask.shape[1])
    print("RAW_FULL:", repr(raw_full))
    print("=" * 60)

    return raw_full


with open(DATA_PATH, "r", encoding="utf-8") as f, open(
    os.path.join(RESULT_DIR, result_file), "w", encoding="utf-8"
) as fout:
    lines = f.readlines()
    for line in tqdm(lines, total=len(lines), desc="Processing entries"):
        entry = json.loads(line)
        try:
            raw_pred = chat(entry)
            pred = extract_choice(raw_pred)
        except Exception as e:
            pred = f"[ERROR] {type(e).__name__}: {e}"

        result_json = {
            "image": entry["image"],
            "question": entry["question"],
            "options": entry["options"],
            "pred": pred,
            "gold": entry["answer"]
        }
        fout.write(json.dumps(result_json, ensure_ascii=False) + "\n")
        fout.flush()