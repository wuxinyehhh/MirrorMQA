import os
import sys
import json
import traceback

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
)

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR, ROLE_PROMPT

sys.path.insert(0, "models/xgen-mm-phi3-mini-instruct-interleave-r-v1.5")
from modeling_xgenmm import process_anyres_image


model_path = "models/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
siglip_path = "models/google_siglip-so400m-patch14-384"

RESULT_DIR = RESULT_DIR["base"]
os.makedirs(RESULT_DIR, exist_ok=True)

run_idx = 7
result_file = f"xgenmm_run{run_idx}.jsonl"
save_path = os.path.join(RESULT_DIR, result_file)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

if hasattr(config, "vision_encoder_config") and hasattr(config.vision_encoder_config, "model_name"):
    config.vision_encoder_config.model_name = siglip_path

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

image_processor = AutoImageProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    config=config,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=DTYPE,
    device_map=None,
)

model = model.to(DEVICE)
model.eval()

print(f"结果保存到: {save_path}")


class _ResizeStub:
    def __init__(self, size):
        self.size = size


class AnyResProcessorAdapter:
    def __init__(self, hf_image_processor):
        self.hf_image_processor = hf_image_processor

        size = getattr(hf_image_processor, "size", [384, 384])
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                size = [size["height"], size["width"]]
            elif "shortest_edge" in size:
                size = [size["shortest_edge"], size["shortest_edge"]]
            else:
                size = [384, 384]

        self.transforms = [_ResizeStub(size)]

    def __call__(self, image_patch):
        out = self.hf_image_processor(
            images=[image_patch],
            return_tensors="pt",
        )
        # 必须返回 [C, H, W]，不能返回 [1, C, H, W]
        pixel = out["pixel_values"][0]
        return pixel


anyres_processor = AnyResProcessorAdapter(image_processor)


def build_prompt(question, options):
    user_query = f"{ROLE_PROMPT}Question: {question}\nOptions: {'; '.join(options)}"
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"USER: <image>\n{user_query}\nASSISTANT:"
    )
    return prompt


@torch.no_grad()
def chat(entry):
    question = entry["question"]
    image_name = entry["image"]
    options = entry["options"]

    image_filepath = os.path.join(IMAGE_DIR, image_name)
    image = Image.open(image_filepath).convert("RGB")
    prompt = build_prompt(question, options)

    text_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(DEVICE)
    attention_mask = text_inputs["attention_mask"].to(DEVICE)

    patches = process_anyres_image(
        image,
        anyres_processor,
        config.vision_encoder_config.anyres_grids,
    )

    # 这里应该是 4 维: [N_patch, C, H, W]
    print("patches shape before unsqueeze:", tuple(patches.shape))

    patches = patches.to(device=DEVICE, dtype=DTYPE)

    # 传给 anyres 路线的单样本结构: list([1, N_patch, C, H, W])
    vision_x = [patches.unsqueeze(0)]
    image_size = [image.size]

    print("vision_x[0] shape:", tuple(vision_x[0].shape))

    outputs = model.generate(
        pixel_values=vision_x,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_size=image_size,
        do_sample=False,
        max_new_tokens=20,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text


with open(DATA_PATH, "r", encoding="utf-8") as f, open(save_path, "w", encoding="utf-8") as fout:
    lines = f.readlines()

    for idx, line in enumerate(tqdm(lines, total=len(lines), desc="Processing entries")):
        entry = json.loads(line)

        try:
            pred = chat(entry)
            if not pred:
                pred = "--"
        except Exception as e:
            err_msg = f"{type(e).__name__}: {repr(e)}"
            print(f"\n[ERROR] idx={idx}, image={entry['image']}")
            print(err_msg)
            traceback.print_exc()
            pred = err_msg

        result_json = {
            "image": entry["image"],
            "question": entry["question"],
            "options": entry["options"],
            "pred": pred,
            "gold": entry["answer"],
        }
        fout.write(json.dumps(result_json, ensure_ascii=False) + "\n")
        fout.flush()