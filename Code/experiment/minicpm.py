import os
import json
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from packaging import version
from transformers import AutoModel, AutoTokenizer, AutoProcessor, PreTrainedModel

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR as RESULT_DIR_CFG, ROLE_PROMPT

# =========================
# 路径配置
# =========================
BASE_MODEL_PATH = "models/MiniCPM-V-4_5"
LORA_PATH = "outputs/lora/MiniCPM/v0-20260403-160059/checkpoint-4500"

if isinstance(RESULT_DIR_CFG, dict):
    RESULT_DIR = RESULT_DIR_CFG.get("lora", RESULT_DIR_CFG.get("base", "./results"))
else:
    RESULT_DIR = RESULT_DIR_CFG

os.makedirs(RESULT_DIR, exist_ok=True)

run_idx = 0
ckpt_name = os.path.basename(LORA_PATH.rstrip("/"))
result_file = f"minicpm_{ckpt_name}_run{run_idx}.jsonl"

# =========================
# 兼容 MiniCPM remote code
# =========================
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    def _get_all_tied_weights_keys(self):
        val = getattr(self, "_all_tied_weights_keys", None)
        if isinstance(val, dict):
            return val
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        if isinstance(keys, (list, tuple, set)):
            return {str(k): True for k in keys}
        return {}

    def _set_all_tied_weights_keys(self, value):
        self._all_tied_weights_keys = value

    PreTrainedModel.all_tied_weights_keys = property(
        _get_all_tied_weights_keys, _set_all_tied_weights_keys
    )

# =========================
# 给 tokenizer 打补丁
# =========================
def patch_minicpm_tokenizer(tok):
    # 基础 token 字符串
    if not hasattr(tok, "im_start"):
        tok.im_start = "<image>"
    if not hasattr(tok, "im_end"):
        tok.im_end = "</image>"
    if not hasattr(tok, "slice_start"):
        tok.slice_start = "<slice>"
    if not hasattr(tok, "slice_end"):
        tok.slice_end = "</slice>"
    if not hasattr(tok, "im_id_start"):
        tok.im_id_start = "<image_id>"
    if not hasattr(tok, "im_id_end"):
        tok.im_id_end = "</image_id>"

    # 基础 id
    if not hasattr(tok, "bos_id"):
        tok.bos_id = tok.bos_token_id
    if not hasattr(tok, "eos_id"):
        tok.eos_id = tok.eos_token_id
    if not hasattr(tok, "unk_id"):
        tok.unk_id = tok.unk_token_id

    # MiniCPM-V 依赖的特殊 id
    if not hasattr(tok, "im_start_id"):
        tok.im_start_id = tok.convert_tokens_to_ids(tok.im_start)
    if not hasattr(tok, "im_end_id"):
        tok.im_end_id = tok.convert_tokens_to_ids(tok.im_end)
    if not hasattr(tok, "slice_start_id"):
        tok.slice_start_id = tok.convert_tokens_to_ids(tok.slice_start)
    if not hasattr(tok, "slice_end_id"):
        tok.slice_end_id = tok.convert_tokens_to_ids(tok.slice_end)
    if not hasattr(tok, "im_id_start_id"):
        tok.im_id_start_id = tok.convert_tokens_to_ids(tok.im_id_start)
    if not hasattr(tok, "im_id_end_id"):
        tok.im_id_end_id = tok.convert_tokens_to_ids(tok.im_id_end)
    if not hasattr(tok, "newline_id"):
        tok.newline_id = tok.convert_tokens_to_ids("\n")

    # 简单检查，防止特殊 token 真没进词表
    must_check = {
        "im_start_id": tok.im_start_id,
        "im_end_id": tok.im_end_id,
        "slice_start_id": tok.slice_start_id,
        "slice_end_id": tok.slice_end_id,
    }
    for k, v in must_check.items():
        if v is None:
            raise ValueError(f"{k} is None, tokenizer special tokens may be broken.")

    return tok

# =========================
# 加载参数
# =========================
load_kwargs = {
    "trust_remote_code": True,
    "attn_implementation": "sdpa",
}

if version.parse(transformers.__version__) >= version.parse("4.56"):
    load_kwargs["dtype"] = torch.bfloat16
else:
    load_kwargs["torch_dtype"] = torch.bfloat16

# =========================
# 1. 加载基底模型
# =========================
print(f"[INFO] Loading base model from: {BASE_MODEL_PATH}")
base_model = AutoModel.from_pretrained(BASE_MODEL_PATH, **load_kwargs)
base_model = base_model.eval().cuda()

# =========================
# 2. 优先加载 processor，再取 tokenizer
# =========================
print(f"[INFO] Loading processor from: {BASE_MODEL_PATH}")
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print(f"[INFO] Loading tokenizer from: {BASE_MODEL_PATH}")
try:
    tokenizer = processor.tokenizer
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

tokenizer = patch_minicpm_tokenizer(tokenizer)

# 让 processor 内部也用这份修补后的 tokenizer
if hasattr(processor, "tokenizer"):
    processor.tokenizer = tokenizer

# =========================
# 3. 挂载 LoRA
# =========================
adapter_config_path = os.path.join(LORA_PATH, "adapter_config.json")
if not os.path.exists(adapter_config_path):
    raise FileNotFoundError(
        f"[ERROR] LoRA checkpoint seems invalid: {LORA_PATH}\n"
        f"adapter_config.json not found."
    )

print(f"[INFO] Loading LoRA adapter from: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.eval()

# 如果后续你确认 merge 没问题，也可打开
# if hasattr(model, "merge_and_unload"):
#     model = model.merge_and_unload()

# =========================
# 推理函数
# =========================
@torch.inference_mode()
def chat(entry):
    question = entry["question"]
    image_name = entry["image"]
    options = entry["options"]

    image_filepath = os.path.join(IMAGE_DIR, image_name)
    image = Image.open(image_filepath).convert("RGB")

    prompt = (
        ROLE_PROMPT
        + f"\nQuestion: {question}\n"
        + f"Options: {'; '.join(options)}\n"
        + "Output:"
    )

    msgs = [
        {
            "role": "user",
            "content": [image, prompt]
        }
    ]

    # 某些版本的 MiniCPM chat 支持 processor，有些不支持
    try:
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            processor=processor,
            max_new_tokens=20,
            do_sample=False
        )
    except TypeError:
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            max_new_tokens=20,
            do_sample=False
        )

    if isinstance(res, tuple):
        res = res[0]

    return str(res).strip()

# =========================
# 批量处理
# =========================
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
            "gold": entry["answer"]
        }
        fout.write(json.dumps(result_json, ensure_ascii=False) + "\n")

print("[INFO] Done.")