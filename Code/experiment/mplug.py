import os
import json
import sys
import types
from enum import Enum

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm
from PIL import Image
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers import GenerationConfig
from transformers.generation import GenerationMixin

from config import DATA_PATH, IMAGE_DIR, RESULT_DIR as RESULT_DIR_CFG, ROLE_PROMPT


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

    if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "seen_tokens"):
        def _get_seen_tokens(self):
            if hasattr(self, "_seen_tokens"):
                return self._seen_tokens
            if hasattr(self, "get_seq_length"):
                return self.get_seq_length()
            return 0

        cache_utils.DynamicCache.seen_tokens = property(_get_seen_tokens)

    if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_max_length"):
        def _get_max_length(self):
            return None

        cache_utils.DynamicCache.get_max_length = _get_max_length

    if hasattr(cache_utils, "DynamicCache") and not hasattr(cache_utils.DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length=None, layer_idx=None):
            if hasattr(self, "get_seq_length"):
                return self.get_seq_length()
            return 0

        cache_utils.DynamicCache.get_usable_length = _get_usable_length


def patch_qwen2_remote_code_compat():
    try:
        from transformers.models.qwen2 import modeling_qwen2 as qwen2_modeling

        orig_qwen2_forward = qwen2_modeling.Qwen2Attention.forward

        def compat_qwen2_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=None,
            **kwargs,
        ):
            if position_embeddings is None:
                if not hasattr(self, "_compat_rotary_emb"):
                    self._compat_rotary_emb = qwen2_modeling.Qwen2RotaryEmbedding(config=self.config)

                if position_ids is None:
                    past_seen_tokens = 0
                    if past_key_value is not None and hasattr(past_key_value, "get_seq_length"):
                        past_seen_tokens = past_key_value.get_seq_length()
                    position_ids = torch.arange(
                        hidden_states.shape[1], device=hidden_states.device
                    ) + past_seen_tokens
                    position_ids = position_ids.unsqueeze(0)

                position_embeddings = self._compat_rotary_emb(hidden_states, position_ids)

            attn_output, attn_weights = orig_qwen2_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_value,
                **kwargs,
            )
            present_key_value = past_key_value if use_cache else None
            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights, present_key_value

        qwen2_modeling.Qwen2Attention.forward = compat_qwen2_forward

        if not hasattr(qwen2_modeling, "Qwen2FlashAttention2") and hasattr(qwen2_modeling, "Qwen2Attention"):
            qwen2_modeling.Qwen2FlashAttention2 = qwen2_modeling.Qwen2Attention

        if not hasattr(qwen2_modeling, "Qwen2SdpaAttention") and hasattr(qwen2_modeling, "Qwen2Attention"):
            qwen2_modeling.Qwen2SdpaAttention = qwen2_modeling.Qwen2Attention
    except Exception:
        pass


def patch_transformers_utils_compat():
    try:
        if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):
            def is_flash_attn_greater_or_equal_2_10():
                return False

            transformers.utils.is_flash_attn_greater_or_equal_2_10 = is_flash_attn_greater_or_equal_2_10
    except Exception:
        pass


def patch_optional_dependencies():
    if "icecream" not in sys.modules:
        icecream = types.ModuleType("icecream")

        def ic(*args, **kwargs):
            if len(args) == 0:
                return None
            return args[0] if len(args) == 1 else args

        icecream.ic = ic
        sys.modules["icecream"] = icecream

    try:
        from mistral_common.protocol.instruct import request as mistral_request
        from mistral_common.tokens.tokenizers import utils as mistral_tokenizer_utils

        if not hasattr(mistral_request, "ReasoningEffort"):
            class ReasoningEffort(str, Enum):
                low = "low"
                medium = "medium"
                high = "high"

            mistral_request.ReasoningEffort = ReasoningEffort

        if not hasattr(mistral_tokenizer_utils, "get_one_valid_tokenizer_file"):
            def get_one_valid_tokenizer_file(files):
                valid_files = mistral_tokenizer_utils._filter_valid_tokenizer_files(files)
                if len(valid_files) == 0:
                    return None
                if "tekken.json" in valid_files:
                    return "tekken.json"
                return sorted(valid_files)[-1]

            mistral_tokenizer_utils.get_one_valid_tokenizer_file = get_one_valid_tokenizer_file
    except Exception:
        pass


patch_transformers_for_peft()
patch_qwen2_remote_code_compat()
patch_transformers_utils_compat()
patch_optional_dependencies()

from peft import PeftModel

BASE_MODEL_PATH = "models/mPLUG-Owl3-7B-241101"
LORA_PATH = "outputs/lora/mPLUG/v3-20260402-121341/checkpoint-1000"

if isinstance(RESULT_DIR_CFG, dict):
    RESULT_DIR = RESULT_DIR_CFG.get("lora", RESULT_DIR_CFG.get("base", "./results"))
else:
    RESULT_DIR = RESULT_DIR_CFG
os.makedirs(RESULT_DIR, exist_ok=True)

# ckpt_name = os.path.basename(LORA_PATH.rstrip("/"))
result_file = f"mplug_lora.jsonl"

adapter_config_path = os.path.join(LORA_PATH, "adapter_config.json")
if not os.path.exists(adapter_config_path):
    raise FileNotFoundError(
        f"[ERROR] LoRA checkpoint seems invalid: {LORA_PATH}\n"
        f"adapter_config.json not found."
    )


def select_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_idx = 0
    best_free_bytes = -1
    for idx in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        gpu_name = torch.cuda.get_device_name(idx)
        print(
            f"[INFO] visible cuda:{idx} ({gpu_name}) "
            f"free={free_bytes / (1024 ** 3):.1f}GiB / total={total_bytes / (1024 ** 3):.1f}GiB"
        )
        if free_bytes > best_free_bytes:
            best_free_bytes = free_bytes
            best_idx = idx

    return torch.device(f"cuda:{best_idx}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if getattr(config, "pad_token_id", None) is None:
    config.pad_token_id = getattr(tokenizer, "pad_token_id", None)
if getattr(config, "pad_token_id", None) is None:
    config.pad_token_id = getattr(tokenizer, "eos_token_id", None)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
if hasattr(base_model, "vision_model") and hasattr(base_model.vision_model, "config"):
    base_model.vision_model.config.output_hidden_states = True


def safe_small_batched_forward(self, pixel_values):
    vision_batch_size = self.vision_batch_size
    image_forward_out = []
    batch_size = len(pixel_values)

    for i in range(0, batch_size, vision_batch_size):
        start_idx = i
        end_idx = min(batch_size, i + vision_batch_size)
        outputs = self.vision_model(
            pixel_values[start_idx:end_idx],
            output_hidden_states=True,
            return_dict=True,
        )

        if getattr(outputs, "hidden_states", None) is not None and len(outputs.hidden_states) >= 2:
            tmp_hs = outputs.hidden_states[-2]
        elif getattr(outputs, "last_hidden_state", None) is not None:
            tmp_hs = outputs.last_hidden_state
        else:
            raise RuntimeError("[ERROR] vision_model returned neither hidden_states nor last_hidden_state.")

        image_forward_out.append(tmp_hs)

    vision_embedding = torch.cat(image_forward_out, dim=0)
    assert vision_embedding.shape[0] == batch_size
    return vision_embedding


base_model._small_batched_forward = types.MethodType(safe_small_batched_forward, base_model)

for module in base_model.modules():
    rotary_emb_core = getattr(module, "rotary_emb_core", None)
    if rotary_emb_core is None or not hasattr(rotary_emb_core, "inv_freq"):
        continue

    inv_freq = rotary_emb_core.inv_freq
    if isinstance(inv_freq, torch.Tensor) and getattr(inv_freq, "is_meta", False):
        rotary_emb_core.inv_freq = 1.0 / (
            rotary_emb_core.base
            ** (torch.arange(0, rotary_emb_core.dim, 2).float() / rotary_emb_core.dim)
        )
        rotary_emb_core._rotary_pos_emb_cache = None
        rotary_emb_core._seq_len_cached = 0

language_model_cls = base_model.language_model.__class__
if not issubclass(language_model_cls, GenerationMixin):
    language_model_cls.__bases__ = language_model_cls.__bases__ + (GenerationMixin,)
if not hasattr(base_model.language_model, "generation_config"):
    base_model.language_model.generation_config = GenerationConfig.from_model_config(
        base_model.language_model.config
    )

model = PeftModel.from_pretrained(base_model, LORA_PATH).eval()
device = select_device()
print(f"[INFO] selected device: {device}")
model.to(device)
processor = base_model.init_processor(tokenizer)

def chat(entry):
    question = entry['question']
    image_name = entry['image']
    options = entry['options']
    image_filepath = os.path.join(IMAGE_DIR, image_name)
    
    image = Image.open(image_filepath).convert("RGB")

    prompt = ROLE_PROMPT + f"Input: Image: <|image|>\nQuestion: {question}, Options: {'; '.join(options)}.\nOutput:"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}
    ]

    inputs = processor(messages, images=[image], videos=None)

    inputs.to(device)
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens':20,
        'decode_text':True,
        'do_sample': False
    })

    g = model.generate(**inputs)
    return g[0].strip()

with open(DATA_PATH, 'r', encoding="utf-8") as f, open(os.path.join(RESULT_DIR, result_file), 'w+', encoding="utf-8") as fout:
    lines = f.readlines()
    num_lines = len(lines)
    for line in tqdm(lines, total=num_lines, desc="Processing entries"):
        entry = json.loads(line)
        pred = chat(entry)
        if len(pred) == 0:
            pred = '--'
        gold_patterns = []
        result_json = {"image": entry['image'], "question": entry['question'], "options": entry['options'], "pred": pred, "gold": entry['answer']}
        fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
