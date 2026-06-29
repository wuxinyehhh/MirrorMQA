"""
================================================================================
                    DMXAPI Gemini 图像分析示例（流式版本）
================================================================================
功能说明：
    使用 Google Gemini API 对本地图片进行智能分析
    支持流式响应，实时获取分析结果

API 提供商：DMXAPI
使用模型：gemini-3-flash-preview
================================================================================
"""

import os
import base64
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import re


# ==============================================================================
# 配置参数区域
# ==============================================================================
MODEL = "gemini-3-flash-preview"                                      # AI 模型名称
API_KEY = os.getenv("DMXAPI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set DMXAPI_API_KEY in the environment.")
BASE_URL = "https://www.dmxapi.cn/v1beta"                           # DMXAPI 基础地址

IMAGE_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/llm/project/wxy/test_openai/image"
OUTPUT_FILE = os.path.join(IMAGE_DIR, "predictions_gemini_2.tsv")

# 默认参数配置
DEFAULT_IMAGE_PATH = os.path.join(IMAGE_DIR, "1-0021.png")           # 默认图片路径
DEFAULT_PROMPT = """    "You are a senior expert in visual pattern recognition and mirror-image reasoning. "
    "You will be given ONE composite image that contains the question image and four option images(labeled A.1, B.2, C.3, D.4) within the same image. "
    "Your task is to choose the single option whose image is the correct mirror of the question image "
    "(either left-right or up-down mirror). Determine the mirror type yourself. "
    "Requirements: Output must be exactly one character: A/B/C/D. No explanation. "
    "Example 1:\n"
    "Image path: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/llm/project/wxy/test_openai/image/1-0021.png. "
    "Answer: D\n\n"
    "Example 2:\n"
    "Image path: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/llm/project/wxy/test_openai/image/6-0035.png. "
    "Answer: D\n\n"
    "If ambiguous, choose the most reasonable.To reiterate, evaluation is not required."
    " Output exactly in this format on the last line: FINAL: X where X is one of A/B/C/D." """                 # 默认提示词


# ==============================================================================
# 工具函数：图片编码
# ==============================================================================
def encode_image_to_base64(image_path):
    """
    将图片文件转换为 base64 编码字符串

    参数:
        image_path (str): 图片文件的路径

    返回:
        str: base64 编码的图片字符串
    """
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        encoded_string = base64.b64encode(image_data).decode('utf-8')
    return encoded_string


# ==============================================================================
# 主功能函数：调用 Gemini API
# ==============================================================================
def call_gemini_api_stream(image_path=DEFAULT_IMAGE_PATH, prompt_text=DEFAULT_PROMPT):
    """
    调用 Gemini API 进行流式图片分析

    参数:
        image_path (str, optional): 图片文件路径，默认使用 DEFAULT_IMAGE_PATH
        prompt_text (str, optional): 提示文本，默认使用 DEFAULT_PROMPT

    返回:
        dict: 包含完整响应文本的字典 {'full_text': str}
        None: 当请求发生错误时返回 None
    """

    # --------------------------------------------------------------------------
    # 步骤 1: 构建 API 请求 URL
    # --------------------------------------------------------------------------
    url = f"{BASE_URL}/models/{MODEL}:streamGenerateContent?key={API_KEY}&alt=sse"

    # --------------------------------------------------------------------------
    # 步骤 2: 编码图片为 base64
    # --------------------------------------------------------------------------
    image_base64 = encode_image_to_base64(image_path)

    # --------------------------------------------------------------------------
    # 步骤 3: 构造请求头和请求体
    # --------------------------------------------------------------------------
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",               # 图片 MIME 类型
                        "data": image_base64                    # base64 编码的图片数据
                    }
                },
                {"text": prompt_text}                           # 分析提示词
            ]
        }]
    }

    # --------------------------------------------------------------------------
    # 步骤 4: 发送请求并处理流式响应
    # --------------------------------------------------------------------------
    try:
        response = requests.post(url, headers=headers, json=payload, stream=False)
        response.raise_for_status()

        full_text = ""

        # 逐行解析 SSE 格式的响应
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')

                # SSE 数据行以 "data: " 开头
                if line_str.startswith('data: '):
                    json_str = line_str[6:]                     # 去掉 "data: " 前缀

                    try:
                        data = json.loads(json_str)

                        # 提取响应中的文本内容
                        if 'candidates' in data:
                            for candidate in data['candidates']:
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    for part in candidate['content']['parts']:
                                        if 'text' in part:
                                            text_chunk = part['text']
                                            full_text += text_chunk

                    except json.JSONDecodeError:
                        continue                                # 跳过无法解析的行

        return {"full_text": full_text}

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None

    except Exception as e:
        print(f"发生错误: {e}")
        return None


def list_images(folder: str) -> list:
    images = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(".png"):
            p = os.path.join(folder, fn)
            if os.path.isfile(p):
                images.append(p)
    return images


def extract_choice(text: str) -> str:
    if not text:
        return "?"
    up = text.strip().upper()

    # 优先从最后一行固定格式中提取：FINAL: X
    final_matches = re.findall(r"(?m)^\\s*FINAL\\s*:\\s*([ABCD])\\s*$", up)
    if final_matches:
        return final_matches[-1]

    # 否则取最后一个出现的 A/B/C/D（避免被 'Based/Analyze' 等前缀干扰）
    all_letters = re.findall(r"[ABCD]", up)
    if all_letters:
        return all_letters[-1]

    # 最后再尝试 1/2/3/4 映射（取最后一个）
    digits = re.findall(r"\b([1-4])\b", up)
    if digits:
        return {"1": "A", "2": "B", "3": "C", "4": "D"}[digits[-1]]

    return "?"


def extract_conclusion_line(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    # 优先找 FINAL: X
    for ln in reversed(lines):
        if re.match(r"^FINAL\\s*:\\s*[ABCD]\\s*$", ln.upper()):
            return ln
    # 其次找包含 CONCLUSION 的行
    for ln in reversed(lines):
        if "CONCLUSION" in ln.upper():
            return ln
    return ""


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    def read_processed(path: str) -> set:
        processed = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 与 predictions_gemini_0.tsv 一致：按制表符分隔
                    parts = line.split("\t")
                    name = parts[0] if parts else ""
                    if name:
                        processed.add(name)
        return processed

    processed = read_processed(OUTPUT_FILE)
    images = [p for p in list_images(IMAGE_DIR) if os.path.basename(p) not in processed]
    if not images:
        print(f"[INFO] {IMAGE_DIR} 下未找到需要处理的 .png 图片。")
        raise SystemExit(0)

    lock = Lock()

    def solve_one(image_path: str):
        name = os.path.basename(image_path)
        result = call_gemini_api_stream(image_path=image_path, prompt_text=DEFAULT_PROMPT)
        text = (result.get("full_text", "") if result else "").strip()
        ans = extract_choice(text)
        conclusion = extract_conclusion_line(text)
        if ans == "?":
            retry_prompt = DEFAULT_PROMPT + " Output only one character: A/B/C/D."
            result = call_gemini_api_stream(image_path=image_path, prompt_text=retry_prompt)
            text = (result.get("full_text", "") if result else "").strip()
            ans = extract_choice(text)
            conclusion = extract_conclusion_line(text)
        return name, ans, conclusion

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(solve_one, p): p for p in images}
            for fut in as_completed(futures):
                p = futures[fut]
                name = os.path.basename(p)
                try:
                    name, ans, conclusion = fut.result()
                except Exception as e:
                    ans = "ERROR"
                    conclusion = ""
                    print(f"[ERROR] {name}: {e}")

                with lock:
                    out.write(f"{name}\t{ans}\n")
                    out.flush()
                if conclusion:
                    print(conclusion)
                print(f"{name}\t{ans}")

    print(f"\nSaved: {OUTPUT_FILE}")
