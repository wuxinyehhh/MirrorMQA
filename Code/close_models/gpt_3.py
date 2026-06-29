from typing import Tuple
import os
import base64
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ========= 配置 =========
# 读取环境变量中的 API Key（避免硬编码）
api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量中读取 API Key
if not api_key:
    raise RuntimeError("请先设置环境变量 OPENAI_API_KEY（不要把 key 写进代码里）。")

# 读取代理设置（如果需要）
http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

# 创建 OpenAI 客户端配置
client_kwargs = {
    "api_key": api_key,
    "base_url": "https://openkey.cloud/v1",  # 你的代理（含 /v1）
    "timeout": 120.0,  # 设置超时时间为120秒（图片上传可能需要更长时间）
    "max_retries": 2,  # 最大重试次数
}

# 如果设置了代理，添加到配置中
if http_proxy or https_proxy:
    import httpx
    # 创建带代理的 httpx 客户端，trust_env=True 会自动从环境变量读取 HTTP_PROXY 和 HTTPS_PROXY
    # 使用同步客户端，因为 OpenAI SDK 的同步接口需要同步的 http_client
    client_kwargs["http_client"] = httpx.Client(
        timeout=120.0,
        trust_env=True  # 自动从环境变量读取代理设置
    )

# 创建 OpenAI 客户端
client = OpenAI(**client_kwargs)

MODEL = "gpt-5.5"   # 必须是支持图像输入的模型（按你的代理实际可用模型修改）
MAX_WORKERS = 15
DATA_DIR = os.path.join(os.path.dirname(__file__), "image")
OUTPUT_FILE = os.path.join(DATA_DIR, "predictions_gpt_3.tsv")

PROMPT = (
    "You are a senior expert in visual pattern recognition and mirror-image reasoning.\n"
    "You will be given ONE composite image that contains the question image and four option images "
    "(labeled A.1, B.2, C.3, D.4) within the same image.\n"
    "Your task is to choose the single option whose image is the correct mirror of the question image "
    "(either left-right or up-down mirror). Determine the mirror type yourself.\n\n"
    "Example 1:\n"
    "Image path: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/llm/project/wxy/test_openai/image/1-0021.png. "
    "Answer: D\n\n"
    "Example 2:\n"
    "Image path: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/llm/project/wxy/test_openai/image/4-0242.png. "
    "Answer: B\n\n"
    "Example 3:\n"
    "Image path: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/llm/project/wxy/test_openai/image/6-0035.png. "
    "Answer: D\n\n"
    "Requirements: Output must be exactly one character: A/B/C/D. No explanation.\n"
    "If ambiguous, choose the most reasonable."
)



# ========= 工具函数 =========
def image_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def build_messages(image_path: str) -> list:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
        ],
    }]

def solve_one(image_path: str) -> Tuple[str, str]:
    messages = build_messages(image_path)
    name = os.path.basename(image_path)
    
    try:
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            raise Exception(f"图片文件不存在: {image_path}")
        
        # 检查图片文件大小（避免文件过大）
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
        if file_size > 20:  # 如果图片大于20MB，给出警告
            print(f"[WARNING] {name}: 图片文件较大 ({file_size:.2f}MB)，可能需要更长时间处理")
        
        # 打印调试信息（可选，用于排查问题）
        # print(f"[DEBUG] {name}: 发送请求到模型 {MODEL}")
        
        # 直接使用流式响应，避免非流式响应在截断时content为None的问题
        # 流式响应可以逐步获取内容，即使被截断也能获取部分内容
        content = None
        finish_reason = None
        
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_completion_tokens=50,  # 兼容部分模型：用 max_completion_tokens 替代 max_tokens
                stream=True,
            )
            
            content_parts = []
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    # 获取delta中的内容
                    if choice.delta and choice.delta.content:
                        content_parts.append(choice.delta.content)
                    # 记录finish_reason
                    if hasattr(choice, 'finish_reason') and choice.finish_reason:
                        finish_reason = choice.finish_reason
            
            if content_parts:
                content = ''.join(content_parts).strip()
                if finish_reason == 'length':
                    print(f"[WARNING] {name}: 响应被截断: {content[:50]}...")
            else:
                # 如果流式响应没有内容，尝试非流式响应
                print(f"[WARNING] {name}: 流式响应未获取到内容，尝试非流式响应")
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=100,  # 兼容部分模型：用 max_completion_tokens 替代 max_tokens
                )
                
                if resp.choices and len(resp.choices) > 0:
                    choice = resp.choices[0]
                    finish_reason = choice.finish_reason
                    message = choice.message
                    if message:
                        content = message.content
                        if content:
                            content = content.strip()
        except Exception as stream_error:
            error_msg_str = str(stream_error)
            # 检查是否是配额不足错误
            if "quota" in error_msg_str.lower() or "insufficient" in error_msg_str.lower() or "403" in error_msg_str:
                raise Exception(f"API配额不足: {error_msg_str}") from stream_error
            
            # 如果流式响应失败，回退到非流式响应
            print(f"[WARNING] {name}: 流式响应失败 ({stream_error})，尝试非流式响应")
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=100,  # 兼容部分模型：用 max_completion_tokens 替代 max_tokens
                )
                
                if resp.choices and len(resp.choices) > 0:
                    choice = resp.choices[0]
                    finish_reason = choice.finish_reason
                    message = choice.message
                    if message:
                        content = message.content
                        if content:
                            content = content.strip()
            except Exception as e:
                error_msg_str = str(e)
                # 检查是否是配额不足错误
                if "quota" in error_msg_str.lower() or "insufficient" in error_msg_str.lower() or "403" in error_msg_str:
                    raise Exception(f"API配额不足: {error_msg_str}") from e
                raise Exception(f"API调用失败: {e}") from e
        
        if content is None:
            # 检查是否有其他字段包含内容
            raise Exception(f"API返回空响应：无法获取content (finish_reason={finish_reason})，请检查模型是否支持图像输入")
        
        text = content.strip()
        # 兜底：只取第一个字母
        ans = (text[:1].upper() if text else "?")
        return name, ans
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # 检查是否是配额不足错误（优先处理）
        if "quota" in error_msg.lower() or "insufficient" in error_msg.lower() or ("403" in error_msg and "quota" in error_msg.lower()):
            raise Exception(f"API配额不足: {error_msg}。请检查API配额或稍后重试") from e
        
        # 提供更详细的错误信息
        if "Connection" in error_type or "Network" in error_msg or "unreachable" in error_msg.lower():
            raise Exception(f"网络连接失败: {error_msg}。请检查网络连接或设置HTTP_PROXY/HTTPS_PROXY环境变量") from e
        elif "timeout" in error_msg.lower():
            raise Exception(f"请求超时: {error_msg}。图片可能过大或网络较慢") from e
        elif "401" in error_msg or "Unauthorized" in error_msg:
            raise Exception(f"认证失败: {error_msg}。请检查API Key是否正确") from e
        elif "404" in error_msg or "Not Found" in error_msg:
            raise Exception(f"模型不存在: {error_msg}。请检查MODEL配置是否正确") from e
        elif "403" in error_msg:
            raise Exception(f"API访问被拒绝: {error_msg}。可能是配额不足或权限问题") from e
        else:
            raise Exception(f"API调用失败 ({error_type}): {error_msg}") from e

def list_images(folder: str, processed_images: set) -> list:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images_to_process = []
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in exts and fn not in processed_images:
            images_to_process.append(p)
    return images_to_process

def read_processed_images() -> set:
    processed_images = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split("\t")
                    if parts:  # 确保有内容
                        processed_images.add(parts[0])
    return processed_images

def clean_tsv():
    """删除 predictions_gpt_3.tsv 中图片文件在 DATA_DIR 下实际不存在的行。"""
    if not os.path.exists(OUTPUT_FILE):
        return
    valid_lines = []
    removed = 0
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            img_name = parts[0]
            if os.path.isfile(os.path.join(DATA_DIR, img_name)):
                valid_lines.append(stripped)
            else:
                removed += 1
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")
    if removed:
        print(f"[INFO] clean_tsv: 已删除 {removed} 行（图片文件不存在），保留 {len(valid_lines)} 行。")

def main():
    # 确保输出目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    clean_tsv()
    
    processed_images = read_processed_images()
    images = list_images(DATA_DIR, processed_images)
    if not images:
        print(f"[INFO] {DATA_DIR} 下未找到未处理的图片文件（png/jpg/jpeg/webp）。")
        return

    results = {}

    quota_exhausted = False
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(solve_one, p): p for p in images}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    name, ans = fut.result()
                    results[name] = ans
                    print(f"{name}\t{ans}")
                    # 实时保存结果，避免中断时丢失
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{name}\t{ans}\n")
                except Exception as e:
                    error_msg = str(e)
                    name = os.path.basename(p)
                    
                    # 检查是否是配额不足错误
                    if "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
                        quota_exhausted = True
                        print(f"[ERROR] {name}: {error_msg}")
                        print(f"[WARNING] 检测到API配额不足，停止处理剩余图片")
                        # 保存错误信息到结果中，标记为配额不足
                        results[name] = "QUOTA_ERROR"
                        # 实时保存错误结果
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            f.write(f"{name}\tQUOTA_ERROR\n")
                        # 取消剩余任务
                        for remaining_fut in futs:
                            if remaining_fut != fut:
                                remaining_fut.cancel()
                        break
                    else:
                        print(f"[ERROR] {name}: {error_msg}")
                        # 保存错误信息到结果中，标记为失败
                        results[name] = "ERROR"
                        # 实时保存错误结果
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            f.write(f"{name}\tERROR\n")
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，已保存当前结果")
    except Exception as e:
        print(f"[ERROR] 主程序异常: {e}")
        import traceback
        traceback.print_exc()
    
    # 如果还有未保存的结果，再次保存（防止遗漏）
    if results:
        existing_results = read_processed_images()
        new_results = {k: v for k, v in results.items() if k not in existing_results}
        if new_results:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for k in sorted(new_results):
                    f.write(f"{k}\t{new_results[k]}\n")
    
    print(f"\nSaved: {OUTPUT_FILE}")
    
    # 如果配额不足，给出提示
    if quota_exhausted:
        print("\n" + "="*80)
        print("警告: API配额已用完，部分图片未处理")
        print("请检查API配额或稍后重试")
        print("="*80)

if __name__ == "__main__":
    main()
