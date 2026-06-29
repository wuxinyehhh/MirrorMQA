from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = str(REPO_ROOT / "Dataset" / "test_wxy_1.jsonl")
IMAGE_DIR = str(REPO_ROOT / "Dataset" / "images")
RESULT_DIR = {"base": str(REPO_ROOT / "outputs" / "experiment")}
ROLE_PROMPT = "You are currently a senior expert in visual reasoning.\nGiven an Image, a Question, and Options, your task is to choose the correct answer.\nNote that you only need to choose one option from all options without explaining any reason.\n"