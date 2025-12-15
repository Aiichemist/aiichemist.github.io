import re
from pathlib import Path

# ========= 配置区 =========
BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "./full.md"
OUTPUT_DIR = BASE_DIR / "chapters"

FILENAME_LIST = [
    "01_DLRevolution",
    "02_Probability",
    "03_StandardDist",
    "04_SL-NR",
    "05_SL-NC",
    "06_DNN",
    "07_GD",
    "08_Backpropagation",
    "09_Regularization",
    "10_CN",
    "11_StructuredDist",
    "12_Transformers",
    "13_GNN",
    "14_Sampling",
    "15_DLV",
    "16_CLV",
    "17_GAN",
    "18_NF",
    "19_Autoencoders",
    "20_DM",
    "21_Appendix"
]
# =========================

OUTPUT_DIR.mkdir(exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# 只按一级标题拆分
sections = [
    s for s in re.split(r'(?m)(?=^# )', text)
    if s.strip()
]

if len(sections) != len(FILENAME_LIST):
    raise ValueError(
        f"一级标题数量({len(sections)}) 与 文件名数量({len(FILENAME_LIST)}) 不一致"
    )

for section, name in zip(sections, FILENAME_LIST):
    md_path = OUTPUT_DIR / f"{name}.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(section.strip() + "\n")

    print(f"生成文件：{md_path.name}")

print("全部一级章节拆分完成")
