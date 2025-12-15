from pathlib import Path
import re

# ===== 配置 =====
BASE_DIR = Path(__file__).parent
INPUT_FILE = "./dl/DLFC/full.md"
# =================

titles = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        # 只匹配「一级标题 # 」
        if re.match(r'^# ', line):
            title = line[2:].strip()
            titles.append(title)

# 输出结果
print("一级标题如下：")
for t in titles:
    print(t)

01_The Deep Learning Revolution
02_Probability
03_Standard Distributions
04_Single-Layer Networks_Regression
05_Single-Layer Networks_Classification
06_Deep Neural Networks
07_Gradient Descent
08_Backpropagation
09_Regularization
10_Convolutional Networks
11_Structured Distributions
12_Transformers
13_Graph Neural Networks
14_Sampling
15_Discrete Latent Variables
16_Continuous Latent Variables
17_Generative Adversarial Networks
18_Normalization Flows
19_Autoencoders
20_Diffusion Models