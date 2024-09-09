# 第二章 统计决策方法
---

### 性能度量指标
![](1042406-20161024154443875-2037260202.jpg)

- TP(True Positives)：预测为正样本，实际也为正样本的特征数
- FP(False Positives)：预测为正样本，实际为负样本的特征数
- TN(True Negatives)：预测为负样本，实际也为负样本的特征数
- FN(False Negatives)：预测为负样本，实际为正样本的特征数

>[!tip] **技巧：** XX=真假+**预测** T代表预测与实际相同，F代表预测与实际相反

图如上所示，里面**绿色的半圆**就是**TP**, **红色的半圆**就是**FP**, **左边的灰色长方形**（不包括绿色半圆），就是**FN**。**右边的浅灰色长方形**（不包括红色半圆），就是**TN**。这个绿色和红色组成的圆内代表我们分类得到模型结果认为是正值的样本。

1. **灵敏度**（Sensitivity、True Positive Rate、Recall）：
    - 定义：**真正例的比例**，表示模型对正例的识别能力。
    - 计算：
    $$Sn=\dfrac{TP}{TP+FN}=1-\beta$$
1. **特异度**（Specificity）：
    - 定义：**真负例的比例**，表示模型对负例的识别能力。
    - 计算：$$Sp=\dfrac{TN}{TN+FP}=1-\alpha$$
2. **第一类错误**（Type I Error、False Positive Rate）：
    - 定义：将**负例**错误地分类为**正例**的概率（**假正例率**、**误报率**）
    - 计算：
    $$\alpha=\dfrac{FP}{FP+TN}$$
1. **第二类错误**（Type II Error、False Negative Rate）：
    - 定义：将**正例**错误地分类为**负例**的概率（**假阴性率**、**漏报率**）
    - 计算：$$\beta=\dfrac{FN}{FN+TP}$$

7. **准确率**（Precision）：
    - 定义：被模型判定为**正例的观察**中，**实际为正例**的比例。
    - 计算：
    $$Pre=\frac{TP}{TP+FP}$$
1. **召回率（Recall）：** (同灵敏度)
    - 定义：**实际正例**中**被判定为正例**的比例。
    - 计算：
    $$Rec=\dfrac{TP}{TP+FN}$$
1. **正确率**（Accuracy）：
    - 定义：**所有正确分类**的观察占**总观察**的比例。
    - 计算：
    $$ACC=\dfrac{TP+TN}{TP+FP+TN+FN}$$

**Neyman-Pearson决策：**
    
    - 定义：一种基于统计学的假设检验框架，通常应用于二元假设检验问题，其中将控制第一类错误的概率（显著性水平）设定为先验决定的值。这种方法强调对错误的控制，并在给定显著性水平的条件下最大化功效（1 - 第二类错误率）。

**举例说明：**

假设有一个二元分类问题，模型对100个样本进行了预测，其中有30个正例。以下是混淆矩阵：

|      | 实际正例    | 实际负例   |
| ---- | ------- | ------ |
| 预测正例 | 20(TP)  | 10(FP) |
| 预测负例 | 5  (FN) | 65(TN) |

根据混淆矩阵，我们可以计算上述指标：

- 灵敏度（Recall）：$\frac{20}{20+5}=0.80$
- 特异度（Specificity）：$\frac{65}{65+10}=0.87$
- 第一类错误率（False Positive Rate）：$\frac{10}{10+65}=0.13$
- 第二类错误率（False Negative Rate）：$\frac{5}{5+20}=0.2$
- 准确率（Precision）：$\frac{20}{20+10}=0.67$
- 正确率（Accuracy）：$\frac{20+65}{100}=0.85$

### 一、三种贝叶斯决策

#### 1. 最小错误率贝叶斯决策

##### (1) 对于$c$类的判别

1. 判别函数：

$$对于\omega_i类，g_i(\boldsymbol{x}) = p(\mathbf{x}|\omega_i)P(\omega_i)$$

2. 判别规则：

$$若g_i(\mathbf{x}) = \max\limits_{j=1,\cdots,c}{g_j(\mathbf{x})}，则\mathbf{x}\in\omega_i$$

3. 决策面：

$$g_i(\mathbf{x}) = g_j(\mathbf{x})$$

##### (2) 对于两类的判别

1. 判别函数：

$$g(\mathbf{x}) = p(\mathbf{x}|\omega_1)P(\omega_1) - p(\mathbf{x}|\omega_2)P(\omega_2)$$

2. 判别规则：

$$若l(\mathbf{x}) = \frac{p(\mathbf{x}|\omega_1)}{p(\mathbf{x}|\omega_2)} \gtrless \lambda = \frac{P(\omega_2)}{P(\omega_1)}，则\mathbf{x}\in\begin{cases} \omega_1 \\ \omega_2 \end{cases}$$

其中，

$$p(\mathbf{x}|\omega_1)叫似然度，l(\mathbf{x})叫似然比$$

3. 决策面：

$$g(\mathbf{x}) = 0$$

#### 2. 最小风险贝叶斯决策

##### (1) 对于$c$类的判别

判别规则：

$$若R(\alpha_i|\mathbf{x}) = \min\limits_{j=1,\cdots,k}R(\alpha_j|\mathbf{x})，则决策\alpha=\alpha_i$$

其中，

$$R(\alpha_i|\mathbf{x})=\sum_{j=1}^c\lambda(\alpha_i,\omega_j)P(\omega_j|\mathbf{x})$$

##### (2) 对于两类的判别

判别规则：

$$若l(\mathbf{x}) = \frac{p(\mathbf{x}|\omega_1)}{p(\mathbf{x}|\omega_2)} \gtrless \lambda = \frac{P(\omega_2)}{P(\omega_1)}\cdot \frac{\lambda_{12}-\lambda_{22}}{\lambda_{21}-\lambda{11}}，则\mathbf{x}\in\begin{cases} \omega_1 \\ \omega_2\\\end{cases}$$

其中，

$$\lambda_{ij}=\lambda(\alpha_i, \omega_j)，即实际情况为\omega_j时决策为\alpha_i的风险。$$

#### 3. Neyman-Pearson决策

即固定一类错误率的情况下最小化另一类错误率。

判别规则：

$$若l(\mathbf{x}) = \frac{p(\mathbf{x}|\omega_1)}{p(\mathbf{x}|\omega_2)} \gtrless \lambda，则\mathbf{x}\in\begin{cases} \omega_1 \\ \omega_2\\\end{cases}$$

$\lambda$由固定的一类错误率计算出，假设固定第二类错误率（假阴性）为$\epsilon_0$，则决策边界$\lambda$保证$\displaystyle \int_{\mathscr{R}_1}p(\mathbf{x}|\omega_2)\text{d}\mathbf{x}=\varepsilon_0$

### 二、正态分布时的统计决策

#### 1. 正态分布概率密度公式

$$p(\mathbf{x})=\frac{1}{(2\pi)^{\frac{d}{2}}|\boldsymbol{\Sigma}|^{\frac{1}{2}}}\exp\left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\}$$

#### 2. 正态分布下最小错误率贝叶斯决策的性质

设所有类别的概率密度都服从正态分布，$p(\mathbf{x}|\omega_i) \sim N(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$

##### (1) 若$\boldsymbol{\Sigma}_i = \sigma^2\mathbf{I}$，$\mathbf{I}$是单位矩阵

$\omega_i$和$\omega_j$的决策面是平面，并且与$\boldsymbol{\mu}_i$和$\boldsymbol{\mu}_j$连线正交（垂直）。

1. 所有先验概率$P(\omega_i)$相同

决策面不仅与$\boldsymbol{\mu}_i$和$\boldsymbol{\mu}_j$连线正交，还过$\boldsymbol{\mu}_i$和$\boldsymbol{\mu}_j$连线的中点，是垂直平分线（面）。

此情况下**最小错误率**贝叶斯决策等价于**最小距离分类器**。

2. 先验概率$P(\omega_i)$不同

决策面向先验概率小的类偏移，即先验概率大的类占据更大的决策空间。

##### (2) 若$\boldsymbol{\Sigma}_i = \boldsymbol{\Sigma}$

$\omega_i$和$\omega_j$的决策面是平面，但是和$\boldsymbol{\mu}_i$和$\boldsymbol{\mu}_j$连线**不**正交（**不**垂直）。

1. 所有先验概率$P(\omega_i)$相同

决策面过$\boldsymbol{\mu}_i$和$\boldsymbol{\mu}_j$连线的中点。

2. 先验概率不同

决策面向先验概率小的类偏移，即先验概率大的类占据更大的决策空间。

对于两类情况，决策面为：

$$\begin{aligned}
g(\mathbf{x}) &= \mathbf{w}^\text{T}\mathbf{x} + w_0 = 0 \\
\mathbf{w} &= \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) \\
w_0 &= -\frac{1}{2}(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_2)^\text{T}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_1) - \ln\frac{P(\omega_2)}{P(\omega_1)}
\end{aligned}$$

##### (3) 一般情况

决策面是超二次曲面。

### 三、错误率的计算

#### 1. 贝叶斯错误率计算公式

##### (1) 两类情况

$$\begin{aligned}
P(e) &= P(\omega_1)\int_{\mathscr{R}_2}p(\mathbf{x}|\omega_1)\text{d}\mathbf{x} + P(\omega_2)\int_{\mathscr{R}_1}p(\mathbf{x}|\omega_2)\text{d}\mathbf{x} \\
    &= P(\omega_1)P_1(e)+P(\omega_2)P_2(e)\\    
\end{aligned}$$

其中

$$\begin{aligned}
P_1(e) &= \alpha \quad (假阳性) \\
P_2(e) &= \beta \quad (假阴性)\\
\end{aligned}$$

另外

$$\begin{aligned}
灵敏度\text{Sn} &= \frac{\text{TP}}{\text{TP}+\text{FN}} = 1-\beta \\
特异度\text{Sp} &= \frac{\text{TN}}{\text{TN}+\text{FP}} = 1-\alpha
\end{aligned}$$


##### (2) 多类情况

$$
P(e) = \int P(e|\mathbf{x})p(\mathbf{x})\text{d}\mathbf{x} = \int 1-\max\limits_{i}\{P(\omega_i|\mathbf{x})\}\text{d}\mathbf{x}
$$

#### 2. ROC曲线

横轴为$\alpha = 1-\text{Sp}$，纵轴为$1-\beta = \text{Sn}$

#### 3. 正态分布且各类协方差矩阵相等情况下错误率的计算

$$\begin{aligned}
P_1(e) &= \int_t^{+\infty} p(h|\omega_1)\text{d}h \\
     &= 1-\Phi(\frac{t+\eta}{\sigma}) \\
P_2(e) &= \int_{-\infty}^t p(h|\omega_2)\text{d}h \\
     &= \Phi(\frac{t-\eta}{\sigma}) \\
\end{aligned}$$  
其中，
$$\begin{aligned}
t     &= \ln\frac{P(\omega_1)}{P(\omega_2)} \\
\eta   &= \frac{1}{2}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2) \\
\sigma  &= \sqrt{2\eta}
\end{aligned}$$

#### 4. 高维独立随机变量时错误率的估计

$d$维随机变量$\mathbf{x}$各分量相互独立时，用中心极限定理把$h(\mathbf{x})$近似为正态分布，按照上面正态分布的公式计算错误率。

近似认为$h(\mathbf{x}|\omega_i) \sim N(\sum_{i=1}^{d}\eta_{il},\sum_{i=1}^{d}\sigma_{il}^2)$。

其中，$\eta_{il}$是$\omega_i$类第$l$个分量的对数似然比$p(h(x_l)|\omega_i)$的期望，$\sigma_{il}^2$是$\omega_i$类第$l$个分量的对数似然比$p(h(x_l)|\omega_i)$的方差。

### 四、一阶马尔科夫链

对数几率比：

$$S(x)=\sum_{i=1}^L \log\frac{a_{x_{i-1} x_i}^+}{a_{x_{i-1} x_i}^-} = \sum_{i=1}^L \beta_{x_{i-1}x_i}$$

阈值根据不同决策方法确定。