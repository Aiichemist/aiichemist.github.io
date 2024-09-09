# 入门知识

对概率的诠释有两大学派，一种是**频率派**，另一种是**贝叶斯派**。后面我们对观测集采用下面记号：
$$
\begin{split}
X_{N\times p}&=[x_{1},x_{2},\cdots,x_{N}]^{T}\\
&=\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1p}\\
x_{21} & x_{22} & \cdots & x_{2p}\\
\vdots & \vdots & \ddots & \vdots\\
x_{N1} & x_{N2} & \cdots & x_{Np}\\
\end{bmatrix}_{N\times p}
\end{split}
$$
 这个记号表示有 $N$ 个样本，每个样本都是 $p$ 维向量。其中每个观测都是由 $p(x|\theta)$ 生成的。

假设数据服从概率模型
$$X\sim p(x|\theta)$$
其中 $\theta$ 表示概率模型参数

## 频率派的观点

认为 $p(x|\theta)$ 中的 ==$\theta$ 是一个未知常量==。对于 $N$ 个观测来说观测集的概率为 $p(X|\theta)\mathop{=}\limits _{iid}\prod\limits _{i=1}^{N}p(x_{i}|\theta))$ 。

每个 $x_i$ 独立同分布于 $p(x|\theta)$

为了求 $\theta$ 的大小，我们采用**最大对数似然MLE**的方法：
$$
\theta_{MLE}
=\arg\max\limits _{\theta}\log p(X|\theta)
\mathop{=}\limits ^{iid}\mathop{argmax}\limits _{\theta}\sum\limits _{i=1}^{N}\log p(x_{i}|\theta)
$$
## 贝叶斯派的观点

贝叶斯派认为 $p(x|\theta)$ 中的 $\theta$ 不是一个常量。这个 ==$\theta$ 满足一个预设的先验的分布 $\theta\sim p(\theta)$== 。于是根据贝叶斯定理依赖观测集参数的**后验概率**可以写成：

$$
p(\theta|X)=\frac{p(X|\theta)\cdot p(\theta)}{p(X)}=\frac{p(X|\theta)\cdot p(\theta)}{\int\limits _{\theta}p(X|\theta)\cdot p(\theta)d\theta}
$$
为了求 $\theta$ 的值，我们要最大化这个参数**后验MAP**：

$$
\theta_{MAP}=\arg\max\limits _{\theta}p(\theta|X)=\arg\max\limits _{\theta}p(X|\theta)\cdot p(\theta)
$$
其中第二个等号是由于分母和 $\theta$ 没有关系。求解这个 $\theta$ 值后计算$\frac{p(X|\theta)\cdot p(\theta)}{\int\limits _{\theta}p(X|\theta)\cdot p(\theta)d\theta}$ ，就得到了参数的后验概率。其中 $p(X|\theta)$ 叫**似然**，是我们的**模型分布**。得到了参数的后验分布后，我们可以将这个分布用于**贝叶斯预测**：
$$
p(\tilde{x}|X)
=\int\limits_{\theta}p(\tilde{x},\theta|X)d\theta
=\int\limits _{\theta}p(\tilde{x}|\theta)\cdot p(\theta|X)d\theta
$$
 其中积分中的被乘数是模型，乘数是后验分布。

## 小结

频率派和贝叶斯派分别给出了一系列的机器学习算法。

频率派的观点导出了一系列的**统计机器学习算法（优化问题）**
- 设计模型：概率/非概率
- 设计损失函数
- 使用优化算法
而贝叶斯派导出了**概率图理论**，需要求积分。

在应用频率派的 MLE 方法时最优化理论占有重要地位。而贝叶斯派的算法无论是后验概率的建模还是应用这个后验进行推断时积分占有重要地位。因此采样积分方法如 MCMC 有很多应用。

# 数学基础

## 高斯分布

### 一维情况 MLE
此时模型参数 $\theta=(\mu,\Sigma)=(\mu,\sigma^{2})$

在 MLE 方法中：
$$
\theta_{MLE}=\mathop{argmax}\limits _{\theta}\log p(X|\theta)\mathop{=}\limits _{iid}\mathop{argmax}\limits _{\theta}\sum\limits _{i=1}^{N}\log p(x_{i}|\theta)
$$
一般地，高斯分布的概率密度函数PDF写为：
$$
p(x|\mu,\Sigma)=\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}e^{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}
$$
考虑一维的情况，带入 MLE：
$$
\log p(X|\theta)=\sum\limits _{i=1}^{N}\log p(x_{i}|\theta)=\sum\limits _{i=1}^{N}\log\frac{1}{\sqrt{2\pi}\sigma}\exp(-(x_{i}-\mu)^{2}/2\sigma^{2})
$$
首先对 $\mu$ 的极值 可以得到 ：
$$
\mu_{MLE}=\arg\max\limits _{\mu}\log p(X|\theta)=\arg\min\limits _{\mu}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2}
$$
 于是：
$$
\frac{\partial}{\partial\mu}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2}=0\quad\Rightarrow\quad\mu_{MLE}=\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}
$$
其次对 $\theta$ 中的另一个参数 $\sigma$ ，有：
$$
\begin{align}
\sigma_{MLE}=\arg\max\limits _{\sigma}\log p(X|\theta)&=\arg\max\limits _{\sigma}\sum\limits _{i=1}^{N}[-\log\sigma-\frac{1}{2\sigma^{2}}(x_{i}-\mu)^{2}]\nonumber\\
&=\arg\max\limits _{\sigma}\sum\limits _{i=1}^{N}[\log\sigma+\frac{1}{2\sigma^{2}}(x_{i}-\mu)^{2}]
\end{align}
$$
于是：
$$
\frac{\partial}{\partial\sigma}\sum\limits _{i=1}^{N}[\log\sigma+\frac{1}{2\sigma^{2}}(x_{i}-\mu)^{2}]=0\quad\Rightarrow\quad\sigma_{MLE}^{2}=\frac{1}{N}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2}
$$
值得注意的是，上面的推导中，首先对 $\mu$ 求 MLE， 然后利用这个结果求 $\sigma_{MLE}$ ，因此可以预期的是对数据集求期望时 $\mathbb{E}_{\mathcal{D}}[\mu_{MLE}]$ 是**无偏差**的：
$$
\mathbb{E}_{\mathcal{D}}[\mu_{MLE}]=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}]=\frac{1}{N}\sum\limits _{i=1}^{N}\mathbb{E}_{\mathcal{D}}[x_{i}]=\mu
$$
但是当对 $\sigma_{MLE}$ 求期望的时候由于使用了单个数据集的 $\mu_{MLE}$，因此对所有数据集求期望的时候我们会发现 $\sigma_{MLE}$ 是 **有偏**的：

$$
\begin{align}
\mathbb{E}_{\mathcal{D}}[\sigma_{MLE}^{2}]&=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}(x_{i}-\mu_{MLE})^{2}]=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}(x_{i}^{2}-2x_{i}\mu_{MLE}+\mu_{MLE}^{2})\nonumber
\\&=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}^{2}-\mu_{MLE}^{2}]=\mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}^{2}-\mu^{2}+\mu^{2}-\mu_{MLE}^{2}]\nonumber\\
&= \mathbb{E}_{\mathcal{D}}[\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}^{2}-\mu^{2}]-\mathbb{E}_{\mathcal{D}}[\mu_{MLE}^{2}-\mu^{2}]=\sigma^{2}-(\mathbb{E}_{\mathcal{D}}[\mu_{MLE}^{2}]-\mu^{2})\nonumber\\&=\sigma^{2}-(\mathbb{E}_{\mathcal{D}}[\mu_{MLE}^{2}]-\mathbb{E}_{\mathcal{D}}^{2}[\mu_{MLE}])=\sigma^{2}-Var[\mu_{MLE}]\nonumber\\&=\sigma^{2}-Var[\frac{1}{N}\sum\limits _{i=1}^{N}x_{i}]=\sigma^{2}-\frac{1}{N^{2}}\sum\limits _{i=1}^{N}Var[x_{i}]=\frac{N-1}{N}\sigma^{2}
\end{align}
$$
所以：
$$\hat{\sigma}^{2}=\frac{1}{N-1}\sum\limits _{i=1}^{N}(x_{i}-\mu)^{2}$$

### 多维情况

多维高斯分布表达式为：
$$p(x|\mu,\Sigma)=\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}\exp\left({-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)}\right)$$
其中 $x,\mu\in\mathbb{R}^{p},\Sigma\in\mathbb{R}^{p\times p}$ ，$\Sigma$ 为协方差矩阵，一般而言也是半正定矩阵。这里我们只考虑正定矩阵。首先我们处理指数上的数字，指数上的数字可以记为 $x$ 和 $\mu$ 之间的**马氏距离**。对于对称的协方差矩阵可进行特征值分解，$\Sigma=U\Lambda U^{T}=[u_{1},u_{2},\cdots,u_{p}]diag(\lambda_{i})[u_{1},u_{2},\cdots,u_{p}]^{T}=\sum\limits _{i=1}^{p}u_{i}\lambda_{i}u_{i}^{T}$ ，于是：

$$\Sigma^{-1}=\sum\limits _{i=1}^{p}u_{i}\frac{1}{\lambda_{i}}u_{i}^{T}$$
$$\Delta=(x-\mu)^{T}\Sigma^{-1}(x-\mu)=\sum\limits _{i=1}^{p}(x-\mu)^{T}u_{i}\frac{1}{\lambda_{i}}u_{i}^{T}(x-\mu)=\sum\limits _{i=1}^{p}\frac{y_{i}^{2}}{\lambda_{i}}$$
我们注意到 $y_{i}$ 是 $x-\mu$ 在特征向量 $u_{i}$ 上的投影长度，因此上式子就是 $\Delta$ 取不同值时的**同心椭圆**。

下面我们看多维高斯模型在实际应用时的两个问题

>一、参数 $\Sigma,\mu$ 的自由度为 $O(p^{2})$ 对于维度很高的数据其自由度太高
>
>解决方案：
>高自由度的来源是 $\Sigma$ 有 $\frac{p(p+1)}{2}$ 个自由参数
>
>1. 可以假设其是**对角矩阵**（此时椭圆长短轴与坐标轴平行）
>
>1. 甚至假设**各向同性**（即各个特征值均相等$\lambda_1=\lambda_2=\cdots=\lambda_p$，此时椭圆退化为正圆）
>
>前一种的算法有**因子分析Factor Analysis**，后一种有**概率PCA**(p-PCA) 。


>二、单个高斯分布是单峰的，对有多个峰的数据分布不能得到好的结果
>
>解决方案：高斯混合GMM 模型。

下面对多维高斯分布的常用定理进行介绍。

我们记 $x=\begin{bmatrix}x_1\\x_2\\\vdots\\x_p\end{bmatrix}=\begin{bmatrix}x_{a(m\times1)}\\x_{b(n\times1)}\end{bmatrix}\qquad\mu=\begin{bmatrix}\mu_{a(m\times1)}\\\mu_{b(n\times1)}\end{bmatrix}\qquad\Sigma=\begin{bmatrix}\Sigma_{aa}&\Sigma_{ab}\\\Sigma_{ba}&\Sigma_{bb}\end{bmatrix}$

已知 $x\sim N(\mu,\Sigma)$。

首先是一个高斯分布的定理：

>**定理：** 已知 $x\sim N(\mu,\Sigma), y=Ax+b$，那么 $y\sim N(A\mu+b, A\Sigma A^T)$。
>
>证明：$\mathbb{E}[y]=\mathbb{E}[Ax+b]=A\mathbb{E}[x]+b=A\mu+b$
>
>$Var[y]=Var[Ax+b]=Var[Ax]=A\cdot Var[x]\cdot A^T$。

下面利用这个定理得到 $p(x_a),p(x_b),p(x_a|x_b),p(x_b|x_a)$ 这四个量。

#### 1. $p(x_a)$
$x_a=\underbrace{\begin{bmatrix}\mathbb{I}_{m\times m}&\mathbb{O}_{m\times n}\end{bmatrix}}_A\underbrace{\begin{bmatrix}x_a\\x_b\end{bmatrix}}_x$，代入定理中得到：

$$\mathbb{E}[x_a]=\begin{bmatrix}\mathbb{I}&\mathbb{O}\end{bmatrix}\begin{bmatrix}\mu_a\\\mu_b\end{bmatrix}=\mu_a\\Var[x_a]=\begin{bmatrix}\mathbb{I}&\mathbb{O}\end{bmatrix}\begin{bmatrix}\Sigma_{aa}&\Sigma_{ab}\\\Sigma_{ba}&\Sigma_{bb}\end{bmatrix}\begin{bmatrix}\mathbb{I}\\\mathbb{O}\end{bmatrix}=\Sigma_{aa}$$
所以 $x_a\sim N(\mu_a,\Sigma_{aa})$。

#### 2. $p(x_b)$

同样的，$x_b\sim\mathcal{N}(\mu_b,\Sigma_{bb})$。

#### 3. $p(x_b|x_a)$
对于两个条件概率，我们引入三个量：
$$\begin{split}
x_{b\cdot a}&=x_b-\Sigma_{ba}\Sigma_{aa}^{-1}x_a\\
\mu_{b\cdot a}&=\mu_b-\Sigma_{ba}\Sigma_{aa}^{-1}\mu_a\\
\Sigma_{bb\cdot a}&=\Sigma_{bb}-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}\\
\end{split}$$
特别的，最后一个式子叫做 $\Sigma_{bb}$ 的**舒尔补Schur Complementary**。可以看到：
$$x_{b\cdot a}=\underbrace{\begin{bmatrix}-\Sigma_{ba}\Sigma_{aa}^{-1}&\mathbb{I}_{n\times n}\end{bmatrix}}_{A}\underbrace{\begin{bmatrix}x_a\\x_b\end{bmatrix}}_{x}$$
所以：
$$\begin{split}
\mathbb{E}[x_{b\cdot a}]&=\begin{bmatrix}-\Sigma_{ba}\Sigma_{aa}^{-1}&\mathbb{I}_{n\times n}\end{bmatrix}\begin{bmatrix}\mu_a\\\mu_b\end{bmatrix}=\mu_{b\cdot a}\\
Var[x_{b\cdot a}]&=\begin{bmatrix}-\Sigma_{ba}\Sigma_{aa}^{-1}&\mathbb{I}_{n\times n}\end{bmatrix}\begin{bmatrix}\Sigma_{aa}&\Sigma_{ab}\\\Sigma_{ba}&\Sigma_{bb}\end{bmatrix}\begin{bmatrix}-\Sigma_{aa}^{-1}\Sigma_{ba}^T\\\mathbb{I}_{n\times n}\end{bmatrix}=\Sigma_{bb\cdot a}
\end{split}$$
利用这三个量可以得到 $x_b=x_{b\cdot a}+\Sigma_{ba}\Sigma_{aa}^{-1}x_a$。因此：
$$\mathbb{E}[x_b|x_a]=\mu_{b\cdot a}+\Sigma_{ba}\Sigma_{aa}^{-1}x_a$$
$$Var[x_b|x_a]=\Sigma_{bb\cdot a}$$

这里同样用到了定理。

#### 4. $p(x_a|x_b)$
同样：
$$x_{a\cdot b}=x_a-\Sigma_{ab}\Sigma_{bb}^{-1}x_b\\
\mu_{a\cdot b}=\mu_a-\Sigma_{ab}\Sigma_{bb}^{-1}\mu_b\\
\Sigma_{aa\cdot b}=\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}
$$
所以：
$$\mathbb{E}[x_a|x_b]=\mu_{a\cdot b}+\Sigma_{ab}\Sigma_{bb}^{-1}x_b$$
$$Var[x_a|x_b]=\Sigma_{aa\cdot b}$$
下面利用上边四个量，求解线性模型：

> 已知：$p(x)=\mathcal{N}(\mu,\Lambda^{-1}),p(y|x)=\mathcal{N}(Ax+b,L^{-1})$，求解：$p(y),p(x|y)$。
>
>   解：令 $y=Ax+b+\epsilon,\epsilon\sim\mathcal{N}(0,L^{-1})$，所以 $\mathbb{E}[y]=\mathbb{E}[Ax+b+\epsilon]=A\mu+b$，$Var[y]=A \Lambda^{-1}A^T+L^{-1}$，因此：
>   $$
>   p(y)=\mathcal{N}(A\mu+b,L^{-1}+A\Lambda^{-1}A^T)
>   $$
>   引入 $z=\begin{bmatrix}x\\y\end{bmatrix}$，我们可以得到 $Cov[x,y]=\mathbb{E}[(x-\mathbb{E}[x])(y-\mathbb{E}[y])^T]$。对于这个协方差可以直接计算：
>   $$
>   \begin{align}
>   Cov(x,y)&=\mathbb{E}[(x-\mu)(Ax-A\mu+\epsilon)^T]=\mathbb{E}[(x-\mu)(x-\mu)^TA^T]=Var[x]A^T=\Lambda^{-1}A^T
>   \end{align}
>   $$
>   注意到协方差矩阵的对称性，所以 $p(z)=\mathcal{N}\begin{bmatrix}\mu\\A\mu+b\end{bmatrix},\begin{bmatrix}\Lambda^{-1}&\Lambda^{-1}A^T\\A\Lambda^{-1}&L^{-1}+A\Lambda^{-1}A^T\end{bmatrix})$。根据之前的公式，我们可以得到：
>   $$
>   \mathbb{E}[x|y]=\mu+\Lambda^{-1}A^T(L^{-1}+A\Lambda^{-1}A^T)^{-1}(y-A\mu-b)
>   $$
>
>   $$
>   Var[x|y]=\Lambda^{-1}-\Lambda^{-1}A^T(L^{-1}+A\Lambda^{-1}A^T)^{-1}A\Lambda^{-1}
>   $$

