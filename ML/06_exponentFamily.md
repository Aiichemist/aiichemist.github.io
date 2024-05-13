# 指数族分布

指数族是一类分布，包括**高斯分布**、**伯努利分布**（**类别分布**）、**二项分布**（**多项分布**）、**泊松分布**、**Beta 分布**、**Dirichlet 分布**、**Gamma 分布**等一系列分布。

指数族分布可以写为统一的形式：
$$\begin{split}
p(x|\eta)&=h(x)\exp(\eta^T\phi(x)-A(\eta))\\
&=\frac{1}{\exp(A(\eta))}h(x)\exp(\eta^T\phi(x))
\end{split}$$
其中，$\eta$ 是**参数向量**，$A(\eta)$ 是**对数配分函数**（归一化因子）。

在这个式子中，$\phi(x)$ 叫做**充分统计量**，包含样本集合所有的信息，例如高斯分布中的均值和方差。充分统计量在在线学习中有应用，对于一个数据集，只需要记录样本的充分统计量即可。

对于一个模型分布假设（似然），那么我们在求解中，常常需要寻找一个**共轭先验**，使得先验与后验的形式相同，例如选取似然是二项分布，可取先验是 Beta 分布，那么后验也是 Beta 分布。指数族分布常常具有共轭的性质，于是我们在模型选择以及推断具有很大的便利。

共轭先验的性质便于计算，同时，指数族分布满足最大熵的思想（无信息先验），也就是说对于经验分布利用最大熵原理导出的分布就是指数族分布。

观察到指数族分布的表达式类似线性模型，事实上，指数族分布很自然地导出广义线性模型：
$$y=f(w^Tx)\quad y|x\sim \text{Exp Family}$$
在更复杂的概率图模型中，例如在无向图模型中如受限玻尔兹曼机中，指数族分布也扮演着重要作用。

在推断的算法中，例如变分推断中，指数族分布也会大大简化计算。

## 一维高斯分布

一维高斯分布可以写成：
$$p(x|\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
将这个式子改写：
$$\begin{split}
&\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{1}{2\sigma^2}
(x^2-2\mu x+\mu^2)
\right)\\
=&\exp\left(
\log(2\pi\sigma^2)^{-1/2}
\right)\exp\left(
-\frac{1}{2\sigma^2}
\begin{bmatrix}
-2\mu&1
\end{bmatrix}
\begin{bmatrix}
x\\
x^2
\end{bmatrix}
-\frac{\mu^2}{2\sigma^2}\right)\\
=&\exp\left(
\underbrace{\begin{bmatrix}
\frac{\mu}{\sigma^2}&-\frac{1}{2\sigma^2}
\end{bmatrix}}_{\eta^T}
\underbrace{\begin{bmatrix}
x\\
x^2
\end{bmatrix}}_{\phi(x)}
-\underbrace{\left(\frac{\mu^2}{2\sigma^2}+\frac{1}{2}\log2\pi\sigma^2\right)}_{A(\eta)}\right)
\end{split}$$
所以：
$$\eta=\begin{bmatrix}\eta_1\\\eta_2\end{bmatrix}=\begin{bmatrix}\dfrac{\mu}{\sigma^2}\\-\dfrac{1}{2\sigma^2}\end{bmatrix},\qquad\phi(x)=\begin{bmatrix}x\\x^2\end{bmatrix}$$
于是 $A(\eta)$：
$$\begin{split}
A(\eta)
&=\frac{\mu^2}{2\sigma^2}+\frac{1}{2}\log2\pi\sigma^2\\
&=-\frac{\eta_1^2}{4\eta_2}+\frac{1}{2}\log(-\frac{\pi}{\eta_2})\end{split}$$

## 充分统计量和对数配分函数的关系
对于：
$$p(x|\eta)=\frac{1}{\exp(A(\eta))}h(x)\exp(\eta^T\phi(x))$$

其中 $\exp(A(\eta))$ 为归一化因子，对概率密度函数求积分可得：
$$\begin{align}
\exp(A(\eta))&=\int h(x)\exp(\eta^T\phi(x))dx\nonumber
\end{align}$$
两边对参数求导：
$$\begin{split}\exp(A(\eta))A'(\eta)&=\int h(x)\exp(\eta^T\phi(x))\phi(x)dx\\
\Rightarrow A'(\eta)
&=\frac{1}{\exp(A(\eta))}\int h(x)\exp(\eta^T\phi(x))\phi(x)\mathrm{d}x\\
&=\int \underbrace{h(x)\exp(\eta^T\phi(x)-A(\eta))}_{P(x|\eta)}\phi(x)\mathrm{d}x\\
&=\int P(x|\eta)\phi(x)\mathrm{d}x\\
&=\boldsymbol{E}_{p(x|\eta)}[\phi(x)]\end{split}$$
即 ==$A'(\eta)$ 是 $P(x|\eta)$ 关于 $\phi(x)$ 的期望==

类似的：
$$
A''(\eta)=Var_{p(x|\eta)}[\phi(x)]
$$
由于方差为正，于是 $A(\eta)$ 一定是凸函数。

## 充分统计量和极大似然估计

对于独立全同采样得到的数据集 $\mathcal{D}=\{x_1,x_2,\cdots,x_N\}$。
$$\begin{align}\eta_{MLE}
&=\mathop{argmax}_\eta P(D|\eta)\\
&=\mathop{argmax}_\eta \log\prod\limits_{i=1}^Np(x_i|\eta)\\
&=\mathop{argmax}_\eta\sum\limits_{i=1}^N\log p(x_i|\eta)\nonumber\\
&=\mathop{argmax}_\eta\sum\limits_{i=1}^N\log\left[h(x_i)\exp(\eta^T\phi(x_i)-A(\eta))\right]\\
&=\mathop{argmax}_\eta\sum\limits_{i=1}^N\left[\underbrace{\log h(x_i)}_{与\eta无关}+\eta^T\phi(x_i)-A(\eta)\right]\\
&=\mathop{argmax}_\eta\sum\limits_{i=1}^N\left[\eta^T\phi(x_i)-A(\eta)\right]\nonumber\\
\frac{\partial}{\partial\eta}\sum\limits_{i=1}^N\left[\eta^T\phi(x_i)-A(\eta)\right]&=\sum\limits_{i=1}^N\frac{\partial}{\partial\eta}\left[\eta^T\phi(x_i)-A(\eta)\right]\\
&=\sum\limits_{i=1}^N\phi(x_i)-NA'(\eta)\stackrel{\mathrm{set}}{=}0\\
\Rightarrow
\textcolor{red}{A'(\eta_{MLE})}&\textcolor{red}{=\frac{1}{N}\sum\limits_{i=1}^N\phi(x_i)}
\end{align}
$$
由此可以看到，**为了估算参数，只需要知道充分统计量**就可以了。

## 最大熵

信息熵为信息量关于分布本身的期望：
$$\begin{split}
Entropy=E_{p(x)}[-\log p]
&=\int-p(x)\log(p(x))dx\\
&=\sum_{i=1}^N-p(x)\log p(x)
\end{split}$$

一般地，对于完全随机的变量（等可能），信息熵最大。
$$H(P)=-\sum_{x}p(x)\log p(x)$$
我们的假设为最大熵原则，假设数据是离散分布的，$k$ 个特征的概率分别为 $p_k$，最大熵原理可以表述为：
$$\max\{H(p)\}=\min\{\sum\limits_{k=1}^Kp_k\log p_k\}\ s.t.\ \sum\limits_{k=1}^Kp_k=1$$
即寻找一个 $p$，使得熵最大
$$\begin{split}
\hat{p}&=\mathop{argmax}H[P]\\
&=\mathop{argmin}\sum_{i=1}^kp_i\log p_i
\end{split}$$
其中 $P=\begin{bmatrix}p_1&p_2&\cdots&p_k\end{bmatrix}^T$

利用 Lagrange 乘子法：
$$L(p,\lambda)=\sum\limits_{k=1}^Kp_k\log p_k+\lambda(1-\sum\limits_{k=1}^Kp_k)$$
于是可得：
$$p_1=p_2=\cdots=p_K=\frac{1}{K}$$
因此**等可能的情况熵最大**。

一个数据集 $\mathcal{D}$，在这个数据集上的经验分布为 $\hat{p}(x)=\dfrac{Count(x)}{N}$，实际不可能满足所有的经验概率相同，于是在上面的最大熵原理中还需要加入这个**经验分布的约束**。

对任意一个**函数向量** $f(x)=\begin{bmatrix}f_1(x)\\f_2(x)\\\vdots\\f_Q(x)\end{bmatrix}$，**经验分布**的经验期望可以求出(假设为已知量 $\Delta$)：
$$\boldsymbol{E}_{\hat{p}}[f(x)]=\Delta$$
其中 $\Delta=\begin{bmatrix}\Delta_1&\Delta_2&\cdots&\Delta_Q\end{bmatrix}^T$

于是：
$$\max\{H(p)\}=\min\{\sum\limits_{k=1}^Np_k\log p_k\}\quad s.t.\begin{cases}\sum\limits_{k=1}^Np_k=1\\\\\boldsymbol{E}_p[f(x)]=\Delta\end{cases}$$
Lagrange 函数为：
$$L(p,\lambda_0,\lambda)=\sum\limits_{k=1}^Np_k\log p_k+\lambda_0(1-\sum\limits_{k=1}^Np_k)+\lambda^T(\Delta-\mathbb{E}_p[f(x)])$$
求导得到：
$$\frac{\partial}{\partial p(x)}L=\sum\limits_{k=1}^N(\log p(x)+1)-\sum\limits_{k=1}^N\lambda_0-\sum\limits_{k=1}^N\lambda^Tf(x)\\
\Longrightarrow\sum\limits_{k=1}^N\log p(x)+1-\lambda_0-\lambda^Tf(x)=0$$
由于数据集是任意的，对数据集求和也意味着求和项里面的每一项都是0：
$$\begin{split}
p(x)&=\exp(\lambda^Tf(x)+\lambda_0-1)\\
&=\exp\left(\underbrace{\lambda^T}_{\eta^T}\underbrace{f(x)}_{\phi(x)}+(\underbrace{\lambda_0-1}_{A(\eta)})\right)
\end{split}$$
这就是指数族分布。

