# 降维

我们知道，解决过拟合的问题除了**正则化**和**添加数据**之外，降维就是最好的方法。降维的思路来源于维度灾难的问题：

设有一个半径 $R$ 为1的 $D$ 维超球体，其体积为：
$$V_{球}=KR^D=K$$
在其内部有一个宽度为 $\varepsilon$ 的环形带（$0\lt\varepsilon\lt1$），其体积为：
$$V_{环形带}=V_{外}-V_{内}=K-K(1-\varepsilon)^D$$
当维度 $D$ 增长时：
$$\lim\limits_{D\to\infty}\frac{V_{环}}{V_{球}}
=\lim\limits_{D\to\infty}\frac{K-K(1-\varepsilon)^D}{K}
=\lim\limits_{D\to\infty}1-(1-\varepsilon)^D=1$$
即==当数据维度足够高，无论环形带宽度多小，数据分布均分布在 $n$ 维超球体的边缘区域==。

这就是所谓的维度灾难，在高维数据中，主要样本都分布在边缘，所以数据集更加稀疏。

>[!note]+ 降维的算法分为：
>1.  直接降维：特征选择
>
>2.  线性降维：PCA，MDS等
>
>3.  非线性降维：流形包括 Isomap，LLE 等

假设数据 $X=(x_1,x_2,\cdots,x_N)^T_{N\times p}=\begin{bmatrix}x_1^T\\x_2^T\\\vdots\\x_N^T\end{bmatrix}=\begin{bmatrix}x_{11}&x_{12}&\cdots&x_{1p}\\x_{21}&x_{22}&\cdots&x_{2p}\\\vdots&\vdots&\ddots&\vdots\\x_{N1}&x_{N2}&\cdots&x_{Np}\end{bmatrix}$ 

样本均值 $\overline{X}_{p\times 1}=\dfrac{1}{N}\sum\limits_{i=1}^Nx_i=\dfrac{1}{N}\underbrace{(x_1,x_2,\cdots,x_N)}_{X^T}\begin{pmatrix}1\\1\\\vdots\\1\end{pmatrix}_{N\times 1}=\dfrac{1}{N}X^T\cdot\boldsymbol{I}_{N\times 1}$

样本协方差矩阵：
$$\begin{align}
S&=\frac{1}{N}\sum\limits_{i=1}^N(x_i-\overline{x})(x_i-\overline{x})^T\nonumber\\
&=\dfrac{1}{N}\underbrace{(x_1-\bar{x},x_2-\bar{x},\cdots,x_N-\bar{x})}_{X^T-\overline{X}\cdot\boldsymbol{1}_{N\times1}}
\begin{pmatrix}(x_1-\bar{x})^T\\(x_2-\bar{x})^T\\\vdots\\(x_N-\bar{x})^T\end{pmatrix}\\
&=\frac{1}{N}(X^T-\frac{1}{N}X^T\boldsymbol{1}_{N\times1}\boldsymbol{1}_{N\times1}^T)(X^T-\frac{1}{N}X^T\boldsymbol{1}_{N\times1}\boldsymbol{1}_{N\times1}^T)^T\nonumber\\
&=\frac{1}{N}X^T
\underbrace{(\boldsymbol{I}_{N\times N}-\frac{1}{N}\boldsymbol{1}_{N\times1}\boldsymbol{1}_{N\times1}^T)}_{H_{N\times N}}
\underbrace{(E_N-\frac{1}{N}\boldsymbol{1}_{N\times1}\boldsymbol{1}_{N\times1}^T)^T}_{H_{N\times N}^T}
X\nonumber\\
&=\frac{1}{N}X^TH_{N\times N}H_{N\times N}^TX\nonumber\\
&=\frac{1}{N}X^TH_{N\times N}H_{N\times N}X\\
&=\frac{1}{N}X^TH_{N\times N}X
\end{align}$$
其中 $H=\boldsymbol{I}_{N\times N}-\dfrac{1}{N}\boldsymbol{1}_{N\times1}\boldsymbol{1}_{N\times1}^T$ 被称为**中心矩阵**，能够将数据中心移至原点，且$H^T=H,H=H^2=H^n$

这个式子利用了中心矩阵 $H$ 的对称性，这也是一个投影矩阵。

## 线性降维-主成分分析 PCA

主成分分析中，我们的基本想法是将所有数据投影到一个子空间中，从而达到降维的目标。

一个中心：
- 对原始特征空间的重构：将一组可能线性相关的变量通过正交变换转化为线性无关的变量

两个基本点：
- 最大投影方差
- 最小重构代价
### 最大投影方差

原来的数据很有可能各个维度之间是相关的，于是我们希望找到一组 $p$ 个新的线性无关的单位基 $u_i$，降维就是取其中的 $q$ 个基。

于是对于一个样本 $x_i$，经过这个坐标变换后：
$$\hat{x}_i
=\sum\limits_{j=1}^p \underbrace{\text{proj}_{u_j}x_i}_{u_i\cdot x_i}\cdot u_j
=\sum\limits_{j=1}^p\underbrace{(u_j^Tx_i)}_{投影长度}\underbrace{u_j}_{方向向量}
=\underbrace{\sum\limits_{j=1}^q(u_j^Tx_i)u_j}_{前q个基}+\underbrace{\sum\limits_{j=q+1}^p(u_j^Tx_i)u_j}_{后p-q个基}$$
即 $\hat{x}_i$ 在每个 $u_j$ 分量上的投影长度为 $u_j^Tx_i$

对于数据集来说，我们首先将其**中心化**然后再取上面的式子的第一项，由于中心化后其均值为0，方差为投影长度平方求和，即**其系数的平方平均**
$$\begin{align}
J
&=\frac{1}{N}\sum\limits_{i=1}^N
\underbrace{\sum\limits_{j=1}^q\left[(x_i-\overline{x})^Tu_j\right]^2}_{单个点x_i的投影方差}\nonumber\\
&=\frac{1}{N}\sum\limits_{i=1}^N
\sum\limits_{j=1}^q\left[u_j^T(x_i-\overline{x})(x_i-\overline{x})^Tu_j\right]^2\nonumber\\
&=\sum\limits_{j=1}^qu_j^T\left[\frac{1}{N}\sum_{i=1}^N(x_i-\overline{x})(x_i-\overline{x})^T\right]u_j\\
&=\sum\limits_{j=1}^qu_j^T\ S\ u_j
\end{align}$$
其中 $u_j^Tu_j=1$

由于每个基都是线性无关的，于是每一个 $u_j$ 的求解可以分别进行，使用拉格朗日乘子法：
$$\hat{u}_j=\mathop{argmax}_{u_j}\ L(u_j,\lambda)=\mathop{argmax}_{u_j}\ u_j^T\cdot S\cdot u_j+\lambda(1-u_j^Tu_j)$$
求导得：
$$\frac{\partial}{\partial u_j}L(u_j,\lambda)=2S\cdot u_j-2\lambda u_j=0$$
于是：
$$S\cdot u_j=\lambda u_j$$
可见，我们需要的基就是协方差矩阵的**特征向量**。损失函数最大取在**特征值**前 $q$ 个最大值。

### 最小重构代价

下面看最小重构代价这个条件，即损失的信息最少

原始数据：
$$x_i=\sum\limits_{j=1}^p(x_i^Tu_j)u_j$$
降维后的数据重构：
$$\hat{x}_i=\sum\limits_{j=1}^q(x_i^Tu_j)u_j$$
用 $||x_i-\hat{x}_i||^2$ 表示重构代价，中心化后其损失函数为：
$$\begin{align}
J
&=\frac{1}{N}\sum_{i=1}^N||x_i'-\hat{x}'_i||^2\\
&=\frac{1}{N}\sum_{i=1}^N\left\Vert\sum_{j=q+1}^p\left[(x_i-\bar{x})^Tu_j\right]u_j\right\Vert^2\\
&=\frac{1}{N}\sum_{i=1}^N\sum_{j=q+1}^p\left[(x_i-\bar{x})^Tu_j\right]^2\\
&=\sum\limits_{j=q+1}^p\underbrace{\frac{1}{N}\sum\limits_{i=1}^N\left[(x_i-\overline{x})^Tu_j\right]^2}_{u_j\ S\ u_j}\nonumber\\
&=\sum\limits_{j=q+1}^pu_j^T\ S\ u_j
\end{align}$$
其中 $u_j^Tu_j=1$

同样的，由于每个基都是线性无关的，于是每一个 $u_j$ 的求解可以分别进行，使用拉格朗日乘子法：
$$\hat{u}_j=\mathop{argmin}_{u_j}\ L(u_j,\lambda)=\mathop{argmin}_{u_j}\ u_j^TSu_j+\lambda(1-u_j^Tu_j)
$$
损失函数最小取在特征值剩下的个最小的几个值。数据集的协方差矩阵可以写成 $S=U\Lambda U^T$，直接对这个表达式分解可以得到特征值。

### 奇异值分解SVD 与 主坐标分析PCoA

下面使用实际训练时常常使用的 SVD 直接求得这个 $q$ 个特征值。

对中心化后的数据集 $HX$ 进行奇异值分解（SVD）：
$$HX=U\Sigma V^T$$
其中 $\Sigma$ 为 $N\times p$ 的对角矩阵，$U^TU=\boldsymbol{I}_N,V^TV=\boldsymbol{I}_p$ 

于是：
$$
S=\frac{1}{N}X^THX=\frac{1}{N}X^TH^THX=\frac{1}{N}V\Sigma U^TU\Sigma V^T=\frac{1}{N}V\Sigma^2V^T
$$
因此，我们直接对**中心化后的数据集**进行**奇异值分解**，就相当于对**样本方差矩阵**进行**特征值分解**，就可以得到特征值和特征向量 $V$ 即主成分，然后做投影，在新坐标系中的坐标就是：
$$HX\cdot V=U\Sigma V^TV=\underbrace{U\Sigma}_{坐标矩阵}$$
由上面的推导，我们也可以得到另一种方法 PCoA **主坐标分析**，定义一个矩阵 $T$ 并进行特征值分解：
$$T=HXX^TH=U\Sigma^2U^T$$
即 $T$ 与 $S$ 具有相同的特征值

由于：
$$T\underbrace{U\Sigma}_{坐标矩阵}
=U\Sigma^2U^TU\Sigma=U\Sigma^3
=\underbrace{U\Sigma}_{坐标矩阵}\underbrace{\Sigma^2}_{特征值矩阵}$$
于是可以直接得到坐标。这两种方法都可以得到主成分，但是由于方差矩阵是 $p\times p$ 的，而 $T$ 是 $N\times N$ 的，所以对样本量较少的时候可以采用 PCoA的方法。

### p-PCA

下面从概率的角度对 PCA 进行分析，概率方法也叫 p-PCA。

我们使用**线性高斯模型**，类似之前 LDA，我们选定一个方向，对原数据 $x\in\mathbb{R}^p$ ，降维后的数据为 $z\in\mathbb{R}^q,q<p$。

假设 $z$ 服从高斯分布先验：
$$z\sim N(\boldsymbol{O}_{q\times1},\boldsymbol{I}_{q\times q})$$
降维通过一个矩阵变换（投影）进行：
$$x=Wz+\mu+\varepsilon$$
其中 $\varepsilon\sim N(0,\sigma^2\boldsymbol{I}_{p\times p})$

p-PCA有两个问题：
1. 推断inference：$p(z|x)$

2. 学习learning：$W$、$\mu$、$\sigma^2$，使用EM算法

在进行推断的时候需要求得 $p(z|x)$，推断的求解过程和线性高斯模型类似。
$$\begin{split}&\begin{cases}
E[x|z]=E[Wz+\mu+\varepsilon]=Wz+\mu\\
Var[x|z]=Var[Wz+\mu+\varepsilon]=\sigma^2\boldsymbol{I}\\
\end{cases}\\
\Rightarrow&x|z\sim N(Wz+\mu,\sigma^2\boldsymbol{I})
\end{split}$$
$$\begin{split}
&\begin{cases}
E[x]=E[Wz+\mu+\varepsilon]=E[Wz+\mu]+E[\varepsilon]=\mu\\
Var[x]VVar[Wz]+Var[\varepsilon]=WW^T+\sigma^2\boldsymbol{I}_{p\times p}
\end{cases}\\
\Rightarrow& x\sim N(\mu,WW^T+\sigma^2)\\
\end{split}$$
$x$ 与 $z$ 的协方差矩阵：
$$\begin{split}
Cov(x,z)&=E[(x-\mu)(z-0)^T]\\
&=E[(Wz+\varepsilon)z^T]\\
&=E[Wzz^T+\varepsilon z^T]\\
&=E[Wzz^T]+\underbrace{E[\varepsilon]}_{=0}E[z^T]\\
&=W\underbrace{E[zz^T]}_{Var[z]}\\
&=WI=W
\end{split}$$
$x$ 与 $z$ 的联合分布：
$$\begin{bmatrix}
x\\z
\end{bmatrix}
\sim N\left(
\begin{bmatrix}
\mu\\0
\end{bmatrix},
\begin{bmatrix}
WW^T+\sigma^2\boldsymbol{I}&W\\
W^T&\boldsymbol{I}\\
\end{bmatrix}\right)$$

由多维高斯分布中 $p(x_b|x_a)$ 的求解：
>[!tip]- $p(x_b|x_a)$ 的求解
>![[01_数学入门#3. $p(x_b x_a)$]]

可得$x$ 与 $z$ 的条件分布：
$$z|x\sim N(\underline{W^T(WW^T+\sigma^2\boldsymbol{I})^{-1}(x-\mu)},\underline{\boldsymbol{I}-W^T(WW^T+\sigma^2\boldsymbol{I})^{-1}W})$$

## 小结

降维是解决维度灾难和过拟合的重要方法，除了直接的特征选择外，我们还可以采用算法的途径对特征进行筛选，线性的降维方法以 PCA 为代表，在 PCA 中，我们只要直接对数据矩阵进行中心化然后求奇异值分解或者对数据的协方差矩阵进行分解就可以得到其主要维度。非线性学习的方法如流形学习将投影面从平面改为超曲面。

