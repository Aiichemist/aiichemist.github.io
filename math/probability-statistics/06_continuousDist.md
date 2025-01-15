# 连续分布
## 连续均匀分布：离散均匀分布的连续版
### 概率密度函数
如图 1 所示，连续随机变量 $X$ 在区间 $[a, b]$ 内取得任意一个实数的概率密度函数满足：
$$f_X(x)=\begin{cases}
\dfrac{1}{b-a}&\text{for }a\le x\le b\\
0&\text{for }x\lt a\text{ or }x\gt b
\end{cases}$$
则称 $X$ 区间 $[a, b]$ 上服从**连续均匀分布** (continuous uniform distribution)。这个连续分布常记做 $Uniform(a, b)$ 或 $U(a, b)$，比如 $[0, 1]$ 区间上的均匀分布可以记做 $Uniform(0, 1)$ 或 $U(0, 1)$。
![](/img/Pasted%20image%2020230908120348.png ':size=70%')

### 期望、方差
服从上式连续均匀分布 $X$ 的期望和方差分别为：
$$E(X)=\dfrac{a+b}{2},\quad var(X)=\dfrac{(b-a)^2}{12}$$

### 随机数
利用随机数发生器，我们可以获得满足连续均匀分布的随机数。图 2 (a) 所示为满足连续均匀 分布随机数的直方图。 

图 2 (b) 所 示 为 随 机 数 的 **经 验 累 积 分 布 函 数** (Empirical Cumulative Distribution Function, ECDF)。不难看出 ECDF 的取值范围为 $[0, 1]$。经验分布函数是在所有 $n$ 个样本点上都跳跃 $1/n$ 的 阶跃函数。对于某个特定样本，它的 ECDF 为样本中小于或等于该值的样本所占的比例。
![](/img/Pasted%20image%2020230908120656.png ':size=70%')

## 高斯分布：最重要的概率分布，没有之一
**高斯分布** (Gaussian distribution)，也叫**正态分布** (normal distribution)，仿佛是整个纷繁复杂宇宙表象下的终极秩序。实际上，高斯分布是由德国数学家和天文学家亚伯拉罕·棣莫弗 (Abraham de Moivre) 于 1733 年首先提出。

### 一元高斯分布
**一元高斯分布** (univariate normal distribution) 的概率密度函数为：
$$f_X(x)=\dfrac{1}{\sigma\sqrt{2\pi}}\exp\left(\dfrac{-1}{2}\left(\dfrac{x-\mu}{\sigma}\right)^2\right)$$

其中，$μ$ 为均值/期望值，$σ$ 为标准差。满足上式的高斯分布常记做 $N(μ, σ^2)$。

也就是说，连续随机变量 $X$ 服从 $N(μ, σ^2)$，即 $X ~ N(μ, σ^2)$，则 $X$ 的期望和方差为：
$$E(X)=\mu,\quad var(X)=\sigma^2$$
图3所示为三个不同一元高斯分布 PDF、CDF 图像。可以发现，一元高斯分布 PDF 关于 $x = μ$ 对称，当 $x$ 远离 $μ$，概率密度函数高度迅速下降。
![](/img/Pasted%20image%2020230908171448.png ':size=70%')

### 形状
$μ$ 和 $σ$ 两个参数确定了一元高斯分布 PDF 的位置和形状。如图 4 所示，$μ$ 决定概率密度曲线 $p(x)$ 的位置，$σ$ 影响曲线的胖瘦。特别地，当 $μ = 0$，且 $σ = 1$ 时，得到的高斯分布为标准正态分布 (standard normal distribution)。
![](/img/Pasted%20image%2020230908171529.png ':size=70%')

### 二元高斯分布
**二元高斯分布** (bivariate Gaussian distribution)，也叫二元正态分布，它的概率密度函数解析式 如下：
$$f_{X_1,X_2}(x_1,x_2)=\dfrac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho_{1,2}^2}}\times\exp\left(\dfrac{-1}{2}\left(\overbrace{\dfrac{1}{(1-\rho_{1,2}^2)}\left(\left(\dfrac{x_1-\mu_1}{\sigma_1}\right)^2-2\rho_{1,2}\left(\dfrac{x_1-\mu_1}{\sigma_1}\right)\left(\dfrac{x_2-\mu_2}{\sigma_2}\right)+\left(\dfrac{x_2-\mu_2}{\sigma_2}\right)^2\right)}^{Ellipse}\right)\right)$$
其中，$μ_1$ 和 $μ_2$ 分别为 $X_1$ 和 $X_2$ 的期望值，$σ_1$ 和 $σ_2$ 为 $X_1$ 和 $X_2$ 的标准差，$ρ_{1,2}$ 为两者的线性相关系数。

注意，上式中 $ρ_{1,2}$ 取值范围为 $(−1, 1)$。

连续随机变量 $(X_1, X_2)$ 服从上述二元正态分布，记做：
$$\begin{bmatrix}
X_1\\
X_2\\
\end{bmatrix}\thicksim
N\left(\mathop{\begin{bmatrix}
\mu_1\\
\mu_2\\
\end{bmatrix}}_{\vec{\mu}},
\underbrace{\begin{bmatrix}
\sigma_1^2 & \rho_{1,2}\sigma_1\sigma_2\\
\rho_{1,2}\sigma_1\sigma_2 & \sigma_2^2\\
\end{bmatrix}}_{\Sigma}
\right)
=N(\vec{\mu},\Sigma)$$
图 5 所示为方差和相关性系数取不同值时，二元正态分布概率密度函数椭圆等高线以及边缘 分布形状。注意，图中 $σ_{1,1}$ 和 $σ_{2,2}$ 代表方差，即标准差的平方。
![](/img/Pasted%20image%2020230908172553.png ':size=70%')
![](/img/Pasted%20image%2020230908172613.png ':size=70%')

### 多元高斯分布
多元高斯分布 PDF 每个不同成分的含义：
$$\begin{split}
d=\sqrt{(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})}&:\text{马氏距离}\\
\|\vec{z}\|&:\text{z分数}\\
\vec{z}=\Lambda^{\frac{-1}{2}}V^T(\vec{x}-\vec{\mu})&:\text{平移-旋转-缩放}\\
\left[\Lambda^{\frac{-1}{2}}V^T(\vec{x}-\vec{\mu})\right]^T\Lambda^{\frac{-1}{2}}V^T(\vec{x}-\vec{\mu})&:\text{特征分解}\\
(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})&:\text{椭圆}\\
\\
f_\chi(\vec{x})
=\frac
{\overbrace{\exp(-\frac{1}{2}}^{\text{距离度量—>亲近度}}\overbrace{(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})}^{\text{椭圆}})}
{\underbrace{(2\pi)^{\frac{D}{2}}}_{\begin{split}&\text{归一化}\\&\text{多变量微积分}\end{split}}\quad\underbrace{|\Sigma|^{\frac{1}{2}}}_{\begin{split}&\text{缩放}\\&\text{特征值}\end{split}}}
\end{split}$$
### 拉普拉斯分布
拉普拉斯分布的概率密度函数为：
$$f_X(x)=\dfrac{1}{2b}\exp\left(-\dfrac{|x-\mu|}{b}\right)$$
形式上，拉普拉斯分布和高斯分布很类似，只不过拉普拉斯分布的 PDF 图像在对称轴处存在尖点。很容易发现，参数 $μ$ 决定概率密度分布位置。如图 6 所示，参数 $b$ 决定分布形状。
![](/img/Pasted%20image%2020230908172958.png ':size=70%')
如果连续随机变量 $X$ 满足拉普拉斯分布，$X$ 期望和方差为：
$$E(X)=\mu,\quad var(X)=2b^2$$

## 逻辑分布：类似高斯分布
**一元逻辑分布** (univariate logistic distribution) 的 PDF 为：
$$f_X(x)=\dfrac{\exp\left(\dfrac{-(x-\mu)}{s}\right)}{s\left(1+\exp\left(\dfrac{-(x-\mu)}{s}\right)\right)^2}$$
其中，$μ$ 为位置参数，$s$ 为形状参数。

相比 PDF，逻辑函数的 CDF 更常用：
$$F_X(x)=\dfrac{1}{1+\exp\left(\dfrac{-(x-\mu)}{s}\right)}$$
![](/img/Pasted%20image%2020230908173651.png ':size=70%')

### 逻辑分布 vs 高斯分布
逻辑分布和高斯分布 PDF、CDF 长得很相似。为了比较逻辑函数和高斯函数，用标准正态分布 $N(0, 1)$ 的 PDF 和 CDF 图像，而逻辑分布的位置参数 $μ = 0$。特别选取参数 $s$ 使得逻辑分布 PDF 和标准正态分布 PDF 在 $x = 0$ 处高度一致。 

如图 8 所示，相比标准正态分布，逻辑分布 PDF 中心部位“稍瘦”，而**厚尾** (fat tail)。厚尾，也叫肥尾，指的是和正态分布相比，尾部分布较厚的分布。下一节介绍的学生 t-分布就是典型的厚尾分布。
![](/img/Pasted%20image%2020230908173853.png ':size=70%')

## 学生t-分布：厚尾分布
**学生 t-分布** (Student's t-distribution) 也称**学生分布**，或 **t 分布**，是由戈赛特 (William Sealy Gosset) 于 1908 年提出的，Student 一词源自于他发表论文时用的化名。

学生 t-分布是常用的一类厚尾分布。学生 t-分布多应用于根据小样本数据来估计呈正态分布且方差未知的总体的均值，本书第 17 章将简要介绍相关内容。

一元学生 t-分布的 PDF 为：
$$f_X(x)=\frac{\Gamma\left(\frac{v+1}{2}\right)}{\sqrt{v \pi} \cdot \Gamma\left(\frac{v}{2}\right)}\left(1+\frac{x^2}{v}\right)^{\frac{-(v+1)}{2}}$$

其中，$v$ 为自由度 (number of degrees of freedom 或 df)，$v = n – 1$，$n$ 为样本数；$\Gamma$ 是 Gamma 函数 (Gamma function)。

### Gamma函数
Gamma 函数是从阶乘的概念推广而来的，它将阶乘的概念推广到了实数和复数的范围。

$v$ 为正整数时，Gamma 方程类似于阶乘表达式，正整数 $ν$ 的 Gamma 函数表达式为：
$$\Gamma(v)=(v-1)!$$
$v$ 取特殊分数，比如 $1/2$ 和 $3/2$ 时，$v$ 的 Gamma 函数的值：
$$\begin{split}
\Gamma\left(\frac{1}{2}\right)&=\sqrt{\pi}\\
\Gamma\left(\frac{3}{2}\right)&=\dfrac{1}{2}\sqrt{\pi}
\end{split}$$
图 9 所示为 Gamma 函数图像，其中红色 $×$ 是取正整数时 Gamma 函数的取值。
![](/img/Pasted%20image%2020230908183656.png ':size=70%')
![](/img/Pasted%20image%2020230908183702.png ':size=70%')

一般情况，当 $v$ 为偶数时，上式中系数部分为：
$$\dfrac{\Gamma\left(\dfrac{v+1}{2}\right)}{\sqrt{v\pi}\Gamma\left(\dfrac{v}{2}\right)}=\dfrac{(v-1)(v-3)\cdots 5\cdot 3}{2\sqrt{v}(v-2)(v-4)\cdots 4\cdot 2}$$
当 $v$ 为奇数时：
$$\dfrac{\Gamma\left(\dfrac{v+1}{2}\right)}{\sqrt{v\pi}\Gamma\left(\dfrac{v}{2}\right)}=\dfrac{(v-1)(v-3)\cdots 4\cdot 2}{\pi\sqrt{v}(v-2)(v-4)\cdots 5\cdot 3}$$
Gamma 函数存在如下递推关系：
$$\Gamma(v+1)=\Gamma(v)\cdot v$$
上式和 $ν$ 取值无关。Gamma 函数在概率分布中具有重要的作用，尤其是在 Gamma 分布、卡 方分布、t 分布、Beta 分布、Dirichlet 分布等定义和性质中都涉及到 Gamma 函数。

### 自由度
图 10 所示为 $v$ 从 1 变化到 30 时，学生 t-分布 PDF 和 CDF 图像。图 10 中黑色的曲线对应正态分布。当自由度 $v$ 不断提高时，厚尾现象逐渐消失，学生 t-分布逐渐接近标准正态分布 (黑色)。 很明显，学生 t-分布的偏度为 $0$。
![](/img/Pasted%20image%2020230908184057.png ':size=70%')

### 多元学生 t-分布
类似多元高斯分布，多元学生 t-分布的概率密度函数为：
$$f_X(\vec{x})
=\dfrac{\Gamma[(v+D)/2]}
{\Gamma\left(\dfrac{v}{2}\right)v^{\frac{D}{2}}\pi^{\frac{D}{2}}|\Sigma_t|^{\frac{1}{2}}}
\left[
1+\dfrac{1}{v}\underbrace{
(\vec{x}-\vec{\mu})^T\Sigma_t^{-1}(\vec{x}-\vec{\mu})
}_{Ellipse}
\right]^{\frac{-(v+D)}{2}}$$
其中，$v$ 为自由度，$D$ 为维数。

上式中$Σ_t$ 和多元高斯分布的协方差矩阵关系为：
$$\Sigma_t=\frac{v}{v-2}\Sigma$$

## 对数正态分布：源自正态分布
### 定义
如果随机变量 $X$ 的对数 $\ln X$ 服从正态分布，则 $X$ 服从对数正态分布 (logarithmic normal distribution)。

对于 $x > 0$，对数正态分布的 PDF 为：
$$f_X(x)=\frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right)$$
其中，$μ$ 是 $X$ 对数的平均值，$σ$ 是 $X$ 对数的标准差。

如果 $X$ 满足上式的对数正态分布，则 $X$ 期望和方差为：
$$E(X)=\exp\left(\mu+\frac{\sigma^2}{2}\right),\quad var(X)=\left[\exp(\sigma^2)-1\right]\exp(2\mu+\sigma^2)$$
### 图像
图 11 给出对数正态分布的图像。对数正态分布的最大特点是右偏，即正偏。对于右偏的对数 正态分布，其平均值大于其众数。

再次强调，对数正态分布的随机变量取值只能为正值。
![](/img/Pasted%20image%2020230908221839.png ':size=70%')
图 12 对比正态分布和对数正态分布。
![](/img/Pasted%20image%2020230908221852.png ':size=70%')

## 指数分布：泊松分布的连续随机变量版
### 定义
与泊松分布相比，**指数分布**重要特点是随机变量连续。而泊松分布是针对随机事件发生次数定义的，发生次数是离散的。

指数分布的概率密度函数为：
$$f_X(x)=\begin{cases}
\lambda\exp(-\lambda x)\quad &x\ge 0\\
0 & x\lt 0
\end{cases}$$
指数分布的期望和方差分别为：
$$E(X)=\frac{1}{\lambda},\quad var(X)=\frac{1}{\lambda^2}$$
### 图像
 图 13 所示为 $λ$ 取不同值时，指数分布 PDF 和 CDF 图像。
 
![](/img/Pasted%20image%2020230909161131.png ':size=70%')

## 卡方分布：若干IID标准正态分布平方和
### 定义
**卡方分布** (chi-square distribution 或 $\chi^2$-distribution) 先是德国统计学家赫尔默特 (Friedrich Robert Helmert) 在 1875 年提出。

若 $n$ 个相互独立的随机变量 $Z_1$、$Z_2$、……、$Z_k$ 均服从标准正态分布，即：
$$Z_i\thicksim N(0,1),\quad \forall i=1,\ldots,k$$
这 $n$ 个随机变量的平方和构成一个新的随机变量 $X$，$X$ 服从自由度为 $k$ 的卡方分布：
$$X=\sum_{i=1}^kZ_i^2\thicksim\chi_k^2$$
其中 ，$k$ 称为自由度。自由度为 $k$ 的卡方分布一般标记为$\chi_k^2$ 。

如果随机变量 $X$ 满足上式的卡方分布，$X$ 的期望值和方差为：
$$E(X)=k,\quad var(X)=2k$$
### 图像
如图 14 所示，卡方分布的值均为正值，且呈现右偏态，随着自由度 $n$ 的增大，卡方分布趋近于正态分布。当自由度大于 30 时，已经非常类似于正态分布。
![](/img/Pasted%20image%2020230909161620.png ':size=70%')


## F-分布：和两个服从卡方分布的独立随机变量有关
### 定义
F-分布是两个服从卡方分布的独立随机变量各除以其自由度后的比值的抽样分布。

如果随机变量 $X$ 满足参数为 $d_1$ 和 $d_2$ 的 F-分布，记做 $X\thicksim F(d_1, d_2)$。随机变量 $X$ 为：
$$X=\dfrac{S_1/d_1}{S_2/d_2}$$
其中，随机变量 $S_1$ 和 $S_2$ 分别服从自由度为 $d_1$、$d_2$ 的卡方分布。

如果 $X\thicksim F(d_1, d_2)$，$X$ 的 PDF 为：
$$f_X(x;d_1,d_2)=\dfrac{1}{B\left(\dfrac{d_1}{2},\dfrac{d_2}{2}\right)}\left(\dfrac{d_1}{d_2}\right)^{\frac{d_1}{2}}x^{\frac{d_1}{2}-1}\left(1+\frac{d_1}{d_2}x\right)^{\frac{-(d_1+d_2)}{2}}$$
其中，$B()$ 叫做 Beta 函数。$B(α, β)$ 函数和 Gamma 函数的关系为：
$$B(\alpha,\beta)=\int_0^1x^{\alpha-1}(1-x)^{\beta-1}dx=\dfrac{\Gamma(\alpha)\cdot\Gamma(\beta)}{\Gamma(\alpha+\beta)}$$
### 图像
图 15 所示为 $B(α, β)$ 函数取值随 $α$ 和 $β$ 变化的火柴梗图、三维散点图。下一节的 Beta 分布中 也会用到 $B(α, β)$ 函数。
![](/img/Pasted%20image%2020230909162500.png ':size=70%')

如图 16 所示，F-分布是一种非对称分布，且 $d_1$、$d_2$ 的位置不可随意互换。
![](/img/Pasted%20image%2020230909162525.png ':size=70%')

## Beta分布：概率的概率
**贝叶斯推断** (Bayesian inference) 是数据科学和机器学习重要的数学工具，而 Beta 分布在贝叶斯推断中扮演重要角色。
(对应离散分布中的二项分布)
### 定义
Beta 分布定义在 $(0, 1)$ 或 $[0, 1]$ 区间的连续概率分布，它有两个参数 $α$、$β$。$Beta(α, β)$ 分布的概率密度函数为：
$$f_X(x;\alpha,\beta)=\dfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}$$
其中，$x^{\alpha-1}(1-x)^{\beta-1}$ 决定 PDF 曲线的形状。

注意到，这个 PDF 概率密度曲线有两个区间，原因是 $α、β$ 当取不同值时 $x$ 的取值范围不同。举个例子，当 $α、β$ 均为 $0.1$ 时，Beta 分布的定义域为 $(0, 1)$。

可以在上述解析式中看到 $B(α, β)$ 函数。利用 $B(α, β)$，上式可以写成：
$$f_X(x;\alpha,\beta)=\dfrac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$$

而 $B(α, β)$ 是让$x^{\alpha-1}(1-x)^{\beta-1}$ 成为概率密度函数的归一化因子。白话说，$B(α, β)$ 让 PDF 曲线和 横轴围成的图形面积为 1。

如果 $α、β$ 都是大于 1 的正整数，$B(α, β)$ 可以展开写成：
$$B(\alpha,\beta)=\dfrac{(\alpha-1)!(\beta-1)!}{(\alpha+\beta-1)!}$$
### 图像
图 17 所示为参数 $α、β$ 取不同值时 Beta 分布 PDF 图像。

容易发现 $Beta(α, β)$ 分布实际上代表了一系列分布。举个例子，连续均匀分布 $U(0, 1)$ 便是 $Beta(1, 1)$。

注意图 17 对角线上的图像，即 $α = β$，这些 PDF 图像对称，对应的分布相当于 $Beta(α, α)$。
![](/img/Pasted%20image%2020230909163245.png ':size=70%')

### 众数 vs 期望
如果 $X$ 服从 $Beta(α, β)$ 分布，$X$ 的期望为：
$$E(X)=\dfrac{\alpha}{\alpha+\beta}$$
常常用到的是 $Beta(α, β)$ 分布的众数：
$$\dfrac{\alpha-1}{\alpha+\beta-2},\quad\alpha,\beta\gt 1$$
众数是概率密度函数曲线最大值所在位置。这一点在贝叶斯推断格外重要。

### 推导期望
推导 $Beta(α, β)$ 的期望其实很容易，甚至不需要积分。

连续随机变量 $X$ 的期望为：
$$E(X)=\int\limits_xx\cdot f_X(x)dx$$
将 $Beta(α, β)$ 的概率密度函数代入上式，得到：
$$\begin{split}
E(X)
&=\int\limits_x x\cdot\dfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}dx\\
&=\dfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\underbrace{\int\limits_xx^{\alpha}(1-x)^{\beta-1}dx}_{Beta(\alpha+1,\beta)}

\end{split}$$
容易看出来，上式中积分部分可以整理成为 $Beta(α + 1, β)$ 分布的 PDF 解析式。缺的就是归一化系数。

补充这个归一化系数，上式可以写成：
$$\begin{split}
E(X)
&=\dfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\dfrac{\Gamma(\alpha+1)\Gamma(\beta)}{\Gamma(\alpha+\beta+1)}
\underbrace{\int\limits_x\dfrac{\Gamma(\alpha+\beta+1)}{\Gamma(\alpha+1)\Gamma(\beta)}x^{\alpha}(1-x)^{\beta-1}dx}_{=1}\\
&=\dfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\dfrac{\Gamma(\alpha+1)\Gamma(\beta)}{\Gamma(\alpha+\beta+1)}

\end{split}$$
根据 Gamma 函数的递推关系 $\Gamma(v+1)=\Gamma(v)\cdot v$ ，上式进一步整理为：
$$\begin{split}
E(X)&=\dfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\dfrac{\Gamma(\alpha)\cdot\alpha\cdot\Gamma(\beta)}{\Gamma(\alpha+\beta)\cdot(\alpha+\beta)}\\
&=\dfrac{\alpha}{\alpha+\beta}
\end{split}$$
### 方差、标准差
$Beta(α, β)$ 的方差为：
$$var(X)=\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$
$Beta(α, β)$ 的标准差为方差的平方根：
$$std(X)=\sqrt{\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}}$$
为了方便和下文的 Dirichlet 分布对照，令
$$\alpha_0=\alpha+\beta$$
$Beta(α, β)$ 的可以进一步写成：
$$\begin{split}
var(X)
&=\dfrac{\alpha(\alpha_0-\alpha)}{\alpha_0^2(\alpha_0+1)}\\
&=\dfrac{\dfrac{\alpha}{\alpha_0}\left(1-\dfrac{\alpha}{\alpha_0}\right)}{\alpha_0+1}
\end{split}$$

## Dirichlet 分布：多元 Beta 分布
Dirichlet 分布也叫狄利克雷分布，它本质上是**多元 Beta 分布** (multivariate Beta distribution)。 Dirichlet 分布常作为贝叶斯统计的先验概率。
(对应离散分布中的多项分布)

Dirichlet 分布概率密度函数为：
$$f_{X_1,\ldots,X_K}(x_1,\ldots,x_K;\alpha_1,\ldots,\alpha_K)=\dfrac{1}{B(\alpha_1,\ldots,\alpha_K)}\prod_{i=1}^Kx_i^{\alpha_i-1},\quad\sum_{i=1}^Kx_i=1$$
注意，$x_i (i = 1, 2, …, K)$ 的取值范围为 $[0, 1]$，而且它们的和为 $1$。这个分布常记做 $Dir(α)$ 或 $Dir(α_1, α_2, …, α_K)$。后文在贝叶斯推断中，会用 $θ$ 代替 $x$。

$K$ 元 $B()$ 函数的定义为：
$$B(\alpha_1,\ldots,\alpha_K)=\dfrac{\prod\limits_{i=1}^K\Gamma(\alpha_i)}{\Gamma(\sum\limits_{i=1}^K\alpha_i)}$$
### 举个例子
当 $K = 3$ 时，$x_1、x_2、x_3$ 满足：
$$x_1+x_2+x_3=1$$
并且，$x_1、x_2、x_3$ 都在区间 $[0, 1]$ 内。显然，$x_1、x_2、x_3$ 在一个平面上。

白话说，$x_1+x_2+x_3=1$ 好比三维空间撑起的一张“画布”，概率密度等高线则必须画在这张画布上。

后文将采用五种可视化方案展示 Dirichlet 分布概率密度函数。如图 18 所示，这五种可视 化方案主要分成两大类。由于上式关系，给定 $x_1、x_2$，则 $x_3$ 确定。因此，我们可以用图 18 (a) 的 $x_1x_2$ 平面展示 Dirichlet 分布 PDF 图像。

![](/img/Pasted%20image%2020230909165630.png ':size=70%')

Dirichlet 分布非常重要，因此下文用图 19~图 23 五种可视化方案展示 Dirichlet 分布的分布特征。
![](/img/Pasted%20image%2020230909165722.png ':size=70%')
![](/img/Pasted%20image%2020230909165735.png ':size=70%')

![](/img/Pasted%20image%2020230909165750.png ':size=70%')
![](/img/Pasted%20image%2020230909165802.png ':size=70%')

![](/img/Pasted%20image%2020230909165812.png ':size=70%')

### 边缘分布
Dirichlet 分布的边缘分布服从 Beta 分布：
$$X_i\thicksim Beta(\alpha_i,\alpha_0-\alpha_i)$$
其中，
$$\alpha_0=\sum_{i=1}^K\alpha_i$$
以图 19 中 (d) 组为例，三个 Dirichlet 分布的边缘分布 PDF 如图 24 所示。

$X_i$ 的期望为：
$$E(X_i)=\dfrac{\alpha_i}{\sum\limits_{k=1}^K\alpha_k}=\dfrac{\alpha_i}{\alpha_0}$$
$X_i$ 的众数为：
$$\frac{\alpha_i-1}{\sum\limits_{k=1}^K\alpha_k-K}
=\frac{\alpha_i-1}{\alpha_0-K},\quad \alpha_i\gt 1$$
![](/img/Pasted%20image%2020230909170305.png ':size=70%')





$$\begin{aligned}
& w_1^*=\frac{\overline{x y}-\overline{x y}}{\overline{x^2}-(\bar{x})^2} ; w_0^*=\bar{y}-w_1^* \bar{x} \\
& \hat{w}^*=\left(X^{\mathrm{T}} X\right)^{-1} X^{\mathrm{T}} y
\end{aligned}$$
