# 第3章标准分布

在本章中，我们将通过具体示例来讨论一些概率分布和它们的性质。这些分布除了常规的用途之外，还是构建更复杂模型的基石，因此本书将广泛使用它们。

本章所讨论的分布有一个作用，即在给定随机变量  $x$  的有限个观测值  $x_{1},\dots ,x_{N}$  的情况下，对其分布  $p(\pmb {x})$  进行建模。这就是密度估计（density estimation）问题。需要强调的是，密度估计问题基本上是不适定的，因为可能存在无数个概率分布都能产生这样一组观测值。事实上，任意在  $x_{1},\dots ,x_{N}$  处非零的分布  $p(x)$  都有可能是真实的概率分布。选择合适分布的问题与模型选择有关，而该问题我们已经在多项式曲线拟合这一机器学习核心问题中遇到过（参见1.2节）。

在对连续变量的高斯分布进行考察之前，首先考虑离散变量的分布。这些分布是参数分布的典型例子，之所以如此命名，是因为这些分布仅由很少的可调参数决定，如高斯分布中的均值和方差等。为了将这些模型应用于密度估计问题，我们需要一种在给定观测数据集的情况下确定合适参数值的方法（主要聚焦于最大化似然函数）。

在本章中，我们假设观测数据是独立同分布的（independent and identically distributed, i.i.d.）。在后续章节中，我们还将探索包含结构化数据的更复杂情形，到那时，这一假设将不再成立。

参数化方法的一个局限在于，它将分布假设为一种特定的函数形式，而这一做法可能在特定的应用中并不合适。另一种方法是非参数（nonparametric）密度估计方法，此时，分布的形式通常取决于数据集的大小。这类模型仍然包含一些参数，但这些参数将控制模型的复杂度而不是分布的形式。在本章的最后，我们将简述3种典型的非参数化方法，这3种方法分别基于直方图、最近邻和核函数。诸如此类的非参数化技术的一个主要局限，在于它们涉及存储所有训练数据。换言之，参数的数量将随数据量的增大而增加。因此，这些方法在面对大型数据集时将变得非常低效。神经网络拥有大量但数量固定的参数，基于神经网络的灵活分布，深度学习同时具备参数模型的高效和非参数模型的泛化性。

## 3.1 离散变量

首先考虑离散变量的简单分布，以二元变量为起点，随后扩展到多状态变量。

### 3.1.1 伯努利分布

考虑一个二元随机变量  $x \in \{0,1\}$  。例如， $x$  可能描述的是抛掷硬币的结果，其中  $x = 1$  表示“正面”、 $x = 0$  表示“反面”。如图2.2所示，如果这枚硬币有所损坏，则落地时正面朝上的概率和反面朝上的概率未必相等。 $x = 1$  的概率可以用参数  $\mu$  来表示：

$$
p (x = 1 \mid \mu) = \mu \tag {3.1}
$$

其中  $0 \leqslant \mu \leqslant 1$  ，于是可以得到  $p(x = 0|\mu) = 1 - \mu$  。 $x$  的概率分布因此可以写成如下形式：

$$
\operatorname {B e r n} (x \mid \mu) = \mu^ {x} (1 - \mu) ^ {1 - x} \tag {3.2}
$$

这就是伯努利（Bernoulli）分布（见习题3.1）。伯努利分布是归一化的，且具有如下均值和方差：

$$
\mathbb {E} [ x ] = \mu \tag {3.3}
$$

$$
\operatorname {v a r} [ x ] = \mu (1 - \mu) \tag {3.4}
$$

假设我们有一个包含  $x$  的观测值的数据集  $\mathcal{D} = \{x_1, \dots, x_N\}$  。基于所有观测值都是独立地从  $p(x|\mu)$  生成的，我们可以建立以  $\mu$  为参数的似然方程，使得

$$
p (\mathcal {D} \mid \mu) = \prod_ {n = 1} ^ {N} p \left(x _ {n} \mid \mu\right) = \prod_ {n = 1} ^ {N} \mu^ {x _ {n}} (1 - \mu) ^ {1 - x _ {n}} \tag {3.5}
$$

我们可以通过最大化似然方程，或等价地最大化对数似然函数（对数似然函数是

单调函数）来估计  $\mu$  的值。伯努利分布的对数似然函数形式如下：

$$
\ln p (\mathcal {D} \mid \mu) = \sum_ {n = 1} ^ {N} \ln p (x _ {n} \mid \mu) = \sum_ {n = 1} ^ {N} \left\{x _ {n} \ln \mu + \left(1 - x _ {n}\right) \ln (1 - \mu) \right\} \tag {3.6}
$$

注意对数似然函数仅依赖  $x_{n}$  的  $N$  个观测值的和  $\sum_{n = 1}^{N}x_{n}$  。该值给出了数据在这一分布下的充分统计量（sufficient statistic）（参见3.4节）。令  $\ln p(\mathcal{D}|\mu)$  关于  $\mu$  的导数为0，就可以得到最大似然估计：

$$
\mu_ {\mathrm {M L}} = \frac {1}{N} \sum_ {n = 1} ^ {N} x _ {n} \tag {3.7}
$$

它又称为样本均值（sample mean）。如果观测到  $x = 1$  （正面）的次数为  $m$ ，则式（3.7）又可写作

$$
\mu_ {\mathrm {M L}} = \frac {m}{N} \tag {3.8}
$$

这样硬币落地时正面朝上的概率就可以在上述最大化似然的框架下，通过在数据集中计算正面朝上次数的比例得到。

### 3.1.2 二项分布

我们同样可以给出在包含  $N$  个观测值的数据集中，有  $N$  个观测值为  $x = 1$  的二元变量  $x$  的分布。这一分布称为二项（binomial）分布，由式（3.5）可以看出，它与  $\mu^{m}(1 - \mu)^{N - m}$  成比例。为得到归一化系数，注意在抛掷硬币  $\mu^{m}(1 - \mu)^{N - m}$  次的过程中，我们需要将所有可能掷出  $m$  次正面朝上的情况加总起来，因此二项分布可写为如下形式：

$$
\operatorname {B i n} (m \mid N, \mu) = \binom {N} {m} \mu^ {m} (1 - \mu) ^ {N - m} \tag {3.9}
$$

其中

$$
\binom {N} {m} \equiv \frac {N !}{(N - m) ! m !} \tag {3.10}
$$

是从  $N$  个相同物品中无放回地选择  $m$  个物品的方法总数。图3.1展示了一个  $N = 10$ 、 $\mu = 0.25$  的二项分布（见习题3.3）。

对于独立的事件，和的均值就是均值的和，而和的方差就是方差的和。用这些结果就能找到二项分布的均值和方差（见习题2.10）。由于  $m = x_{1} + \dots +x_{N}$  且任意观测的均值和方差均可由式（3.3）和式（3.4）给出，故有

$$
\mathbb {E} [ m ] \equiv \sum_ {m = 0} ^ {N} m \operatorname {B i n} (m | N, \mu) = N \mu \tag {3.11}
$$

![](img/d2d999407c32005ac46caedf7f9cc110664779f3072585b1d8be956f34b1ca56.jpg)  
图3.1 当  $N = 10$  且  $\mu = 0.25$  时，二项分布式（3.9）作为  $m$  的函数的直方图

$$
\operatorname {v a r} [ m ] \equiv \sum_ {m = 0} ^ {N} (m - \mathbb {E} [ m ]) ^ {2} \operatorname {B i n} (m \mid N, \mu) = N \mu (1 - \mu) \tag {3.12}
$$

这些结论也可以用微积分进行证明（见习题3.4）。

### 3.1.3 多项分布

二元变量可以用于描述只有两种可能取值的量。然而通常情况下，我们遇到的离散变量可能包含  $K$  个相互无关的状态。尽管存在多种表示这些变量的方法，但我们很快将会看到一种非常便捷的“ $K$  中之一”（1-of- $K$ ）的表示方案，有时又称“独热编码”。在该方案中，变量可以表示为一个  $K$  维的向量  $\pmb{x}$ ， $\pmb{x}$  中的一个元素  $x_{k}$  取值为1，其余元素均取值为0。因此，如果我们有一个变量存在  $K = 6$  种状态，且该变量特定的一次观测值刚好对应状态  $x_{3} = 1$ ，则  $\pmb{x}$  可以表示为

$$
\boldsymbol {x} = (0, 0, 1, 0, 0, 0) ^ {\mathrm {T}} \tag {3.13}
$$

注意，这些元素满足  $\sum_{k=1}^{K} x_k = 1$ 。如果我们将  $x_k = 1$  的概率记为  $\mu_k$ ，则  $\pmb{x}$  的分布可以由下式给出：

$$
p (\boldsymbol {x} \mid \boldsymbol {\mu}) = \prod_ {k = 1} ^ {K} \mu_ {k} ^ {x _ {k}} \tag {3.14}
$$

其中  $\pmb{\mu} = (\mu_{1},\dots ,\mu_{K})^{\mathrm{T}}$  ，且因为参数  $\mu_{k}$  表示的是概率，它们满足约束  $\mu_{k}\geqslant 0$  和 $\sum_{k = 1}^{K}\mu_{k} = 1$  。式（3.14）所示的分布可以看作将伯努利分布扩展到可以描述超过两种状态。可以看出，这一分布是归一化的：

$$
\sum_ {\boldsymbol {x}} p (\boldsymbol {x} \mid \boldsymbol {\mu}) = \sum_ {k = 1} ^ {K} \mu_ {k} = 1 \tag {3.15}
$$

且

$$
\mathbb {E} [ \boldsymbol {x} \mid \boldsymbol {\mu} ] = \sum_ {\boldsymbol {x}} p (\boldsymbol {x} \mid \boldsymbol {\mu}) \boldsymbol {x} = \boldsymbol {\mu} \tag {3.16}
$$

考虑包括  $N$  个独立观测值  $\pmb{x}_1,\dots ,\pmb{x}_N$  的数据集  $\mathcal{D}$  。相应的似然函数有如下形式：

$$
p (\mathcal {D} \mid \boldsymbol {\mu}) = \prod_ {n = 1} ^ {N} \prod_ {k = 1} ^ {K} \mu_ {k} ^ {x _ {n k}} = \prod_ {k = 1} ^ {K} \mu_ {k} ^ {\left(\sum_ {n} x _ {n k}\right)} = \prod_ {k = 1} ^ {K} \mu_ {k} ^ {m _ {k}} \tag {3.17}
$$

从中可以看出，包含  $N$  个数据点的似然函数其实可以只通过  $K$  个量来定义：

$$
m _ {k} = \sum_ {n = 1} ^ {N} x _ {n k} \tag {3.18}
$$

它表示观测值  $x_{k} = 1$  的数量，称为分布的充分统计量（sufficient statistic）（参见3.4节）。注意变量  $m_{k}$  满足如下约束：

$$
\sum_ {k = 1} ^ {K} m _ {k} = N \tag {3.19}
$$

为找到  $\mu$  的最大似然解，我们需要在式（3.15）的约束下（也就是让  $\mu_{k}$  的和为1），关于  $\mu_{k}$  最大化  $\ln p(\mathcal{D}|\mu)$  （参见附录C）。这可以通过使用拉格朗日乘子  $\lambda$  并最大化

$$
\sum_ {k = 1} ^ {K} m _ {k} \ln \mu_ {k} + \lambda \left(\sum_ {k = 1} ^ {K} \mu_ {k} - 1\right) \tag {3.20}
$$

来实现求解。

令式（3.20）关于  $\mu_{k}$  的导数为零，可得

$$
\mu_ {k} = - m _ {k} / \lambda \tag {3.21}
$$

将式（3.21）代入约束  $\sum_{k}\mu_{k} = 1$  来求拉格朗日乘子  $\lambda$  。由此可得  $\mu_{k}$  的最大似然解为

$$
\mu_ {k} ^ {\mathrm {M L}} = \frac {m _ {k}}{N} \tag {3.22}
$$

它实际上是  $N$  个观测值中  $x_{k} = 1$  的观测值所占的比例。

已知参数向量  $\mu$  和观测样本总数  $N$  ，可以考虑量  $m_{1},\dots ,m_{K}$  的联合条件概率分布。由式（3.17）可得，这一联合条件概率分布有如下形式：

$$
\operatorname {M u l t} \left(m _ {1}, m _ {2}, \dots , m _ {K} \mid \mu , N\right) = \binom {N} {m _ {1} m _ {2} \dots m _ {K}} \prod_ {k = 1} ^ {K} \mu_ {k} ^ {m _ {k}} \tag {3.23}
$$

这就是多项分布。归一化系数是将  $N$  个对象分为  $K$  组的方法数，其中每组分别有  $m_{1}, \dots, m_{K}$  个对象，由下式给出：

$$
\binom {N} {m _ {1} m _ {2} \dots m _ {K}} = \frac {N !}{m _ {1} ! m _ {2} ! \cdots m _ {K} !} \tag {3.24}
$$

注意，包含两个状态的量可以表示为二元变量并用二项分布式（3.9）建模，或表示为二中之一（1-of-2）变量并用分布式（3.14）建模，其中令  $K = 2$ 。

## 3.2 多元高斯分布

高斯分布又称正态分布，已广泛用于建模连续变量的分布。我们已经看到对于单个变量  $x$  （参见2.3节），高斯分布可以写为如下形式：

$$
\mathcal {N} (x \mid \mu , \sigma^ {2}) = \frac {1}{(2 \pi \sigma^ {2}) ^ {1 / 2}} \exp \left\{- \frac {1}{2 \sigma^ {2}} (x - \mu) ^ {2} \right\} \tag {3.25}
$$

其中  $\mu$  是均值， $\sigma^2$  是方差。对于一个  $D$  维向量  $\pmb{x}$ ，多元高斯分布有如下形式：

$$
\mathcal {N} (\boldsymbol {x} \mid \boldsymbol {\mu}, \boldsymbol {\Sigma}) = \frac {1}{(2 \pi) ^ {D / 2}} \frac {1}{| \boldsymbol {\Sigma} | ^ {1 / 2}} \exp \left\{- \frac {1}{2} (\boldsymbol {x} - \boldsymbol {\mu}) ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {x} - \boldsymbol {\mu}) \right\} \tag {3.26}
$$

其中  $\pmb{\mu}$  是  $D$  维均值向量， $\pmb{\Sigma}$  是  $D\times D$  的协方差矩阵， $\operatorname{det}(\pmb{\Sigma})$  表示  $\pmb{\Sigma}$  的行列式。

在很多不同的场景中都可以看到高斯分布，因此可以从各种不同的角度来理解它的作用（参见2.5节）。例如，我们已经看到，对于单个实数变量，能够使它的熵最大化的分布就是高斯分布。这一性质对于多元高斯分布同样适用（见习题3.8）。

当我们考虑多个随机变量之和的时候，同样会用到高斯分布。中心极限定理（central limit theorem）告诉我们，在某些温和的条件下，一组随机变量的和（它本身也是一个随机变量）的分布会随着其中随机变量数量的增加而越来越接近于高斯分布（Walker, 1969）。我们可以这样解释这一现象：有  $N$  个随机变量  $x_{1}, \dots, x_{N}$ ，每个随机变量都在  $[0,1]$  区间上服从均匀分布，考虑均值  $(x_{1} + \dots + x_{N}) / N$  的分布。对于较大的  $N$ ，如图3.2所示，这一分布趋向于高斯分布。现实中，随着  $N$  的增大，分布会很快地收敛到高斯分布。这一结果揭示了式（3.9）中关于变量  $m$  的二项分布，其中  $m$  是二元随机变量  $x$  的  $N$  个观测样本的和，分布会在  $N \to \infty$  时趋向于高斯分布（见图3.1中  $N = 10$  的情形）。

![](img/b86b4f65d6d73264f322c726df05242ecd7954158bfae374643fa0808b137f46.jpg)  
图3.2  $N$  个均匀分布的数字的均值直方图。我们可以看到，随着  $N$  的增加，分布趋向于高斯分布

![](img/d58e118708a9c43dccf77e20e73d764d11370aca5810b5d2a3b88a2ca87b4e8b.jpg)

![](img/72fd284190302de64b9a7cea48839da1a4712bd22da6ffa028c0691ae233c5df.jpg)

高斯分布有很多重要的分析性质，我们将会考虑其中一些性质的具体细节。因此，本节比前几节更具技术性，读者需要熟悉多种矩阵相关的等式（参见附录A）。

### 3.2.1 高斯几何

下面我们考虑高斯分布的几何形式。高斯分布对  $x$  的函数依赖是通过二次形式来呈现的：

$$
\Delta^ {2} = (\boldsymbol {x} - \mu) ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {x} - \mu) \tag {3.27}
$$

其中，量  $\pmb{\varDelta}$  称为  $\pmb{\mu}$  到  $\pmb{x}$  的马哈拉诺比斯距离（Mahalanobis distance）。当  $\pmb{\Sigma}$  为单位矩阵时，它退化为欧氏距离。高斯分布在  $\pmb{x}$  空间的曲面上是常数，因为该二次型为常数。

首先，注意可以不失一般性地假设矩阵  $\pmb{\Sigma}$  为对称矩阵，因为任何非对称的成分都会从指数中消失（见习题3.11）。考虑协方差矩阵的特征方程：

$$
\boldsymbol {\Sigma} \boldsymbol {u} _ {i} = \lambda_ {i} \boldsymbol {u} _ {i} \tag {3.28}
$$

其中  $i = 1, \dots, D$  。由于  $\pmb{\Sigma}$  是实对称矩阵，其特征值是实数，并且其特征向量可以正交化（见习题3.12），故有

$$
\boldsymbol {u} _ {i} ^ {\mathrm {T}} \boldsymbol {u} _ {j} = I _ {i j} \tag {3.29}
$$

其中  $I_{ij}$  是单位矩阵中的元素，满足

$$
I _ {i j} = \left\{ \begin{array}{l l} 1, & \text {当} i = j \text {时} \\ 0, & \text {其 他} \end{array} \right. \tag {3.30}
$$

协方差矩阵  $\pmb{\Sigma}$  可以由其特征向量扩展获得，并且可以表示为如下形式（参见习题3.13）：

$$
\boldsymbol {\Sigma} = \sum_ {i = 1} ^ {D} \lambda_ {i} \boldsymbol {u} _ {i} \boldsymbol {u} _ {i} ^ {\mathrm {T}} \tag {3.31}
$$

类似地，协方差矩阵的逆矩阵  $\pmb{\Sigma}^{-1}$  可以表示为

$$
\boldsymbol {\Sigma} ^ {- 1} = \sum_ {i = 1} ^ {D} \frac {1}{\lambda_ {i}} \boldsymbol {u} _ {i} \boldsymbol {u} _ {i} ^ {\mathrm {T}} \tag {3.32}
$$

将式（3.32）代入式（3.27），二次型变为

$$
\Delta^ {2} = \sum_ {i = 1} ^ {D} \frac {y _ {i} ^ {2}}{\lambda_ {i}} \tag {3.33}
$$

其中

$$
y _ {i} = \boldsymbol {u} _ {i} ^ {\mathrm {T}} (\boldsymbol {x} - \boldsymbol {\mu}) \tag {3.34}
$$

我们可以将  $\{y_{i}\}$  解释为一个由正交向量  $\pmb{u}_{i}$  定义的新坐标系，它由原始  $x_{i}$  的坐标经过平移和旋转得到。构造向量  $\pmb{y} = (y_{1},\dots ,y_{D})^{\mathrm{T}}$  ，有

$$
\boldsymbol {y} = \boldsymbol {U} (\boldsymbol {x} - \boldsymbol {\mu}) \tag {3.35}
$$

其中， $\pmb{U}$  是以  $\pmb{u}_i^{\mathrm{T}}$  为行向量的矩阵。由式（3.29）可知  $\pmb{U}$  是正交矩阵，满足  $\pmb{U}\pmb{U}^{\mathrm{T}} = \pmb{U}^{\mathrm{T}}\pmb{U} = \pmb{I}$ ，其中  $\pmb{I}$  是单位矩阵（参见附录A）。

由于式（3.33）为常数，因此二次型（即高斯密度）在曲面上也为常数。如果所有特征值  $\lambda_{i}$  均为正，则这些曲面表示以  $\lambda_{i}$  为中心、以  $\pmb{u}_{i}$  的方向为坐标轴且以  $\lambda_{i}^{1 / 2}$  为半轴长的椭球，如图3.3所示。

为了给出良定的高斯分布，需要保证协方差矩阵的所有特征值  $\lambda_{i}$  均严格为正，否则分布就无法被正确归一化。特征值全部严格为正的矩阵是正定的（参见第16章）。在

![](img/57efee229c413d73132e1d4d4c0fc88f35a1293b777cd3e191f52fdc08a1c6af.jpg)  
图3.3 红色曲线表示高斯分布在二维空间  $x = (x_{1}, x_{2})$  中具有恒定概率密度的椭圆面，椭圆面上的密度是其在  $x = \mu$  处取值的  $\exp(-1/2)$  倍。椭圆的轴由协方差矩阵的特征向量  $\pmb{u}_{i}$  定义，具有相应的特征值  $\lambda_{i}$

讨论隐变量模型时，我们将遇到高斯分布的一个或多个特征值为零的情形。在这种情况下，分布是奇异的且限制在一个更低维的子空间中。如果所有的特征值非负，则称这样的协方差矩阵为半正定的。

考虑高斯分布在由  $y_{i}$  定义的新坐标系下的形式。为了从  $\pmb{x}$  坐标系转换到  $\pmb{y}$  坐标系，我们使用了雅可比矩阵  $\pmb{J}$ ，其元素由下式给出：

$$
J _ {i j} = \frac {\partial x _ {i}}{\partial y _ {j}} = U _ {j i} \tag {3.36}
$$

其中， $U_{ji}$  是矩阵  $\mathbf{U}^{\mathrm{T}}$  中的元素。运用矩阵  $\mathbf{U}$  的正交性质，可以得到雅可比矩阵行列式的平方为

$$
| \boldsymbol {J} | ^ {2} = | \boldsymbol {U} ^ {\mathrm {T}} | ^ {2} = | \boldsymbol {U} ^ {\mathrm {T}} | | \boldsymbol {U} | = | \boldsymbol {U} ^ {\mathrm {T}} \boldsymbol {U} | = | \boldsymbol {I} | = 1 \tag {3.37}
$$

从而有  $|\boldsymbol{J}| = 1$  。同样，协方差矩阵的行列式  $|\Sigma|$  可以写成特征值的积，于是有

$$
\left| \boldsymbol {\Sigma} \right| ^ {1 / 2} = \prod_ {j = 1} ^ {D} \lambda_ {j} ^ {1 / 2} \tag {3.38}
$$

因此，在  $y_{i}$  的坐标系下，高斯分布有如下形式：

$$
p (\boldsymbol {y}) = p (\boldsymbol {x}) \mid \boldsymbol {J} \mid = \prod_ {j = 1} ^ {D} \frac {1}{\left(2 \pi \lambda_ {j}\right) ^ {1 / 2}} \exp \left\{- \frac {y _ {j} ^ {2}}{2 \lambda_ {j}} \right\} \tag {3.39}
$$

它是  $D$  个独立的一元高斯分布的积。于是特征向量经过平移和旋转便定义了一组新的坐标，这些坐标使得联合概率分布可以分解为一组独立分布的积。于是在坐标系  $\pmb{y}$  下，分布的积分为

$$
\int p (y) \mathrm {d} y = \prod_ {j = 1} ^ {D} \int_ {- \infty} ^ {\infty} \frac {1}{\left(2 \pi \lambda_ {j}\right) ^ {1 / 2}} \exp \left\{- \frac {y _ {j} ^ {2}}{2 \lambda_ {j}} \right\} \mathrm {d} y _ {j} = 1 \tag {3.40}
$$

这里用到了前面关于一元高斯分布归一化的结论式（2.51），从而验证了多元高斯分布（式3.26）确实是归一化的。

### 3.2.2 矩

下面考虑高斯分布的矩，并给出对参数  $\pmb{\mu}$  和  $\pmb{\Sigma}$  的一种解释。  $\pmb{x}$  在高斯分布下的期望由下式给出：

$$
\begin{array}{l} \mathbb {E} [ x ] = \frac {1}{(2 \pi) ^ {D / 2}} \frac {1}{| \boldsymbol {\Sigma} | ^ {1 / 2}} \int \exp \left\{- \frac {1}{2} (\boldsymbol {x} - \boldsymbol {\mu}) ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {x} - \boldsymbol {\mu}) \right\} x d x \\ = \frac {1}{(2 \pi) ^ {D / 2}} \frac {1}{| \boldsymbol {\Sigma} | ^ {1 / 2}} \int \exp \left\{- \frac {1}{2} z ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} z \right\} (\boldsymbol {z} + \boldsymbol {\mu}) \mathrm {d} z \tag {3.41} \\ \end{array}
$$

这里进行了变量替换：  $z = x - \mu$  。注意指数项是  $z$  的偶函数，且由于是在  $(-\infty, \infty)$  上求积分，因此  $(z + \mu)$  中的  $z$  将由于对称性而消去。于是有

$$
\mathbb {E} [ \boldsymbol {x} ] = \boldsymbol {\mu} \tag {3.42}
$$

$\pmb{\mu}$  可以视为高斯分布的均值。

接下来考虑高斯分布的二阶矩。在单变量的情形下，我们知道二阶矩由  $\mathbb{E}[x^2]$  给出。对于多变量的高斯分布，存在由  $\mathbb{E}[x_i x_j]$  给出的  $D^2$  个二阶矩。我们可以将它们组合成矩阵  $\mathbb{E}[\boldsymbol{x}\boldsymbol{x}^{\mathrm{T}}]$ ，该矩阵可以写作

$$
\begin{array}{l} \mathbb {E} [ \boldsymbol {x x} ^ {\mathrm {T}} ] = \frac {1}{(2 \pi) ^ {D / 2}} \frac {1}{| \boldsymbol {\Sigma} | ^ {1 / 2}} \int \exp \left\{- \frac {1}{2} (\boldsymbol {x} - \boldsymbol {\mu}) ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {x} - \boldsymbol {\mu}) \right\} \boldsymbol {x x} ^ {\mathrm {T}} d \mathbf {x} \\ = \frac {1}{(2 \pi) ^ {D / 2}} \frac {1}{| \boldsymbol {\Sigma} | ^ {1 / 2}} \int \exp \left\{- \frac {1}{2} \boldsymbol {z} ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {z} \right\} (\boldsymbol {z} + \boldsymbol {\mu}) (\boldsymbol {z} + \boldsymbol {\mu}) ^ {\mathrm {T}} d z \tag {3.43} \\ \end{array}
$$

这里再一次进行了变量替换：  $z = x - \mu$  。注意交叉项中涉及的  $\mu z^{\mathrm{T}}$  和  $\mu^{\mathrm{T}}z$  也会由于对称性而消去。项  $\mu \mu^{\mathrm{T}}$  取常值，因此可以拿到积分号之外。由于高斯分布是归一化的，因此它是单位矩阵。可以再一次利用式（3.28）给出的协方差矩阵的特征向量扩展，以及特征向量集的完备性，写出

$$
\boldsymbol {z} = \sum_ {j = 1} ^ {D} y _ {j} \boldsymbol {u} _ {j} \tag {3.44}
$$

其中  $y_{j} = \pmb{u}_{j}^{\mathrm{T}}\pmb{z}$  ，从而有

$$
\begin{array}{l} \frac {1}{(2 \pi) ^ {D / 2}} \frac {1}{| \boldsymbol {\Sigma} | ^ {1 / 2}} \int \exp \left\{- \frac {1}{2} \boldsymbol {z} ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {z} \right\} \boldsymbol {z} \boldsymbol {z} ^ {\mathrm {T}} \mathrm {d} \boldsymbol {z} \\ = \frac {1}{(2 \pi) ^ {D / 2}} \frac {1}{| \boldsymbol {\Sigma} | ^ {1 / 2}} \sum_ {i = 1} ^ {D} \sum_ {j = 1} ^ {D} \boldsymbol {u} _ {i} \boldsymbol {u} _ {j} ^ {\mathrm {T}} \int \exp \left\{- \sum_ {k = 1} ^ {D} \frac {y _ {k} ^ {2}}{2 \lambda_ {k}} \right\} y _ {i} y _ {j} d y \tag {3.45} \\ = \sum_ {i = 1} ^ {D} \boldsymbol {u} _ {i} \boldsymbol {u} _ {i} ^ {\mathrm {T}} \lambda_ {i} = \boldsymbol {\Sigma} \\ \end{array}
$$

这里用到了特征向量等式[式（3.28）]，以及对称性导致中间行除  $i = j$  之外的积分项消去的事实。式（3.45）的最后一行用到了式（2.53）和式（3.38）的结果，并结合了式（3.31）。于是有

$$
\mathbb {E} \left[ \boldsymbol {x x} ^ {\mathrm {T}} \right] = \boldsymbol {\mu} \boldsymbol {\mu} ^ {\mathrm {T}} + \boldsymbol {\Sigma} \tag {3.46}
$$

在定义单个随机变量的方差时，我们在计算二阶矩之前减去了均值。类似地，在多元的情形中，我们再次减去均值，从而将随机变量  $x$  的方差定义为

$$
\operatorname {c o v} [ \boldsymbol {x} ] = \mathbb {E} \left[ (\boldsymbol {x} - \mathbb {E} [ \boldsymbol {x} ]) (\boldsymbol {x} - \mathbb {E} [ \boldsymbol {x} ]) ^ {\mathrm {T}} \right] \tag {3.47}
$$

对于特定的高斯分布的情形，我们可以利用  $\mathbb{E}[x] = \mu$  并结合式（3.46），得到

$$
\operatorname {c o v} [ \boldsymbol {x} ] = \boldsymbol {\Sigma} \tag {3.48}
$$

由于参数矩阵  $\pmb{\Sigma}$  决定了高斯分布下  $\pmb{x}$  的协方差，因此我们称其为协方差矩阵。

### 3.2.3 局限性

尽管高斯分布式（3.26）经常用作简单的密度模型，但它也存在一些明显的局限性。考虑分布中自由参数的数量。一个常规的对称协方差矩阵  $\pmb{\Sigma}$  含有  $D(D + 1) / 2$  个独立的参数（见习题3.15）， $\pmb{\mu}$  中还另外存在  $D$  个独立的参数，因此总共有  $D(D + 3) / 2$  个参数。对于较大的  $D$  ，参数量按  $D$  的平方增长，对较大矩阵的操作和取逆计算代价巨大。解决这个问题的一种方法是使用协方差矩阵的受限形式。考虑到协方差矩阵是对角的， $\pmb{\Sigma} = \mathrm{diag}(\sigma_i^2)$ ，因此密度模型中总共有  $2D$  个独立参数。相应的概率密度等值线由轴对齐的椭球体给出。我们可以进一步将协方差矩阵限制为与单位矩阵成正比， $\pmb{\Sigma} = \sigma^2\pmb{I}$ ，即所谓各向同性的（isotropic）协方差，这时模型中有  $D + 1$  个独立参数，且概率密度等值线是一个球面。常规协方差矩阵、对角协方差矩阵和各向同性协方差矩阵这3种情形如图3.4所示。遗憾的是，虽然这些方法限制了分布中的自由度，并使协方差矩阵的求逆变得更快，但它们也极大限制了概率密度的形式，并限制了模型从数据中捕获有趣相关性的能力。

![](img/b9cb547d7c563c39ae6be5204e1625c4dff9340b21991bfc135cd84ec47eba94.jpg)  
(a)

![](img/6b572c8f492fc78d989c2b1a0f996d65ee7171bda2b2d50b78edafc3a497aa1c.jpg)  
(b)  
图3.4二维高斯分布的概率密度等值线，其中协方差矩阵为：(a)常规协方差矩阵；(b)对角协方差矩阵，在这种情况下，椭圆等值线与坐标轴对齐；(c)各向同性协方差矩阵，与单位矩阵成正比，在这种情况下，等值线是同心圆

![](img/c25fe8ed34899b646151f2f78caa0267e7f1f85217c3ce6e23179b311ea6594b.jpg)  
(c)

高斯分布的另一个局限性是，它本质上是单峰的（即具有单个最大值），因此无法为多峰分布提供良好的近似。高斯分布既存在由于具有太多参数而过于灵活的问题，又存在所能表示的分布太过局限的问题。稍后我们将看到，引入潜变量（也称隐变量或无法观测的变量）可以同时解决上述问题。特别地，通过引入离散潜变量实现高斯混合，可以获得一大类多峰分布（参见3.2.9小节）。同样，连续潜变量的引入将使模型可以独立于数据空间的维度  $D$  ，控制自由参数的数量，同时仍然允许模型捕获数据集中的主要相关性（参见第16章）。

### 3.2.4 条件分布

多元高斯分布的一个重要性质是，如果两组变量服从联合高斯分布，那么以其中一组变量为条件的另一组变量的条件分布是高斯分布。同样，其中任意一组变量的边缘分布也是高斯分布。

首先考虑条件分布的情况。假设  $D$  维向量  $\pmb{x}$  服从高斯分布  $\mathcal{N}(\pmb{x}|\pmb{\mu},\pmb{\Sigma})$ ，将  $\pmb{x}$  划分为两个不相交的子集  $\pmb{x}_a$  和  $\pmb{x}_b$ 。在不失一般性的情况下，我们可以取  $\pmb{x}_a$  来构成  $\pmb{x}$  的

前  $M$  个分量，并取  $\pmb{x}_b$  作为剩余的  $D - M$  个分量，于是

$$
\boldsymbol {x} = \left( \begin{array}{l} \boldsymbol {x} _ {a} \\ \boldsymbol {x} _ {b} \end{array} \right) \tag {3.49}
$$

对均值向量  $\pmb{\mu}$  定义相应的分量：

$$
\boldsymbol {\mu} = \left( \begin{array}{l} \boldsymbol {\mu} _ {a} \\ \boldsymbol {\mu} _ {b} \end{array} \right) \tag {3.50}
$$

对协方差矩阵  $\pmb{\Sigma}$  也定义相应的分量：

$$
\boldsymbol {\Sigma} = \left( \begin{array}{l l} \boldsymbol {\Sigma} _ {a a} & \boldsymbol {\Sigma} _ {a b} \\ \boldsymbol {\Sigma} _ {b a} & \boldsymbol {\Sigma} _ {b b} \end{array} \right) \tag {3.51}
$$

注意，协方差矩阵的对称性  $\pmb{\Sigma}^{\mathrm{T}} = \pmb{\Sigma}$  意味着  $\pmb{\Sigma}_{aa}$  和  $\pmb{\Sigma}_{bb}$  也是对称的，且  $\pmb{\Sigma}_{ba} = \pmb{\Sigma}_{ab}^{\mathrm{T}}$ 。在许多情况下，使用协方差矩阵的逆矩阵会更方便：

$$
\boldsymbol {\Lambda} \equiv \boldsymbol {\Sigma} ^ {- 1} \tag {3.52}
$$

这称为精度矩阵（precision matrix）。事实上，我们将看到高斯分布的某些性质用协方差来表示是最自然的，而对于其他一些性质，精度则是更简单的形式。精度矩阵的分量形式为

$$
\boldsymbol {\Lambda} = \left( \begin{array}{l l} \boldsymbol {\Lambda} _ {a a} & \boldsymbol {\Lambda} _ {a b} \\ \boldsymbol {\Lambda} _ {b a} & \boldsymbol {\Lambda} _ {b b} \end{array} \right) \tag {3.53}
$$

这对应于向量  $x$  的分量形式[式（3.49）]。因为对称矩阵的逆矩阵也是对称的（见习题3.16），所以  $\pmb{\Lambda}_{aa}$  和  $\pmb{\Lambda}_{bb}$  也是对称的，且有  $\pmb{\Lambda}_{ba} = \pmb{\Lambda}_{ab}^{\mathrm{T}}$  。这里需要强调的是， $\pmb{\Lambda}_{ba} = \pmb{\Lambda}_{ab}^{\mathrm{T}}$  不是简单地由  $\pmb{\Sigma}_{aa}$  的逆矩阵给出。事实上，我们将很快研究分量矩阵的逆矩阵与其分量的逆矩阵之间的关系。

首先找到条件分布  $p(\boldsymbol{x}_a|\boldsymbol{x}_b)$  的表达式。从概率的乘积法则中，我们看到只需要将  $\boldsymbol{x}_b$  固定为观测值，并归一化表达式以获得关于  $\boldsymbol{x}_a$  的有效概率分布，就可以基于联合分布  $p(\boldsymbol{x}) = p(\boldsymbol{x}_a,\boldsymbol{x}_b)$  得到条件分布。有别于显式地执行归一化，我们可以通过考虑式（3.27）给出的高斯分布指数中的二次形式，然后在计算结束时恢复归一化系数，从而更高效地获得解。如果使用分量式（3.49）、式（3.50）、式（3.53），则可以得到

$$
\begin{array}{l} - \frac {1}{2} (\boldsymbol {x} - \mu) ^ {\mathrm {T}} \Sigma^ {- 1} (\boldsymbol {x} - \mu) = \\ - \frac {1}{2} \left(\boldsymbol {x} _ {a} - \mu_ {a}\right) ^ {\mathrm {T}} \Lambda_ {a a} \left(\boldsymbol {x} _ {a} - \mu_ {a}\right) - \frac {1}{2} \left(\boldsymbol {x} _ {a} - \mu_ {a}\right) ^ {\mathrm {T}} \Lambda_ {a b} \left(\boldsymbol {x} _ {b} - \mu_ {b}\right) - \tag {3.54} \\ \frac {1}{2} \left(\boldsymbol {x} _ {b} - \boldsymbol {\mu} _ {b}\right) ^ {\mathrm {T}} \Lambda_ {b a} \left(\boldsymbol {x} _ {a} - \boldsymbol {\mu} _ {a}\right) - \frac {1}{2} \left(\boldsymbol {x} _ {b} - \boldsymbol {\mu} _ {b}\right) ^ {\mathrm {T}} \Lambda_ {b b} \left(\boldsymbol {x} _ {b} - \boldsymbol {\mu} _ {b}\right) \\ \end{array}
$$

可以看到，作为  $x_{a}$  的函数，这又是一个二次形式。因此，相应的条件分布  $p\left(x_{a} \mid x_{b}\right)$  将是高斯分布。由于这种分布完全由其均值和协方差确定，因此我们的目标是通过检查式（3.54）来确定  $p\left(x_{a} \mid x_{b}\right)$  的均值和协方差的表达式。

这是一个与高斯分布相关的相当常见的运算示例，有时称为“完全平方”，我们得到了一个定义了高斯分布中指数项的二次形式。我们需要确定相应的均值和协方差。注意，一般高斯分布  $\mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu},\boldsymbol{\Sigma})$  中的指数可以写成

$$
- \frac {1}{2} (\boldsymbol {x} - \boldsymbol {\mu}) ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {x} - \boldsymbol {\mu}) = - \frac {1}{2} \boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {x} + \boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {\mu} + \text {c o n s t} \tag {3.55}
$$

其中“const”表示独立于  $x$  的常数项。再利用  $\pmb{\Sigma}$  的对称性，这些问题就可以得到解决。因此，如果我们采用一般的二次形式，并用式（3.55）右侧给出的形式表示它，则可以立即使  $x$  中二阶项的系数矩阵等于协方差矩阵的逆矩阵  $\pmb{\Sigma}^{-1}$ ，然后使  $x$  中线性项的系数等于  $\pmb{\Sigma}^{-1}\pmb{\mu}$ ，即可得到  $\pmb{\mu}$ 。

将此过程应用于条件高斯分布  $p\left(\boldsymbol{x}_a \mid \boldsymbol{x}_b\right)$ ，指数中的二次形式由式（3.54）给出。分别用  $\mu_{a|b}$  和  $\Sigma_{a|b}$  表示该分布的均值和协方差。考虑式（3.54）对  $\boldsymbol{x}_a$  的函数依赖性，其中  $\boldsymbol{x}_b$  可以视为常数。如果我们挑选出  $\boldsymbol{x}_a$  中的所有二阶项，则得到

$$
- \frac {1}{2} \boldsymbol {x} _ {a} ^ {\mathrm {T}} \boldsymbol {A} _ {a a} \boldsymbol {x} _ {a} \tag {3.56}
$$

从中我们可以得出一个结论，  $p\left(\boldsymbol{x}_a \mid \boldsymbol{x}_b\right)$  的协方差（逆精度）由下式给出：

$$
\boldsymbol {\Sigma} _ {a b} = \boldsymbol {\Lambda} _ {a a} ^ {- 1} \tag {3.57}
$$

考虑式（3.54）中所有在  $x_{a}$  中呈线性的项：

$$
\boldsymbol {x} _ {a} ^ {\mathrm {T}} \left\{\Lambda_ {a a} \boldsymbol {\mu} _ {a} - \Lambda_ {a b} \left(\boldsymbol {x} _ {b} - \boldsymbol {\mu} _ {b}\right) \right\} \tag {3.58}
$$

这里用到了表达式  $\pmb{\Lambda}_{ba}^{\mathrm{T}} = \pmb{\Lambda}_{ab}$  。根据我们对一般形式[式（3.55）]的讨论，该表达式中  $\pmb{x}_a$  的系数必须等于  $\pmb{\Sigma}_{a|b}^{-1}\pmb{\mu}_{a|b}$ ，故有

$$
\begin{array}{l} \boldsymbol {\mu} _ {a \mid b} = \boldsymbol {\Sigma} _ {a \mid b} \left\{\boldsymbol {\Lambda} _ {a a} \boldsymbol {\mu} _ {a} - \boldsymbol {\Lambda} _ {a b} \left(\boldsymbol {x} _ {b} - \boldsymbol {\mu} _ {b}\right) \right\} \tag {3.59} \\ = \boldsymbol {\mu} _ {a} - \boldsymbol {\Lambda} _ {a a} ^ {- 1} \boldsymbol {\Lambda} _ {a b} \left(\boldsymbol {x} _ {b} - \boldsymbol {\mu} _ {b}\right) \\ \end{array}
$$

这里用到了表达式（3.57）。

式（3.57）和式（3.59）给出的结果可以用原始联合分布  $p\left(\boldsymbol{x}_a,\boldsymbol{x}_b\right)$  的分量精度矩阵来表示。也可以用相应的分量协方差矩阵来表示这些结果。为此，对分量矩阵的逆矩阵使用以下恒等式（见习题3.18）：

$$
\left( \begin{array}{l l} \boldsymbol {A} & \boldsymbol {B} \\ \boldsymbol {C} & \boldsymbol {D} \end{array} \right) ^ {- 1} = \left( \begin{array}{c c} \boldsymbol {M} & - \boldsymbol {M B D} ^ {- 1} \\ - \boldsymbol {D} ^ {- 1} \boldsymbol {C M} & \boldsymbol {D} ^ {- 1} + \boldsymbol {D} ^ {- 1} \boldsymbol {C M B D} ^ {- 1} \end{array} \right) \tag {3.60}
$$

其中

$$
\boldsymbol {M} = \left(\boldsymbol {A} - \boldsymbol {B} \boldsymbol {D} ^ {- 1} \boldsymbol {C}\right) ^ {- 1} \tag {3.61}
$$

量  $M^{-1}$  称为式（3.60）左侧矩阵相对于子矩阵  $\pmb{D}$  的舒尔补（Schur complement）。利用定义

$$
\left( \begin{array}{l l} \boldsymbol {\Sigma} _ {a a} & \boldsymbol {\Sigma} _ {a b} \\ \boldsymbol {\Sigma} _ {b a} & \boldsymbol {\Sigma} _ {b b} \end{array} \right) ^ {- 1} = \left( \begin{array}{l l} \boldsymbol {\Lambda} _ {a a} & \boldsymbol {\Lambda} _ {a b} \\ \boldsymbol {\Lambda} _ {b a} & \boldsymbol {\Lambda} _ {b b} \end{array} \right) \tag {3.62}
$$

并使用式（3.60），有

$$
\Lambda_ {a a} = \left(\sum_ {a a} - \sum_ {a b} \sum_ {b b} ^ {- 1} \sum_ {b a}\right) ^ {- 1} \tag {3.63}
$$

$$
\boldsymbol {\Lambda} _ {a b} = - \left(\boldsymbol {\Sigma} _ {a a} - \boldsymbol {\Sigma} _ {a b} \boldsymbol {\Sigma} _ {b b} ^ {- 1} \boldsymbol {\Sigma} _ {b a}\right) ^ {- 1} \boldsymbol {\Sigma} _ {a b} \boldsymbol {\Sigma} _ {b b} ^ {- 1} \tag {3.64}
$$

条件分布  $p\left(\boldsymbol{x}_a \mid \boldsymbol{x}_b\right)$  的均值和协方差可以通过以下表达式获得：

$$
\boldsymbol {\mu} _ {a \mid b} = \boldsymbol {\mu} _ {a} + \boldsymbol {\Sigma} _ {a b} \boldsymbol {\Sigma} _ {b b} ^ {- 1} \left(\boldsymbol {x} _ {b} - \boldsymbol {\mu} _ {b}\right) \tag {3.65}
$$

$$
\boldsymbol {\Sigma} _ {a \mid b} = \boldsymbol {\Sigma} _ {a a} - \boldsymbol {\Sigma} _ {a b} \boldsymbol {\Sigma} _ {b b} ^ {- 1} \boldsymbol {\Sigma} _ {b a} \tag {3.66}
$$

比较式（3.57）和式（3.66），可以看到条件分布  $p\left(\boldsymbol{x}_a \mid \boldsymbol{x}_b\right)$  在用分量精度矩阵表示时，比用分量协方差矩阵表示时更简单。注意，由式（3.65）给出的条件分布  $p\left(\boldsymbol{x}_a \mid \boldsymbol{x}_b\right)$  的均值是  $\boldsymbol{x}_b$  的线性函数，而式（3.66）给出的协方差与  $\boldsymbol{x}_b$  无关。这是线性高斯（linear-Gaussian）模型的一个示例（参见11.1.4小节）。

### 3.2.5 边缘分布

我们已经看到，如果联合分布  $p(\mathbf{x}_a,\mathbf{x}_b)$  是高斯分布，则条件分布  $p(\mathbf{x}_a,\mathbf{x}_b)$  也是高斯分布。现在我们接着讨论由下式给出的边缘分布：

$$
p \left(\boldsymbol {x} _ {a}\right) = \int p \left(\boldsymbol {x} _ {a}, \boldsymbol {x} _ {b}\right) \mathrm {d} \boldsymbol {x} _ {b} \tag {3.67}
$$

正如我们将看到的，它也是高斯分布。与之前相同，我们计算这种分布的策略是关注联合分布指数中的二次形式，从而确定边缘分布  $p(x_{a})$  的均值和协方差。

联合分布指数中的二次形式可以用式（3.54）中的分量精度矩阵来表示。我们的目标是通过积分消去  $x_{b}$ ，而最容易实现的是首先考虑涉及  $x_{b}$  的项，然后完全平方以完成积分。通过选出那些涉及  $x_{b}$  的项，有

$$
- \frac {1}{2} \boldsymbol {x} _ {b} ^ {\mathrm {T}} \boldsymbol {\Lambda} _ {b b} \boldsymbol {x} _ {b} + \boldsymbol {x} _ {b} ^ {\mathrm {T}} \boldsymbol {m} = - \frac {1}{2} \left(\boldsymbol {x} _ {b} - \boldsymbol {\Lambda} _ {b b} ^ {- 1} \boldsymbol {m}\right) ^ {\mathrm {T}} \boldsymbol {\Lambda} _ {b b} \left(\boldsymbol {x} _ {b} - \boldsymbol {\Lambda} _ {b b} ^ {- 1} \boldsymbol {m}\right) + \frac {1}{2} \boldsymbol {m} ^ {\mathrm {T}} \boldsymbol {\Lambda} _ {b b} ^ {- 1} \boldsymbol {m} \tag {3.68}
$$

其中

$$
\boldsymbol {m} = \Lambda_ {b b} \boldsymbol {\mu} _ {b} - \Lambda_ {b a} \left(\boldsymbol {x} _ {a} - \boldsymbol {\mu} _ {a}\right) \tag {3.69}
$$

可以看到，对  $x_{b}$  的依赖性已转换为高斯分布的标准二次形式，对应于式（3.68）右侧的第一项加上一个不依赖于  $x_{b}$  （但依赖于  $x_{a}$  ）的项。因此，当我们取这个二次形式的指数时，可以看到式（3.67）所需的对  $x_{b}$  的积分将采用以下形式：

$$
\int \exp \left\{- \frac {1}{2} \left(\boldsymbol {x} _ {b} - \boldsymbol {\Lambda} _ {b b} ^ {- 1} \boldsymbol {m}\right) ^ {\mathrm {T}} \boldsymbol {\Lambda} _ {b b} \left(\boldsymbol {x} _ {b} - \boldsymbol {\Lambda} _ {b b} ^ {- 1} \boldsymbol {m}\right) \right\} \mathrm {d} \boldsymbol {x} _ {b} \tag {3.70}
$$

这种积分很容易实现，由于该分布是未归一化的高斯分布，因此积分的结果将是归一化系数的倒数。由式（3.26）给出的归一化高斯形式可知，该系数与均值无关，而仅取决于协方差矩阵的行列式。因此，通过计算完全关于  $x_{b}$  的平方，可以积分消去  $x_{b}$ ，于是式（3.68）左侧唯一剩下的依赖于  $x_{a}$  的项就是式（3.68）右侧的最后一项，其中  $m$  由式（3.69）给出。将该项与式（3.54）中依赖于  $x_{a}$  的其余项相结合，可以得到

$$
\begin{array}{l} \frac {1}{2} \left[ \boldsymbol {\Lambda} _ {b b} \boldsymbol {\mu} _ {b} - \boldsymbol {\Lambda} _ {b a} \left(\boldsymbol {x} _ {a} - \boldsymbol {\mu} _ {a}\right) \right] ^ {\mathrm {T}} \boldsymbol {\Lambda} _ {b b} ^ {- 1} \left[ \boldsymbol {\Lambda} _ {b b} \boldsymbol {\mu} _ {b} - \boldsymbol {\Lambda} _ {b a} \left(\boldsymbol {x} _ {a} - \boldsymbol {\mu} _ {a}\right) \right] - \\ \frac {1}{2} \boldsymbol {x} _ {a} ^ {\mathrm {T}} \boldsymbol {\Lambda} _ {a a} \boldsymbol {x} _ {a} + \boldsymbol {x} _ {a} ^ {\mathrm {T}} \left(\boldsymbol {\Lambda} _ {a a} \boldsymbol {\mu} _ {a} + \boldsymbol {\Lambda} _ {a b} \boldsymbol {\mu} _ {b}\right) + \text {c o n s t} \tag {3.71} \\ = - \frac {1}{2} \boldsymbol {x} _ {a} ^ {\mathrm {T}} \left(\boldsymbol {\Lambda} _ {a a} - \boldsymbol {\Lambda} _ {a b} \boldsymbol {\Lambda} _ {b b} ^ {- 1} \boldsymbol {\Lambda} _ {b a}\right) \boldsymbol {x} _ {a} + \boldsymbol {x} _ {a} ^ {\mathrm {T}} \left(\boldsymbol {\Lambda} _ {a a} - \boldsymbol {\Lambda} _ {a b} \boldsymbol {\Lambda} _ {b b} ^ {- 1} \boldsymbol {\Lambda} _ {b a}\right) \boldsymbol {\mu} _ {a} + \text {c o n s t} \\ \end{array}
$$

其中“const”表示独立于  $x_{a}$  的常数项。同样，通过与式（3.55）进行比较，可以看到边缘分布  $p(x_{a})$  的协方差由下式给出：

$$
\boldsymbol {\Sigma} _ {a} = \left(\boldsymbol {\Lambda} _ {a a} - \boldsymbol {\Lambda} _ {a b} \boldsymbol {\Lambda} _ {b b} ^ {- 1} \boldsymbol {\Lambda} _ {b a}\right) ^ {- 1} \tag {3.72}
$$

均值则由下式给出：

$$
\boldsymbol {\Sigma} _ {a} \left(\boldsymbol {\Lambda} _ {a a} - \boldsymbol {\Lambda} _ {a b} \boldsymbol {\Lambda} _ {b b} ^ {- 1} \boldsymbol {\Lambda} _ {b a}\right) \boldsymbol {\mu} _ {a} = \boldsymbol {\mu} _ {a} \tag {3.73}
$$

这里用到了式（3.72）。协方差式（3.72）是用式（3.53）给出的分量精度矩阵来表示的。就像我们对条件分布所做的那样，我们可以根据式（3.51）给出的协方差矩阵的相应分量来重写它。这些分量矩阵有如下关系：

$$
\left( \begin{array}{l l} \boldsymbol {\Lambda} _ {a a} & \boldsymbol {\Lambda} _ {a b} \\ \boldsymbol {\Lambda} _ {b a} & \boldsymbol {\Lambda} _ {b b} \end{array} \right) ^ {- 1} = \left( \begin{array}{c c} \boldsymbol {\Sigma} _ {a a} & \boldsymbol {\Sigma} _ {a b} \\ \boldsymbol {\Sigma} _ {b a} & \boldsymbol {\Sigma} _ {b b} \end{array} \right) \tag {3.74}
$$

利用式（3.60），有

$$
\left(\boldsymbol {A} _ {a a} - \boldsymbol {A} _ {a b} \boldsymbol {A} _ {b b} ^ {- 1} \boldsymbol {A} _ {b a}\right) ^ {- 1} = \boldsymbol {\Sigma} _ {a a} \tag {3.75}
$$

这样我们就得到了直观上令人满意的结果，即边缘分布  $p(x_{a})$  具有下式给出的均值和协方差：

$$
\mathbb {E} \left[ \boldsymbol {x} _ {a} \right] = \boldsymbol {\mu} _ {a} \tag {3.76}
$$

$$
\operatorname {c o v} \left[ \boldsymbol {x} _ {a} \right] = \Sigma_ {a a} \tag {3.77}
$$

可以看到，对于边缘分布，均值和协方差用分量协方差矩阵表达最简单；而对于条件分布，均值和协方差用分量精度矩阵表达最简单。

我们可以将分量形式高斯分布的边缘分布和条件分布的结果总结如下。对于联合高斯分布  $\mathcal{N}\left(x|\boldsymbol{\mu},\boldsymbol{\Sigma}\right)$ ，给定  $\varLambda\equiv\Sigma^{-1}$  和下列分量矩阵：

$$
\boldsymbol {x} = \left( \begin{array}{l} \boldsymbol {x} _ {a} \\ \boldsymbol {x} _ {b} \end{array} \right), \quad \boldsymbol {\mu} = \left( \begin{array}{l} \boldsymbol {\mu} _ {a} \\ \boldsymbol {\mu} _ {b} \end{array} \right) \tag {3.78}
$$

$$
\boldsymbol {\Sigma} = \left( \begin{array}{l l} \boldsymbol {\Sigma} _ {a a} & \boldsymbol {\Sigma} _ {a b} \\ \boldsymbol {\Sigma} _ {b a} & \boldsymbol {\Sigma} _ {b b} \end{array} \right), \quad \boldsymbol {\Lambda} = \left( \begin{array}{l l} \boldsymbol {\Lambda} _ {a a} & \boldsymbol {\Lambda} _ {a b} \\ \boldsymbol {\Lambda} _ {b a} & \boldsymbol {\Lambda} _ {b b} \end{array} \right) \tag {3.79}
$$

则条件分布由下式给出：

$$
p \left(\boldsymbol {x} _ {a} \mid \boldsymbol {x} _ {b}\right) = \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {\mu} _ {a | b}, \boldsymbol {A} _ {a a} ^ {- 1}\right) \tag {3.80}
$$

$$
\boldsymbol {\mu} _ {a, b} = \boldsymbol {\mu} _ {a} - \boldsymbol {\Lambda} _ {a a} ^ {- 1} \boldsymbol {\Lambda} _ {a b} \left(\boldsymbol {x} _ {b} - \boldsymbol {\mu} _ {b}\right) \tag {3.81}
$$

边缘分布由下式给出：

$$
p \left(\boldsymbol {x} _ {a}\right) = \mathcal {N} \left(\boldsymbol {x} _ {a} \mid \boldsymbol {\mu} _ {a}, \boldsymbol {\Sigma} _ {a a}\right) \tag {3.82}
$$

图3.5用一个涉及两个变量的示例，对多元高斯分布的边缘分布和条件分布的思想进行了阐释。

![](img/5f5eeefea0536b45e3cad32a901c2fe82bdf94ba2b3ec896e4c44ad9f330e4e4.jpg)  
(a)

![](img/215b45e508065bdec04aff10f1fcd4bb7d4281623779ed484914a221440b0a23.jpg)  
(b)  
图3.5 (a) 一个涉及两个变量的高斯分布  $p(x_{a}, x_{b})$  的等值线。(b)  $p(x_{a})$  的边缘分布（蓝色曲线）和  $x_{b} = 0.7$  时的条件分布  $p(x_{a} | x_{b})$  （红色曲线）

### 3.2.6 贝叶斯定理

在3.2.4小节和3.2.5小节中，我们考虑了高斯分布  $p(\pmb {x})$  ，其中我们将向量  $\pmb{x}$  分成了两个子向量  $\pmb {x} = (x_{a},x_{b})$  ，然后找到了条件分布  $p\left(x_{a}\mid x_{b}\right)$  和边缘分布  $p\left(x_{a}\right)$  的表达式。注意条件分布  $p\left(x_{a}\mid x_{b}\right)$  的均值是  $\pmb{x}_{b}$  的线性函数。在这里，假设我们有高斯边缘分布  $p(\pmb {x})$  和高斯条件分布  $p(y|x)$  ，其中  $p(y|x)$  的均值是  $\pmb{x}$  的线性函数，协方差与  $\pmb{x}$  独立。这是线性高斯模型的一个例子（Roweis and Ghahramani,1999）（参见11.1.4小节）。我们希望找到边缘分布  $p(y)$  和条件分布  $p\bigl (x|y\bigr)$  。这种结构在好几种生成式模型中都有出现（参见第16章），因此在这里推导出一般结论会很有帮助。

我们将边缘分布和条件分布取如下形式：

$$
p (\boldsymbol {x}) = \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {\mu}, \boldsymbol {\Lambda} ^ {- 1}\right) \tag {3.83}
$$

$$
p (\boldsymbol {y} \mid \boldsymbol {x}) = \mathcal {N} (\boldsymbol {y} \mid A \boldsymbol {x} + \boldsymbol {b}, L ^ {- 1}) \tag {3.84}
$$

其中  $\pmb{\mu}$  、  $A$  和  $\pmb{b}$  是控制均值的参数，  $A$  和  $L$  是精度矩阵。如果  $\pmb{x}$  的维度为  $M$  ，  $\pmb{y}$  的维度为  $D$  ，则矩阵  $\pmb{A}$  的大小为  $D\times M$  。

首先找到  $x$  和  $y$  上联合分布的表达式。为此，我们定义

$$
\boldsymbol {z} = \left( \begin{array}{l} \boldsymbol {x} \\ \boldsymbol {y} \end{array} \right) \tag {3.85}
$$

然后考虑联合分布的对数：

$$
\begin{array}{l} \ln p (z) = \ln p (\boldsymbol {x}) + \ln p (\boldsymbol {y} | \boldsymbol {x}) \\ = - \frac {1}{2} (\boldsymbol {x} - \boldsymbol {\mu}) ^ {\mathrm {T}} \boldsymbol {A} (\boldsymbol {x} - \boldsymbol {\mu}) - \frac {1}{2} (\boldsymbol {y} - \boldsymbol {A x} - \boldsymbol {b}) ^ {\mathrm {T}} \boldsymbol {L} (\boldsymbol {y} - \boldsymbol {A x} - \boldsymbol {b}) + \text {c o n s t} \tag {3.86} \\ \end{array}
$$

其中“const”表示独立于  $x$  和  $y$  的常数项。如前所述，这是  $z$  分量的二次函数，因此  $p(z)$  是高斯分布。为了找到这个高斯分布的精度矩阵，式（3.86）中的二阶项可以写成

$$
\begin{array}{l} - \frac {1}{2} \boldsymbol {x} ^ {\mathrm {T}} (\boldsymbol {A} + \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {A}) \boldsymbol {x} - \frac {1}{2} \boldsymbol {y} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {y} + \frac {1}{2} \boldsymbol {y} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {A} \boldsymbol {x} + \frac {1}{2} \boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {y} \\ = - \frac {1}{2} \left( \begin{array}{c} \boldsymbol {x} \\ \boldsymbol {y} \end{array} \right) ^ {\mathrm {T}} \left( \begin{array}{c c} \boldsymbol {A} + \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {A} & - \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \\ - \boldsymbol {L} \boldsymbol {A} & \boldsymbol {L} \end{array} \right) \left( \begin{array}{c} \boldsymbol {x} \\ \boldsymbol {y} \end{array} \right) = - \frac {1}{2} \boldsymbol {z} ^ {\mathrm {T}} \boldsymbol {R} \boldsymbol {z} \tag {3.87} \\ \end{array}
$$

因此，  $z$  上的高斯分布的精度（逆协方差）矩阵由下式给出：

$$
\boldsymbol {R} = \left( \begin{array}{c c} \boldsymbol {A} + \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {A} & - \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \\ - \boldsymbol {L} \boldsymbol {A} & \boldsymbol {L} \end{array} \right) \tag {3.88}
$$

协方差矩阵是通过取精度矩阵的逆矩阵来求的，这可以通过使用逆矩阵公式（3.60）来实现（见习题3.23），即

$$
\operatorname {c o v} [ z ] = \boldsymbol {R} ^ {- 1} = \left( \begin{array}{l l} \Lambda^ {- 1} & \Lambda^ {- 1} \boldsymbol {A} ^ {\mathrm {T}} \\ \boldsymbol {A} \Lambda^ {- 1} & \boldsymbol {L} ^ {- 1} + \boldsymbol {A} \Lambda^ {- 1} \boldsymbol {A} ^ {\mathrm {T}} \end{array} \right) \tag {3.89}
$$

同样，也可以通过识别式（3.86）中的线性项来计算关于  $z$  的高斯分布的均值，它由下式给出：

$$
\boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {\Lambda} \boldsymbol {\mu} - \boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {b} + \boldsymbol {y} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {b} = \left( \begin{array}{c} \boldsymbol {x} \\ \boldsymbol {y} \end{array} \right) ^ {\mathrm {T}} \left( \begin{array}{c} \boldsymbol {\Lambda} \boldsymbol {\mu} - \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {b} \\ \boldsymbol {L} \boldsymbol {b} \end{array} \right) \tag {3.90}
$$

使用我们之前通过在多元高斯分布的二次型上完全平方得到的结果式（3.55），可以发现  $z$  的均值由下式给出：

$$
\mathbb {E} [ z ] = \boldsymbol {R} ^ {- 1} \left( \begin{array}{c} A \boldsymbol {\mu} - \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {b} \\ \boldsymbol {L} \boldsymbol {b} \end{array} \right) \tag {3.91}
$$

利用式（3.89）（见习题3.24），可以得到

$$
\mathbb {E} [ z ] = \left( \begin{array}{c} \boldsymbol {\mu} \\ \boldsymbol {A} \boldsymbol {\mu} + \boldsymbol {b} \end{array} \right) \tag {3.92}
$$

接下来，我们在  $x$  上进行边缘化，得到  $p(\mathbf{y})$  的表达式。回想一下，当使用分量协方差矩阵来表达服从高斯分布的随机向量子集的边缘分布时，我们可以采用非常简单的形式（参见3.2节）。具体来说，其均值和协方差分别由式（3.76）和式（3.77）给出。利用式（3.89）和式（3.92）可以看到，边缘分布  $p(\mathbf{y})$  的均值和协方差由下式给出：

$$
\mathbb {E} [ \boldsymbol {y} ] = \boldsymbol {A} \boldsymbol {\mu} + \boldsymbol {b} \tag {3.93}
$$

$$
\operatorname {c o v} [ \mathbf {y} ] = \boldsymbol {L} ^ {- 1} + \boldsymbol {A} \boldsymbol {A} ^ {- 1} \boldsymbol {A} ^ {\mathrm {T}} \tag {3.94}
$$

这个结果的一个特例出现在  $A = I$  时，在这种情况下，边缘分布退化为两个高斯分布的卷积。由此可以看到，卷积的均值是两个高斯分布的均值之和，卷积的协方差则是它们的协方差之和。

最后推导条件分布  $p(x|y)$  的表达式。回想一下，条件分布的结果可以很容易地通过式（3.57）和式（3.59），用分量精度矩阵来表示（参见3.2节）。将这些结果应用于式（3.89）和式（3.92），可以看到条件分布  $p(x|y)$  具有以下均值和协方差：

$$
\mathbb {E} [ \boldsymbol {x} \mid \boldsymbol {y} ] = (\boldsymbol {\Lambda} + \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {A}) ^ {- 1} \left\{\boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} (\boldsymbol {y} - \boldsymbol {b}) + \boldsymbol {\Lambda} \boldsymbol {\mu} \right\} \tag {3.95}
$$

$$
\operatorname {c o v} [ \boldsymbol {x} \mid \boldsymbol {y} ] = \left(\boldsymbol {A} + \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {A}\right) ^ {- 1} \tag {3.96}
$$

对这种条件分布进行估计可以看作贝叶斯定理的一个示例。在贝叶斯定理中，我们将  $p(x)$  称为  $\pmb{x}$  的先验分布。如果观测到变量  $\pmb{y}$ ，则条件分布  $p(\pmb{x}|\pmb{y})$  表示  $\pmb{x}$  所对应的后验分布。得到边缘分布和条件分布后，我们可以有效地将联合分布  $p(z) = p(x)p(y|x)$  表示为  $p(\pmb{x}|\pmb{x})p(\pmb{y})$  的形式。

这些结果可以总结如下。给定关于  $x$  的边缘分布和给定  $x$  时关于  $y$  的条件分布：

$$
p (\boldsymbol {x}) = \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {\mu}, \boldsymbol {\Lambda} ^ {- 1}\right) \tag {3.97}
$$

$$
p (\boldsymbol {y} \mid \boldsymbol {x}) = \mathcal {N} (\boldsymbol {y} \mid \boldsymbol {A x} + \boldsymbol {b}, \boldsymbol {L} ^ {- 1}) \tag {3.98}
$$

则  $y$  的边缘分布和给定  $\pmb{y}$  时  $\pmb{x}$  的条件分布由下式给出：

$$
p (\boldsymbol {y}) = \mathcal {N} \left(\boldsymbol {y} \mid A \boldsymbol {\mu} + \boldsymbol {b}, \boldsymbol {L} ^ {- 1} + A \boldsymbol {A} ^ {- 1} \boldsymbol {A} ^ {\mathrm {T}}\right) \tag {3.99}
$$

$$
p (\boldsymbol {x} \mid \boldsymbol {y}) = \mathcal {N} \left(\boldsymbol {x} \mid \Sigma \left\{\boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} (\boldsymbol {y} - \boldsymbol {b}) + \boldsymbol {\Lambda} \boldsymbol {\mu} \right\}, \boldsymbol {\Sigma}\right) \tag {3.100}
$$

其中

$$
\boldsymbol {\Sigma} = \left(\boldsymbol {A} + \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {L} \boldsymbol {A}\right) ^ {- 1} \tag {3.101}
$$

### 3.2.7 最大似然

给定数据集  $X = (x_{1},\dots ,x_{N})^{\mathrm{T}}$  ，假设其中的观测值  $\{x_{n}\}$  服从多元高斯分布且相互独立，则可以通过最大似然来估计分布的参数。对数似然函数由下式给出：

$$
\ln p (\boldsymbol {X} \mid \boldsymbol {\mu}, \boldsymbol {\Sigma}) = - \frac {N D}{2} \ln (2 \pi) - \frac {N}{2} \ln | \boldsymbol {\Sigma} | - \frac {1}{2} \sum_ {n = 1} ^ {N} \left(\boldsymbol {x} _ {n} - \boldsymbol {\mu}\right) ^ {\mathrm {T}} \boldsymbol {\Sigma} ^ {- 1} \left(\boldsymbol {x} _ {n} - \boldsymbol {\mu}\right) \tag {3.102}
$$

通过简单地重排，可以看到似然函数与数据集相关的部分仅依赖于这两个量：

$$
\sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n} \text {和} \sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n} \boldsymbol {x} _ {n} ^ {\mathrm {T}} \tag {3.103}
$$

这称为高斯分布的充分统计量（sufficient statistic）。使用式（A.19）（参见附录A），对数似然关于  $\pmb{\mu}$  的导数由下式给出：

$$
\frac {\partial}{\partial \boldsymbol {\mu}} \ln p (\boldsymbol {X} \mid \boldsymbol {\mu}, \boldsymbol {\Sigma}) = \sum_ {n = 1} ^ {N} \boldsymbol {\Sigma} ^ {- 1} \left(\boldsymbol {x} _ {n} - \boldsymbol {\mu}\right) \tag {3.104}
$$

令导数为零，即可得到均值的最大似然估计解：

$$
\boldsymbol {\mu} _ {\mathrm {M L}} = \frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n} \tag {3.105}
$$

这是观测数据点集的平均值。关于  $\pmb{\Sigma}$  最大化式（3.102）可以起到更大的作用。最简单的方法是忽略对称约束，正如所要求的那样，得到的解是对称的（参见习题3.28）。我们在Magnus and Neudecker（1999）中可以找到对这一结果的替代推导，这些推导显式地施加了对称性和正定性约束。结果正如预期的那样，具有如下形式：

$$
\boldsymbol {\Sigma} _ {\mathrm {M L}} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left(\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {\mathrm {M L}}\right) \left(\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {\mathrm {M L}}\right) ^ {\mathrm {T}} \tag {3.106}
$$

其中涉及  $\pmb{\mu}_{\mathrm{ML}}$  ，因为这是相对于  $\pmb{\mu}$  和  $\pmb{\Sigma}$  联合最大化的结果。注意，  $\pmb{\mu}_{\mathrm{ML}}$  的解[式（3.105）]并不依赖于  $\pmb{\Sigma}_{\mathrm{ML}}$  ，因此我们可以首先估计  $\pmb{\mu}_{\mathrm{ML}}$  ，然后用它来估计  $\pmb{\Sigma}_{\mathrm{ML}}$  。

如果我们估计真实分布下最大似然解的期望（见习题3.29），则可以得到以下结果：

$$
\mathbb {E} \left[ \boldsymbol {\mu} _ {\mathrm {M L}} \right] = \boldsymbol {\mu} \tag {3.107}
$$

$$
\mathbb {E} \left[ \Sigma_ {\mathrm {M L}} \right] = \frac {N - 1}{N} \Sigma \tag {3.108}
$$

可以看到，均值的最大似然估计期望等于真实均值。但是，协方差的最大似然估计期望小于真实值，因此它是有偏的。通过定义一个不同的估计量  $\tilde{\Sigma}$ ，可以修正这一偏差，其由下式给出：

$$
\tilde {\Sigma} = \frac {1}{N - 1} \sum_ {n = 1} ^ {N} \left(\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {\mathrm {M L}}\right) \left(\boldsymbol {x} _ {n} - \boldsymbol {\mu} _ {\mathrm {M L}}\right) ^ {\mathrm {T}} \tag {3.109}
$$

显然，从式（3.106）和式（3.108）可以看出， $\tilde{\Sigma}$  的期望与  $\pmb{\Sigma}$  的期望相等。

### 3.2.8 序贯估计

我们对最大似然解的讨论采用了批处理（batch processing）方法，即一次性考虑整个训练集。也可以使用序贯（sequential）方法，该方法允许一次处理一个数据点，然后将其丢弃。这对于在线应用和大数据非常重要，因为一次性批量处理所有数据在这种情况下是不可行的。

考虑式（3.105）给出的最大似然估计的均值  $\pmb{\mu}_{\mathrm{ML}}$  ，当它建立在  $N$  个观测值之上时，我们将它表示为  $\pmb{\mu}_{\mathrm{ML}}^{(N)}$  。如果我们分解出最末尾的数据点  $\pmb{x}_N$  的贡献，则可以得到

$$
\begin{array}{l} \boldsymbol {\mu} _ {\mathrm {M L}} ^ {(N)} = \frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n} \\ = \frac {1}{N} x _ {N} + \frac {1}{N} \sum_ {n = 1} ^ {N - 1} x _ {n} \tag {3.110} \\ = \frac {1}{N} \boldsymbol {x} _ {N} + \frac {N - 1}{N} \boldsymbol {\mu} _ {\mathrm {M L}} ^ {(N - 1)} \\ = \boldsymbol {\mu} _ {\mathrm {M L}} ^ {(N - 1)} + \frac {1}{N} \left(\boldsymbol {x} _ {N} - \boldsymbol {\mu} _ {\mathrm {M L}} ^ {(N - 1)}\right) \\ \end{array}
$$

这个结果可以很容易地按如下方式解释。在观测到  $N - 1$  个数据点后，我们用 $\pmb{\mu}_{\mathrm{ML}}^{(N - 1)}$  来估计  $\pmb{\mu}$  。而在观测到数据点  $\pmb{x}_N$  后，我们可以通过将旧的估计值在“误差信号” $(x_{N} - \pmb{\mu}_{\mathrm{ML}}^{(N - 1)})$  的方向上移动一个较小的量来获得修订后的估计值  $\pmb{\mu}_{\mathrm{ML}}^{(N)}$  ，这个较小的量与

$1 / N$  成正比。注意，随着  $N$  的增加，来自后续数据点的贡献会越来越小。

### 3.2.9 高斯混合

尽管高斯分布具有一些重要的分析特性，但在用于对真实数据集建模时，它存在很大的局限性。考虑图3.6(a)中的示例，它又称为“Old Faithful”数据集，其中包括美国黄石国家公园老忠实喷泉喷发的272个观测值，每个观测值都以分钟为单位给出喷发的持续时间（横轴）和下一次喷发的时间（以分钟为单位）（纵轴）。我们可以看到该数据集形成了两个主要的簇，并且使用简单的高斯分布无法捕捉这种结构。

![](img/21ce8612fa56e77c119889bd2f200624d995db48ccdcef7a91239e13938663d6.jpg)  
(a)

![](img/d2e6f8be12a00b7237ee56dfe341f1ea462836d8ac3d2943ac5d004d17c80bb5.jpg)  
(b)  
图3.6 老忠实喷泉数据的图示，其中红色曲线是等概率密度的等值线。(a) 在单一高斯分布上使用最大似然拟合数据得到的分布。注意，此分布无法捕捉数据中的两个簇，并且确实将其大部分概率质量放置在了数据相对稀疏的团块间的中心区域。(b) 由两个高斯分布的线性组合得到的分布，其也由最大似然拟合，这样可以更好地反映数据

可以预期，叠加两个高斯分布将能够更好地表示该数据集的结构，情况也确实如此，如图3.6(b)所示。这种叠加是通过对更基本的分布（如高斯分布等）进行线性组合来实现

![](img/41629ad8bdb433478550a89a4a8bd8871b356ff265815448553bc69ea17a1972.jpg)  
图3.7 一维高斯混合分布，蓝色曲线显示了3个高斯分布（每个高斯分布按系数缩放），红色曲线显示了它们的叠加效果

的，可以称作混合分布（mixture distribution）的概率模型（参见第15章）。在本节中，我们将考虑用高斯分布来说明混合模型的框架。更一般地，混合模型可以包含其他分布的线性组合，例如二元变量的伯努利分布的混合分布等。在图3.7中，我们看到高斯分布的线性组合可以产生非常复杂的密度。通过使用足够数量的高斯分布，并调整它们的均值和协方差以及线性组合中的系数，我们可以把几乎任何连续分布近似到任意准确度。

考虑形如下式的  $K$  个高斯密度的叠加：

$$
p (\boldsymbol {x}) = \sum_ {k = 1} ^ {K} \pi_ {k} \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}\right) \tag {3.111}
$$

这称为高斯混合（mixture of Gaussians）。每个高斯密度  $\mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right)$  则称为混合分布的一个分量（component），且具有自己的均值  $\boldsymbol{\mu}_k$  和协方差  $\boldsymbol{\Sigma}_k$  。图3.8显示了具

有3个分量的二元高斯混合分布的等值线和曲面图。

![](img/4e0f8b443cf4af7d10dc91a5db53ff22eb68e58ea6b0b22f8a6eec0dbd2cc810.jpg)  
(a)

![](img/d13f73b42143a3bcf07badb3c91e580a40600ca9da632f8142555b2de7441840.jpg)  
(b)

![](img/9e38dc9a9ba0d16098158a3689bf7847ae4004e48cac0b009d5f9a36e8f20e3e.jpg)  
(c)  
图3.8 3个二元高斯混合分布的示意图。(a)每种混合情况的密度等值线，3种情况分别表示为红色曲线、蓝色曲线和绿色曲线，混合系数的值显示在每个分量的下方。(b)混合分布的边缘概率密度  $p(x)$  的等值线。(c)分布 $p(x)$  的曲面图

式（3.111）中的参数  $\pi_{k}$  称为混合系数（mixing coefficient）。如果在式（3.111）的两边对  $x$  进行积分，并注意到  $p(x)$  和单个高斯分量都可以归一化，便可得到

$$
\sum_ {k = 1} ^ {K} \pi_ {k} = 1 \tag {3.112}
$$

此外，给定  $\mathcal{N}\big(x|\pmb{\mu}_k,\pmb{\Sigma}_k\big)\geqslant 0$  ，则  $p(\pmb {x})\geqslant 0$  的充分条件是对所有  $k$  都有  $\pi_k\geqslant 0$  。将其与条件式（3.112）相结合，可以得到

$$
0 \leqslant \pi_ {k} \leqslant 1 \tag {3.113}
$$

因此，混合系数满足将其作为概率的要求。下面我们将证明这种对混合分布的概率解释是非常强大的（参见第15章）。

根据概率的加和法则与乘积法则，边缘密度可以写为

$$
p (\boldsymbol {x}) = \sum_ {k = 1} ^ {K} p (k) p (\boldsymbol {x} \mid k) \tag {3.114}
$$

这等价于式（3.111），其中可以将  $\pi_k = p(k)$  视为选择第  $k$  个分量的先验概率，并将密度  $\mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k) = p(\boldsymbol{x}|k)$  视为以  $k$  为条件时  $\pmb{x}$  的概率。我们将在后面的章节中看到，相应的后验概率  $p(k|x)$  会起到重要作用，因此也称责任。根据贝叶斯定理，这些可以由下式给出：

$$
\begin{array}{l} \gamma_ {k} (\boldsymbol {x}) \equiv p (k \mid \boldsymbol {x}) \\ = \frac {p (k) p (\boldsymbol {x} \mid k)}{\sum_ {l} p (l) p (\boldsymbol {x} \mid l)} = \frac {\pi_ {k} \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {\mu} _ {k} , \boldsymbol {\Sigma} _ {k}\right)}{\sum_ {l} \pi_ {l} \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {\mu} _ {l} , \boldsymbol {\Sigma} _ {l}\right)} \tag {3.115} \\ \end{array}
$$

高斯混合分布的形式由参数  $\pmb{\pi}$  和  $\pmb{\Sigma}$  控制，其中  $\pmb{\pi} \equiv \{\pi_1, \dots, \pi_K\}$ ， $\pmb{\mu} \equiv \{\pmb{\mu}_1, \dots, \pmb{\mu}_K\}$ ， $\pmb{\Sigma} \equiv \{\pmb{\Sigma}_1, \dots, \pmb{\Sigma}_K\}$ 。设置这些参数值的一种方法是使用最大似然。根据式（3.111），似然函数的对数由下式给出：

$$
\ln p (X \mid \boldsymbol {\pi}, \boldsymbol {\mu}, \boldsymbol {\Sigma}) = \sum_ {n = 1} ^ {N} \ln \left\{\sum_ {k = 1} ^ {K} \pi_ {k} \mathcal {N} \left(\boldsymbol {x} _ {n} \mid \boldsymbol {\mu} _ {k}, \boldsymbol {\Sigma} _ {k}\right) \right\} \tag {3.116}
$$

其中  $X = \{x_{1},\dots ,x_{N}\}$  。我们发现由于是在对数中对  $k$  求和，现在的情况比单个高斯分布要复杂得多。参数的最大似然解不再具有闭式解析解。一种最大化似然函数的方法是使用迭代数值优化技术。或者，我们也可以采用一个称作最大期望（expectation maximization）的强大框架（参见第15章），它广泛适用于各种不同的深度生成式模型。

## 3.3 周期变量

尽管高斯分布本身具有重要的实际意义，并且作为构建更复杂概率模型的基础也十分有用，但在某些情况下，它们并不适合作为连续变量的密度模型。实际应用中存在的周期变量就属于这种情况。

特定地理位置的风向是周期变量的一个例子。例如，我们在测量多个地理位置的风向后，希望使用参数分布来总结这些数据。另一个例子是日历时间，我们想建模那些可能在24小时或一年内具有周期性的量。这样的量可以使用  $0 \leqslant \theta < 2\pi$  范围内的角（极）坐标来方便地表示。

我们可以选择某个方向作为起始方向，然后应用高斯分布等常规分布来处理周期变量。然而，这种方法所产生的结果极其依赖于我们随意选择的起始方向。例如，假设有两个观测值  $\theta_{1} = 1^{\circ}$  和  $\theta_{2} = 359^{\circ}$ ，并且我们使用标准的单元高斯分布对它们建模。如果将起始方向设置为  $0^{\circ}$ ，那么该数据集的样本均值为  $180^{\circ}$ ，标准差为  $179^{\circ}$ ；而如果将起始方向设置为  $180^{\circ}$ ，那么该数据集的样本均值为  $0^{\circ}$ ，标准差为  $1^{\circ}$ 。显然，我们需要一种专门用来处理周期变量的方法。

### 冯·米塞斯分布

考虑估计周期变量  $\theta$  的一组观测值  $\mathcal{D} = \{\theta_1,\dots ,\theta_N\}$  的均值，其中  $\theta$  的单位为弧度。简单计算均值  $(\theta_{1} + \dots +\theta_{N}) / N$  将与坐标系强相关。为了找到均值的不变度量，我们注意到观测值可以视为单位圆上的点，因此可以用二维单位向量  $x_{1},\dots ,x_{N}$  来描述它们，其中对于  $x_{1},\dots ,x_{N}$  ，  $\| x_{n}\| = 1$  ，如图3.9所示。对向量  $\{x_{n}\}$  求平均值，可得

$$
\bar {\boldsymbol {x}} = \frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {x} _ {n} \tag {3.117}
$$

![](img/703c7ccc6d97d21e8cc9f2b51921c0f160e8f420e3921f3517804cea35505aa1.jpg)  
图3.9将周期变量的值  $\theta_{n}$  表示为位于单位圆上的二维向量  $x_{n}$  的示意图，同时还展示了这些向量的平均值  $\overline{x}$

然后找到与平均值对应的角度  $\overline{\theta}$  。显然，上述定义将确保均值的位置与角坐标的起始方向无关。注意  $\overline{x}$  通常位于单位圆内。观测值的笛卡儿坐标由  $x_{n} = (\cos \theta_{n},\sin \theta_{n})$  给出，样本均值的笛卡儿坐标可以写成  $\overline{x} = (\overline{r}\cos \overline{\theta},\overline{r}\sin \overline{\theta})$  。代入式（3.117）并使  $x_{1}$  和  $x_{2}$  分量相等，则有

$$
\bar {x} _ {1} = \bar {r} \cos \bar {\theta} = \frac {1}{N} \sum_ {n = 1} ^ {N} \cos \theta_ {n}, \quad \bar {x} _ {2} = \bar {r} \sin \bar {\theta} = \frac {1}{N} \sum_ {n = 1} ^ {N} \sin \theta_ {n} \tag {3.118}
$$

对二者取比值，并使用恒等式  $\tan \theta = \sin \theta / \cos \theta$ ， $\overline{\theta}$  可由下式求解：

$$
\bar {\theta} = \tan^ {- 1} \left\{\frac {\sum_ {n} \sin \theta_ {n}}{\sum_ {n} \cos \theta_ {n}} \right\} \tag {3.119}
$$

读者很快就会看到这个结果是如何作为最大似然估计量而产生的。

首先定义高斯分布的一个周期性推广，称为冯·米塞斯（von Mises）分布。接下来仅着重于一元分布，尽管在任意维度的超球面上也可以找到类似的周期分布（Mardia and Jupp, 2000）。

按照惯例，考虑分布  $p(\theta)$ ，其具有  $2\pi$  周期。任何定义在  $\theta$  上的概率密度  $p(\theta)$  都必须是非负的且积分等于  $p(\theta)$ ，同时还必须具有周期性。因此， $p(\theta)$  必须满足以下3个条件：

$$
p (\theta) \geqslant 0 \tag {3.120}
$$

$$
\int_ {0} ^ {2 \pi} p (\theta) \mathrm {d} \theta = 1 \tag {3.121}
$$

$$
p (\theta + 2 \pi) = p (\theta) \tag {3.122}
$$

从式（3.122）可知，对于任何整数  $M$  ，  $p(\theta + 2M\pi) = p(\theta)$  。

我们可以很容易地获得满足上述3个条件的类高斯分布：考虑二元变量  $\pmb{x} = (x_{1}, x_{2})$

上的高斯分布，其均值为  $\pmb{\mu} = (\mu_1, \mu_2)$ ，协方差矩阵为  $\pmb{\Sigma} = \sigma^2\pmb{I}$ ，其中  $\pmb{I}$  是  $2 \times 2$  的单位矩阵，故有

$$
p \left(x _ {1}, x _ {2}\right) = \frac {1}{2 \pi \sigma^ {2}} \exp \left\{- \frac {\left(x _ {1} - \mu_ {1}\right) ^ {2} + \left(x _ {2} - \mu_ {2}\right) ^ {2}}{2 \sigma^ {2}} \right\} \tag {3.123}
$$

常数  $p(x)$  的等值线为圆形，如图3.10所示。

考虑沿着固定半径的圆来取这个分布上的值。通过这种构造方法，尽管没有被归一化，但这个分布仍将是周期性的。我们可以通过从笛卡儿坐标  $(x_{1}, x_{2})$  转换到极坐标  $(r, \theta)$  来确定该分布的形式：

$$
x _ {1} = r \cos \theta , \quad x _ {2} = r \sin \theta \tag {3.124}
$$

同时将均值  $\mu$  也映射到极坐标：

$$
\mu_ {1} = r _ {0} \cos \theta_ {0}, \quad \mu_ {2} = r _ {0} \sin \theta_ {0} \tag {3.125}
$$

接下来，我们设单位圆  $r = 1$  ，将这些变换代入二维高

![](img/4961e38ee30a8eaa0b7add721da895a2561067ae9e2e4dafdb78774ce5c7414e.jpg)  
图3.10冯·米塞斯分布可通过式（3.123）所示的二维高斯分布导出。图中蓝色的圆是其密度等值线，红色的单位圆是其条件

斯分布式（3.123），以研究该分布与  $\theta$  的依赖关系。关注高斯分布中的指数部分，则有

$$
\begin{array}{l} - \frac {1}{2 \sigma^ {2}} \left\{\left(r \cos \theta - r _ {0} \cos \theta_ {0}\right) ^ {2} + \left(r \sin \theta - r _ {0} \sin \theta_ {0}\right) ^ {2} \right\} \\ = - \frac {1}{2 \sigma^ {2}} \left\{1 + r _ {0} ^ {2} - 2 r _ {0} \cos \theta \cos \theta_ {0} - 2 r _ {0} \sin \theta \sin \theta_ {0} \right\} \tag {3.126} \\ = \frac {r _ {0}}{\sigma^ {2}} \cos (\theta - \theta_ {0}) + \text {c o n s t} \\ \end{array}
$$

其中“const”表示独立于  $\theta$  的常数项。此处的推导使用了以下三角恒等式：

$$
\cos^ {2} A + \sin^ {2} A = 1 \tag {3.127}
$$

$$
\cos A \cos B + \sin A \sin B = \cos (A - B) \tag {3.128}
$$

如果定义  $m = r_0 / \sigma^2$  ，则可以得到  $p(\theta)$  沿单位圆  $r = 1$  这个分布的最终表达式，形式为

$$
p \left(\theta \mid \theta_ {0}, m\right) = \frac {1}{2 \pi I _ {0} (m)} \exp \left\{m \cos \left(\theta - \theta_ {0}\right) \right\} \tag {3.129}
$$

这称为冯·米塞斯分布或圆形正态（circular normal）分布。这里的参数  $\theta_0$  对应于分布的均值， $m$  则称为聚焦（concentration）参数，类似于高斯分布的逆方差（即精度）。式（3.129）中的归一化系数用  $I_0(m)$  表示，它是第一类零阶修正贝塞尔函数

（Abramowitz and Stegun, 1965），定义如下：

$$
I _ {0} (m) = \frac {1}{2 \pi} \int_ {0} ^ {2 \pi} \exp \{m \cos \theta \} d \theta \tag {3.130}
$$

当  $m$  较大时，该分布近似为高斯分布（见习题3.31）。冯·米塞斯分布如图3.11所示，函数  $I_0(m)$  如图3.12所示。

![](img/9554ac2d4e75dcb63ff4f65ddac150606803697d5bebebe4e8d0b46994bd5969.jpg)  
图3.11 两个不同参数值下的冯·米塞斯分布。左图在笛卡儿坐标下，右图在相应的极坐标下

![](img/d522171c61b29bbde507833e5e9345b3f307fe99470a94be7391393bdee7cc7a.jpg)

![](img/16b827a6cb8c9b3255a45525ee26184d78fe8e491b872de91a383eb4d549d877.jpg)  
图3.12 贝塞尔函数  $I_0(m)$  [式（3.130）]和函数  $A(m)$  [式（3.136）]的图示

![](img/6ea202152aad37ebe19d84f678260e1546e4f751b1e7cddabeec94cd9dfb5478.jpg)

考虑冯·米塞斯分布中参数  $\theta_0$  和  $m$  的最大似然估计量。对数似然函数由下式给出：

$$
\ln p \left(\mathcal {D} \mid \theta_ {0}, m\right) = - N \ln (2 \pi) - N \ln I _ {0} (m) + m \sum_ {n = 1} ^ {N} \cos \left(\theta_ {n} - \theta_ {0}\right) \tag {3.131}
$$

令式（3.131）对  $\theta_0$  的导数为零，可得

$$
\sum_ {n = 1} ^ {N} \sin \left(\theta_ {n} - \theta_ {0}\right) = 0 \tag {3.132}
$$

为了求解  $\theta_0$  ，使用如下三角不等式：

$$
\sin (A - B) = \cos B \sin A - \cos A \sin B \tag {3.133}
$$

由此可得（见习题3.32）

$$
\theta_ {0} ^ {\mathrm {M L}} = \tan^ {- 1} \left\{\frac {\sum_ {n} \sin \theta_ {n}}{\sum_ {n} \cos \theta_ {n}} \right\} \tag {3.134}
$$

可以发现，这与之前二维笛卡儿空间中观测值均值的结果［式（3.119）］一致。

类似地，关于  $m$  最大化式（3.131），并利用  $I_0^{\prime}(m) = I_1(m)$  （Abramowitz and Stegun, 1965），可得

$$
A \left(m _ {\mathrm {M L}}\right) = \frac {1}{N} \sum_ {n = 1} ^ {N} \cos \left(\theta_ {n} - \theta_ {0} ^ {\mathrm {M L}}\right) \tag {3.135}
$$

此处已经代入了  $\theta_0$  的最大似然解（注意我们现在正在对  $\theta$  和  $m$  进行联合优化），并定义了

$$
A (m) = \frac {I _ {1} (m)}{I _ {0} (m)} \tag {3.136}
$$

函数  $A(m)$  如图3.12所示。利用式（3.128）中的三角恒等式，可以将式（3.135）写成如下形式：

$$
A \left(m _ {\mathrm {M L}}\right) = \left(\frac {1}{N} \sum_ {n = 1} ^ {N} \cos \theta_ {n}\right) \cos \theta_ {0} ^ {\mathrm {M L}} + \left(\frac {1}{N} \sum_ {n = 1} ^ {N} \sin \theta_ {n}\right) \sin \theta_ {0} ^ {\mathrm {M L}} \tag {3.137}
$$

式（3.137）的右侧部分很容易计算，而函数  $A(m)$  可以通过数值反演得到。冯·米塞斯分布的一个局限性是，它是单峰的。我们可以通过构造一个包含多个冯·米塞斯分布的混合分布来获得一个更为灵活的周期变量建模框架，以处理多峰数据。

为了讨论的完整性，下面简要提及一些构造周期分布的替代技术。最简单的方法是使用观测值的直方图，其中角坐标被划分为固定的区间。这种方法的优点是简单且灵活，但也存在显著的局限性，这一点读者在后文对直方图方法的详细讨论中将会看到（参见3.5节）。另一种方法类似于冯·米塞斯分布：从欧氏空间上的高斯分布开始，但现在边缘化到单位圆上，而不是条件化（Mardia and Jupp, 2000）。然而，这会导致分布形式更加复杂，因此我们不再做进一步讨论。最后，实轴上的任何有效分布（如高斯分布）都可以通过将宽度为  $2\pi$  的连续区间映射到周期变量  $(0, 2\pi)$  上，成为周期分布，这相当于将实轴“包裹”到单位圆上（环绕分布）。同样，由此得到的分布比冯·米塞斯分布更复杂。

## 3.4 指数族分布

到目前为止，本章所研究的概率分布（混合模型除外）都是指数族（exponential

family）分布（Duda and Hart, 1973; Bernardo and Smith, 1994）这一大类里的具体例子。指数族分布的成员具有许多共同的重要性质，对这些性质的一般性讨论可以带来不少启发。

给定参数  $\eta$  ，关于  $x$  的指数族分布可以定义为以下形式的分布的集合：

$$
p (\boldsymbol {x} \mid \boldsymbol {\eta}) = h (\boldsymbol {x}) g (\boldsymbol {\eta}) \exp \left\{\boldsymbol {\eta} ^ {\mathrm {T}} \boldsymbol {u} (\boldsymbol {x}) \right\} \tag {3.138}
$$

其中  $\pmb{x}$  可以是标量或矢量，它可以是离散的或连续的。  $\eta$  则称为分布的自然参数（natural parameters），  $\pmb{u}(\pmb{x})$  是  $\pmb{x}$  的某个函数。函数  $g(\eta)$  可以解释为用来确保分布被归一化的系数，因此满足

$$
g (\boldsymbol {\eta}) \int h (\boldsymbol {x}) \exp \left\{\boldsymbol {\eta} ^ {\mathrm {T}} \boldsymbol {u} (\boldsymbol {x}) \right\} \mathrm {d} \boldsymbol {x} = 1 \tag {3.139}
$$

其中  $x$  若为离散变量，则积分改为求和。

下面我们从本章前面介绍的分布示例入手，说明它们确实是指数族分布的成员。首先考虑伯努利分布：

$$
p (x \mid \mu) = \operatorname {B e r n} (x \mid \mu) = \mu^ {x} (1 - \mu) ^ {1 - x} \tag {3.140}
$$

将式（3.140）的右侧表示为对数的指数，可得

$$
\begin{array}{l} p (x \mid \mu) = \exp \left\{x \ln \mu + (1 - x) \ln (1 - \mu) \right\} \\ = (1 - \mu) \exp \left\{\ln \left(\frac {\mu}{1 - \mu}\right) x \right\} \tag {3.141} \\ \end{array}
$$

与式（3.138）比较后，我们可以确定

$$
\eta = \ln \left(\frac {\mu}{1 - \mu}\right) \tag {3.142}
$$

对  $\mu$  进行求解可得  $\mu = \sigma (\eta)$  ，其中

$$
\sigma (\eta) = \frac {1}{1 + \exp (- \eta)} \tag {3.143}
$$

它又称为 sigmoid 函数。因此，我们可以使用以下标准表达式来描述伯努利分布：

$$
p (x \mid \eta) = \sigma (- \eta) \exp (\eta x) \tag {3.144}
$$

这里使用了等式  $1 - \sigma(\eta) = \sigma(-\eta)$ ，其易由式（3.143）证得。与式（3.138）比较可得

$$
u (x) = x \tag {3.145}
$$

$$
h (x) = 1 \tag {3.146}
$$

$$
g (\eta) = \sigma (- \eta) \tag {3.147}
$$

接下来考虑多项分布，对于单个观测  $x$  ，其形式为

$$
p (\boldsymbol {x} \mid \boldsymbol {\mu}) = \prod_ {k = 1} ^ {M} \mu_ {k} ^ {x _ {k}} = \exp \left\{\sum_ {k = 1} ^ {M} x _ {k} \ln \mu_ {k} \right\} \tag {3.148}
$$

其中  $\pmb{x} = (x_{1},\dots ,x_{M})^{\mathrm{T}}$  。式（3.148）同样可以写成标准表达式：

$$
p (\boldsymbol {x} \mid \boldsymbol {\eta}) = \exp \left(\boldsymbol {\eta} ^ {\mathrm {T}} \boldsymbol {x}\right) \tag {3.149}
$$

其中  $\eta_{k} = \ln \mu_{k}$ ，并且我们定义  $\pmb{\eta} = (\eta_{1},\dots ,\eta_{M})^{\mathrm{T}}$ 。与式（3.138）比较可得

$$
\boldsymbol {u} (\boldsymbol {x}) = \boldsymbol {x} \tag {3.150}
$$

$$
\boldsymbol {h} (\boldsymbol {x}) = 1 \tag {3.151}
$$

$$
g (\boldsymbol {\eta}) = 1 \tag {3.152}
$$

注意，参数  $\eta_{k}$  不是独立的，因为参数  $\mu_{k}$  受到下式的约束：

$$
\sum_ {k = 1} ^ {M} \mu_ {k} = 1 \tag {3.153}
$$

因此，对于给定的任意  $M - 1$  个参数  $\mu_{k}$  ，剩余参数的值被固定。在某些情况下，通过仅用  $M - 1$  个参数表达分布，可以方便地去除这一约束。这可以通过使用关系式（3.153）将  $\mu_{M}$  用剩余的  $\{\mu_k\}$  （其中  $k = 1,\dots ,M - 1$  ）表示，从而留下  $M - 1$  个参数来实现。请注意，这些剩余参数仍然受到下式的约束：

$$
0 \leqslant \mu_ {k} \leqslant 1, \quad \sum_ {k = 1} ^ {M - 1} \mu_ {k} \leqslant 1 \tag {3.154}
$$

利用约束式（3.153），多项分布变为

$$
\begin{array}{l} \exp \left\{\sum_ {k = 1} ^ {M} x _ {k} \ln \mu_ {k} \right\} \\ = \exp \left\{\sum_ {k = 1} ^ {M - 1} x _ {k} \ln \mu_ {k} + \left(1 - \sum_ {k = 1} ^ {M - 1} x _ {k}\right) \ln \left(1 - \sum_ {k = 1} ^ {M - 1} \mu_ {k}\right) \right\} \tag {3.155} \\ = \exp \left\{\sum_ {k = 1} ^ {M - 1} x _ {k} \ln \left(\frac {\mu_ {k}}{1 - \sum_ {j = 1} ^ {M - 1} \mu_ {j}}\right) + \ln \left(1 - \sum_ {k = 1} ^ {M - 1} \mu_ {k}\right) \right\} \\ \end{array}
$$

现在我们确定了

$$
\ln \left(\frac {\mu_ {k}}{1 - \sum_ {j} \mu_ {j}}\right) = \eta_ {k} \tag {3.156}
$$

可以通过首先对式（3.156）两边的  $k$  求和，然后重新排列和反向代入来求解  $k$  ：

$$
\mu_ {k} = \frac {\exp \left(\eta_ {k}\right)}{1 + \sum_ {j} \exp \left(\eta_ {j}\right)} \tag {3.157}
$$

它又称为softmax函数或归一化指数（normalized exponential）。因此在这种表示中，多项分布将采用以下形式：

$$
p (\boldsymbol {x} \mid \boldsymbol {\eta}) = \left(1 + \sum_ {k = 1} ^ {M - 1} \exp \left(\eta_ {k}\right)\right) ^ {- 1} \exp \left(\boldsymbol {\eta} ^ {\mathrm {T}} \boldsymbol {x}\right) \tag {3.158}
$$

这是带有参数向量  $\eta = (\eta_{1},\dots ,\eta_{M - 1})^{\mathrm{T}}$  的指数族分布的标准形式。与式（3.138）比较可得

$$
\boldsymbol {u} (\boldsymbol {x}) = \boldsymbol {x} \tag {3.159}
$$

$$
h (\boldsymbol {x}) = 1 \tag {3.160}
$$

$$
g (\eta) = \left(1 + \sum_ {k = 1} ^ {M - 1} \exp \left(\eta_ {k}\right)\right) ^ {- 1} \tag {3.161}
$$

最后考虑高斯分布。对于一元高斯分布，我们有

$$
\begin{array}{l} p (x \mid \mu , \sigma^ {2}) = \frac {1}{(2 \pi \sigma^ {2}) ^ {1 / 2}} \exp \left\{- \frac {1}{2 \sigma^ {2}} (x - \mu) ^ {2} \right\} (3.162) \\ = \frac {1}{\left(2 \pi \sigma^ {2}\right) ^ {1 / 2}} \exp \left\{- \frac {1}{2 \sigma^ {2}} x ^ {2} + \frac {\mu}{\sigma^ {2}} x - \frac {1}{2 \sigma^ {2}} \mu^ {2} \right\} (3.163) \\ \end{array}
$$

经过一些简单的重排，就可以得到标准指数族分布的形式[式（3.138）]（见习题3.35）：

$$
\eta = \left( \begin{array}{c} \mu / \sigma^ {2} \\ - 1 / 2 \sigma^ {2} \end{array} \right) \tag {3.164}
$$

$$
\boldsymbol {u} (x) = \left( \begin{array}{l} x \\ x ^ {2} \end{array} \right) \tag {3.165}
$$

$$
h (\boldsymbol {x}) = (2 \pi) ^ {- 1 / 2} \tag {3.166}
$$

$$
g (\eta) = \left(- 2 \eta_ {2}\right) ^ {1 / 2} \exp \left(\frac {\eta_ {1} ^ {2}}{4 \eta_ {2}}\right) \tag {3.167}
$$

有时我们会使用式（3.138）所示的带约束形式，其中我们选择  $\pmb{u}(\pmb{x}) = \pmb{x}$  。不过，我们也可以将其稍微推广一下。如果  $f(x)$  是一个归一化的密度函数，则

$$
\frac {1}{s} f \left(\frac {1}{s} x\right) \tag {3.168}
$$

也是一个归一化的密度函数，其中  $s > 0$  是尺度参数。结合这些，我们可以得到指数族分布的类-条件密度的一个约束集，形式为

$$
p \left(\boldsymbol {x} \mid \lambda_ {k}, s\right) = \frac {1}{s} h \left(\frac {1}{s} \boldsymbol {x}\right) g \left(\lambda_ {k}\right) \exp \left\{\frac {1}{s} \lambda_ {k} ^ {\mathrm {T}} \boldsymbol {x} \right\} \tag {3.169}
$$

注意，我们允许每个类别都有自己的参数向量  $\lambda_{k}$ ，并且我们假设这些类别共享相同的尺度参数  $s$ 。

### 充分统计量

考虑使用最大似然估计的方法来估计一般指数族分布[式（3.138）]中的参数向量  $\pmb{\eta}$  。在式（3.139）的两边对  $\pmb{\eta}$  求导，可得

$$
\nabla g (\boldsymbol {\eta}) \int h (\boldsymbol {x}) \exp \left\{\boldsymbol {\eta} ^ {\mathrm {T}} \boldsymbol {u} (\boldsymbol {x}) \right\} \mathrm {d} \boldsymbol {x} + g (\boldsymbol {\eta}) \int h (\boldsymbol {x}) \exp \left\{\boldsymbol {\eta} ^ {\mathrm {T}} \boldsymbol {u} (\boldsymbol {x}) \right\} \boldsymbol {u} (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} = 0 \tag {3.170}
$$

重排并再次使用式（3.139），可得

$$
- \frac {1}{g (\boldsymbol {\eta})} \nabla g (\boldsymbol {\eta}) = g (\boldsymbol {\eta}) \int h (x) \exp \left\{\boldsymbol {\eta} ^ {\mathrm {T}} \boldsymbol {u} (x) \right\} \boldsymbol {u} (x) \mathrm {d} x = \mathbb {E} [ \boldsymbol {u} (x) ] \tag {3.171}
$$

从而得到

$$
- \nabla \ln g (\boldsymbol {\eta}) = \mathbb {E} [ \boldsymbol {u} (\boldsymbol {x}) ] \tag {3.172}
$$

注意， $u(x)$  的协方差可以用  $g(\eta)$  的二阶导数来表示，并且对于更高阶矩来说也是类似的（见习题3.36）。因此，如果可以对指数族分布进行归一化，则总是可以通过对函数进行简单的微分来找到它的矩。

考虑一组由  $X = \{x_{1},\dots ,x_{n}\}$  表示的独立同分布数据，其似然函数为

$$
p (\boldsymbol {X} \mid \boldsymbol {\eta}) = \left(\prod_ {n = 1} ^ {N} h \left(\boldsymbol {x} _ {n}\right)\right) g (\boldsymbol {\eta}) ^ {N} \exp \left\{\boldsymbol {\eta} ^ {\mathrm {T}} \sum_ {n = 1} ^ {N} \boldsymbol {u} \left(\boldsymbol {x} _ {n}\right) \right\} \tag {3.173}
$$

令  $\ln p(X|\eta)$  对  $\eta$  的梯度为零，可以得出最大似然估计量  $\eta_{\mathrm{ML}}$  必须满足以下条件：

$$
- \nabla \ln g (\boldsymbol {\eta} _ {\mathrm {M L}}) = \frac {1}{N} \sum_ {n = 1} ^ {N} \boldsymbol {u} (\boldsymbol {x} _ {n}) \tag {3.174}
$$

从而原则上可以解出  $\eta_{\mathrm{ML}}$  。可以看到，最大似然估计量的解仅通过  $\sum_{n}\eta_{\mathrm{ML}}$  依赖于数据，因此又称为分布式（3.138）的充分统计量。我们不需要存储整个数据集本身，而只需要存储充分统计量的值。例如，对于伯努利分布，函数  $u(x)$  仅由  $x$  给出，因此只需要保留数据点  $\{x_n\}$  的和；而对于高斯分布， $u(x) = (x,x^2)^{\mathrm{T}}$  ，因此应该保留  $\{x_n\}$  的和与  $\{x_n^2\}$  的和。

考虑极限  $N \to \infty$ ，式（3.174）的右边将变为  $\mathbb{E}\left[u(x)\right]$ ，通过与式（3.172）进行比较，我们可以看到在此极限下， $\eta_{\mathrm{ML}}$  将等于真实值  $\pmb{\eta}$ 。

## 3.5 非参数化方法

本章专注于使用由少量参数决定的具有特定函数形式的概率分布，这些参数的值可从数据集中确定。这称为参数化（parametric）的密度估计方法。这种方法的一个关键局限是，所选择的密度模型可能不适合数据分布，这可能导致预测性能不佳。例如，如果生成数据的过程是多峰的，那么高斯分布永远无法捕捉到分布这个角度的特征，因为高斯分布必然是单峰的。本节将考虑一些非参数化（nonparametric）的密度估计方法，这些方法对分布的形式几乎不作任何假设。

### 3.5.1 直方图

首先讨论密度估计的直方图方法。我们在前面关于边缘分布与条件分布（见图2.5）以及中心极限定理（见图3.2）的讨论中已经见到过这种方法。下面将更详细地探讨直方图密度模型的特性，并重点关注具有单个连续变量  $x$  的情况。标准直方图简单将  $x$  划分为互不相交的宽度为  $\Delta_{i}$  的分箱，然后计算落在不同分箱中的观测值  $x$  的数量  $n_i$  。为了将此计数值转换为归一化的概率密度，我们可以将其简单地除以观测值总数  $N$  和分箱宽度  $\Delta_{i}$  的乘积，以获得落入每个分箱的概率值：

$$
p _ {i} = \frac {n _ {i}}{N \Delta_ {i}} \tag {3.175}
$$

可以看出，  $\int p(x)\mathrm{d}x = 1$  。这种方法就给出了密度  $p(x)$  的一种模型，在每个分箱的宽度上，密度是恒定的。通常情况下，我们选择具有相同宽度的分箱，  $\varDelta_{i}=\varDelta$  。

图3.13展示了直方图密度估计的示例。这里的数据来自绿色曲线所对应的分布，该分布由两个高斯分布混合形成。这3个直方图密度估计的例子对应着3种不同的分箱宽度  $\Delta$  。我们可以看到，当  $\Delta$  非常小时（图3.13的顶部图），得到的密度模型非常毛糙，具有许多在生成数据集的底层分布中存在的结构。相反，如果  $\Delta$  太大（图3.13的底部图），得到的密度模型将太过平滑，因此无法捕捉到绿色曲线的双峰特性。最佳结果是  $\Delta$  取某个中间值（图3.13的中间图）。原则上，直方图密度模型还取决于分箱边界的选择，

![](img/e24e24b3031d9ece3a5dec168d3eb5432833a13813fa2b59c0c5b9d3ec9b3717.jpg)  
图3.13 直方图密度估计的示例，其中根据绿色曲线所示的分布生成含有50个数据点的数据集。这里的直方图密度估计基于式（3.175），具有相同的分箱宽度  $\varDelta$  。图中展示了不同分箱宽度的结果

但这相较于分箱宽度  $\Delta$  的选择来说通常不那么重要。

注意直方图方法具有一个特性（与稍后讨论的其他密度估计方法不同），即一旦直方图计算完成，数据集本身就可以丢弃，这一点在数据集很大时会十分有利。如果数据点是顺序到达的，则应用直方图方法也将比较容易。

在实践中，直方图技术可以用于快速可视化一维或二维数据，但其不适用于大多数密度估计应用。一个明显的问题是，分箱边界导致估计的密度是不连续的，而这并不是由生成数据的底层分布的任何性质引起的。直方图方法的一个主要局限是，规模会随维度变化显著改变。如果将  $D$  维空间中的每个变量划分为  $M$  个区间，那么分箱将有  $M^D$  个。这种随  $D$  指数级扩展的特性是维度诅咒（curse of dimensionality）的一个例子（参见6.11节）。在高维空间中，为了提供有意义的局部概率密度估计，所需的数据量将是我们无法承受的。

然而，直方图方法也给密度估计带来了两条重要的经验。首先，要在特定位置估计概率密度，就应该考虑落在该位置某个局部邻域内的数据点。注意局部性的概念要求我们假设某种距离度量，而在这里我们假设的是欧氏距离。对于直方图来说，这种邻域性质是由分箱定义的，并且存在一个自然的“平滑”参数用于描述局部区域的空间大小，本例中即为分箱宽度。其次，为了获得良好的结果，平滑参数的值既不应该太大也不应该太小。这会让人联想到多项式回归中模型复杂性的选择（参见第1章），其中多项式的阶数  $M$  或者正则化参数  $\lambda$  在某些中间值上是最优的，既不太大也不太小。有了这些洞察，下面我们来讨论两种使用广泛的非参数化密度估计技术——核密度和最近邻，它们在维度变化时的扩展性比简单的直方图模型要好。

### 3.5.2 核密度

假设观测值是从某个  $D$  维空间（这里将其视为欧氏空间）中的某个未知概率密度  $p(\boldsymbol{x})$  中产生的，我们希望估计  $p(\boldsymbol{x})$  的值。根据先前关于局部性的讨论，考虑包含  $\boldsymbol{x}$  的一个小区域  $\mathcal{R}$  。与该区域相关联的概率质量为

$$
P = \int_ {\mathcal {R}} p (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} \tag {3.176}
$$

假设我们已经收集了一个包含  $N$  个从  $p(x)$  中产生的观测值的数据集。因为每个数据点在  $\mathcal{R}$  内的概率为  $P$ ，所以  $\mathcal{R}$  内的点的总数  $K$  将服从二项分布（参见3.12节）：

$$
\operatorname {B i n} (K \mid N, P) = \frac {N !}{K ! (N - K) !} P ^ {K} (1 - P) ^ {N - K} \tag {3.177}
$$

利用式（3.11）可以看出，落入该区域的点的平均占比为  $\mathbb{E}[K / N] = P$  。类似地，利用式（3.12）可以看出，此均值对应的方差为  $\mathrm{var}\big[K / N\big] = P(1 - P) / N$  。对于较大的 $N$  ，该分布在均值处将变得十分尖锐，因此

$$
K \approx N P \tag {3.178}
$$

然而，如果我们同时假设区域  $\mathcal{R}$  足够小，以至于概率密度  $p(x)$  在该区域内大致保持恒定，则有

$$
P \approx p (\boldsymbol {x}) V \tag {3.179}
$$

其中  $V$  是区域  $\mathcal{R}$  的体积。结合式（3.178）和式（3.179），可以得到密度估计的形式为

$$
p (\boldsymbol {x}) = \frac {K}{N V} \tag {3.180}
$$

请注意，式（3.180）的有效性依靠两个相互矛盾的假设：区域  $\mathcal{R}$  足够小，以至于密度在该区域内近似恒定；但区域  $\mathcal{R}$  又足够大（与该区域内密度的值相关），使得落入该区域的点的数量  $K$  足以使二项分布变得十分尖锐。

我们可以通过两种不同的方式利用式（3.180）。一种是固定  $K$  并根据数据确定  $V$  的值，这将引出后面讨论的  $K$  近邻技术；另一种是固定  $V$  并根据数据确定  $K$  的值，从而引出核密度估计技术。可以证明，当  $N \to \infty$  时，只要  $V$  随着  $N$  以合适的速率缩小，而  $K$  随着  $N$  以合适的速率增长， $K$  近邻密度估计和核密度估计都会收敛到真实概率密度（Duda and Hart, 1973）。

下面详细讨论核密度估计技术。首先，对于希望确定概率密度的点  $x$  ，将区域  $\mathcal{R}$  取为以点  $x$  为中心的超小立方体。为了计算落入该区域的点的数量  $K$  ，定义如下函数：

$$
k (\boldsymbol {u}) = \left\{ \begin{array}{l l} 1, & \left| u _ {i} \right| \leqslant 1 / 2, i = 1, \dots , D \\ 0, & \text {其 他} \end{array} \right. \tag {3.181}
$$

它表示以原点为中心的单位立方体。函数  $k(\pmb{u})$  是一个核函数，在这里，它又称为Parzen窗（Parzen window）。根据式（3.181），如果数据点  $\pmb{x}_n$  位于以点  $\pmb{x}$  为中心、边长为  $h$  的立方体内，那么量  $k((\pmb{x} - \pmb{x}_n) / h)$  将为1，否则为0。因此，位于该立方体内的数据点的总数为

$$
K = \sum_ {n = 1} ^ {N} k \left(\frac {\boldsymbol {x} - \boldsymbol {x} _ {n}}{h}\right) \tag {3.182}
$$

将式（3.182）代入式（3.180），可得点  $\pmb{x}$  处估计的密度为

$$
p (\boldsymbol {x}) = \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {1}{h ^ {D}} k \left(\frac {\boldsymbol {x} - \boldsymbol {x} _ {n}}{h}\right) \tag {3.183}
$$

这里使用了  $D$  维空间中边长为  $h$  的超小立方体的体积公式  $V = h^{D}$  。利用函数  $k(\pmb{u})$  的对称性，我们可以不再将其视为以点  $\pmb{x}$  为中心的单个立方体，而是视为以  $N$  个数据点  $\pmb{x}_n$  为中心的  $N$  个立方体的总和，进而重新解释这个方程。

就目前情况来看，核密度估计[式（3.183）]将遇到与直方图方法相同的一个问题，即存在人为的不连续性，此处体现在立方体的边界上。如果我们选择更平滑的核函数，

则可以获得更平滑的密度模型。我们通常选择使用高斯函数，导出的核密度模型为

$$
p (\boldsymbol {x}) = \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {1}{\left(2 \pi h ^ {2}\right) ^ {D / 2}} \exp \left\{- \frac {\| \boldsymbol {x} - \boldsymbol {x} _ {n} \| ^ {2}}{2 h ^ {2}} \right\} \tag {3.184}
$$

其中  $h$  代表所使用的高斯分量的标准差。因此，我们得到密度模型的方式是，首先在每个数据点上放置一个高斯分布，并将整个数据集的贡献相加，然后除以  $N$  来正确地归一化密度。在图3.14中，核密度模型［式（3.184）]被应用于之前展示直方图方法的数据集。正如预期的那样，我们可以看到参数  $h$  充当了平滑参数的角色，并且 $h$  较小时对噪声敏感，与  $h$  较大时过度平滑之间存在权衡。同样，类似于直方图密度估计中的分箱选择或曲线拟合中所使用多项式的阶数选择，  $h$  的优化是模型复杂性的问题。

![](img/33b72c24c21899ff2094201a2540ca504738bf6ef835264ba5ec2a9df4d435f2.jpg)  
图3.14 核密度模型[式（3.184）]的图示。这里使用与图3.13中用于展示直方图方法相同的数据集。我们可以看到  $h$  充当了平滑参数的角色，如果  $h$  设置得太小（顶部图），则结果是噪声很大的核密度模型；而如果  $h$  设置得太大（底部图），那么所生成数据的底层分布（绿色曲线）的双峰性质就会被淡化。通过对  $h$  取某个中间值，可以获得最佳的核密度模型（中间图）

我们可以在式（3.138）中选择任何其他核函数  $k(\pmb{u})$  ，只要满足以下两个条件即可。

$$
k (\boldsymbol {u}) \geqslant 0 \tag {3.185}
$$

$$
\int k (\boldsymbol {u}) \mathrm {d} \boldsymbol {u} = 1 \tag {3.186}
$$

这两个条件确保了得到的概率分布在任何地方都是非负的，并且积分为1。由式（3.183）给出的这类密度模型称为核密度估计器或Parzen估计器。它们有一个很大的优点，就是在“训练”阶段不涉及计算，因为只需要将训练集存储起来即可。然而，这同时也是这类密度模型的一个巨大缺点，因为评估密度的计算成本会随着数据集的增大而线性增长。

### 3.5.3 最近邻

使用核方法进行密度估计的一个难点在于，对所有核来说控制核宽度的参数  $h$  是固定的。在数据密度较高的区域，较大的  $h$  值可能导致过度平滑，并且可能淡化从数

据中本应提取出的结构。然而，减小  $h$  值可能会导致在其他数据密度较低的区域产生噪声估计。因此， $h$  的最佳选择可能取决于数据空间中的位置。这个问题可以通过使用最近邻方法进行密度估计来解决。

因此，下面我们重新回到局部密度估计的一般结果[式（3.180）]。我们不使用固定  $V$  值并从数据中确定  $V$  值的方式，而是考虑固定  $K$  值并使用数据找到适当的  $V$  值。为此，对于希望估计密度  $p(\boldsymbol{x})$  的点  $\boldsymbol{x}$ ，考虑以点  $\boldsymbol{x}$  为中心的小球，并允许小球的半径增长，直至恰好包含  $K$  个数据点。然后，密度  $p(\boldsymbol{x})$  的估计由式（3.180）给出，其中  $V$  设置为小球的体积。这种技术称为  $K$  近邻（ $K$  nearest neighbor）。图3.15展示了在使用与图3.13和图3.14中相同的数据集的情况下，选择不同参数  $K$  时的  $K$  近邻密度估计。我们可以看到  $K$  值决定了模型的平滑程度，而且同样存在一个既不太大也不太小的最佳  $K$  值。注意  $K$  近邻产生的模型并不是一个真正的密度模型，因为其在整个空间上的积分是发散的（见习题3.38）。

![](img/f70edca25286ac09c7acf3eaa69d9f3e6b9cee7502ea69f4007a94bc2a335bae.jpg)  
图3.15  $K$  近邻密度估计的图示，使用与图3.13和图3.14中相同的数据集。可以看到参数  $K$  控制了模型的平滑程度，因此较小的  $K$  值会导致噪声极大的密度模型（顶部图），而较大的  $K$  值则会平滑掉所生成数据的真实分布（绿色曲线）的双峰性质（底部图）

下面通过展示  $K$  近邻技术如何扩展到分类问题来结束本章。为此，我们需要将  $K$  近邻密度估计分别应用于每个类别，然后利用贝叶斯定理。假设数据集包含  $N_{k}$  个属于类别  $\mathcal{C}_k$  的点，总共有  $N$  个点，即  $\sum_{k}N_{k} = N$  。如果希望对一个新点  $\pmb{x}$  进行分类，则可以在这个点的周围绘制一个恰好包含  $K$  个点的球体，这里不考虑它们的类别。假设这个球体的体积为  $V$  ，并且包含来自类别  $\mathcal{C}_k$  的  $K_{k}$  个点。式（3.180）提供了与每个类别相关联的密度估计值：

$$
p \left(\boldsymbol {x} \mid \mathcal {C} _ {k}\right) = \frac {K _ {k}}{N _ {k} V} \tag {3.187}
$$

类似地，无条件密度由下式给出：

$$
p (\boldsymbol {x}) = \frac {K}{N V} \tag {3.188}
$$

类别先验由下式给出：

$$
p \left(\mathcal {C} _ {k}\right) = \frac {N _ {k}}{N} \tag {3.189}
$$

我们现在可以使用贝叶斯定理结合式（3.187）～式（3.189）得到类别成员的后验概率：

$$
p \left(\mathcal {C} _ {k} \mid \boldsymbol {x}\right) = \frac {p \left(\boldsymbol {x} \mid \mathcal {C} _ {k}\right) p \left(\mathcal {C} _ {k}\right)}{p (\boldsymbol {x})} = \frac {K _ {k}}{K} \tag {3.190}
$$

我们可以通过将测试点  $x$  分配给具有最大后验概率的类别来最小化错误分类的概率，这对应于  $K_{k} / K$  的最大值。因此，在对一个新点进行分类时，我们首先需要识别训练集中与新点最近的  $K$  个点，然后将新点分配给在这组点中拥有最多代表的类别。如果多个类别拥有的代表数量一致，则可以将新点随机分配给其中一个类别。  $K = 1$  的特殊情况称为最近邻法则，因为测试点被简单地分配给了训练集中离新点最近的点所属的那个类别。图3.16对这些概念进行了说明。

![](img/a6b018c5f4d2ccfad798034bc0ee709c82b69952a51b7e6d0cd7d364566949e4.jpg)  
(a)

![](img/9c0481930e1bd0e87018adf5913e957183d727ccf37400a82111f4f0ea8e23a7.jpg)  
(b)  
图3.16 (a) 在  $K$  近邻分类器中，一个用黑色菱形表示的新点根据  $K$  个最接近的训练数据点被归类为类别成员最多的那个类别，在本例中  $K = 3$  。(b) 在最近邻  $(K = 1)$  分类器中，所得到的决策边界由超平面组成，每个超平面是由来自不同类别点对的垂直平分线构造的

最近邻（ $K = 1$ ）分类器的一个有趣特性是，在  $N \to \infty$  的情况下，错误率永远不会超过最优分类器（即使用真实类别分布的分类器）的最小可实现错误率的两倍（Cover and Hart, 1967）。

到目前为止，我们已经讨论了  $K$  近邻方法和核密度估计，它们都需要存储整个训练集。如果数据集很大，计算成本将十分高昂。为了抵消这种影响，可以构建基于树的搜索结构，引入一些一次性的额外计算，以便高效地找到（近似的）近邻，而无须对整个数据集进行全量搜索。然而，这些非参数化方法的使用场景仍然受到严重制约。另外，我们已经看到简单的参数化模型仅能表示非常有限的分布形式。因此我们需要找到一些非常灵活的密度模型，它们的复杂度可以独立于训练集的大小而进行控制，深度神经网络可以实现这一点。

## 习题

3.1（ $\star$ ）验证伯努利分布［式（3.2）］满足下列性质：

$$
\sum_ {x = 0} ^ {1} p (x \mid \mu) = 1 \tag {3.191}
$$

$$
\mathbb {E} [ x ] = \mu \tag {3.192}
$$

$$
\operatorname {v a r} [ x ] = \mu (1 - \mu) \tag {3.193}
$$

并证明符合伯努利分布的随机二元变量  $x$  的熵  $H[x]$  由下式给出：

$$
H [ x ] = - \mu \ln \mu - (1 - \mu) \ln (1 - \mu) \tag {3.194}
$$

3.2（ $\star \star$ ）式（3.2）给出的伯努利分布形式在两个  $x$  值之间不对称。在某些情况下，使用另一种等价的形式会更方便，其中  $x \in \{-1, 1\}$ ，此时该分布可以写成

$$
p \{x \mid \mu \} = \left(\frac {1 - \mu}{2}\right) ^ {(1 - x) / 2} \left(\frac {1 + \mu}{2}\right) ^ {(1 + x) / 2} \tag {3.195}
$$

其中  $\mu \in \{-1,1\}$  。证明式（3.195）给出的分布是归一化的，并计算它的均值、方差和熵。

3.3（ $\star \star$ ）证明二项分布[式（3.9）]是归一化的。为此，首先使用定义式（3.10），即从总数为  $N$  的物体中选择  $m$  个相同物体的组合数量，证明下式：

$$
\binom {N} {m} + \binom {N} {m - 1} = \binom {N + 1} {m} \tag {3.196}
$$

然后利用上述结论，通过归纳证明以下结论：

$$
(1 + x) ^ {N} = \sum_ {m = 0} ^ {N} \binom {N} {m} x ^ {m} \tag {3.197}
$$

这个称为二项式定理（binomial theorem）的结论对所有实数  $x$  都成立。最后，证明二项分布是归一化的，从而使

$$
\sum_ {m = 0} ^ {N} \binom {N} {m} \mu^ {m} (1 - \mu) ^ {N - m} = 1 \tag {3.198}
$$

这可以通过首先从求和中提取因子  $(1 - \mu)^{N}$ ，然后利用二项式定理来完成。

3.4（ $\star \star$ ）证明二项分布的均值由式（3.11）给出。为此，在归一化条件式（3.198）的两边对  $\mu$  求导，并重排以获得  $\mu$  的均值表达式。类似地，通过对式（3.198）关于  $\mu$  进行二次求导，并利用式（3.11）给出的二项分布均值的结果，证明二项分布方差的结果[式（3.12）]。  
3.5（ $\star$ ）证明多元高斯分布式（3.26）的众数为  $\mu$ 。  
3.6（ $\star \star$ ）假设  $x$  服从均值为  $\mu$ 、协方差为  $\pmb{\Sigma}$  的高斯分布。证明线性变换后的变量  $Ax + b$  也服从高斯分布，并给出它的均值和协方差。  
3.7（ $\star \star \star$ ）证明两个高斯分布  $q(\pmb{x}) = \mathcal{N}(\pmb{x}|\pmb{\mu}_q, \pmb{\Sigma}_q)$  和  $p(\pmb{x}) = \mathcal{N}(\pmb{x}|\pmb{\mu}_p, \pmb{\Sigma}_p)$  之间的Kullback-Leibler散度为

$$
\operatorname {K L} (q (\boldsymbol {x}) \| p (\boldsymbol {x})) = \frac {1}{2} \left\{\ln \frac {\left| \boldsymbol {\Sigma} _ {p} \right|}{\left| \boldsymbol {\Sigma} _ {q} \right|} - D + \operatorname {t r} \left(\boldsymbol {\Sigma} _ {p} ^ {- 1} \boldsymbol {\Sigma} _ {q}\right) + \left(\boldsymbol {\mu} _ {p} - \boldsymbol {\mu} _ {q}\right) ^ {\mathrm {T}} \boldsymbol {\Sigma} _ {p} ^ {- 1} \left(\boldsymbol {\mu} _ {p} - \boldsymbol {\mu} _ {q}\right) \right\} (3. 1 9 9)
$$

其中  $\operatorname{tr}(\cdot)$  表示矩阵的迹， $D$  为  $\pmb{x}$  的维度。

3.8（ $\star \star$ ）这个习题表明，对于给定的协方差，具有最大熵的多元分布是高斯分布。分布  $p(x)$  的熵由下式给出：

$$
H [ \boldsymbol {x} ] = - \int p (\boldsymbol {x}) \ln p (\boldsymbol {x}) d \boldsymbol {x} \tag {3.200}
$$

我们希望在所有分布  $p(\boldsymbol{x})$  中最大化  $H[x]$ ，约束如下：满足  $p(x)$  是归一化的，且具有特定的均值和协方差，使得

$$
\int p (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} = 1 \tag {3.201}
$$

$$
\int p (x) x \mathrm {d} x = \mu \tag {3.202}
$$

$$
\int p (\boldsymbol {x}) (\boldsymbol {x} - \boldsymbol {\mu}) (\boldsymbol {x} - \boldsymbol {\mu}) ^ {\mathrm {T}} \mathrm {d} \boldsymbol {x} = \Sigma \tag {3.203}
$$

通过对式（3.200）进行变分最大化，并使用拉格朗日乘子来确保约束条件式（3.201）～式（3.203），证明最大似然分布由高斯分布式（3.26）给出。

3.9（ $\star \star \star$ ）证明多元高斯分布  $\mathcal{N}(\pmb{x}|\pmb{\mu},\pmb{\Sigma})\partial$  的熵由下式给出：

$$
H [ \boldsymbol {x} ] = \frac {1}{2} \ln | \boldsymbol {\Sigma} | + \frac {D}{2} (1 + \ln (2 \pi)) \tag {3.204}
$$

其中  $D$  为  $\pmb{x}$  的维度。

3.10（ $\star \star \star$ ）考虑两个随机变量  $x_{1}$  和  $x_{2}$ ，它们的分布都是高斯分布，均值分别为  $\mu_{1}$  和  $\mu_{2}$ ，精度分别为  $\tau_{1}$  和  $\tau_{2}$ 。推导变量  $x = x_{1} + x_{2}$  的微分熵表达式。为此，首先利用以下关系找到  $x$  的分布：

$$
p (x) = \int_ {- \infty} ^ {\infty} p (x \mid x _ {2}) p (x _ {2}) d x _ {2} \tag {3.205}
$$

并完成指数中的平方项。然后观察到这表示两个高斯分布的卷积，而卷积本身也是高斯分布。最后利用式（2.99）中一元高斯分布的熵的计算结果。

3.11（ $\star$ ）考虑式（3.26）给出的多元高斯分布。通过将精度矩阵（逆协方差矩阵）写成对称矩阵和反对称矩阵的和，证明反对称项不会出现在高斯分布的指数中，所以精度矩阵可以不失一般性地假定为对称的。又因为对称矩阵的逆矩阵也是对称的（见习题3.16），所以协方差矩阵也可以不失一般性地选为对称的。

3.12（ $\star \star \star$ ）考虑一个实对称矩阵  $\pmb{\Sigma}$ ，其特征方程由式（3.28）给出。通过取这个方程的复共轭，并减去原方程，然后计算与特征向量  $\pmb{u}_i$  的内积，证明特征值  $\lambda_i$  是实数。类似地，利用  $\pmb{\Sigma}$  的对称性，证明当  $\lambda_j \neq \lambda_i$  时，特征向量  $\pmb{u}_i$  和  $\pmb{u}_j$  是正交的。

最后，请证明特征向量集可以不失一般性地选择为正交的，使得它们满足式（3.29），即使一些特征值为零。

3.13（ $\star \star$ ）证明式（3.28）这样一个具有特征向量方程的实对称矩阵  $\pmb{\Sigma}$  可以表示为形如式（3.31）的特征向量的展开形式，且系数由特征值给出。类似地，证明逆矩阵  $\pmb{\Sigma}^{-1}$  具有形如式（3.32）的表示形式。

3.14（ $\star \star$ ）正定矩阵  $\pmb{\Sigma}$  可以定义如下：对于任意实向量  $\pmb{a}$ ，其二次型

$$
\boldsymbol {a} ^ {\mathrm {T}} \Sigma \boldsymbol {a} \tag {3.206}
$$

是正的。证明  $\pmb{\Sigma}$  为正定的一个充要条件是， $\pmb{\Sigma}$  的所有特征值  $\lambda_{i}$  都是正的，其中特征值由式（3.28）定义。

3.15（ $\star$ ）证明一个大小为  $D \times D$  的实对称矩阵具有  $D(D + 1) / 2$  个独立的参数。  
3.16（ $\star$ ）证明对称矩阵的逆矩阵也是对称的。  
3.17（ $\star \star$ ）通过使用特征向量展开式（3.31）对坐标系进行对角化，证明与常数马氏距离  $\Delta$  对应的超椭球的体积为

$$
V _ {D} \left| \boldsymbol {\Sigma} \right| ^ {1 / 2} \Delta^ {D} \tag {3.207}
$$

其中  $V_{D}$  是  $D$  维单位球的体积，而马氏距离由式（3.27）定义。

3.18（ $\star \star$ ）利用定义式（3.61），并通过对它的两边乘以下列矩阵，证明恒等式（3.60）成立。

$$
\left( \begin{array}{l l} A & B \\ C & D \end{array} \right) \tag {3.208}
$$

3.19（ $\star \star \star$ ）3.2.4小节和3.2.5小节介绍了多元高斯分布的条件分布和边缘分布。更一般地，可以将  $x$  的分量划分为三组—— $x_{a}$ 、 $x_{b}$  和  $x_{c}$ ，对应地，将均值向量  $\pmb{\mu}$  和协方差矩阵  $\pmb{\Sigma}$  也按照如下形式进行划分：

$$
\boldsymbol {\mu} = \left( \begin{array}{l} \boldsymbol {\mu} _ {a} \\ \boldsymbol {\mu} _ {b} \\ \boldsymbol {\mu} _ {c} \end{array} \right), \boldsymbol {\Sigma} = \left( \begin{array}{l l l} \boldsymbol {\Sigma} _ {a a} & \boldsymbol {\Sigma} _ {a b} & \boldsymbol {\Sigma} _ {a c} \\ \boldsymbol {\Sigma} _ {b a} & \boldsymbol {\Sigma} _ {b b} & \boldsymbol {\Sigma} _ {b c} \\ \boldsymbol {\Sigma} _ {c a} & \boldsymbol {\Sigma} _ {c b} & \boldsymbol {\Sigma} _ {c c} \end{array} \right) \tag {3.209}
$$

利用3.2节的结论，给出条件分布  $p(\boldsymbol{x}_a \mid \boldsymbol{x}_b)$  的表达式，其中  $\boldsymbol{x}_c$  已被边缘化。

3.20（ $\star \star$ ）线性代数中一个非常有用的结论是伍德伯里（Woodbury）矩阵求逆公式，即

$$
\left(\boldsymbol {A} + \boldsymbol {B C D}\right) ^ {- 1} = \boldsymbol {A} ^ {- 1} - \boldsymbol {A} ^ {- 1} \boldsymbol {B} \left(\boldsymbol {C} ^ {- 1} + \boldsymbol {D A} ^ {- 1} \boldsymbol {B}\right) ^ {- 1} \boldsymbol {D A} ^ {- 1} \tag {3.210}
$$

通过对式（3.210）的两边乘以  $\left(A + BCD\right)$ ，证明这个结论的正确性。

3.21（ $\star$ ）假设  $x$  和  $z$  是两个独立的随机向量，故  $p(x, z) = p(x)p(z)$ 。证明这两个随机向量的和  $y = x + z$  的均值等于其中每个随机向量的均值之和。类似地，证明

$\pmb{y}$  的协方差矩阵等于  $\pmb{x}$  与  $\pmb{z}$  的协方差矩阵之和。

3.22 （ $\star \star \star$ ）考虑变量

$$
z = \left( \begin{array}{l} x \\ y \end{array} \right) \tag {3.211}
$$

的联合分布，它的均值和协方差分别由式（3.92）和式（3.89）给出。利用式（3.76）和式（3.77），证明边缘分布  $p(\boldsymbol{x})$  由式（3.83）给出。类似地，利用式（3.65）和式（3.66），证明条件分布  $p(\boldsymbol{y}|\boldsymbol{x})$  由式（3.84）给出。

3.23（ $\star \star$ ）请利用分块矩阵求逆公式[式（3.60）]，证明精度矩阵[式（3.88）]的逆矩阵由协方差矩阵[式（3.89）]给出。  
3.24（ $\star \star$ ）从式（3.91）开始，利用式（3.89），验证式（3.92）。  
3.25（ $\star \star$ ）考虑两个多维随机向量  $x$  和  $z$ ，它们分别服从高斯分布  $p(x) = \mathcal{N}(x|\mu_x,\Sigma_x)$  和  $p(z) = \mathcal{N}(z|\mu_z,\Sigma_z)$ ，定义  $y = x + z$ 。通过考虑由边缘分布  $p(x)$  和条件分布  $p(y|x)$  的乘积组成的线性高斯模型，并利用式（3.93）和式（3.94），证明边缘分布  $p(y)$  由下式给出：

$$
p (\boldsymbol {y}) = \mathcal {N} \left(\boldsymbol {y} \mid \mu_ {x} + \mu_ {z}, \Sigma_ {x} + \Sigma_ {z}\right) \tag {3.212}
$$

3.26（ $\star \star \star$ ）本习题和下一习题提供了关于操作线性高斯模型中出现的二次型的实践，它们也可以作为对正文中结论的独立检查。考虑由式（3.83）和式（3.84）给出的边缘分布和条件分布定义的联合分布  $p(x, y)$  。通过检查联合分布指数中的二次型，并使用3.2节讨论的“完成平方”技术，给出边缘分布  $p(y)$  的均值表达式和协方差表达式，其中变量  $x$  已经被积分消除。为此，利用伍德伯里（Woodbury）矩阵求逆公式[式（3.210）]，验证这些结果与式（3.93）和式（3.94）的一致性。  
3.27（ $\star \star \star$ ）考虑与习题3.26中相同的联合分布，但使用“完成平方”技术来找到条件分布  $p(x|y)$  的均值表达式和协方差表达式。再次验证这些结果与式（3.95）和式（3.96）的一致性。  
3.28（ $\star \star$ ）为了找到多元高斯分布的协方差矩阵的最大似然解，需要最大化对于  $\pmb{\Sigma}$  的对数似然函数[式（3.102）]，注意协方差矩阵必须是对称且正定的。此处忽略这些约束，直接进行最大化。利用附录A中的式（A.21）、式（A.26）和式（A.28），证明使对数似然函数[式（3.102）]最大化的协方差矩阵  $\pmb{\Sigma}$  是由样本协方差[式（3.106）]给出的。注意最终的结果必然是对称且正定的（假设样本协方差是非奇异的）。  
3.29  $(\star \star)$  使用结果式（3.42）证明式（3.46）。然后，利用结果式（3.42）和式（3.46）证明

$$
\mathbb {E} \left[ \boldsymbol {x} _ {n} \boldsymbol {x} _ {m} ^ {\mathrm {T}} \right] = \mu \mu^ {\mathrm {T}} + I _ {n m} \Sigma \tag {3.213}
$$

其中  $x_{n}$  表示从均值为  $\mu$  、协方差为  $\Sigma$  的高斯分布中采样得到的数据点，而  $I_{nm}$  表示单位矩阵中位于  $(n,m)$  的元素。最后证明式（3.108）。

3.30（ $\star$ ）本章在讨论周期变量时使用的各种三角恒等式可以很容易地从下面的关系中得到证明：

$$
\exp (\mathrm {i} A) = \cos A + \mathrm {i} \sin A \tag {3.214}
$$

其中  $\mathrm{i} = \sqrt{-1}$  。通过考虑恒等式

$$
\exp (\mathrm {i} A) \exp (- \mathrm {i} A) = 1 \tag {3.215}
$$

证明式（3.127）。类似地，使用恒等式

$$
\cos (A - B) = \Re \exp \left\{\mathrm {i} (A - B) \right\} \tag {3.216}
$$

证明式（3.128），其中  $\Re$  表示实部。最后，使用  $\sin (A - B) = \Im \exp \left\{\mathrm{i}(A - B)\right\}$  （其中  $\Im$  表示虚部）证明式（3.133）。

3.31（ $\star \star$ ）对于较大的  $m$ ，冯·米塞斯分布[式（3.129）]将会在众数  $\theta_0$  处变得十分尖锐。通过定义  $\xi = m^{1/2} (\theta - \theta_0)$ ，并利用余弦函数的泰勒展开式，即

$$
\cos \alpha = 1 - \frac {\alpha^ {2}}{2} + O \left(\alpha^ {4}\right) \tag {3.217}
$$

证明当  $m \to \infty$  时，冯·米塞斯分布趋向于高斯分布。

3.32（ $\star$ ）利用三角恒等式（3.133），证明式（3.132）对于  $\theta_0$  的解由式（3.134）给出。  
3.33（ $\star$ ）通过计算式（3.129）给出的冯·米塞斯分布的一阶导数和二阶导数，并利用  $I_0(m) > 0 (m > 0)$ ，证明该分布的极大值发生在  $\theta = \theta_0$  处，极小值发生在  $\theta = \theta_0 + \pi (\bmod 2\pi)$  处。  
3.34（ $\star$ ）通过利用结果式（3.118），并结合式（3.134）和三角恒等式（3.128），证明冯·米塞斯分布的聚焦参数的最大似然解  $m_{\mathrm{ML}}$  满足  $A(m_{\mathrm{ML}}) = \overline{r}$ ，其中  $\overline{r}$  是观测值的均值的半径。如图3.9所示，它可以视为二维欧氏平面上的单位向量。  
3.35（ $\star$ ）验证多元高斯分布可以表示为指数族分布的形式[式（3.138）]，并为  $\pmb{\eta}$ 、 $\pmb{u}(\pmb{x})$ 、 $h(\pmb{x})$  和  $g(\pmb{\eta})$  推导类似于式（3.164）～式（3.167）的表达式。  
3.36（ $\star$ ）式（3.172）表明，指数族分布中  $\ln g(\pmb{\eta})$  的负梯度由  $\pmb{u}(\pmb{x})$  的期望给出。通过对式（3.139）求二阶导数，证明

$$
- \nabla V \ln g (\boldsymbol {\eta}) = \mathbb {E} \left[ \boldsymbol {u} (\boldsymbol {x}) \boldsymbol {u} (\boldsymbol {x}) ^ {\mathrm {T}} \right] - \mathbb {E} [ \boldsymbol {u} (\boldsymbol {x}) ] \mathbb {E} \left[ \boldsymbol {u} (\boldsymbol {x}) ^ {\mathrm {T}} \right] = \operatorname {c o v} [ \boldsymbol {u} (\boldsymbol {x}) ] \tag {3.218}
$$

3.37（ $\star \star$ ）考虑一个类似直方图的密度模型，其中空间  $\pmb{x}$  被划分为固定的区域，密度  $p(x)$  在第  $i$  个区域上取常数值  $h_i$ 。第  $i$  个区域的体积表示为  $\Delta_{i}$ 。假设有一个集

合，里面包含  $N$  个  $\pmb{x}$  的观测值，其中  $n_i$  个观测值落在第  $i$  个区域。利用拉格朗日乘子确保密度的归一化约束，并推导  $\{h_i\}$  的最大似然估计量的表达式。

3.38（★）证明  $K$  近邻密度模型定义了一个反常分布，其在整个空间上的积分是发散的。
