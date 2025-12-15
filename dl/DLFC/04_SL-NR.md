# 第4章 单层网络：回归

![](img/18aa573525c59742ae1f947095c295b252c8ea63a0b52801365ed950cca80278.jpg)

在之前的多项式曲线拟合问题中，我们曾简单了解过线性回归（参见1.2节）。而在本章中，我们将继续通过它的框架来讨论神经网络背后的一些基本思想。你将看到，线性回归模型对应简单的具有单层可学习参数的神经网络。虽然单层网络的应用场景非常有限，但它们具有一些简单的解析性质，并为后续引入许多核心概念提供了一个优秀的框架，这些概念将为我们在后续章节中讨论深度神经网络奠定基础。

## 4.1 线性回归

给定  $D$  维输入变量  $\pmb{x}$ ，回归的目标是预测一个或多个连续的目标变量  $t$ 。通常我们会有一个包含  $N$  个观测值的训练数据集  $\{x_{n}\}$ ，其中  $n = 1, \dots, N$ ，以及相应的目标值  $\{t_{n}\}$ ，目标是为新的  $\pmb{x}$  值预测对应的  $t$  值。为此，我们定义了函数  $y(x, w)$  来预测新的输入变量  $\pmb{x}$  所对应的  $t$  值，其中  $w$  表示可以从训练数据中学习的参数向量。

最简单的回归模型是输入变量的线性组合：

$$
y (\boldsymbol {x}, \boldsymbol {w}) = w _ {0} + w _ {1} x _ {1} + \dots + w _ {D} x _ {D} \tag {4.1}
$$

其中  $\pmb{x} = (x_{1},\dots ,x_{D})^{\mathrm{T}}$  。线性回归（linear regression）有时特指这种形式的模型。这种模型的关键性质在于其不仅是参数  $w_{0},\dots ,w_{D}$  的线性函数，而且也是输入变量  $x_{i}$  的线性函数。然而，正是其作为输入变量  $x_{i}$  的线性函数这一特性，给模型带来了显著的限制。

### 4.1.1 基函数

我们可以通过输入变量的固定非线性函数的线性组合来扩展式（4.1）定义的模型：

$$
y (\boldsymbol {x}, \boldsymbol {w}) = w _ {0} + \sum_ {j = 1} ^ {M - 1} w _ {j} \phi_ {j} (\boldsymbol {x}) \tag {4.2}
$$

其中  $\phi_j(x)$  称为基函数。如果用  $M - 1$  表示索引  $j$  的最大值，则模型中参数的总数为  $M$ 。

参数  $w_{0}$  代表数据中一个任意的固定偏移量，有时称为偏置参数（bias）（注意不要与统计学意义上的偏差弄混淆）。通过定义一个额外的虚拟基函数  $\phi_0(x) = 1$  ，式（4.2）可以统一为（参见4.3节）

$$
y (\boldsymbol {x}, \boldsymbol {w}) = \sum_ {j = 0} ^ {M - 1} w _ {j} \phi_ {j} (\boldsymbol {x}) = \boldsymbol {w} ^ {\mathrm {T}} \phi (\boldsymbol {x}) \tag {4.3}
$$

其中  $\pmb{w} = (w_{0},\dots ,w_{M - 1})^{\mathrm{T}}$  ，  $\phi = (\phi_0,\dots ,\phi_{M - 1})^{\mathrm{T}}$  。我们可以用神经网络图来表示该模型[式（4.3）]，如图4.1所示。

通过使用非线性基函数，函数  $y(x, w)$  变为关于输入变量  $x$  的非线性函数。尽管如此，形如式（4.2）的函数仍然称为线性模型，因为它们关于  $w$  是线性的。正是这种关于参数线性的性质大大简化了对这类模型的分析。然而，它也带来了一些显著的限制

（参见6.1节）。

![](img/d75185086876b80e8d018c5d28ec348595e4928030cbf91e2d54ae5d18be0f04.jpg)  
图4.1 线性回归模型[式（4.3）]可以表示为一个简单的由单层参数构成的神经网络。这里的每个基函数  $\phi_j(x)$  用一个输入节点表示，实心节点表示“偏置”基函数  $\phi_0$  ，函数  $y(x, w)$  用输出节点表示。每个参数  $w_j$  由一条连接相应基函数和输出的线表示

在深度学习出现之前，机器学习中通常会对输入变量  $x$  进行某种形式的固定预处理，称为特征提取（feature extraction），其可以用一组基函数  $\{\phi_j(x)\}$  来表示。特征提取的目标是选择一组足够强大的基函数，从而可以使用简单的网络模型解决学习任务。遗憾的是，除了最简单的应用之外，人工选取合适的基函数是非常困难的。深度学习仅通过数据集学习所需的数据非线性变换来避开这个问题。

之前讨论使用多项式进行曲线拟合时，我们

已经见到了一个回归问题的例子（参见第1章）。如果我们考虑单个输入变量  $x$ ，并选择由  $\phi_j(x) = x^j$  定义的基函数，则多项式函数式(1.1)可以表示为式(4.3)的形式。此外，基函数还有其他的形式，例如：

$$
\phi_ {j} (x) = \exp \left\{- \frac {\left(x - \mu_ {j}\right) ^ {2}}{2 s ^ {2}} \right\} \tag {4.4}
$$

其中  $\mu_{j}$  控制基函数在输入空间中的位置，参数  $s$  控制它们的空间尺度。这种形式的基函数通常叫作高斯基函数，虽然这些基函数在形式上可能类似于高斯分布，但它们在使用时并不需要概率化的解释。特别地，这些基函数会被乘以可学习参数  $w_{j}$ ，因此归一化系数不重要。

基函数的另一种形式是 sigmoid 型基函数：

$$
\phi_ {j} (x) = \sigma \left(\frac {x - \mu_ {j}}{s}\right) \tag {4.5}
$$

其中  $\sigma(a)$  是按如下方式定义的 sigmoid 函数：

$$
\sigma (a) = \frac {1}{1 + \exp (- a)} \tag {4.6}
$$

我们可以等价地使用tanh函数，因为其可以通过  $\tanh (a) = 2\sigma (2a) - 1$  与sigmoid函数相关联。sigmoid函数的一般线性组合在某种意义上等价于tanh函数的一般线性组合，故它们可以表示同一类输入-输出函数（见习题4.3）。不同的基函数选择如图4.2所示。

![](img/87924d28f91546d411edba71c536961ffe28f492842eeaf9df1899c20d8e4b6b.jpg)  
图4.2 左图是多项式，中间图是式（4.4）形式的高斯基函数，右图是式（4.5）形式的sigmoid型基函数

![](img/23f4082b1c9ef71b8fff4674e4ad6599c4a0a2b74590a4c66f9d71f710ef8beb.jpg)

![](img/53052650cf2cce8f2a949453384ddeaecbd2f0fb6bc99cef5a5f55656701b476.jpg)

基函数的另一种选择是傅里叶基函数，它对应正弦函数的展开。每个基函数代表无限空间范围内一个特定的频率。相比之下，位于输入空间有限区域的基函数必然包含不同空间频率的频谱。在信号处理的相关应用中，通常需要考虑在空间和频率上都局部化的一类基函数，即小波（wavelet）基函数（Ogden, 1997; Mallat, 1999; Vidakovic, 1999）。为了简化应用，这些函数被定义为相互正交的。当输入值在规则的格（lattice）上时，小波基函数最为适用，例如时间序列中连续的时间点或图像中的

像素。

然而，本章后续大部分的讨论与基函数的选择无关，所以除了数值说明外，我们不会明确基函数的具体形式。此外，为了让符号标记简单，我们将重点关注单目标变量  $t$  的情况，也会简要概述处理多目标变量时所需要做的修改（参见4.1.7小节）。

### 4.1.2 似然函数

我们已经通过最小化平方和误差函数解决了多项式拟合数据的问题，并且我们还发现，平方和误差函数可以用高斯噪声模型假设下的最大似然解来导出（参见1.2节）。下面我们将更详细地介绍最小二乘法，以及它与最大似然的关系。

如前所述，假设目标变量  $t$  由具有加性高斯噪声的确定性函数  $y(x, w)$  给出，即

$$
t = y (x, w) + \varepsilon \tag {4.7}
$$

其中  $\varepsilon$  是方差为  $\sigma^2$  的零均值高斯随机变量。因此，我们可以写出

$$
p (t \mid x, w, \sigma^ {2}) = \mathcal {N} (t \mid y (x, w), \sigma^ {2}) \tag {4.8}
$$

考虑输入数据集  $X = \{x_{1},\dots ,x_{N}\}$  以及对应的目标值  $t_1,\dots ,t_N$  。将目标变量  $\{t_n\}$  构造为列向量，用  $\pmb{\mathfrak{f}}$  表示。假设这些数据点是从分布式（4.8）中独立产生的，我们得到如下似然函数的表达式，它是一个关于可调参数  $\pmb{w}$  和  $\sigma^2$  的函数：

$$
p (\boldsymbol {t} \mid \boldsymbol {X}, \boldsymbol {w}, \sigma^ {2}) = \prod_ {n = 1} ^ {N} \mathcal {N} \left(t _ {n} \mid \boldsymbol {w} ^ {\mathrm {T}} \phi \left(\boldsymbol {x} _ {n}\right), \sigma^ {2}\right) \tag {4.9}
$$

其中用到了式（4.3）。取似然函数的对数，并使用标准形式式（2.49），我们有

$$
\begin{array}{l} \ln p (\boldsymbol {t} \mid X, \boldsymbol {w}, \sigma^ {2}) = \sum_ {n = 1} ^ {N} \ln \mathcal {N} \left(t _ {n} \mid \boldsymbol {w} ^ {\mathrm {T}} \phi (\boldsymbol {x} _ {n}), \sigma^ {2}\right) \\ = - \frac {N}{2} \ln \sigma^ {2} - \frac {N}{2} \ln (2 \pi) - \frac {1}{\sigma^ {2}} E _ {D} (\boldsymbol {w}) \tag {4.10} \\ \end{array}
$$

其中平方和误差函数定义为

$$
E _ {D} (\boldsymbol {w}) = \frac {1}{2} \sum_ {n = 1} ^ {N} \left\{t _ {n} - \boldsymbol {w} ^ {\mathrm {T}} \phi (\boldsymbol {x} _ {n}) \right\} ^ {2} \tag {4.11}
$$

在确定  $w$  时，式（4.10）中的前两项独立于  $w$ ，因此可以视为常数。正如我们在前面所看到的，在高斯噪声分布下，最大化似然函数等价于最小化平方和误差函数[式（4.11）]（参见2.3.4小节）。

### 4.1.3 最大似然

在写出似然函数后，我们可以通过最大似然来确定  $w$  和  $\sigma^2$  。首先对于  $w$  最大化上述似然函数，其对数似然函数[式（4.10）]关于  $w$  的梯度为如下形式：

$$
\nabla_ {w} \ln p (\boldsymbol {t} | X, w, \sigma^ {2}) = \frac {1}{\sigma^ {2}} \sum_ {n = 1} ^ {N} \left\{t _ {n} - w ^ {\mathrm {T}} \phi \left(x _ {n}\right) \right\} \phi \left(x _ {n}\right) ^ {\mathrm {T}} \tag {4.12}
$$

将梯度设置为0，可得

$$
\mathbf {0} = \sum_ {n = 1} ^ {N} t _ {n} \phi \left(\mathbf {x} _ {n}\right) ^ {\mathrm {T}} - \mathbf {w} ^ {\mathrm {T}} \left(\sum_ {n = 1} ^ {N} \phi \left(\mathbf {x} _ {n}\right) \phi \left(\mathbf {x} _ {n}\right) ^ {\mathrm {T}}\right) \tag {4.13}
$$

求解  $\pmb{w}$  可得

$$
\boldsymbol {w} _ {\mathrm {M L}} = \left(\boldsymbol {\Phi} ^ {\mathrm {T}} \boldsymbol {\Phi}\right) ^ {- 1} \boldsymbol {\Phi} ^ {\mathrm {T}} \boldsymbol {t} \tag {4.14}
$$

这称为最小二乘问题的正规方程（normal equation）。其中  $\Phi$  是一个  $N\times M$  的矩阵，称为设计矩阵（design matrix），其元素由  $\Phi_{nj} = \phi_j(x_n)$  给出，即

$$
\boldsymbol {\Phi} = \left( \begin{array}{c c c c} \phi_ {0} \left(\boldsymbol {x} _ {1}\right) & \phi_ {1} \left(\boldsymbol {x} _ {1}\right) & \dots & \phi_ {M - 1} \left(\boldsymbol {x} _ {1}\right) \\ \phi_ {0} \left(\boldsymbol {x} _ {2}\right) & \phi_ {1} \left(\boldsymbol {x} _ {2}\right) & \dots & \phi_ {M - 1} \left(\boldsymbol {x} _ {2}\right) \\ \vdots & \vdots & & \vdots \\ \phi_ {0} \left(\boldsymbol {x} _ {N}\right) & \phi_ {1} \left(\boldsymbol {x} _ {N}\right) & \dots & \phi_ {M - 1} \left(\boldsymbol {x} _ {N}\right) \end{array} \right) \tag {4.15}
$$

其中

$$
\boldsymbol {\Phi} ^ {\dagger} \equiv \left(\boldsymbol {\Phi} ^ {T} \boldsymbol {\Phi}\right) ^ {- 1} \boldsymbol {\Phi} ^ {T} \tag {4.16}
$$

称为矩阵  $\Phi$  的摩尔-彭若斯伪逆（Moore-Penrose pseudo-inverse）（Rao and Mitra, 1971; Golub and Van Loan, 1996）。这可以看作矩阵逆的概念被推广到非方阵的情况。如果  $\Phi$  是方阵且可逆，则利用性质  $(AB)^{-1} = B^{-1}A^{-1}$  可得  $\Phi^{\dagger} \equiv \Phi^{-1}$ 。

至此，我们对偏置参数  $w_{0}$  的作用有了进一步的了解。如果我们显式考虑偏置参数，则误差函数[式（4.11）]就会变成

$$
E _ {D} (\boldsymbol {w}) = \frac {1}{2} \sum_ {n = 1} ^ {N} \left\{t _ {n} - w _ {0} - \sum_ {j = 1} ^ {M - 1} w _ {j} \phi_ {j} (\boldsymbol {x} _ {n}) \right\} ^ {2} \tag {4.17}
$$

使上述误差函数关于  $w_{0}$  的导数为0并求解  $w_{0}$ ，可得

$$
w _ {0} = \bar {t} - \sum_ {j = 1} ^ {M - 1} w _ {j} \overline {{\phi_ {j}}} \tag {4.18}
$$

其中，定义

$$
\bar {t} = \frac {1}{N} \sum_ {n = 1} ^ {N} t _ {n}, \quad \overline {{\phi_ {j}}} = \frac {1}{N} \sum_ {n = 1} ^ {N} \phi_ {j} (x _ {n}) \tag {4.19}
$$

因此，偏置  $w_{0}$  补足了目标值平均值（在训练集上）的加权和与基函数值平均值的加权和之间的差异。

我们还可以使对数似然函数式（4.10）关于方差  $\sigma^2$  最大化，得到

$$
\sigma_ {\mathrm {M L}} ^ {2} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left\{t _ {n} - \boldsymbol {w} _ {\mathrm {M L}} ^ {\mathrm {T}} \phi (\boldsymbol {x} _ {n}) \right\} ^ {2} \tag {4.20}
$$

可以看到，方差参数的最大似然值是由回归函数周围目标值的残差方差给出的。

### 4.1.4 最小二乘的几何表示

下面讨论最小二乘解的几何解释。考虑一个  $n$  维空间，其轴由  $t_n$  给出，因此  $\pmb{t} = (t_1, \dots, t_N)^\mathrm{T}$  是这个空间中的一个向量。如图4.3所示，每个在  $N$  个数据点上求值的基函数  $\phi_j(x_n)$  可以表示为相同空间中的一个向量，用  $\varphi_j$  表示。 $\varphi_j$  是  $\pmb{\Phi}$  的第  $j$  列，而  $\pmb{\Phi}(\pmb{x}_n)$  是  $\pmb{\Phi}$  的第  $n$  行的转置。如果基函数的个数  $M$  小于数据点的个数  $N$ ，则  $M$  个  $\phi_j(x_n)$  将张

![](img/24d7a902ee740b3631e55b6db8a591256cd7617a08e8168f77e994e16ec91ccb.jpg)  
图4.3 在  $N$  维空间中，不同轴的值分别为  $t_1, \dots, t_N$  时最小二乘解的几何解释。最小二乘回归函数是通过确定数据向量  $\pmb{\varepsilon}$  在基函数  $\phi_j(x)$  张成的子空间中的正交投影得到的，其中每个基函数都视为一个长度为  $N$  的向量  $\varphi_j$  ，元素为  $\phi_j(x_n)$

成一个维度为  $M$  的线性子空间  $S$  。我们定义  $\pmb{y}$  是一个  $n$  维向量，其第  $N$  个元素由  $y(x_{n},w)$  给出，其中  $n = 1,\dots ,N$  ，因为  $\pmb{y}$  是向量  $\varphi_{j}$  任意的线性组合，所以它可以存在于  $M$  维子空间中的任意位置。平方和误差［式（4.11）］等于（最高0.5倍）  $\pmb{y}$  和  $\pmb{\tau}$  之间的欧氏距离的平方。 $\pmb{w}$  的最小二乘解对应子空间  $S$  中最接近  $\pmb{\tau}$  的  $\pmb{y}$  。从图4.3中可以直观地看出，这个解对应于  $\pmb{\tau}$  在子空间  $S$  中的正交投影。由于  $\pmb{y}$  的解是由 $\Phi w_{\mathrm{ML}}$  给出的，我们通过正交投影的形式可以很容易地验证这一点（见习题4.4）。

在实践中，当  $\Phi^{\mathrm{T}}\Phi$  接近奇异值时，直接求解正规方程在数值上可能会比较困难。特别是当两个或两个以上的基向量  $\varphi_{j}$  共线或近似共线时，所得到的参数值数量级可能较大。在处理真实数据集时，这种近似退化现象并不少见。由这一现象产生的数值困难可以使用奇异值分解（Singular Value Decomposition，SVD）来解决（Deisenroth, Faisal, and Ong, 2020）。值得注意的是，我们可以通过添加正则项来确保矩阵是非奇异的，即使在存在退化的情况下也可以如此。

### 4.1.5 序贯学习

最大似然解式（4.14）涉及一次性处理整个训练集，称为批量学习。对于大型数据集来说，批处理的计算成本很高。如果数据集非常大，则使用序贯（sequential）学习算法（也称在线算法）是更为合适的。这种算法一个个地考虑数据点，在考虑每个

数据点时模型参数得到更新。序贯学习也适用于实时应用场景，在这种场景中，观测数据连续地流式到达，我们需要在所有数据可获得前就进行预测。

如下所示，我们可以通过随机梯度下降（stochastic gradient descent）技术（参见第7章），也称序贯梯度下降（sequential gradient descent）技术，得到序贯学习算法。如果误差函数包含  $n$  个数据点误差的总和，  $E = \sum_{n}E_{n}$ ，那么在观测  $n$  个数据点之后，随机梯度下降法将使用下式更新参数向量  $\pmb{w}$  ：

$$
\boldsymbol {w} ^ {(\tau + 1)} = \boldsymbol {w} ^ {(\tau)} - \eta \nabla E _ {n} \tag {4.21}
$$

其中  $\tau$  表示迭代次数， $\eta$  是合适的学习率。 $w$  的值被初始化为某个起始向量  $w^{(0)}$ 。通过平方和误差函数式（4.11），可得

$$
\boldsymbol {w} ^ {(\tau + 1)} = \boldsymbol {w} ^ {(\tau)} + \eta \left(t _ {n} - \boldsymbol {w} ^ {(\tau) \mathrm {T}} \phi_ {n}\right) \phi_ {n} \tag {4.22}
$$

其中  $\phi_{n} = \phi (x_{n})$  。这称为最小二乘（least-mean-square）法或LMS算法。

### 4.1.6 正则化最小二乘法

我们之前介绍过在误差函数中添加正则化项来控制过拟合的思想（参见1.2节），因此我们可以采用下列形式来最小化总误差函数：

$$
E _ {D} (\boldsymbol {w}) + \lambda E _ {W} (\boldsymbol {w}) \tag {4.23}
$$

其中  $\lambda$  是控制数据相关的误差  $E_{D}(\pmb{w})$  和正则化项  $E_{W}(\pmb{w})$  的相对重要性的正则化系数。正则化项最为简单的形式之一是权重向量的平方和：

$$
E _ {W} (\boldsymbol {w}) = \frac {1}{2} \sum_ {j} w _ {j} ^ {2} = \frac {1}{2} \boldsymbol {w} ^ {\mathrm {T}} \boldsymbol {w} \tag {4.24}
$$

如果我们还考虑如下平方和误差函数：

$$
E _ {D} (\boldsymbol {w}) = \frac {1}{2} \sum_ {n = 1} ^ {N} \left\{t _ {n} - \boldsymbol {w} ^ {\mathrm {T}} \phi (\boldsymbol {x} _ {n}) \right\} ^ {2} \tag {4.25}
$$

那么总误差函数将变成

$$
\frac {1}{2} \sum_ {n = 1} ^ {N} \left\{t _ {n} - \boldsymbol {w} ^ {\mathrm {T}} \phi (\boldsymbol {x} _ {n}) \right\} ^ {2} + \frac {\lambda}{2} \boldsymbol {w} ^ {\mathrm {T}} \boldsymbol {w} \tag {4.26}
$$

在统计学中，正则化方法会将参数值向0减小，这为参数收缩（parameter shrinkage）提供了一个示例。这一方法的优势是，由于误差函数仍然是  $\pmb{w}$  的二次函数，因此它具有闭式解。具体来说，将式（4.26）关于  $\pmb{w}$  的梯度设为0，并像之前一样求解  $\pmb{w}$  （见习题4.6），可得

$$
\boldsymbol {w} = \left(\lambda \boldsymbol {I} + \boldsymbol {\Phi} ^ {\mathrm {T}} \boldsymbol {\Phi}\right) ^ {- 1} \boldsymbol {\Phi} ^ {\mathrm {T}} \boldsymbol {t} \tag {4.27}
$$

式（4.27）可以看作对最小二乘解式（4.14）的简单扩展。

### 4.1.7 多重输出

到目前为止，我们已经考虑了单目标变量  $t$  的情况。在一些应用中，我们可能希望预测  $K > 1$  个目标变量，它们可以用目标向量  $\pmb{t} = (t_1, \dots, t_K)^{\mathrm{T}}$  来表示。通过为  $\pmb{t}$  的每个分量引入一组不同的基函数，我们得到多个独立的回归问题。然而，更常见的方法是使用相同的一组基函数来对目标向量的所有分量建模，即

$$
y (x, w) = W ^ {\mathrm {T}} \phi (x) \tag {4.28}
$$

其中  $y$  是一个  $K$  维的列向量； $W$  是一个  $M \times K$  大小的参数矩阵； $\phi(\pmb{x})$  是一个  $M$  维的列向量， $\phi_j(\pmb{x})$  则是其中的元素，且  $\phi_0(\pmb{x}) = 1$ 。如图4.4所示，线性回归模型同样可以表示为只有单层参数的神经网络。

![](img/ab020af2e75fa8c8beddb69b4b0c58a6da578fd574b5fe5350b794973129f525.jpg)  
图4.4 将线性回归模型表示为只有单层参数的神经网络。每个基函数由一个节点表示，实心节点表示“偏置”基函数  $\phi_0(x)$  。同样，每个输出也用一个节点表示。节点之间的链接表示对应的权重和偏置参数

假设我们取目标向量的条件分布为如下形式的各向同性的高斯分布：

$$
p (t \mid x, W, \sigma^ {2}) = \mathcal {N} (t \mid W ^ {\mathrm {T}} \phi (x), \sigma^ {2} I) \tag {4.29}
$$

一组观测  $t_1, \dots, t_N$  可以用一个大小为  $N \times K$  的矩阵  $\pmb{T}$  来表示， $t_n^{\mathrm{T}} (n = 1, \dots, N)$  代表其中的第  $n$  行。类似地，我们可以将输入向量  $x_1, \dots, x_N$  转换为矩阵  $\pmb{X}$ ，此时对数似然函数可以表示为如下形式：

$$
\begin{array}{l} \ln p \left(\boldsymbol {T} \mid \boldsymbol {X}, \boldsymbol {W}, \sigma^ {2}\right) = \sum_ {n = 1} ^ {N} \ln \mathcal {N} \left(\boldsymbol {t} _ {n} \mid \boldsymbol {W} ^ {\mathrm {T}} \boldsymbol {\phi} \left(\boldsymbol {x} _ {n}\right), \sigma^ {2} \boldsymbol {I}\right) \\ = - \frac {N K}{2} \ln \left(2 \pi \sigma^ {2}\right) - \frac {1}{2 \sigma^ {2}} \sum_ {n = 1} ^ {N} \left\| t _ {n} - W ^ {\mathrm {T}} \phi \left(x _ {n}\right) \right\| ^ {2} \tag {4.30} \\ \end{array}
$$

和之前一样，我们可以关于  $W$  最大化这个函数，即

$$
\boldsymbol {W} _ {\mathrm {M L}} = \left(\boldsymbol {\Phi} ^ {\mathrm {T}} \boldsymbol {\Phi}\right) ^ {- 1} \boldsymbol {\Phi} ^ {\mathrm {T}} \boldsymbol {T} \tag {4.31}
$$

其中，我们已经用输入的特征向量  $\phi(x_1), \dots, \phi(x_N)$  组成了矩阵  $\Phi$ 。如果我们检查

每个目标变量  $t_k$  的结果，可得

$$
\boldsymbol {w} _ {k} = \left(\boldsymbol {\Phi} ^ {\mathrm {T}} \boldsymbol {\Phi}\right) ^ {- 1} \boldsymbol {\Phi} ^ {\mathrm {T}} \boldsymbol {\xi} _ {k} = \boldsymbol {\Phi} ^ {\dagger} \boldsymbol {\xi} _ {k} \tag {4.32}
$$

其中  $\pmb{t}_k$  是一个  $N$  维的列向量，分量为  $t_{nk}$ ，其中  $n = 1, \dots, N$ 。因此，回归问题的解在不同的目标变量之间是解耦的，我们只需要计算被所有向量  $\pmb{w}_k$  共享的单个伪逆矩阵  $\pmb{\Phi}^{\dagger}$  即可。

我们可以直接将其推广到具有任意协方差矩阵的一般高斯噪声分布上（参见习题4.7）。这也再次将问题解耦成了  $K$  个独立回归问题。由于参数  $\pmb{W}$  只定义了高斯噪声分布的均值，而多元高斯均值的最大似然解与协方差无关（参见3.2.7小节），因此这并不令人惊讶。从现在开始，为了简便起见，我们将只考虑单目标变量  $t$ 。

## 4.2 决策理论

我们将回归任务看作条件概率分布  $p(t|x)$  来建模，并且选择了条件概率的特定形式，即高斯分布[式（4.8）]，其中与  $\pmb{x}$  相关的均值  $y(x,w)$  由参数  $\pmb{w}$  控制，方差由参数 $\sigma^2$  给出。  $\pmb{w}$  和  $\sigma^2$  都可以使用最大似然从数据中学习得到。预测分布（predictivedistribution）的结果如下：

$$
p \left(t \mid x, w _ {\mathrm {M L}}, \sigma_ {\mathrm {M L}} ^ {2}\right) = \mathcal {N} \left(t \mid y \left(x, w _ {\mathrm {M L}}\right), \sigma_ {\mathrm {M L}} ^ {2}\right) \tag {4.33}
$$

预测分布表达了我们对某些新输入  $x$  所对应  $t$  值的不确定性。然而，对于许多实际应用，我们需要预测  $t$  的具体值，而不是返回整个分布，特别是在需要采取特定行动的情况下。例如，如果我们想要确定用于治疗肿瘤的最佳辐射水平，而我们的模型预测了辐射剂量的概率分布，那么我们必须使用该分布来决定给予治疗的具体剂量。因此，我们的任务分为两个阶段。第一个阶段称为推理（inference）阶段，我们使用训练数据来确定预测分布  $p(t|x)$  。第二个阶段称为决策（decision）阶段，我们使用这个预测分布来确定特定的值  $f(t|x)$  。这个特定的值依赖于输入向量  $\mathbf{x}$ ，并且根据某些准则是最优的。我们可以通过最小化一个既依赖于预测分布  $p(t|x)$  又依赖于特定值  $f(t|x)$  的损失函数（也称成本函数）来实现这一点。

直觉上，我们可以选择条件分布的均值，即  $f(\pmb{x}) = y(\pmb{x}, \pmb{w}_{\mathrm{ML}})$ 。在某些情况下，这种直觉是正确的；但在其他情况下，它可能会给出非常糟糕的结果。因此，我们需要将其形式化，这样我们就可以理解它应该何时应用以及在什么假设下应用。该框架称为决策理论（decision theory）。

假设真实值为  $t$  ，我们为预测选择了一个值  $f(\pmb {x})$  ，从而得到某种形式的惩罚或成本。惩罚是由损失（loss）决定的，用  $L(t,f(x))$  表示。当然，由于我们不知道  $t$  的真实值，因此我们并不最小化  $L$  本身，而是最小化平均或期望的损失，即

$$
\mathbb {E} [ L ] = \iint L (t, f (x)) p (x, t) d x d t \tag {4.34}
$$

其中，我们以输入变量和目标变量的联合分布  $p(\mathbf{x},t)$  作为权重对它们的分布进行加权平均。回归问题中常见的损失函数是平方损失函数：  $L\big(t,f(\boldsymbol {x})\big) = \big\{f(\boldsymbol {x}) - t\big\} ^2$  。在这种情况下，期望损失函数可以写成

$$
\mathbb {E} [ L ] = \iint \left\{f (x) - t \right\} ^ {2} p (x, t) d x d t \tag {4.35}
$$

注意不要将平方损失函数与前面介绍的平方和误差函数弄混淆。误差函数用于在训练过程中设置参数，以便确定条件概率分布  $p(t|x)$ ，而损失函数则控制如何使用条件分布来达到预测函数  $f(x)$ ，以预测  $x$  的每个值。

我们的目标是找到能使  $\mathbb{E}[L]$  最小的  $f(x)$  。假设  $f(x)$  是一个完全任意的函数，则可以使用变分法（参见附录B）得到

$$
\frac {\delta \mathbb {E} [ L ]}{\delta f (x)} = 2 \int \left\{f (x) - t \right\} p (x, t) \mathrm {d} t = 0 \tag {4.36}
$$

求解  $f(x)$  并使用概率的加和法则和乘积法则，可以得到

$$
f ^ {\star} (\boldsymbol {x}) = \frac {1}{p (\boldsymbol {x})} \int t p (\boldsymbol {x}, t) \mathrm {d} t = \int t p (t | \boldsymbol {x}) \mathrm {d} t = \mathbb {E} _ {t} [ t | \boldsymbol {x} ] \tag {4.37}
$$

这是  $t$  以  $\pmb{x}$  为条件的条件平均，称为回归函数（regression function）。结果如图4.5所示，这一形式可以很容易地扩展到由向量  $\pmb{t}$  表示的多个目标变量。这时，最优解是条件平均  $f^{*}(x) = \mathbb{E}_{t}[t|x]$  （见习题4.8）。对于形如式（4.8）的高斯条件分布，条件均值可以简化为

$$
\mathbb {E} [ t \mid x ] = \int t p (t \mid x) d t = y (x, w) \tag {4.38}
$$

![](img/74066b00d7b1abe5a5950b6dcdbeb0c12be3a2ceecd468725cf3017621b671cd.jpg)  
图4.5 用于最小化期望平方损失的回归函数  $f^{*}(\mathbf{x})$  是由条件分布  $p(t\mid x)$  的均值给出的

使用变分法推导式（4.37）代表我们正在优化所有可能的函数  $f(\pmb{x})$ 。虽然实践中可以实现的任何参数模型都被限制在其可以表示的函数范围内，但我们在后续章节中广泛讨论的深度神经网络是一类高度灵活的函数。在许多实际应用中，我们可以精准

拟合任何想要的函数。

我们也可以用另一种推导方式得到这个结果，这种推导方式将阐明回归问题的本质。在知道最优解是条件期望的情况下，我们可以将平方项展开为

$$
\begin{array}{l} \left\{f (\boldsymbol {x}) - t \right\} ^ {2} = \left\{f (\boldsymbol {x}) - \mathbb {E} [ t | \boldsymbol {x} ] + \mathbb {E} [ t | \boldsymbol {x} ] - t \right\} ^ {2} \\ = \left\{f (\boldsymbol {x}) - \mathbb {E} [ t | \boldsymbol {x} ] \right\} ^ {2} + 2 \left\{f (\boldsymbol {x}) - \mathbb {E} [ t | \boldsymbol {x} ] \right\} \left\{\mathbb {E} [ t | \boldsymbol {x} ] - t \right\} + \left\{\mathbb {E} [ t | \boldsymbol {x} ] - t \right\} ^ {2} \\ \end{array}
$$

其中，为了保持符号整洁，我们使用  $\mathbb{E}[t|\boldsymbol{x}]$  来表示  $\mathbb{E}_t[t|\boldsymbol{x}]$  。代入损失函数式（4.35）并对  $t$  进行积分后，交叉项消失了，得到如下形式的损失函数的表达式：

$$
\mathbb {E} [ L ] = \int \left\{f (x) - \mathbb {E} [ t | x ] \right\} ^ {2} p (x) d x + \int \operatorname {v a r} [ t | x ] p (x) d x \tag {4.39}
$$

我们需要确定的函数  $f(\pmb{x})$  只出现在式（4.39）右侧的第一项中，当  $f(\pmb{x})$  只等于  $\mathbb{E}[t|\pmb{x}]$  时，该项达到最小值。此时，式（4.39）右侧的这一项就消失了。这只是我们之前推导出来的结果，这表明最优最小二乘的预测是由条件均值给出的。式（4.39）右侧的第二项是分布  $t$  的方差在  $\pmb{x}$  上取平均后的结果，代表目标数据的内在变化，可以视为噪声。因为它独立于  $f(\pmb{x})$  ，所以它代表损失函数不可约的最小值。

平方损失函数并不是回归中唯一可选的损失函数。平方损失的一种泛化形式称为闵可夫斯基（Minkowski）损失，其期望由下式给出：

$$
\mathbb {E} \left[ L _ {q} \right] = \iint | f (\boldsymbol {x}) - t | ^ {q} p (\boldsymbol {x}, t) d \boldsymbol {x} d t \tag {4.40}
$$

当  $q = 2$  时即为期望平方损失。图4.6给出了当  $q$  取不同值时，函数  $\left|f - t\right|^q$  关于 $f - t$  的曲线。 $\mathbb{E}\Big[L_q\Big]$  的最小值由  $q = 2$  时的条件均值、  $q = 1$  时的条件中位数和  $q\to 0$  时的条件众数给出（见习题4.12）。

高斯噪声假设意味着给定  $x$  时  $t$  的条件分布是单峰的，这对于某些应用来说可能是不合适的。在这种情况下，平方损失可能会导致结果变差，我们需要开发更复杂的其他方法。例如，我们可以通过使用混合高斯分布来扩展这个模型，使其成为多峰条件分布（参见6.5节）。这种情况在求解逆问题时会经常出现。本节重点讨论的是回归问题的决策理论，在第5章中，我们将在分类任务中引入类似的概念（参见5.2节）。

![](img/d96fe043a9ad3c819449f20f8ec8d6b303bb4e0e95e745031e1c05e67cea5fbd.jpg)  
图4.6 当  $q$  取不同值时，曲线  $L_{q} = \left|f - l\right|^{\alpha}$  的示意图

![](img/eeed7fbcf2495052526a0443bf1dde7e77412375960ab1e067871fb055a9a5d9.jpg)

![](img/5090cb4b646bbc100b160ac339322a858361f80bf8c9d5d368312e9269f18d62.jpg)  
图4.6 当  $q$  取不同值时，曲线  $L_{q} = \left|f - l\right|^{q}$  的示意图（续）

![](img/4a850c20186ec7eadfd65f144e0f89848a99fc31293d70734f1a046709c37683.jpg)

## 4.3 偏差-方差权衡

到目前为止，在对回归问题的线性模型进行讨论时，我们假设基函数的形式和数量都是给定的（参见1.2节）。我们还看到，如果使用非常有限的数据集训练复杂模型，使用最大似然估计可能会导致严重的过拟合问题。然而，通过限制基函数的数量来避免过拟合有一个副作用，就是限制了模型捕捉数据中有趣和重要趋势的灵活性。尽管正则化项可以控制包含大量参数的模型的过拟合问题，但这又引出了如何确定合适的正则化系数  $\lambda$  的问题。同时关于权重向量  $\pmb{w}$  和正则化系数  $\lambda$  最小化正则化误差函数显然是不正确的，因为这会导致未正则化的解，即  $\lambda = 0$  。

我们可以从频率学派的视角讨论模型复杂性的问题，称为偏差-方差（bias-variance）权衡。虽然我们将在线性基函数模型的背景下介绍这个概念，因为在该背景下可以使用简单的例子很容易地说明这些想法，但我们的讨论具有非常广泛的适用性。过拟合确实是最大似然的缺点，但是当我们在贝叶斯设定中边缘化参数时，则不会出现这种情况（Bishop, 2006）。

在讨论回归问题的决策理论时（参见4.2节），我们考虑了各种损失函数。一旦知道条件分布  $p(t|x)$ ，每个损失函数就都会得到对应的最优预测。一个普遍的选择是平方损失函数，其最优预测由条件期望给出，用  $h(x)$  表示。

$$
h (\boldsymbol {x}) = \mathbb {E} [ t | \boldsymbol {x} ] = \int t p (t | \boldsymbol {x}) d t \tag {4.41}
$$

平方损失的期望可以写成如下形式：

$$
\mathbb {E} [ L ] = \int \left\{f (x) - h (x) \right\} ^ {2} p (x) d x + \iint \left\{h (x) - t \right\} ^ {2} p (x, t) d x d t \tag {4.42}
$$

前面我们曾提到，与  $f(x)$  无关的式（4.42）右侧的第二项来自数据上的固有噪声，代表了期望损失的最小值。式（4.42）右侧的第一项取决于如何选择函数  $f(x)$  。我们将寻找一个  $f(x)$  来最小化这一项。因为这一项是非负的，所以能达到的最小值是0。如果有无限的数据供应（和无限的计算资源），则原则上可以找到任何准确度下的回

归函数  $h(\pmb{x})$  ，它可以代表  $f(x)$  的最优选择。然而在实践中，数据集  $\mathcal{D}$  只包含  $N$  个有限的数据点，因此我们无法精确地知道回归函数  $h(\pmb{x})$  。

如果使用参数向量  $\pmb{w}$  控制的函数对  $h(x)$  进行建模，那么从贝叶斯学派的视角来看，模型中的不确定性可以通过  $\pmb{w}$  上的后验分布来表示。但是频率学派会基于数据集  $\mathcal{D}$  对  $\pmb{w}$  进行点估计，并试图通过以下思想实验（thought experiment）来解释该估计的不确定性。假设我们有大量的数据集，每个数据集的大小为  $N$  且都独立地从分布  $p(t|x)$  中产生。对于任何给定的数据集  $\mathcal{D}$ ，都可以通过我们的算法得到预测函数  $f(\pmb{x};\mathcal{D})$  。集合中的不同数据集将给出不同的函数以及平方损失值。一个特定学习算法的性能，可通过对它在集合中每个数据集上的性能表现取平均值来评估。

式（4.42）右侧第一项的被积函数，在一个特定的数据集  $\mathcal{D}$  上具有以下形式：

$$
\{f (\boldsymbol {x}; \mathcal {D}) - h (\boldsymbol {x}) \} ^ {2} \tag {4.43}
$$

因为它的值依赖于特定的数据集  $\mathcal{D}$ ，我们取其在数据集上的平均值。在式（4.43）的花括号内加减  $\mathbb{E}_D[f(x;\mathcal{D})]$ ，然后展开，可以得到

$$
\begin{array}{l} \left\{f (\boldsymbol {x}; \mathcal {D}) - \mathbb {E} _ {\mathcal {D}} [ f (\boldsymbol {x}; \mathcal {D}) ] + \mathbb {E} _ {\mathcal {D}} [ f (\boldsymbol {x}; \mathcal {D}) ] - h (\boldsymbol {x}) \right\} ^ {2} \\ = \left\{f (\boldsymbol {x}; \mathcal {D}) - \mathbb {E} _ {\mathcal {D}} [ f (\boldsymbol {x}; \mathcal {D}) ] \right\} ^ {2} + \left\{\mathbb {E} _ {\mathcal {D}} [ f (\boldsymbol {x}; \mathcal {D}) ] - h (\boldsymbol {x}) \right\} ^ {2} + \tag {4.44} \\ 2 \left\{f (\boldsymbol {x}; \mathcal {D}) - \mathbb {E} _ {\mathcal {D}} [ f (\boldsymbol {x}; \mathcal {D}) ] \right\} \left\{\mathbb {E} _ {\mathcal {D}} [ f (\boldsymbol {x}; \mathcal {D}) ] - h (\boldsymbol {x}) \right\} \\ \end{array}
$$

取这个表达式对  $\mathcal{D}$  的期望，消去最后一项，可得

$$
\begin{array}{l} \mathbb {E} _ {\mathcal {D}} \left[ \left\{f (\boldsymbol {x}; \mathcal {D}) - h (\boldsymbol {x}) \right\} ^ {2} \right] \\ = \underbrace {\left\{\mathbb {E} _ {\mathcal {D}} [ f (\boldsymbol {x} ; \mathcal {D}) ] - h (\boldsymbol {x}) \right\} ^ {2}} _ {\text {平 方 偏 差}} + \underbrace {\mathbb {E} _ {\mathcal {D}} \left[ \left\{f (\boldsymbol {x} ; \mathcal {D}) - \mathbb {E} _ {\mathcal {D}} [ f (\boldsymbol {x} ; \mathcal {D}) ] \right\} ^ {2} \right]} _ {\text {方 差}} \tag {4.45} \\ \end{array}
$$

$f(x; \mathcal{D})$  和回归函数  $h(x)$  平方差的期望可以表示为两项的和。第一项称为平方偏差，表示所有数据集上的平均预测与期望回归函数的差异程度。第二项称为方差，用于衡量单个数据集的解在其平均值周围变化的程度；换言之，方差衡量的是函数  $f(x; \mathcal{D})$  对被选的特定数据集的敏感程度。接下来我们讨论一个简单的例子，以便对上述定义有一个直觉上的认识。

到目前为止，我们已经考虑了单一输入变量  $x$  的情况。如果我们将这个展开代回式（4.42），则可以得到

$$
\text {期 望 损 失} = \text {平 方 偏 差} + \text {方 差} + \text {噪 声} \tag {4.46}
$$

其中

$$
\text {平 方 偏 差} = \int \left\{\mathbb {E} _ {\mathcal {D}} [ f (\boldsymbol {x}; \mathcal {D}) ] - h (\boldsymbol {x}) \right\} ^ {2} p (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} \tag {4.47}
$$

$$
\text {方 差} = \int \mathbb {E} _ {\mathcal {D}} \left[ \left\{f (\boldsymbol {x}; \mathcal {D}) - \mathbb {E} _ {\mathcal {D}} \left[ f (\boldsymbol {x}; \mathcal {D}) \right] \right\} ^ {2} \right] p (\boldsymbol {x}) d \boldsymbol {x} \tag {4.48}
$$

$$
\text {噪 声} = \int \{h (\pmb {x}) - t \} ^ {2} p (\pmb {x}, t) \mathrm {d} \pmb {x} \mathrm {d} t \tag {4.49}
$$

我们的目标是最小化期望损失，我们已经将其分解为平方偏差项、方差项和常数噪声项的和。偏差和方差之间存在权衡，非常灵活的模型具有低偏差和高方差，而相对稳定的模型具有高偏差和低方差。具有最优预测能力的模型能够在偏差和方差之间达到最佳平衡。这可以用前面介绍过的正弦数据集来说明（参见1.2节）。这里独立生成100个数据集，每个数据集包含  $N = 25$  个数据点，数据点由正弦曲线  $h(x) = \sin (2\pi x)$  产生。数据集的索引是  $l = 1,\dots ,L$  ，其中  $L = 100$  。对于每个数据集  $\mathcal{D}^{(l)}$  ，拟合一个包含  $M = 24$  个高斯基函数和一个常数“偏差”基函数的模型，总共25个参数。如图4.7所示，通过最小化正则化误差函数式（4.26），可以得到预测函数  $f^{(l)}(x)$  。

![](img/b108ebbd0fd4de74e3657b4deeda8aaf81c217cc1d84ffd5dbfca7abbb8fd9ae.jpg)

![](img/1106ed61ce748d90a54a5e7060cb40232cb80e797bb28b20b04b048d84ae2267.jpg)

![](img/1da8cafc80d0944a9b6ae57f272a7ab02deb9b34443975a0a75a56afac89193e.jpg)

![](img/f543f979bcb4212e672afd5af5e5292b2ad5fc6cba7ea39d443e89224b864157.jpg)

![](img/9a78541cffade7ea1017e9ee681d217b7d2e9cf0d80b89eef54de30a65d9f409.jpg)  
图4.7 使用第1章的正弦数据展示模型复杂度上偏差和方差的依赖关系，模型复杂度由正则化参数  $\lambda$  控制。这里有  $L = 100$  个数据集，每个数据集有  $N = 25$  个数据点，模型中有24个高斯基函数，因此包括偏差参数在内的参数总数为  $M = 25$  。左列显示了模型拟合结果随  $\ln \lambda$  变化的结果（为了清晰起见，只显示了100次拟合中的20次）。右列显示了100次拟合的平均结果（红色曲线）以及生成数据集的正弦函数（绿色曲线）

![](img/b80e97e77e66beb9365213988bb635934915ce8f9f2fbf539d126ad7806db923.jpg)

图4.7的最上面一行对应的正则化系数  $\lambda$  很大，即方差很低（因为左图中的红色曲线彼此之间相似），但偏差很高（因为右图中的两条曲线差异很大）。相反，在  $\lambda$  较小的最下面一行，方差较高（表现为左图中红色曲线之间的高变异性），但偏差较低（表现为平均模型与原始正弦函数之间拟合良好）。我们发现复杂模型（ $M = 25$ ）多次拟合平均后的结果对回归函数的拟合效果非常好，这说明平均确实颇有裨益。虽然是关于参数的后验分布的平均，而不是关于多个数据集的平均，但是多个解的加权平均是贝叶斯方法的核心。

我们还可以定量地检查这个示例中的偏差-方差权衡。平均预测是用下式估计的：

$$
\bar {f} (x) = \frac {1}{L} \sum_ {l = 1} ^ {L} f ^ {(l)} (x) \tag {4.50}
$$

而平方偏差和方差则由下式给出：

$$
\text {平 方 偏 差} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left\{\overline {{f}} \left(x _ {n}\right) - h \left(x _ {n}\right) \right\} ^ {2} \tag {4.51}
$$

$$
\text {方 差} = \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {1}{L} \sum_ {l = 1} ^ {L} \left\{f ^ {(l)} \left(x _ {n}\right) - \overline {{f}} \left(x _ {n}\right) \right\} ^ {2} \tag {4.52}
$$

其中，由分布  $p(x)$  加权的对  $x$  的积分，可通过该分布产生的数据点的有限和来近似。图4.8给出了这些量在不同  $\ln \lambda$  取值时的曲线。可以看到，较小的  $\lambda$  值允许模型对每个单独数据集上的噪声进行微调，从而导致较大的方差。相反，较大的  $\lambda$  值会使权重参数向0靠近，从而导致较大的偏差。

![](img/0a31357735f30d1ece9460b27914697b238824ed7e72cad92bb57efe1b49946f.jpg)  
图4.8 根据图4.7所示的结果，绘制的平方偏差、方差以及它们的和的示意图。图中还显示了测试数据集大小为1000个数据点时的平均测试集误差。平方偏差  $+$  方差的最小值出现在  $\ln \lambda = 0.43$  附近，接近于在测试数据上给出最小误差的值

偏差-方差分解的实际价值是有限的，因为它基于数据集的集合的平均，而实际上我们只有单个观测数据集。如果我们有大量给定大小的独立训练集，则更好的方法是将它们组合成单个更大的训练集，从而降低给定复杂度模型的过拟合风险。但尽管如此，偏差-方差分解仍经常提供对模型复杂性问题的有用洞察。虽然我们在本章中

只是从回归问题的角度介绍了它，但其底层逻辑具有广泛的适用性。

## 习题

4.1  $(\star)$  考虑式（1.2）给出的平方和误差函数，其中函数  $y(x, w)$  由多项式（1.1）给出。证明能使该误差函数最小的系数  $w = \{w_{i}\}$  是由下列线性方程组的解给出的：

$$
\sum_ {j = 0} ^ {M} A _ {i j} w _ {j} = T _ {i} \tag {4.53}
$$

其中

$$
A _ {i j} = \sum_ {n = 1} ^ {N} \left(x _ {n}\right) ^ {i + j}, \quad T _ {i} = \sum_ {n = 1} ^ {N} \left(x _ {n}\right) ^ {i} t _ {n} \tag {4.54}
$$

4.2（ $\star$ ）写出类似于式（4.53）的耦合线性方程组，其中系数  $w_{i}$  能使式（1.4）给出的正则化平方和误差函数最小化。

4.3 （ $\star$ ）证明定义如下的tanh函数

$$
\tanh (a) = \frac {\mathrm {e} ^ {a} - \mathrm {e} ^ {- a}}{\mathrm {e} ^ {a} + \mathrm {e} ^ {- a}} \tag {4.55}
$$

和式（4.6）定义的sigmoid函数有如下关系：

$$
\tanh  (a) = 2 \sigma (2 a) - 1 \tag {4.56}
$$

并进一步证明 sigmoid 函数的一般线性组合形式

$$
y (x, w) = w _ {0} + \sum_ {j = 1} ^ {M} w _ {j} \sigma \left(\frac {x - \mu_ {j}}{s}\right) \tag {4.57}
$$

等价于如下形式的tanh函数的线性组合：

$$
y (x, u) = u _ {0} + \sum_ {j = 1} ^ {M} u _ {j} \tanh  \left(\frac {x - \mu_ {j}}{2 s}\right) \tag {4.58}
$$

最后找到将新参数  $\{u_1,\dots ,u_M\}$  与原始参数  $\{w_{1},\dots ,w_{M}\}$  关联起来的表达式。

4.4 （ $\star \star \star$ ）证明矩阵

$$
\boldsymbol {\Phi} \left(\boldsymbol {\Phi} ^ {T} \boldsymbol {\Phi}\right) ^ {- 1} \boldsymbol {\Phi} ^ {T} \tag {4.59}
$$

可将任何向量  $\pmb{\nu}$  投影到  $\pmb{\Phi}$  的列所张成的空间中。使用此结果证明，最小二乘解式（4.14）对应向量  $\pmb{t}$  在流形  $S$  上的正交投影（见图4.3）。

4.5（ $\star$ ）考虑一个每个数据点  $t_n$  都与加权因子  $r_n > 0$  相关联的数据集，平方和误差函

数变为

$$
E _ {D} (\boldsymbol {w}) = \frac {1}{2} \sum_ {n = 1} ^ {N} r _ {n} \left\{t _ {n} - \boldsymbol {w} ^ {\mathrm {T}} \boldsymbol {\phi} \left(\boldsymbol {x} _ {n}\right) \right\} ^ {2} \tag {4.60}
$$

给出最小化该误差函数的解  $w^{\star}$  的表达式。从与数据相关的噪声方差和复制的数据点两方面，给出加权平方和误差函数的两种替代解释。

4.6（ $\star$ ）通过将式（4.26）对  $\pmb{w}$  的梯度设置为0，证明在线性回归中，正则化平方和误差函数的精确最小值由式（4.27）给出。

4.7（ $\star \star$ ）假设多元目标变量  $t$  的线性基函数回归模型具有下式给出的高斯分布：

$$
p (t \mid W, \Sigma) = \mathcal {N} (t \mid y (x, W), \Sigma) \tag {4.61}
$$

其中

$$
\mathbf {y} (\mathbf {x}, \mathbf {W}) = \mathbf {W} ^ {\mathrm {T}} \boldsymbol {\phi} (\mathbf {x}) \tag {4.62}
$$

训练数据集由输入基向量  $\phi(x_{n})$  和对应的目标向量  $t_{n}$  组成，其中  $n = 1, \dots, N$  。证明参数矩阵  $\pmb{W}$  的最大似然解  $\pmb{W}_{\mathrm{ML}}$  具有如下性质：每列由形如式（4.14）的表达式给出，其是各向同性噪声分布的解。注意，它独立于协方差矩阵  $\pmb{\Sigma}$  。证明  $\pmb{\Sigma}$  的最大似然解由下式给出：

$$
\boldsymbol {\Sigma} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left(\boldsymbol {t} _ {n} - \boldsymbol {W} _ {\mathrm {M L}} ^ {\mathrm {T}} \boldsymbol {\phi} \left(\boldsymbol {x} _ {n}\right)\right) \left(\boldsymbol {t} _ {n} - \boldsymbol {W} _ {\mathrm {M L}} ^ {\mathrm {T}} \boldsymbol {\phi} \left(\boldsymbol {x} _ {n}\right)\right) ^ {\mathrm {T}} \tag {4.63}
$$

4.8（ $\star$ ）考虑将单个目标变量  $t$  的平方损失函数式（4.35）推广到下式给出的多目标变量（用向量  $\pmb{t}$  表示），则有

$$
\mathbb {E} [ L (t, f (x)) ] = \iint \left\| f (x) - t \right\| ^ {2} p (x, t) d x d t \tag {4.64}
$$

使用变分演算，证明使预期损失最小化的函数  $f(x)$  由式（4.65）给出：

$$
f (\boldsymbol {x}) = \mathbb {E} _ {t} [ t | \boldsymbol {x} ] \tag {4.65}
$$

4.9（ $\star$ ）通过对式（4.64）中的平方项进行展开，推导类似于式（4.39）的结果。进一步地，证明使多元目标变量  $\pmb{t}$  的预期平方损失最小化的函数  $f(x)$  同样以式（4.65）的形式由  $\pmb{t}$  的条件期望给出。

4.10（ $\star \star$ ）类比式（4.39），通过扩展式（4.64）来重新推导结果式（4.65）。

4.11 （ $\star \star$ ）以下分布

$$
p \left(x \mid \sigma^ {2}, q\right) = \frac {q}{2 \left(2 \sigma^ {2}\right) ^ {1 / q} \Gamma (1 / q)} \exp \left(- \frac {\left| x \right| ^ {q}}{2 \sigma^ {2}}\right) \tag {4.66}
$$

是一元高斯分布的推广形式。其中  $\Gamma (x)$  是由下式定义的伽马函数：

$$
\Gamma (x) = \int_ {- \infty} ^ {\infty} u ^ {x - 1} e ^ {- u} d u \tag {4.67}
$$

证明此分布已经归一化，从而有

$$
\int_ {- \infty} ^ {\infty} p (x | \sigma^ {2}, q) d x = 1 \tag {4.68}
$$

且当  $q = 2$  时，它退化为高斯分布。考虑一个目标变量由  $t = y(\pmb{x},\pmb{w}) + \varepsilon$  给出的回归模型， $\varepsilon$  是从分布式（4.66）中产生的随机噪声变量。证明对于输入向量  $\pmb{X} = \{\pmb{x}_1,\dots ,\pmb{x}_N\}$  和相应的目标变量  $\pmb{f} = (t_{1},\dots ,t_{N})^{\mathrm{T}}$  的数据集，定义在  $\pmb{w}$  和  $\sigma^2$  上的对数似然函数由下式给出：

$$
\ln p (\boldsymbol {t} \mid X, w, \sigma^ {2}) = - \frac {1}{2 \sigma^ {2}} \sum_ {n = 1} ^ {N} | y (x _ {n}, w) - t _ {n} | ^ {q} - \frac {N}{q} \ln \left(2 \sigma^ {2}\right) + \text {c o n s t} \tag {4.69}
$$

其中“const”表示独立于  $\pmb{w}$  和  $\sigma^2$  的常数项。注意，作为  $\pmb{w}$  的函数，该对数似然函数是4.2节中的  $L_{q}$  误差函数。

4.12  $(\star \star)$  考虑由  $L_{q}$  损失函数定义的回归问题的期望损失式（4.40）。写出在最小化  $\mathbb{E}[L_q]$  时， $y(x)$  必须满足的条件。证明当  $q = 1$  时，该解表示条件中位数，即函数  $y(x)$  满足  $t < y(x)$  的概率与  $t \geqslant y(x)$  的概率相同；并证明当  $q \to 0$  时最小期望损失  $L_{q}$  由条件众数给出，即函数  $y(x)$  等于对每个  $x$  而言，能使  $p(t|x)$  最大化的  $t$  值。
