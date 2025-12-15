# 第16章连续潜变量

![](img/97c3864bfa89f596e691869e843645ad1ff287c138b870524ca063173b402bc4.jpg)

在第15章中，我们讨论了拥有离散潜变量的概率模型，例如高斯混合模型。本章探讨部分或全部潜变量为连续变量的模型。研究这类模型的一个重要原因是很多数据集都有这样的特点：数据点都挤在一个比起原始数据空间维度低得多的流形（manifold）上。为什么会这样呢？想象一下，假如我们从MNIST数据集（LeCun et al., 1998）里选出一个显示手写数字的  $64 \times 64$  像素灰度图，将它嵌入一个更大的 $100 \times 100$  像素的图像中，图像边缘用数值为零的像素（对应白色像素）填充。嵌入的同时，随机改变数字的位置和方向，就像图16.1展示的那样，那么每一个生成的图像由高达10000维（因为是  $100 \times 100$  像素）的数据空间中的一个数据点来表示。然而在这样的图像数据集中，仅有三个自由度（degree of freedom）的变化，分别对应于垂直移动、平移及旋转。因此，这些数据点实际上存在于一个内在维度（intrinsic dimensionality）为3的子空间中。请注意，这个流形是非线性的，因为如果我们将数字平移经过一个特定像素，那么这个像素的值将从0（白色）变成1（黑色），然后再

次变回0，这显然是关于数字位置的非线性函数。在这个例子中，平移和旋转的相关参数是潜变量，因为我们只观察到图像向量本身，而不知道用来生成它们的平移或旋转变量的具体值。

![](img/e3f1adb5bf139c9c95fc7851e0fef8d2434b66a8e00af21723181f131b018d98.jpg)  
图16.1 一个合成的数据集，从中获取一个手写数字图像并创建多个副本，在每个副本中，手写数字在某个更大的图像区域内进行随机移动和旋转

在真实的手写数字数据集中，会有更多的自由度出现。例如，不同人的书写风格有差异，而同一个人的书写风格也有变化，这就会导致手写数字在尺寸和其他方面上的变化。不过，即使有这些自由度，它们的数量相较于整个数据集的维度还是显得很少。

在实践中，数据点不会精确地局限于一个光滑的低维流形上，我们可以将数据点偏离流形的现象解释为“噪声”。我们可以自然地从生成性视角（generative view）来理解此类模型。首先根据某种潜变量分布在流形内选择一个点，然后添加从给定潜变量的数据变量的某种条件分布中抽取的噪声，生成一个观测数据点。

对于最简单的连续潜变量模型（continuous latent-variable model），可以假设潜变量和观测变量都遵循高斯分布，并利用线性-高斯依赖性（linear-Gaussian dependence）（见11.1.4小节）来描述观测变量对潜变量状态的依赖。这引出了著名的主成分分析（Principal Component Analysis，PCA）概率公式，还有称为因子分析（factor analysis）的相关模型。本章首先介绍如何使用标准非概率方法来实现PCA，随后揭示PCA如何自然而然地对应于线性-高斯潜变量模型的最大似然解（maximum likelihood solution）（见16.1节）。这种概率化表述带来了许多优势（见第16.2节），比如使用EM算法进行参数估计，规范化地扩展成PCA混合模型，以及通过贝叶斯公式自动从数据中确定主成分的数量（Bishop，2006）。本章还为我们学习具有连续潜变量的非线性模型奠定了基础，这些模型包括标准化流、变分自编码器和扩散模型。

## 16.1 主成分分析

主成分分析（PCA）已广泛应用于降维、有损数据压缩、特征提取和数据可视化等领域（Jolliffe, 2002），它也称为Kosambi-Karhunen-Loève变换。考虑将一个数据集正交投影到一个低维的线性空间［即主子空间（principal subspace）］，如图16.2所示。PCA可以定义为最大化投影数据方差的线性投影（Hotelling, 1933）。同样，PCA也可以定义为最小化平均投影代价的线性投影，平均投影代价则定义为数据点与其投影之间的均方距离（Pearson, 1901）。我们将依次学习这些定义。

![](img/f425b0c89ab833a6fbf7542e73623adec54ae17e8987f1dea3fb6b10ee01e515.jpg)  
图16.2 利用PCA寻求一个低维空间，称为主子空间（principal subspace），用洋红色线条表示，以使数据点（红色点）在主子空间中的正交投影能够最大化投影点（绿色点）的方差。PCA的另一种定义是基于最小化投影误差的平方和给出的，这些误差用蓝色线条表示

### 16.1.1 最大方差表述

考虑一个观测数据集  $\{x_{n}\}$ ，其中  $n = 1, \dots, N$ ，并且  $x_{n}$  是一个维度为  $D$  的欧几里得变量。我们想要把数据投影到维度为  $M < D$  的空间，同时最大化投影数据的方差。目前，我们假设  $M$  的值是给定的。在本章的后面，我们会学习从数据中确定  $M$  的合适值的技术。

首先考虑投影到一维空间  $(M = 1)$  。我们可以使用一个  $D$  维向量  $\pmb{u}_{1}$  来定义这个空间的方向，为了方便（且不失一般性），我们会选择单位向量，因此  $\pmb{u}_{1}^{\mathrm{T}}\pmb{u}_{1} = 1$  （注意我们只对  $\pmb{u}_{1}$  定义的方向感兴趣）。然后将每个数据点  $\pmb{x}_n$  投影到一个标量值  $\pmb{u}_{1}^{\mathrm{T}}\pmb{x}_{n}$  上。投影数据的平均值是  $\pmb{u}_{1}^{\mathrm{T}}\overline{\pmb{x}}$  ，其中  $\overline{\pmb{x}}$  是样本集的平均值，由下式给出：

$$
\overline {{x}} = \frac {1}{N} \sum_ {n = 1} ^ {N} x _ {n} \tag {16.1}
$$

投影数据的方差由下式给出：

$$
\frac {1}{N} \sum_ {n = 1} ^ {N} \left\{\boldsymbol {u} _ {1} ^ {\mathrm {T}} \boldsymbol {x} _ {n} - \boldsymbol {u} _ {1} ^ {\mathrm {T}} \overline {{\mathbf {x}}} \right\} ^ {2} = \boldsymbol {u} _ {1} ^ {\mathrm {T}} \boldsymbol {S} \boldsymbol {u} _ {1} \tag {16.2}
$$

其中  $S$  是数据协方差矩阵，定义为

$$
\boldsymbol {S} = \frac {1}{N} \sum_ {n = 1} ^ {N} \left(\boldsymbol {x} _ {n} - \overline {{\boldsymbol {x}}}\right) \left(\boldsymbol {x} _ {n} - \overline {{\boldsymbol {x}}}\right) ^ {\mathrm {T}} \tag {16.3}
$$

下面我们将投影数据的方差  $\pmb{u}_1^{\mathrm{T}}\pmb{S}\pmb{u}_1$  相对于  $\pmb{u}_{1}$  最大化。显然，这必须是一个受约束的最大化操作，以防止  $\| \pmb {u}_1\| \to \infty$  。适当的约束来源于归一化条件  $\pmb {u}_1^{\mathrm{T}}\pmb {u}_1 = 1$  。为了强制执行这个约束，我们引入一个拉格朗日乘子（见附录C），记作  $\lambda_{1}$  ，并对下式进行无约束的最大化：

$$
\boldsymbol {u} _ {1} ^ {\mathrm {T}} \boldsymbol {S} \boldsymbol {u} _ {1} + \lambda_ {1} \left(1 - \boldsymbol {u} _ {1} ^ {\mathrm {T}} \boldsymbol {u} _ {1}\right) \tag {16.4}
$$

通过将式（16.4）对  $\pmb{u}_{1}$  的导数设为零，我们发现当以下条件满足时，目标函数将取驻点：

$$
\boldsymbol {S} \boldsymbol {u} _ {1} = \lambda_ {1} \boldsymbol {u} _ {1} \tag {16.5}
$$

这意味着  $\pmb{u}_{1}$  必须是  $S$  的一个特征向量。如果将等式[式（16.5）]左乘  $\pmb{u}_{1}^{\mathrm{T}}$  并利用归一化条件  $\pmb{u}_{1}^{\mathrm{T}}\pmb{u}_{1} = 1$  ，可得方差为

$$
\boldsymbol {u} _ {1} ^ {\mathrm {T}} \boldsymbol {S} \boldsymbol {u} _ {1} = \lambda_ {1} \tag {16.6}
$$

因此，当我们将  $u_{1}$  设置为具有最大特征值  $\lambda_{1}$  的特征向量时，方差将达到最大。这个特征向量称为第一主成分。

我们可以通过逐步的方式定义更多的主成分，方法就是选取一个新方向，使其在所有与已考虑方向正交的方向中最大化投影后的方差。考虑  $M$  维投影空间的一般情况，投影数据方差最大化的最优线性投影现在由数据协方差矩阵  $\mathbf{S}$  对应于最大特征值  $\lambda_1,\dots ,\lambda_M$  的  $M$  个特征向量  $\pmb {u}_1,\dots ,\pmb {u}_M$  定义（见习题16.1）。这可以很容易地使用数学归纳法来证明。

综上，PCA涉及计算数据集的均值  $\overline{x}$  和数据协方差矩阵  $S$  ，然后找到  $S$  对应于最大特征值的  $M$  个特征向量。关于寻找特征向量和特征值的算法，以及与特征向量分解相关的附加定理，可以在Golub and Van Loan（1996）中找到。请注意，计算一个 $D\times D$  大小矩阵的完整特征向量分解的计算成本是  $\mathcal{O}(D^3)$  。如果打算将数据投影到前 $M$  个主成分上，则只需要找到前  $M$  个特征值和特征向量。这可以通过使用更有效的方法来完成，比如幂方法（Golub and Van Loan,1996），复杂度为  $\mathcal{O}(MD^2)$  ，当然也可以使用EM算法（见16.3.2小节）。

### 16.1.2 最小误差表述

本小节讨论基于投影误差最小化的PCA的另一种表述（见附录A）。为此，引入一组  $D$  维的完备正交规范的基向量  $\{\pmb{u}_i\}$  ，其中  $i = 1,\dots ,D$  ，它们满足

$$
\boldsymbol {u} _ {i} ^ {\mathrm {T}} \boldsymbol {u} _ {j} = \delta_ {i j} \tag {16.7}
$$

由于这组基向量是完备的，每个数据点都可以通过基向量的线性组合精确表示为

$$
\boldsymbol {x} _ {n} = \sum_ {i = 1} ^ {D} \alpha_ {n i} \boldsymbol {u} _ {i} \tag {16.8}
$$

其中系数  $\alpha_{ni}$  对于不同的数据点是不同的。这只是相当于将坐标系旋转到  $\{\pmb{u}_i\}$  定义的新系统，原始的  $D$  个分量  $\{x_{n1},\dots ,x_{nD}\}$  被等价的一组分量  $\{\alpha_{n1},\dots ,\alpha_{nD}\}$  替代。与  $\pmb{u}_{j}$  内积并利用正交性质，可以得到  $\alpha_{nj} = \pmb{x}_n^{\mathrm{T}}\pmb{u}_j$  。因此，不失一般性地，我们可以写作

$$
\boldsymbol {x} _ {n} = \sum_ {i = 1} ^ {D} \left(\boldsymbol {x} _ {n} ^ {\mathrm {T}} \boldsymbol {u} _ {i}\right) \boldsymbol {u} _ {i} \tag {16.9}
$$

不过，我们的目标是用一个较小的变量数  $M$ （小于原始维度  $D$ ）来近似这个数据点，相当于将它投影到一个低维的子空间。这个  $M$  维的线性子空间可以用前  $M$  个基向量表示，这样我们就可以用这些基向量来近似每一个数据点  $\pmb{x}_n$ ：

$$
\tilde {\boldsymbol {x}} _ {n} = \sum_ {i = 1} ^ {M} z _ {n i} \boldsymbol {u} _ {i} + \sum_ {i = M + 1} ^ {D} b _ {i} \boldsymbol {u} _ {i} \tag {16.10}
$$

其中  $\{z_{ni}\}$  依赖于特定的数据点，而  $\{b_i\}$  是对所有数据点都相同的常数。我们可以自由选择  $\{\pmb{u}_i\}$  、  $\{z_{ni}\}$  和  $\{b_i\}$  ，以最小化由维度降低引入的误差。

作为误差度量，我们将使用原始数据点  $x_{n}$  与其近似值  $\tilde{x}_n$  之间的平方距离，并在数据集上取平均，因此我们的目标是最小化

$$
J = \frac {1}{N} \sum_ {n = 1} ^ {N} \left\| \boldsymbol {x} _ {n} - \tilde {\boldsymbol {x}} _ {n} \right\| ^ {2} \tag {16.11}
$$

首先考虑关于量  $\{z_{ni}\}$  的最小化。将  $\tilde{\pmb{x}}_n$  代入，将  $J$  关于  $z_{nj}$  的导数设为零，利用正交关系，我们得到

$$
z _ {n j} = \boldsymbol {x} _ {n} ^ {\mathrm {T}} \boldsymbol {u} _ {j} \tag {16.12}
$$

其中  $j = 1,\dots ,M$  。类似地，将  $J$  关于  $b_{i}$  的导数设为零，并再次利用正交关系，得到

$$
b _ {j} = \overline {{\boldsymbol {x}}} ^ {\mathrm {T}} \boldsymbol {u} _ {j} \tag {16.13}
$$

其中  $j = M + 1,\dots ,D$  。代入  $z_{ni}$  和  $b_{i}$  ，并利用一般展开式[式（16.9)]，得到

$$
\boldsymbol {x} _ {n} - \tilde {\boldsymbol {x}} _ {n} = \sum_ {i = M + 1} ^ {D} \left\{\left(\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}\right) ^ {\mathrm {T}} \boldsymbol {u} _ {i} \right\} \boldsymbol {u} _ {i} \tag {16.14}
$$

我们看到，从  $x_{n}$  到  $\tilde{\pmb{x}}_n$  的位移向量位于主子空间的正交空间中，因为它是 $i = M + 1,\dots ,D$  的  $\pmb{u}_i$  的线性组合，如图16.2所示。这在预期之内，因为投影点  $\tilde{\pmb{x}}_n$  必须位于主子空间内，但我们可以在这个子空间内自由移动它们，因此最小误差由正交投影给出。

误差度量  $J$  作为纯粹依赖于  $\{\pmb{u}_i\}$  的函数的表达式如下：

$$
J = \frac {1}{N} \sum_ {n = 1} ^ {N} \sum_ {i = M + 1} ^ {D} \left(\boldsymbol {x} _ {n} ^ {\mathrm {T}} \boldsymbol {u} _ {i} - \overline {{\boldsymbol {x}}} ^ {\mathrm {T}} \boldsymbol {u} _ {i}\right) ^ {2} = \sum_ {i = M + 1} ^ {D} \boldsymbol {u} _ {i} ^ {\mathrm {T}} \boldsymbol {S} \boldsymbol {u} _ {i} \tag {16.15}
$$

剩下的任务是调整  $\{\pmb{u}_i\}$  从而最小化  $J$  ，这也必须是一个受约束的最小化操作，否

则我们将得到毫无意义的结果  $\pmb{u}_i = 0$  。约束来源于正交条件，正如我们将看到的，解将以协方差矩阵的特征向量的形式表示出来。

在考虑正式的解之前，让我们通过考虑一个二维数据空间（  $D = 2$  ）和一个一维主子空间（  $M = 1$  ）来直观地理解一下。我们需要选择一个方向  $\pmb{u}_2$  来最小化  $J = \pmb{u}_2^{\mathrm{T}}\pmb{S}\pmb{u}_2$  ，同时也要受归一化约束  $\pmb{u}_2^{\mathrm{T}}\pmb{u}_2 = 1$  的限制。我们使用拉格朗日乘子  $\lambda_{2}$  来强制执行约束，考虑最小化

$$
\tilde {J} = \boldsymbol {u} _ {2} ^ {\mathrm {T}} \boldsymbol {S} \boldsymbol {u} _ {2} + \lambda_ {2} \left(1 - \boldsymbol {u} _ {2} ^ {\mathrm {T}} \boldsymbol {u} _ {2}\right) \tag {16.16}
$$

将  $\tilde{J}$  关于  $\pmb{u}_{2}$  的导数设为零，我们得到  $S\pmb{u}_2 = \lambda_2\pmb{u}_2$  ，  $\pmb{u}_{2}$  是  $s$  的一个特征向量，具有特征值  $\lambda_{2}$  。因此，任何特征向量都将定义误差度量的驻点。要找到  $J$  在最小值点的值，可以将方程对  $\pmb{u}_{2}$  的解回代到误差度量中，得到  $J = \lambda_{2}$  。因此，我们可以通过选择 $\pmb{u}_{2}$  为对应于两个特征值中较小的那个特征向量来获得  $J$  的最小值。我们应该选择让主子空间与具有较大特征值的特征向量对齐。这个结果符合我们的直觉，即为了最小化平均平方投影距离，应该选择主成分子空间，使其通过数据点的平均值并与最大方差的方向对齐。只要特征值相等，任何选择的主方向都将产生相同的  $J$  值。

针对任意  $D$  和任意  $M < D$  最小化  $J$  的一般解，均可以选择  $\pmb{u}_i$  为协方差矩阵的特征向量（见习题16.2），由下式给出：

$$
\boldsymbol {S} \boldsymbol {u} _ {i} = \lambda_ {i} \boldsymbol {u} _ {i} \tag {16.17}
$$

其中  $i = 1, \dots, D$ ，并且像往常一样，特征向量  $\pmb{u}_i$  选择为正交向量。然后，误差度量的相应值由下式给出：

$$
J = \sum_ {i = M + 1} ^ {D} \lambda_ {i} \tag {16.18}
$$

这仅仅是正交于主子空间的那些特征向量的特征值之和。因此，我们通过选择这些特征向量为具有  $D - M$  个最小特征值的那些特征向量，得到了  $J$  的最小值，定义主子空间的特征向量是对应于  $M$  个最大特征值的那些特征向量。

虽然我们仅考虑了  $M < D$  的情况，但如果  $M = D$  ，PCA仍然成立，在这种情况下没有降维，只是简单地旋转坐标轴以与主成分对齐。

最后请注意，有一种相关的线性降维技术称为典型相关分析（Canonical Correlation Analysis，CCA）（Hotelling, 1936; Bach and Jordan, 2002）。PCA只处理单个随机变量，而典型相关分析则考虑多个变量，并尝试找到具有高度互相关性的对应线性子空间对，以使一个子空间内的分量就与另一个子空间内的分量具有相关性。它的解可以用广义特征向量问题来表示。

### 16.1.3 数据压缩

PCA的另一个应用是数据压缩，我们用手写数字图像数据集来举例。因为协方差矩阵的每个特征向量都是原始  $D$  维空间中的向量，所以我们可以将特征向量表示成与数据点大小相同的图像。平均向量和前4个PCA特征向量及对应的特征值如图16.3所示。

![](img/fb75820578892eaecc0af7d927a77dc7fc3152ea1e281a5385a42f25c4e43a64.jpg)  
图16.3 将PCA应用于一个由6000个  $28 \times 28$  像素的手写数字“3”的图像数据集，这里展现了平均向量  $\overline{x}$  和前4个PCA特征向量  $u_{1}, \dots, u_{4}$  及对应的特征值

![](img/4f8f8618fbeb5187336e90d995b117570987dcee7954625430f0b6a79ab02311.jpg)

![](img/088ef5f30398759b0a57dcee18c697fadb36c334c130da7a943f6ac6c93e6a2c.jpg)

![](img/7cfd399d130f93b21adb64ce5e738d5dcc6d904004afa5858342592c972302a2.jpg)

![](img/465aa4f07336313d6a7b6f0589ea0d65107edacf0087a6d1c39cee5943c4e6ef.jpg)

完整特征值谱的排序递减图如图16.4(a)所示。与选择特定的  $M$  值相关联的误差度量  $J$  由第  $M + 1$  到第  $D$  个特征值的和给出，并且不同  $M$  值的情况已绘制在图16.4(b)中。

![](img/53e60e3c36fd881afe23fe0fef23750d81013e375f40350d1ca0ba45df8cbfca.jpg)  
(a)  
图16.4 (a) 图16.3中使用的手写数字图像数据集的特征值谱图；(b) 被抛弃特征值的和代表了通过将数据投影到维度为  $M$  的主子空间而引入的平方和误差  $J$

![](img/81275426e70292a45be9bf0d4cf83030f201e5e942d2a6c0572afbfb5c50f154.jpg)  
(b)

如果我们将式（16.12）和式（16.13）代入式（16.10），就可以将PCA近似于数据向量  $x_{n}$  的形式写为

$$
\begin{array}{l} \tilde {\boldsymbol {x}} _ {n} = \sum_ {i = 1} ^ {M} \left(\boldsymbol {x} _ {n} ^ {\mathrm {T}} \boldsymbol {u} _ {i}\right) \boldsymbol {u} _ {i} + \sum_ {i = M + 1} ^ {D} \left(\overline {{\boldsymbol {x}}} ^ {\mathrm {T}} \boldsymbol {u} _ {i}\right) \boldsymbol {u} _ {i} (16.19) \\ = \overline {{\boldsymbol {x}}} + \sum_ {i = 1} ^ {M} \left(\boldsymbol {x} _ {n} ^ {\mathrm {T}} \boldsymbol {u} _ {i} - \overline {{\boldsymbol {x}}} ^ {\mathrm {T}} \boldsymbol {u} _ {i}\right) \boldsymbol {u} _ {i} (16.20) \\ \end{array}
$$

其中我们利用了如下关系：

$$
\overline {{\boldsymbol {x}}} = \sum_ {i = 1} ^ {D} \left(\overline {{\boldsymbol {x}}} ^ {\mathrm {T}} \boldsymbol {u} _ {i}\right) \boldsymbol {u} _ {i} \tag {16.21}
$$

式（16.21）成立是因为  $\{\pmb{u}_i\}$  的完备性。这代表了数据集的压缩，因为对于每个数据点，我们都会将  $D$  维向量  $\pmb{x}_n$  替换为具有分量  $\left(\pmb{x}_n^{\mathrm{T}}\pmb{u}_i - \overline{\pmb{x}}^{\mathrm{T}}\pmb{u}_i\right)$  的  $M$  维向量。 $M$  值越小，压缩程度越大。手写数字图像数据集的 PCA 重建数据点示例见图 16.5。

![](img/e03d1d0352b2b251371704b06ae8924cf0efd794497a0b09260f5094efac9baa.jpg)  
原始图像

![](img/a09b2fe8a27eb54feb4c17c578be0fb9ecefb115c15ac611a9c9c8cf806fa4be.jpg)  
$M = 1$  
图16.5 手写数字图像数据集中的一个手写数字及其通过保留  $M$  个主成分而获得的PCA重构结果。随着  $M$  值的增加，重构变得更准确，当  $M = D = 28 \times 28 = 784$  时，重构将会是完美的

![](img/fd677a484cdd32b386057d702285c7748eb75609e1d41961058b0df5c541e3a7.jpg)  
$M = 10$

![](img/0a9297cd48e88c7024bfe414ec54294712639200c35da45024c34d0e1dd71313.jpg)  
$M = 50$

![](img/c5853a8208989655a6cf4a4f7b24c53299197596adf1b5bbde15d95f5d4ba93a.jpg)  
$M = 250$

### 16.1.4 数据白化

PCA的另一个应用是数据预处理，其目标不是降低维度，而是转换数据集以标准化某些属性。变化后的数据集能更好地为机器学习算法服务。通常，当原始变量以不同的单位测量或具有显著不同的可变性时，就需要进行此类处理。例如，在老忠实喷泉数据集中，两次喷发之间的间隔时间通常比一次喷发的持续时间长一个数量级（见15.1节）。将  $K$  均值算法应用于这个数据集时，我们首先对各个变量进行了单独的线性重缩放，使得每个变量都具有零均值和单位方差。这称为标准化数据，标准化数据的协方差矩阵由以下元素构成：

$$
\rho_ {i j} = \frac {1}{N} \sum_ {n = 1} ^ {N} \frac {\left(x _ {n i} - \bar {x} _ {i}\right)}{\sigma_ {i}} \frac {\left(x _ {n j} - \bar {x} _ {j}\right)}{\sigma_ {j}} \tag {16.22}
$$

其中  $\sigma_{i}$  是  $x_{i}$  的标准差。原始数据的相关性矩阵的特点是，如果数据的两个分量  $x_{i}$  和  $x_{j}$  完全相关，则  $\rho_{ij} = 1$ ；如果它们不相关，则  $\rho_{ij} = 0$ 。

利用PCA，可以对数据进行更实质性的标准化，使其具有零均值和单位协方差，从而使不同的变量变得不相关。为此，首先将特征向量方程［式（16.17）］写成以下形式：

$$
\boldsymbol {S} \boldsymbol {U} = \boldsymbol {U} \boldsymbol {L} \tag {16.23}
$$

其中  $\pmb{L}$  是一个  $D\times D$  的对角矩阵，矩阵元素为  $\lambda_{i}$  ；  $\pmb{U}$  是一个  $D\times D$  的正交矩阵，其列由  $\pmb{u}_i$  给出。然后我们定义，对于每个数据点  $\pmb{x}_n$  ，转换后的值由下式给出：

$$
\boldsymbol {y} _ {n} = \boldsymbol {L} ^ {- 1 / 2} \boldsymbol {U} ^ {\mathrm {T}} \left(\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}\right) \tag {16.24}
$$

其中  $\overline{x}$  是由式（16.1）定义的样本均值。显然，集合  $\{y_{n}\}$  具有零均值，并且其协方差由单位矩阵给出，因为

$$
\begin{array}{l} \frac {1}{N} \sum_ {n = 1} ^ {N} \mathbf {y} _ {n} \mathbf {y} _ {n} ^ {\mathrm {T}} = \frac {1}{N} \sum_ {n = 1} ^ {N} L ^ {- 1 / 2} U ^ {\mathrm {T}} \left(\mathbf {x} _ {n} - \overline {{\mathbf {x}}}\right) \left(\mathbf {x} _ {n} - \overline {{\mathbf {x}}}\right) ^ {\mathrm {T}} U L ^ {- 1 / 2} \tag {16.25} \\ = \boldsymbol {L} ^ {- 1 / 2} \boldsymbol {U} ^ {\mathrm {T}} \boldsymbol {S} \boldsymbol {U} \boldsymbol {L} ^ {- 1 / 2} = \boldsymbol {L} ^ {- 1 / 2} \boldsymbol {L} \boldsymbol {L} ^ {- 1 / 2} = \boldsymbol {I} \\ \end{array}
$$

这个操作称为白化（whitening）或球化（sphering）数据，图16.6展示了对老忠实喷泉数据集（见15.1节）应用线性预处理的效果。

![](img/76d86d439d47b55dcc571f158b2a5e5bc6ad7100816a89d8d3d919eabca816ce.jpg)  
图16.6对老忠实喷泉数据集应用线性预处理的效果。左图显示了原始数据。中图显示了将各个变量标准化为零均值和单位方差后的结果，同时显示了这个规范化数据集的主轴，它们绘制在范围  $\pm \lambda_i^{1 / 2}$  内。右图显示了白化数据并得到零均值和单位协方差后的结果

![](img/692ad219a1b0d9216925b596aad1d130cfb6f2c3c985c4496343887b065a8d79.jpg)

![](img/b12ce6d82c932366ae279d4235c577acd4ebd3e5b64a06d9095c40a11e27323b.jpg)

### 16.1.5 高维数据

在PCA的某些应用中，数据点的数量小于数据空间的维度。例如，我们可能想对一个有几百张图像的数据集应用PCA，其中的每个图像对应于潜在的几百万维空间中的一个向量（对应于图像中每个像素的三个颜色值）。注意，在  $D$  维空间中， $N$  个点的集合（ $N < D$ ）定义了一个线性子空间，其维度最多为  $N - 1$ ，因此当  $M$  的值大于 $N - 1$ 时，应用PCA没有多大意义。实际上，如果我们执行PCA，就会发现至少有 $D - N + 1$ 个特征值是零，在这些特征值对应的特征向量方向上，数据的方差为零。

此外，寻找  $D \times D$  矩阵的特征向量的典型算法的计算成本达到了  $\mathcal{O}(D^3)$  的规模，因此对图像直接应用PCA在计算上是不可行的。

可以按照以下方式来解决这个问题。首先将  $X$  定义为  $N\times D$  维的中心化数据矩阵，其第  $n$  行由  $\left(x_{n} - \overline{x}\right)^{\mathrm{T}}$  给出。然后可以将协方差矩阵［式（16.3）]写作  $S = N^{-1}X^{\mathrm{T}}X$  ，相应的特征向量方程变为

$$
\frac {1}{N} \boldsymbol {X} ^ {\mathrm {T}} \boldsymbol {X} \boldsymbol {u} _ {i} = \lambda_ {i} \boldsymbol {u} _ {i} \tag {16.26}
$$

对式（16.26）的两边同时乘以  $X$  ，得到

$$
\frac {1}{N} \boldsymbol {X} \boldsymbol {X} ^ {\mathrm {T}} \left(\boldsymbol {X} \boldsymbol {u} _ {i}\right) = \lambda_ {i} \left(\boldsymbol {X} \boldsymbol {u} _ {i}\right) \tag {16.27}
$$

如果定义  $\pmb{v}_i = \pmb{X}\pmb{u}_i$  ，则可以得到

$$
\frac {1}{N} \boldsymbol {X} \boldsymbol {X} ^ {\mathrm {T}} \boldsymbol {v} _ {i} = \lambda_ {i} \boldsymbol {v} _ {i} \tag {16.28}
$$

这是  $N \times N$  矩阵  $N^{-1}XX^{\mathrm{T}}$  的特征向量方程，它与原始协方差矩阵具有相同的  $N - 1$  个特征值（它本身还有额外的  $D - N + 1$  个特征值为零）。因此，我们可以在维度较低的空间中解决特征向量问题，计算成本为  $\mathcal{O}(N^3)$  而不是  $\mathcal{O}(D^3)$  。要确定特征向量，我们可以对式（16.28）的两边乘以  $X^{\mathrm{T}}$  ，得到

$$
\left(\frac {1}{N} \boldsymbol {X} ^ {\mathrm {T}} \boldsymbol {X}\right) \left(\boldsymbol {X} ^ {\mathrm {T}} \boldsymbol {v} _ {i}\right) = \lambda_ {i} \left(\boldsymbol {X} ^ {\mathrm {T}} \boldsymbol {v} _ {i}\right) \tag {16.29}
$$

从中可以看出  $\left(\boldsymbol{X}^{\mathrm{T}}\boldsymbol{v}_i\right)$  是具有特征值  $\lambda_{\mathrm{i}}$  的  $S$  的特征向量。但请注意，这些特征向量未必是归一化的。为了对它们进行适当的归一化，重新调整  $\boldsymbol{u}_i \propto \boldsymbol{X}^{\mathrm{T}}\boldsymbol{v}_i$ ，使得  $\left\| \boldsymbol{u}_i \right\| = 1$ （见习题16.3），并且假设  $\boldsymbol{v}_{\mathrm{i}}$  已经归一化为单位长度，同时有

$$
\boldsymbol {u} _ {i} = \frac {1}{\left(N \lambda_ {i}\right) ^ {1 / 2}} \boldsymbol {X} ^ {\mathrm {T}} \boldsymbol {v} _ {i} \tag {16.30}
$$

总之，要应用这种方法，我们需要首先计算  $XX^{\mathrm{T}}$  ，然后找到它的特征向量和特征值，最后使用式（16.30）在原始数据空间中计算特征向量。

## 16.2 概率潜变量

我们在16.1节中已经看到PCA可以定义为数据在低维子空间上的线性投影，这个低维子空间的维度比原始数据空间的维度小。每个数据点被映射到由式（16.12）定义的唯一的量  $z_{nj}$ ，我们可以将这些量视为确定性的潜变量。为了引入和论证连续概率潜变量，本节会展示，PCA也可以表示为概率潜变量模型的最大似然解。这种PCA的重新表述，称为概率PCA，与传统PCA相比，有如下优势。

■ 概率PCA模型表示了高斯分布的一种约束形式，在仍然能够捕捉数据集中主要相关性的同时，还可以限制自由参数的数量。  
■我们可以为PCA推导出一种EM算法，该算法在只需要少数几个主特征向量的情况下计算效率很高，而且能避免将计算数据协方差矩阵作为中间步骤进行计算（见16.3.2小节）。  
■ 概率PCA模型与EM算法的结合使我们能够处理数据集中缺失的值。  
■可以原则性地构造概率PCA模型的混合模型并使用EM算法进行训练。  
■ 似然函数的存在使得我们可以与其他概率密度模型直接进行比较。相比之下，传统PCA模型会给予靠近主成分子空间（也可简称主子空间）且距离训练数据

任意远的数据点一个较低的重建成本。

■ 概率 PCA 可用于建模类 - 条件密度，因此可应用于分类问题。  
■ 概率PCA模型可以生成性地运行以提供来自分布的样本。  
■ 概率 PCA 构成了基于贝叶斯方法处理 PCA 的基础，我们可以从数据中自动找到主成分分子空间的维度（Bishop, 2006）。

将PCA作为概率模型的这种表述是由Tipping and Bishop（1997;1999）以及Roweis（1998）独立提出的。正如我们稍后将看到的，它与因子分析（factor analysis）（Basilevsky, 1994）密切相关。

### 16.2.1 生成式模型

概率PCA是线性-高斯框架的一个简单例子，其中所有的边缘分布和条件分布都是高斯分布。我们可以首先通过引入对应于主成分子空间的、明确的  $M$  维潜变量  $\pmb{z}$  来构建概率PCA。接下来，我们定义一个关于潜变量的高斯先验分布  $p(z)$ ，以及基于潜变量值的  $D$  维观测变量  $\pmb{x}$  的高斯条件分布  $p(x|z)$ 。具体来说，关于  $\pmb{z}$  的先验分布由零均值、单位协方差的高斯分布给出：

$$
p (z) = \mathcal {N} (z | \mathbf {0}, I) \tag {16.31}
$$

同样，观测变量  $x$  的条件分布，基于潜变量  $z$  的值，也符合高斯分布：

$$
p (\boldsymbol {x} \mid \boldsymbol {z}) = \mathcal {N} \left(\boldsymbol {x} \mid W z + \boldsymbol {\mu}, \sigma^ {2} I\right) \tag {16.32}
$$

其中  $x$  的均值是  $z$  的一般线性函数，受  $D \times M$  维矩阵  $W$  和  $D$  维向量  $\mu$  的控制。注意，这关于  $x$  的元素是因子化的（见11.2.3小节）。换句话说，这是一个朴素贝叶斯模型的例子。正如我们稍后将看到的， $W$  的列跨越了数据空间内的线性子空间，对应于主子空间。这个模型中的另一个参数是控制条件分布方差的标量  $\sigma^2$  。请注意，潜分布  $p(z)$  是零均值、单位协方差的高斯分布，这一假设不失一般性，因为更普遍的高斯分布也会产生一个等价的概率模型（见习题16.4）。

我们可以从生成性视角来看概率PCA模型。首先选择一个潜变量，然后基于这个潜变量的值采样观测变量，得到观测变量的样本值。具体来说， $D$  维观测变量  $\pmb{x}$  是通过为  $M$  维潜变量  $\pmb{z}$  加上加性高斯噪声的线性变换来定义的，所以有

$$
\boldsymbol {x} = \boldsymbol {W} \boldsymbol {z} + \boldsymbol {\mu} + \varepsilon \tag {16.33}
$$

其中  $z$  是一个  $M$  维的高斯潜变量， $\varepsilon$  是一个  $D$  维的零均值高斯分布噪声变量，协方差为  $\sigma^2 I$ 。这个生成过程如图16.7所示。注意，这个框架基于从潜空间到数据空间的映射，而不是前面讨论的常规PCA视角。稍后，我们将利用贝叶斯定理得出从数据空间到潜空间的反向映射。

![](img/a70946861c7f9f89af51b8818e8396d67e4cd17b2b0fdec77d56a735cb14fdb9.jpg)  
图16.7 从常规PCA视角看一个针对两维数据空间和一维潜空间的概率PCA模型的生成过程。生成观测数据点  $x$  的过程是：首先从其先验分布  $p(z)$  中抽取潜变量的一个值  $\hat{z}$ ，然后从具有均值  $\omega \hat{z} + \mu$  和协方差  $\sigma^2 I$  的各向同性高斯分布（见红色圆圈）中抽取  $x$  的值。绿色椭圆显示了边缘分布  $p(x)$  的密度等高线

### 16.2.2 似然函数

假设我们希望通过最大似然法确定参数  $W$  、  $\pmb{\mu}$  和  $\sigma^2$  的值。为了写下似然函数，我们需要观测变量的边缘分布  $p(x)$  的表达式。根据概率的加和法则与乘积法则，这可以表示为

$$
p (\boldsymbol {x}) = \int p (\boldsymbol {x} | \boldsymbol {z}) p (\boldsymbol {z}) \mathrm {d} \boldsymbol {z} \tag {16.34}
$$

由于对应于线性-高斯模型，这个边缘分布同样是高斯分布，由下式给出：

$$
p (\boldsymbol {x}) = \mathcal {N} (\boldsymbol {x} \mid \boldsymbol {\mu}, \boldsymbol {C}) \tag {16.35}
$$

其中  $D\times D$  的协方差矩阵  $\pmb{C}$  定义如下：

$$
\boldsymbol {C} = \boldsymbol {W} \boldsymbol {W} ^ {\mathrm {T}} + \sigma^ {2} \boldsymbol {I} \tag {16.36}
$$

注意预测分布也是高斯分布，因此通过使用式（16.33）计算其均值和协方差就能更直接地得出同样的结果，于是有

$$
\mathbb {E} [ \boldsymbol {x} ] = \mathbb {E} [ \boldsymbol {W z} + \boldsymbol {\mu} + \varepsilon ] = \boldsymbol {\mu} \tag {16.37}
$$

$$
\begin{array}{l} \operatorname {c o v} [ \boldsymbol {x} ] = \mathbb {E} \left[ (\boldsymbol {W} \boldsymbol {z} + \varepsilon) (\boldsymbol {W} \boldsymbol {z} + \varepsilon) ^ {\mathrm {T}} \right] \\ = \mathbb {E} \left[ \boldsymbol {W} z \boldsymbol {z} ^ {\mathrm {T}} \boldsymbol {W} ^ {\mathrm {T}} \right] + \mathbb {E} \left[ \varepsilon \varepsilon^ {\mathrm {T}} \right] (16.38) \\ = \boldsymbol {W} \boldsymbol {W} ^ {\mathrm {T}} + \sigma^ {2} \boldsymbol {I} (16.39) \\ \end{array}
$$

这里我们利用了如下事实：  $z$  和  $\varepsilon$  是独立随机变量，因此它们不相关。

直观上，我们可以认为分布  $p(\boldsymbol{x})$  是通过取一个各向同性的高斯“喷雾罐”，然后在主成分子空间上移动并喷洒以  $\sigma^2$  决定密度的高斯墨水，并通过先验分布来加权的。累积的墨水密度将产生一个代表边缘密度  $p(\boldsymbol{x})$  的“松饼”状分布。

预测分布  $p(x)$  受参数  $\mu$  、  $W$  和  $\sigma^2$  的控制。然而，这种参数化中存在冗余，相当于潜空间坐标发生旋转。为了说明这一点，考虑一个矩阵  $\widetilde{W} = WR$  ，其中  $R$  是一个正交矩阵。利用正交性质  $RR^{\mathrm{T}} = I$  ，我们看到出现在协方差矩阵  $C$  中的量  $\widetilde{W}\widetilde{W}^{\mathrm{T}}$  具有以下形式：

$$
\widetilde {\boldsymbol {W}} \widetilde {\boldsymbol {W}} ^ {\mathrm {T}} = \boldsymbol {W} \boldsymbol {R} \boldsymbol {R} ^ {\mathrm {T}} \boldsymbol {W} ^ {\mathrm {T}} = \boldsymbol {W} \boldsymbol {W} ^ {\mathrm {T}} \tag {16.40}
$$

正因为独立于  $\pmb{R}$ ，所以存在一整套矩阵  $\widetilde{\pmb{W}}$ ，所有这些矩阵都会导致相同的预测分布。这种不变性可以理解为潜空间内的旋转。稍后我们将返回讨论该模型中独立参数的数量问题。

当计算预测分布时，我们需要  $C^{-1}$ ，它涉及计算一个  $D \times D$  矩阵的逆[参见式（A.7）]：

$$
\boldsymbol {C} ^ {- 1} = \sigma^ {- 2} \boldsymbol {I} - \sigma^ {- 2} \boldsymbol {W} \boldsymbol {M} ^ {- 1} \boldsymbol {W} ^ {\mathrm {T}} \tag {16.41}
$$

其中  $M \times M$  矩阵  $M$  定义如下：

$$
\boldsymbol {M} = \boldsymbol {W} ^ {\mathrm {T}} \boldsymbol {W} + \sigma^ {2} \boldsymbol {I} \tag {16.42}
$$

因为我们求  $M$  的逆而不是直接求  $C$  的逆，所以计算  $C^{-1}$  的成本从  $\mathcal{O}(D^3)$  降低到了  $\mathcal{O}(M^3)$ 。

除了预测分布  $p(\pmb {x})$  ，我们还需要后验分布  $p(z|\pmb {x})$  ，后者可以直接再次使用线性-高斯模型的结果［式（3.100）]写出：

$$
p (z | \boldsymbol {x}) = \mathcal {N} \left(\boldsymbol {z} \mid \boldsymbol {M} ^ {- 1} \boldsymbol {W} ^ {\mathrm {T}} (\boldsymbol {x} - \boldsymbol {\mu}), \sigma^ {2} \boldsymbol {M} ^ {- 1}\right) \tag {16.43}
$$

请注意，后验分布的均值依赖于  $x$  ，而后验分布的协方差独立于  $x$  。

### 16.2.3 最大似然法

接下来考虑使用最大似然法确定模型参数。给定一组观测数据点的数据集  $X = \{x_{n}\}$ ，概率PCA模型可以表示为一个有向图，如图16.8所示。相应的对数似然函数由式（16.35）给出，形式为

![](img/c93a0b13c5f6e2fd59113d5c9285137be7de3bd634f1628b51074a13bb7f738f.jpg)  
图16.8 对于包含  $x$  的  $N$  次观测的一个数据集，概率PCA模型可以表示为一个有向图，其中的每个观测值  $x_{n}$  与潜变量的一个值  $z_{n}$  相关联

$$
\begin{array}{l} \ln p \left(X \mid \boldsymbol {\mu}, W, \sigma^ {2}\right) = \sum_ {n = 1} ^ {N} \ln p \left(\boldsymbol {x} _ {n} \mid W, \boldsymbol {\mu}, \sigma^ {2}\right) \\ = - \frac {N D}{2} \ln (2 \pi) - \frac {N}{2} \ln | C | - \frac {1}{2} \sum_ {n = 1} ^ {N} \left(x _ {n} - \mu\right) ^ {\mathrm {T}} C ^ {- 1} \left(x _ {n} - \mu\right) \quad (1 6. 4 4) \\ \end{array}
$$

将对数似然函数关于  $\pmb{\mu}$  的导数设为零便可得到预期结果  $\pmb {\mu} = \overline{\pmb{x}}$  ，其中  $\overline{x}$  是由式（16.1）定义的数据均值。由于对数似然函数是  $\pmb{\mu}$  的二次函数，这个解代表了唯一的最大值，可以通过计算二阶导数来验证（见练习16.9）。代入原式后，我们可以将对数似然函数写成以下形式：

$$
\ln p \left(\boldsymbol {X} \mid \boldsymbol {W}, \boldsymbol {\mu}, \sigma^ {2}\right) = - \frac {N}{2} \left\{D \ln (2 \pi) + \ln | \boldsymbol {C} | + \operatorname {t r} \left(\boldsymbol {C} ^ {- 1} \boldsymbol {S}\right) \right\} \tag {16.45}
$$

其中  $S$  是由式（16.3）定义的数据协方差矩阵。

虽然似然函数关于  $W$  和  $\sigma^2$  的最大化更为复杂，但仍有精确的闭合解。Tipping and Bishop（1999）指出，对数似然函数的所有驻点都可以写成

$$
\boldsymbol {W} _ {\mathrm {M L}} = \boldsymbol {U} _ {M} \left(\boldsymbol {L} _ {M} - \sigma^ {2} \boldsymbol {I}\right) ^ {1 / 2} \boldsymbol {R} \tag {16.46}
$$

其中  $U_{M}$  是一个  $D \times M$  矩阵，其列由数据协方差矩阵  $\pmb{S}$  的任意子集（大小为  $M$ ）的特征向量给出。 $M \times M$  对角矩阵  $\pmb{L}_{M}$  的元素由对应的特征值  $\lambda_{1}$  给出。 $\pmb{R}$  是一个任意的  $M \times M$  正交矩阵。

此外，Tipping and Bishop（1999）还指出，当选择那些对应于前  $M$  个最大的特征值的特征向量时，就会得到似然函数的最大值（所有其他解都是鞍点）。Roweis（1998）也独立提出了类似的猜想，尽管没有给出证明。同样，假设特征向量已按照对应特征值递减的顺序排列，则  $M$  个主特征向量是  $\pmb{u}_1, \dots, \pmb{u}_M$  。在这种情况下， $\pmb{W}$  的列定义了标准PCA的主子空间。对应的  $\sigma^2$  的最大似然解为

$$
\sigma_ {\mathrm {M L}} ^ {2} = \frac {1}{D - M} \sum_ {i = M + 1} ^ {D} \lambda_ {i} \tag {16.47}
$$

$\sigma_{\mathrm{ML}}^{2}$  是与被丢弃维度相关联的平均方差。

$\pmb{R}$  是正交的，因此可以解释为  $M$  维潜空间中的旋转矩阵。如果我们将  $\pmb{W}$  的解代入  $\pmb{C}$  的表达式并利用正交性质  $\pmb{R}\pmb{R}^{\mathrm{T}} = \pmb{I}$ ，就可以看到  $\pmb{C}$  独立于  $\pmb{R}$ 。这表明预测密度不受潜空间内部旋转的影响。对于特殊情况  $\pmb{R} = \pmb{I}$ ，我们可以看到  $\pmb{W}$  的列是主成分特征向量，按方差参数  $\lambda_{i} - \sigma^{2}$  缩放。一旦认识到独立高斯分布（在这种情况下是潜空间分布和噪声模型）的卷积的方差是可加的，这些缩放因子的解释也就清楚了。因此，沿着特征向量  $\pmb{u}_{i}$  方向的方差  $\lambda_{i}$  由以下两部分之和构成：一是单位方差潜空间分布通过  $\pmb{W}$  的对应列投影到数据空间的贡献  $\lambda_{i} - \sigma^{2}$ ，二是噪声模型在所有方向上添加的各向同性方差贡献  $\sigma^{2}$ 。

我们有必要花些时间研究式（16.36）给出的数据协方差矩阵的形式。考虑预测分布沿某个由单位向量  $\pmb{\nu}$  指定的方向的方差，其中  $\pmb{v}^{\mathrm{T}}\pmb{\nu} = 1$  ，由  $\pmb{v}^{\mathrm{T}}\pmb{C}\pmb{\nu}$  给出。首先假设  $\pmb{\nu}$  与主子空间正交，换句话说，  $\pmb{\nu}$  是由被丢弃特征向量的某种线性组合给出的。由于  $\pmb{v}^{\mathrm{T}}\pmb{U} = \mathbf{0}$  因此  $\pmb{v}^{\mathrm{T}}\pmb{C}\pmb{\nu} = \sigma^{2}$  。该模型预测了正交于主子空间的噪声方差，从式（16.47）来看，它

正是被丢弃特征值的平均值。假设  $\pmb{v} = \pmb{u}_i$  ，其中  $\pmb{u}_i$  是定义主子空间的保留特征向量之一，则有  $\pmb{v}^{\mathrm{T}}\pmb{C}\pmb{v} = (\lambda_i - \sigma^2) + \sigma^2 = \lambda_i$  。

换句话说，这个模型正确地捕捉了数据沿主轴的方差，并用单个平均值  $\sigma^2$  近似所有其余方向的方差。

构建最大似然密度模型的一种方法是，简单地找到数据协方差矩阵的特征向量和特征值，然后使用上述结果计算  $W$  和  $\sigma^2$  。在这种情况下，选择  $R = I$  会更为方便。然而，如果我们通过数值优化似然函数来找到最大似然解，例如使用共轭梯度算法（Fletcher, 1987; Nocedal and Wright, 1999）或EM算法，那么得到的  $R$  矩阵的值本质上是任意的（见16.3.2小节）。这意味着矩阵  $W$  的列不必是正交的。如果需要正交基，则可以适当地对  $W$  矩阵进行后处理（Golub and Van Loan, 1996）。也可以修改EM算法，以直接产生按相应特征值大小排序的正交主成分方向（Ahn and Oh, 2003）。

潜空间中的旋转不变性表示了一种统计上的不可识别性，类似于我们在离散潜变量混合模型中遇到的情况。这里，参数的连续性使得任何值都会导致相同的预测密度，这与混合设置中组件重标记相关的离散不可识别性形成了对比。

考虑  $M = D$  ，也就是没有降维的情况，则  $\mathbf{U}_M = \mathbf{U}$  且  $\pmb {L}_M = \pmb{L}$  。利用正交性质 $\pmb {U}\pmb{U}^{\mathrm{T}} = \pmb{I}$  和  $\pmb {R}\pmb{R}^{\mathrm{T}} = \pmb{I}$  ，  $\pmb{x}$  的边缘分布的数据协方差矩阵  $C$  将变为

$$
\boldsymbol {C} = \boldsymbol {U} \left(\boldsymbol {L} - \sigma^ {2} \boldsymbol {I}\right) ^ {1 / 2} \boldsymbol {R} \boldsymbol {R} ^ {\mathrm {T}} \left(\boldsymbol {L} - \sigma^ {2} \boldsymbol {I}\right) ^ {1 / 2} \boldsymbol {U} ^ {\mathrm {T}} + \sigma^ {2} \boldsymbol {I} = \boldsymbol {U} \boldsymbol {L} \boldsymbol {U} ^ {\mathrm {T}} = \boldsymbol {S} \tag {16.48}
$$

于是我们得到无约束高斯分布的标准最大似然解，其中数据协方差矩阵由样本协方差给出。

传统PCA通常构造为从  $D$  维数据空间到  $M$  维线性子空间的点的投影。然而，概率PCA可以最自然地表示为从潜空间到数据空间的映射。对于可视化和数据压缩等应用，我们可以使用贝叶斯定理反转这个映射。随后，数据空间中的任何点就可以由其在潜空间中的后验均值和协方差来概括。从式（16.43）可知，均值由下式给出：

$$
\mathbb {E} [ \boldsymbol {z} | \boldsymbol {x} ] = \boldsymbol {M} ^ {- 1} \boldsymbol {W} _ {\mathrm {M L}} ^ {\mathrm {T}} (\boldsymbol {x} - \overline {{\boldsymbol {x}}}) \tag {16.49}
$$

其中  $M$  由式（16.42）给出。这会将点投影到由下式给出的数据空间中。

$$
\boldsymbol {W} \mathbb {E} [ \boldsymbol {z} | \boldsymbol {x} ] + \boldsymbol {\mu} \tag {16.50}
$$

请注意，这与正则化线性回归的方程相同，并且是最大化线性-高斯模型的似然函数的结果（见4.1.6小节）。同样，从式（16.43）可以看出，后验协方差由  $\sigma^2 M^{-1}$  给出，且与  $x$  无关。

取极限  $\sigma^2\to 0$  ，后验均值可以简化为

$$
\left(\boldsymbol {W} _ {\mathrm {M L}} ^ {\mathrm {T}} \boldsymbol {W} _ {\mathrm {M L}}\right) ^ {- 1} \boldsymbol {W} _ {\mathrm {M L}} ^ {\mathrm {T}} (\boldsymbol {x} - \bar {\boldsymbol {x}}) \tag {16.51}
$$

这代表数据点在潜空间中的正交投影，因此我们恢复了标准PCA模型（见练习

16.11)。然而，在这个极限下，后验协方差为零，密度变得奇异。对于  $\sigma^2 > 0$  ，潜投影相对于正交投影向原点移动（见练习16.12）。

最后请注意，概率PCA模型的一个重要作用是定义了一个多元高斯分布，其中自由度的数量（也就是独立参数的数量）是可以控制的，同时仍允许模型捕获数据中的主要相关性。请回想一下，一个普通的高斯分布在协方差矩阵中有  $D(D + 1) / 2$  个独立参数（另外还有  $D$  个参数在均值中）（见3.2节）。因此，参数的数量随着  $D$  的增加而按平方增长，并且在高维空间中可能会变得过多。如果我们限制协方差矩阵为对角矩阵，那便只有  $D$  个独立参数，参数的数量随维度线性增长。然而，这会将变量当作独立变量，因此已无法再表达它们之间的任何相关性。概率PCA模型提供了一个优雅的折中方案，其中  $M$  个最重要的相关性可以捕获，同时确保参数的数量只随维度线性增长。我们可以通过计算概率PCA模型中自由度数的量看到这一点。数据协方差矩阵  $\pmb{C}$  取决于  $\pmb{W}$ ，大小为  $D\times M$ ，再加上  $\sigma^2$ ，总共有  $DM + 1$  个参数。然而，我们已经看到这种参数化中存在与潜空间坐标系的旋转相关的一些冗余。表达这些旋转的正交矩阵  $\pmb{R}$  的大小为  $M\times M$ 。在这个矩阵的第一列中，有  $M - 1$  个独立参数（因为列向量必须归一化为单位长度）；而在第二列中，有  $M - 2$  个独立参数（因为列必须归一化并且与前一列正交），以此类推。将这个等差数列求和后，我们看到  $\pmb{R}$  总共有  $M(M - 1) / 2$  个独立参数。因此，数据协方差矩阵  $\pmb{C}$  中自由度的数量由下式给出：

$$
D M + 1 - M (M - 1) / 2 \tag {16.52}
$$

在固定  $M$  的情况下，该模型中独立参数的数量仅随维度线性增长。取  $M = D - 1$  ，便可得到全协方差高斯分布的标准结果（见习题16.14）。在这种情况下，沿  $D - 1$  个线性无关方向的方差由  $\pmb{W}$  的列控制，而沿剩余方向的方差由  $\sigma^2$  给出。当  $M = 0$  时，该模型等价于各向同性协方差的情况。

### 16.2.4 因子分析

因子分析模型是一个线性-高斯潜变量模型，它与概率PCA模型密切相关。因子分析的独特性体现为，其使得给定潜变量  $z$  的观测变量  $\pmb{x}$  的条件分布具有对角协方差而不是各向同性协方差，即

$$
p (\boldsymbol {x} \mid z) = \mathcal {N} (\boldsymbol {x} \mid W z + \boldsymbol {\mu}, \boldsymbol {\Psi}) \tag {16.53}
$$

其中  $\Psi$  是一个  $D\times D$  的对角矩阵。注意，因子分析模型和概率PCA模型一样，也假设观测变量  $x_{1},\dots ,x_{D}$  在给定潜变量  $\pmb{z}$  时是相互独立的。因子分析模型的核心在于，它通过用矩阵  $\Psi$  表示与每个坐标相关的独立方差，并在矩阵  $\pmb{W}$  中捕获变量之间的协方差，来解释观测到的数据协方差结构。在因子分析文献中，捕获观测变量之间相关性的  $\pmb{W}$  的列称为因子载荷（factor loading），对角矩阵  $\Psi$  中代表每个变量独立噪声方差的对角元素则称为唯一性（uniquenesses）元素。

在 Everitt（1984）、Bartholomew（1987）和 Basilevsky（1994）等人的文献中可

以找到关于因子分析的论述。Lawley（1953）和 Anderson（1963）研究了因子分析与 PCA 之间的联系，他们揭示了在似然函数的驻点处，对于一个  $\Psi = \sigma^2 I$  的因子分析模型， $\pmb{W}$  的列是样本协方差矩阵的缩放特征向量，而  $\sigma^2$  是被丢弃特征值的平均值。后来，Tipping and Bishop（1999）指出，在将组成  $\pmb{W}$  的特征向量选为主特征向量时，对数似然函数的最大值便会出现。

利用式（16.34），我们可以看到观测变量的边缘分布是由  $p(\pmb {x}) = \mathcal{N}(\pmb {x}|\pmb {\mu},\pmb {C})$  给出的，其中

$$
\boldsymbol {C} = \boldsymbol {W} \boldsymbol {W} ^ {\mathrm {T}} + \boldsymbol {\Psi} \tag {16.54}
$$

与概率PCA模型一样，该模型对潜空间中的旋转是不变的（见习题16.16）。

历史上，当人们尝试用因子分析方法对各个因子（ $z$  空间中的坐标）进行解释时，曾引发争议。因子分析与这个空间中的旋转相关联，导致不可识别性，因而这种尝试被证明是有问题的。然而，从我们的角度看，我们将因子分析视为一种潜变量密度模型，关注的重点是其中潜空间的结构形式，而不是用于描述该空间的具体坐标。如果希望消除与潜空间旋转所带来的不确定性，则必须考虑非高斯潜变量分布，以产生独立成分分析模型（参见16.2.5小节）。

概率PCA和因子分析的另一个区别是它们在数据集变换下的行为不同（见习题16.17）。对于PAC和概率PCA，如果旋转数据空间中的坐标系，则可以得到与数据完全相同的拟合结果，只不过  $\pmb{W}$  矩阵会由相应的旋转矩阵变换而成的。然而，对于因子分析，相似的性质是，如果我们对数据向量逐个分量地进行重新缩放，则这个操作会被吸收进相应的  $\pmb{\Psi}$  元素的重新缩放中。

### 16.2.5 独立成分分析

线性高斯潜变量模型的一种扩展是这样的模型：观测变量与潜变量线性相关，但潜变量分布是非高斯分布。这类模型的一个重要分支是独立成分分析（Independent Component Analysis，ICA）模型，其潜变量的联合分布能够分解为各分量的独立分布之积，即

$$
p (z) = \prod_ {j = 1} ^ {M} p \left(z _ {j}\right) \tag {16.55}
$$

为了理解这些模型的作用，可以设想以下场景：有两个人在同时说话，我们使用两个麦克风录制他们的声音。如果忽略时间延迟和回声等效果，则任何时刻麦克风接收到的信号都将由这两个声音的振幅的线性组合给出。这个线性组合的系数是恒定的，如果我们能从样本数据中推断出它们的值，我们就可以反转混合过程（假设它是非奇异的），从而得到两个干净的信号，每个信号只包含一个人的声音。这是一个盲源分离的问题，“盲”指的是我们只给出了混合数据，而没有观察到原始来源或混合系数（Cardoso, 1998）。

这类问题有时可以采取以下方法来解决（MacKay, 2003）。该方法忽略信号的时间特性，并将连续的样本视为独立同分布的。考虑一个生成式模型，其中有两个潜变量，对应于未观测语音信号振幅；还有两个观测变量，由麦克风处信号值给出。潜变量具有可以进行因式分解的联合分布，并且观测变量由潜变量的线性组合给出。没有必要包括噪声分布，因为潜变量的数量等于观测变量的数量，而观测变量的边缘分布通常不具有奇异性，所以观测变量仅仅是潜变量的确定性线性组合。给定一组观测数据，该模型的似然函数是线性组合中系数的一个函数。在使用基于梯度的优化方法最大化对数似然函数时，会产生特定版本的ICA。

这种方法的成功依赖于潜变量具有非高斯分布。为了理解这一点，请回想一下，在概率PCA（以及因子分析）中，潜空间分布由零均值、各向同性的高斯分布给出。因此模型无法区分仅仅通过潜空间中的旋转就可以互相转换的两个不同的潜变量。我们可以直接验证这一点，注意边缘密度［式（16.35）］及由此产生的似然函数在我们做  $W\rightarrow WR$  变换时会保持不变（其中  $\pmb{R}$  是一个满足  $\boldsymbol {R}\boldsymbol{R}^{\mathrm{T}} = \boldsymbol{I}$  的正交矩阵），这是因为式（16.36）给出的矩阵  $c$  本身是不变的。即使扩展模型以允许更通用的高斯潜变量分布，此结论也不会改变，因为我们已经看到，这样的模型等价于零均值、各向同性的高斯潜变量模型。

对于为什么高斯潜变量分布在线性模型中不足以找到独立成分，我们还可以用另一种方式来理解。我们注意到，主成分代表了数据空间中坐标系的旋转，目的是使协方差矩阵对角化，从而使新坐标中的数据分布不再相关。尽管零相关是独立性的必要条件，但它不是充分条件（见习题2.39）。

在实践中，常用的潜变量分布由下式给出：

$$
p \left(z _ {j}\right) = \frac {1}{\pi \cos \left(z _ {j}\right)} = \frac {2}{\pi \left(\mathrm {e} ^ {z _ {j}} + \mathrm {e} ^ {- z _ {j}}\right)} \tag {16.56}
$$

与高斯分布相比，它具有重尾（heavy tail），我们在许多真实世界的分布中也能观察到这种特性。

最初的ICA模型（Bell and Sejnowski, 1995）是采用信息最大化定义的目标函数进行优化。概率潜变量表述的一个优势是，它有助于启发并构建ICA的扩展。例如，独立因子分析（independent factor analysis）（Attias, 1999）考虑了一个模型，在该模型中，潜变量和观测变量的数量可以不同，观测变量是有噪声的，潜变量则具有高斯混合模型化的灵活分布。可以使用EM算法最大化这个模型的对数似然函数，并使用变分法近似重构潜变量。研究人员考虑了许多其他类型的模型，如今关于独立成分分析及其应用的文献已经不胜枚举了（Jutten and Herault, 1991; Comon, Jutten, and Herault, 1991; Amari, Cichocki, and Yang, 1996; Pearlmutter and Parra, 1997; Hyvarinen and Oja, 1997; Hinton et al., 2001; Miskin and MacKay, 2001; Hojen-Sorensen, Winther, and Hansen, 2002; Choudrey and Roberts, 2003; Chan, Lee, and Sejnowski, 2003; Stone, 2004）。

### 16.2.6 卡尔曼滤波器

到目前为止，我们假设数据值是独立同分布的。一种常见的非独立同分布情况是，数据点形成一个有序序列。我们已经看到，隐马尔可夫模型可以视为混合模型的扩展，旨在允许数据中存在序列相关性（参见15.3.1小节）。类似地，连续潜变量模型可以通过连接潜变量形成马尔可夫链来扩展，以处理序列数据，参见图16.9所示的图模型。这种模

![](img/28e8466f139aabe8b394438149fba578e9dfe6189a0ffd2267591e8faddd1108.jpg)  
图16.9 用于序列数据的概率图模型称为线性动态系统或卡尔曼滤波器，其中的潜变量形成了一个马可夫链

型称为线性动态系统（linear dynamical system）或卡尔曼滤波器（Zarchan and Musoff, 2005）。注意，这与隐马尔可夫模型的图结构相同（参见15.3.1小节）。值得注意的是，隐马尔可夫模型和线性动态系统是独立发展的。然而，一旦它们都表示为图模型，它们之间的深层关系就会水落石出。卡尔曼滤波器已广泛地应用在许多实时跟踪应用中，如使用雷达信号跟踪飞机。

在这样的最简单模型中，图16.9中的分布  $p\left(x_{n} \mid z_{n}\right)$  代表了该特定观测结果的线性-高斯潜变量模型，就像我们之前讨论的独立同分布数据一样。然而，潜变量  $z_{n}$  不再视为独立的，而是形成了一个马尔可夫链，链中每个潜变量的分布  $p\left(z_{n} \mid z_{n-1}\right)$  取决于链中前一个潜变量的状态。在线性-高斯潜变量模型中， $z_{n}$  的分布是高斯分布，均值则由  $z_{n-1}$  的线性函数给出。通常，分布  $p\left(x_{n} \mid z_{n}\right)$  的所有参数是共享的，分布  $p\left(z_{n} \mid z_{n-1}\right)$  的所有参数也是共享的，因此模型中总的参数数量是固定的，不依赖于序列的长度。这些参数可以通过最大似然法从数据中来学习。这需要用到在图模型上传播消息的有效算法（Bishop, 2006）。在本章的其余部分，我们将专注于独立同分布数据。

## 16.3 证据下界

在讨论离散潜变量模型时，我们推导出了边缘对数似然的证据下界（ELBO），并展示了它是如何构成期望最大化（EM）算法的基础的，包括其扩展，例如变分推断（参见15.4节）。相同的框架同样适用于连续潜变量模型，以及结合了离散和连续变量的模型。这里我们提供一个稍微不同的ELBO推导，并且假设潜变量  $z$  是连续的。

考虑一个带有观测变量  $x$  、潜变量  $z$  和可学习参数向量  $\pmb{w}$  的模型  $p(x, z \mid w)$  。只要引入潜变量上的任意一个分布  $q(z)$ ，我们就可以将对数似然函数  $\ln p(x \mid w)$  写成两项之和的形式（见习题16.18）：

$$
\ln p (\boldsymbol {x} \mid \boldsymbol {w}) = \mathcal {L} (\boldsymbol {w}) + \operatorname {K L} (q (\boldsymbol {z}) \| p (\boldsymbol {z} \mid \boldsymbol {x}, \boldsymbol {w})) \tag {16.57}
$$

其中

$$
\mathcal {L} (q, \boldsymbol {w}) = \int q (z) \ln \left\{\frac {p (\boldsymbol {x} , z \mid \boldsymbol {w})}{q (z)} \right\} d z \tag {16.58}
$$

$$
\operatorname {K L} \left(q (z) \| p (z | x, w)\right) = - \int q (z) \ln \left\{\frac {p (z | x , w)}{q (z)} \right\} d z \tag {16.59}
$$

$\mathrm{KL}\big(q(z)\big\| p(z|x,w)\big)$  是一种Kullback-Leibler散度，满足特性  $\mathrm{KL}(\cdot \| \cdot)\geqslant 0$  （参见2.5.5小节），从中可以得出

$$
\ln p (x \mid w) \geqslant \mathcal {L} (w) \tag {16.60}
$$

式（16.58）给出的  $\mathcal{L}(q, w)$  构成了对数似然的下界，称为证据下界（ELBO）。 $\mathcal{L}(q, w)$  采用了与离散情况相同的形式[式（15.53）]，但是将求和替换成了积分。

我们可以使用EM算法的两阶段迭代过程来最大化对数似然函数。在EM算法中，我们可以交替地针对  $q(z)$  （E步骤）和  $\pmb{w}$  （M步骤）最大化  $\mathcal{L}(q,w)$  。首先初始化参数  $\pmb{w}^{\mathrm{old}}$  。然后在E步骤中保持  $\pmb{w}$  固定，并针对  $q(z)$  最大化下界。我们注意到，可以通过最小化式（16.59）中的Kullback-Leibler散度来获得下界的最高值。换言之，当Kullback-Leibler散度为零时，  $q(z) = p\big(z|x;\pmb{w}^{\mathrm{old}}\big)$  。在M步骤中，我们保持所选的 $q(z)$  固定，并针对  $\pmb{w}$  最大化  $\mathcal{L}(q,w)$  。将  $q(z)$  代入式（16.58），可以得到

$$
\begin{array}{l} \mathcal {L} (q, w) = \int p (z | x, w ^ {\text {o l d}}) \ln p (x, z | w) \mathrm {d} z - \tag {16.61} \\ \int p (z \mid x, w ^ {\text {o l d}}) \ln p (z \mid x, w ^ {\text {o l d}}) \mathrm {d} z \\ \end{array}
$$

在M步骤中保持  $\pmb{w}^{\mathrm{old}}$  固定，并针对  $\pmb{w}$  最大化  $\mathcal{L}(q, w)$  。请注意，式（16.61）右侧的第二项独立于  $\pmb{w}$ ，因此在M步骤中可以忽略。式（16.61）右侧的第一项则是完整数据对数似然的期望，该期望来自我们用  $\pmb{w}^{\mathrm{old}}$  计算得到的  $z$  的后验分布（参见15.3节）。

如果我们有一个由独立同分布观测值组成的数据集  $x_{1},\dots ,x_{N}$  ，则似然函数的形式为

$$
\ln p (\boldsymbol {X} \mid \boldsymbol {w}) = \sum_ {n = 1} ^ {N} \ln p \left(\boldsymbol {x} _ {n} \mid \boldsymbol {w}\right) \tag {16.62}
$$

其中数据矩阵  $X$  包含  $x_{1},\dots ,x_{N}$  ，并且参数  $\pmb{w}$  在所有数据点之间是共享的。对于每一个数据点，引入一个相应的潜变量  $z_{n}$  及其关联的分布  $q(z_{n})$  ，并且通过遵循类似于推导式（16.58）时使用的步骤，我们得到了以下形式的ELBO（参见19.2节）：

$$
\mathcal {L} (q, w) = \sum_ {n = 1} ^ {N} \int q (z _ {n}) \ln \left\{\frac {p (x _ {n} , z _ {n} \mid w)}{q (z _ {n})} \right\} d z _ {n} \tag {16.63}
$$

当讨论变分自编码器（见19.2节）时，我们将遇到一个模型，其E步骤的精确解

无法实现，因此我们改为使用深度神经网络对  $q(z)$  进行建模，然后使用ELBO来学习网络的参数。

### 16.3.1 EM算法

在本小节，我们将使用EM算法，通过迭代地最大化证据下界来学习概率PCA模型的参数。这看起来有些多余，因为我们已经得到了一个精确的闭式解来求解最大似然参数值，然而在高维空间中，与直接处理样本协方差矩阵相比，使用迭代的EM过程可能在计算上有优势。这个EM过程也可以扩展到没有闭式解的因子分析模型（参见16.2.4小节）。最后，它允许以合理的方式处理缺失数据。

我们可以按照EM算法的通用框架来导出概率PCA的EM算法（参见15.3节）。为此，我们写下完整数据对数似然函数并根据用“旧”参数值评估的潜变量分布的后验分布来获取期望，然后最大化这个完整数据对数似然函数的期望以产生“新”的参数值。因为我们假定数据点是独立的，所以完整数据对数似然函数的形式为

$$
\ln p \left(X, Z \mid \boldsymbol {\mu}, W, \sigma^ {2}\right) = \sum_ {n = 1} ^ {N} \left\{\ln p \left(\boldsymbol {x} _ {n} \mid \boldsymbol {z} _ {n}\right) + \ln p \left(\boldsymbol {z} _ {n}\right) \right\} \tag {16.64}
$$

其中矩阵  $Z$  的第  $n$  行由  $\mathbf{z}_n$  给出。我们已经知道  $\mu$  的精确最大似然解是由样本均值  $\overline{x}$  [式（16.1）]给出的，在这个阶段替代  $\mu$  是很方便的。利用式（16.31）和式（16.32）所示的潜变量分布和条件分布，并且考虑到对潜变量的后验分布的期望，我们得到

$$
\begin{array}{l} \mathbb {E} \left[ \ln p \left(\boldsymbol {X}, \boldsymbol {Z} \mid \boldsymbol {\mu}, \boldsymbol {W}, \sigma^ {2}\right) \right] = - \sum_ {n = 1} ^ {N} \left\{\frac {D}{2} \ln \left(2 \pi \sigma^ {2}\right) + \frac {1}{2} \operatorname {T r} \left(\mathbb {E} \left[ z _ {n} z _ {n} ^ {\mathrm {T}} \right]\right) + \right. \\ \frac {1}{2 \sigma^ {2}} \| \boldsymbol {x} _ {n} - \boldsymbol {\mu} \| ^ {2} - \frac {1}{\sigma^ {2}} \mathbb {E} [ \boldsymbol {z} _ {n} ] ^ {\mathrm {T}} \boldsymbol {W} ^ {\mathrm {T}} (\boldsymbol {x} _ {n} - \boldsymbol {\mu}) + \tag {16.65} \\ \left. \frac {1}{2 \sigma^ {2}} \operatorname {T r} \left(\mathbb {E} \left[ z _ {n} z _ {n} ^ {\mathrm {T}} \right] \boldsymbol {W} ^ {\mathrm {T}} \boldsymbol {W}\right) + \frac {M}{2} \ln (2 \pi) \right\} \\ \end{array}
$$

注意，此情况仅通过高斯分布的充分统计量与后验分布产生关联。因此，在E步骤中，我们使用旧的参数值来计算期望：

$$
\mathbb {E} \left[ \boldsymbol {z} _ {n} \right] = \boldsymbol {M} ^ {- 1} \boldsymbol {W} ^ {\mathrm {T}} \left(\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}\right) \tag {16.66}
$$

$$
\mathbb {E} \left[ z _ {n} z _ {n} ^ {\mathrm {T}} \right] = \sigma^ {2} M ^ {- 1} + \mathbb {E} \left[ z _ {n} \right] \mathbb {E} \left[ z _ {n} \right] ^ {\mathrm {T}} \tag {16.67}
$$

以上这些可以直接从后验分布［式（16.43）］和标准结果  $\mathbb{E}[z_n z_n^{\mathrm{T}}] = \operatorname{cov}[z_n] + \mathbb{E}[z_n]\mathbb{E}[z_n]^{\mathrm{T}}$  中得出。这里的  $M$  由式（16.42）定义。

在  $M$  步骤中，我们固定后验统计量，并针对  $\pmb{W}$  和  $\sigma^2$  进行最大化。针对  $\sigma^2$  的最大化相对简单，针对  $\pmb{W}$  的最大化（见习题16.21）则需要我们利用式（A.24）获得M步骤的如下方程：

$$
\begin{array}{l} \boldsymbol {W} _ {\text {n e w}} = \left[ \sum_ {n = 1} ^ {N} \left(\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}\right) \mathbb {E} \left[ \boldsymbol {z} _ {n} \right] ^ {\mathrm {T}} \right] \left[ \sum_ {n = 1} ^ {N} \mathbb {E} \left[ \boldsymbol {z} _ {n} \boldsymbol {z} _ {n} ^ {\mathrm {T}} \right] \right] ^ {- 1} (16.68) \\ \sigma_ {\text {n e w}} ^ {2} = \frac {1}{N D} \sum_ {n = 1} ^ {N} \left\{\left\| \boldsymbol {x} _ {n} - \bar {\boldsymbol {x}} \right\| ^ {2} - 2 \mathbb {E} \left[ \boldsymbol {z} _ {n} \right] ^ {\mathrm {T}} \boldsymbol {W} _ {\text {n e w}} ^ {\mathrm {T}} \left(\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}\right) + \right. (16.69) \\ \left. \operatorname {t r} \left(\mathbb {E} \left[ z _ {n} z _ {n} ^ {\mathrm {T}} \right] W _ {\text {n e w}} ^ {\mathrm {T}} W _ {\text {n e w}}\right) \right\} \\ \end{array}
$$

概率PCA的EM算法将首先初始化参数，然后交替地在E步骤中使用式（16.66）和式（16.67）计算潜空间后验分布的充分统计量，并在  $M$  步骤中使用式（16.68）和式（16.69）修正参数值。

概率PCA的EM算法的一个好处是，其在大规模应用中的计算效率较高（Roweis,1998）。与基于样本协方差矩阵的特征向量分解的传统PCA不同，EM算法是迭代的，因此可能看起来不那么吸引人。然而，EM算法的每个循环在高维空间中可能比传统PCA的计算效率要高得多。为了理解这一点，请注意，协方差矩阵的特征分解需要 $\mathcal{O}\big(D^3\big)$ 次计算。通常我们只对前  $M$  个特征向量及对应的特征值感兴趣，在这种情况下我们可以使用计算成本仅为  $\mathcal{O}\big(MD^2\big)$  的算法。然而，计算协方差矩阵需要  $\mathcal{O}\big(ND^2\big)$  的计算成本，其中  $N$  是数据点的数量。快照方法（Sirovich,1987）等算法假设特征向量是数据向量的线性组合，这样虽然避免了直接计算协方差矩阵，但是需要  $\mathcal{O}\big(N^3\big)$  的计算成本，因此不适用于大数据集。这里描述的EM算法也没有显式构造协方差矩阵。相反，计算最密集的步骤涉及数据集的求和，其复杂度为  $\mathcal{O}\big(NDM\big)$  。对于很大的维度 $D$  ，  $M$  要远小于  $D$  ，与  $\mathcal{O}\big(ND^2\big)$  相比，这可以显著节省计算成本，并且可以抵消EM算法的迭代成本。

请注意，这个EM算法可以通过在线方式实现，依次读取、处理每个  $D$  维数据点，然后丢弃并考虑下一个数据点。请注意，那些在E步骤中计算的量（一个  $M$  维向量和一个  $M \times M$  矩阵）可以针对每个数据点单独计算。在M步骤中，需要对数据点进行累加求和，这可以逐步进行。如果  $N$  和  $D$  都很大，这种方法可能会有优势。

我们现在有了一个完全概率化的PCA模型。我们可以处理缺失数据，前提是这些数据是随机缺失的（missing at random）。换句话说，决定哪些值缺失的过程不依赖于任何观测到或未观测到的变量的值。这样的数据集可以通过边缘化未观测变量的分布来处理，生成的似然函数可以使用EM算法来最大化（见习题16.22）。

### 16.3.2 PCA的EM算法

EM算法的另一个优雅特性是，我们可以取极限  $\sigma^2\to 0$  ，这对应于传统PCA，并且仍然是一个有效的类EM算法（Roweis,1998）。从式（16.67）中可以看到，在E步骤中唯一需要计算的量是  $\mathbb{E}\bigl [z_n\bigr ]$  。此外，因为  $M = W^{\mathrm{T}}W$  ，所以M步骤也简化了。为了强调算法的简洁性，定义  $\widetilde{X}$  为一个大小为  $N\times D$  的矩阵，其第  $n$  行由向量  $x_{n} - \overline{x}$  给

出。类似地，定义  $\Omega$  为一个大小为  $M\times N$  的矩阵，其第  $n$  列由  $\mathbb{E}\bigl [z_n\bigr ]$  给出。PCA的EM算法的E步骤［式（16.66）]变为

$$
\boldsymbol {\Omega} = \left(\boldsymbol {W} _ {\text {o l d}} ^ {\mathrm {T}} \boldsymbol {W} _ {\text {o l d}}\right) ^ {- 1} \boldsymbol {W} _ {\text {o l d}} ^ {\mathrm {T}} \widetilde {\boldsymbol {X}} ^ {\mathrm {T}} \tag {16.70}
$$

$M$  步骤[式（16.68）]变为

$$
\boldsymbol {W} _ {\text {n e w}} = \widetilde {\boldsymbol {X}} ^ {\mathrm {T}} \boldsymbol {\Omega} ^ {\mathrm {T}} \left(\boldsymbol {\Omega} \boldsymbol {\Omega} ^ {\mathrm {T}}\right) ^ {- 1} \tag {16.71}
$$

以上这些也可以通过在线学习的方式实现。这些方程的简单解释如下。根据我们之前的讨论，E步骤涉及将数据点正交投影到当前估计的主子空间。相应地，M步骤表示在固定投影的情况下重新估计主子空间以最小化重建误差。

我们可以用一个简单的物理类比理解这个EM算法，它在  $D = 2$  和  $M = 1$  时很容易描绘。考虑二维空间中的一系列数据点，用一根坚硬的棒子代表一维主子空间。然后通过符合胡克定律（力与弹簧伸长量成正比，因此储能与弹簧伸长量的平方成正比）的弹簧将每个数据点连接到棒子上。在E步骤中，我们保持棒子固定并允许连接点沿着棒子上下滑动以最小化能量。这会导致每个连接点（独立地）将自己定位在相应数据点到棒子的正交投影位置。在M步骤中，我们保持连接点固定，然后释放棒子，并允许棒子移动到最小能量位置。然后重复E步骤和M步骤，直至满足适当的收敛标准，如图16.10所示。

![](img/f89910bd9b5e3e7343d3bd12d9bff7b89245552ac8facd4c0dc8232ebe69b0b3.jpg)

![](img/dfe9dae06eeebfa3490f66d529c02c77ae6c28b99b3bcc9aae1b39818765ce0c.jpg)

![](img/5acf4b1f93accb891962ebe8c74f68fdd2f5d3364aa64be2ad6465012b810d5b.jpg)

![](img/fc0d070373d7ca0d59edaba7b6cb8302ec066c497078ee68b01fcab0a97a2d5e.jpg)  
图16.10 用一个简单的物理类比来说明PCA的EM算法。(a)一组绿色的数据点以及真实的主成分（显示为按特征值平方根缩放的特征向量）。(b)由  $W$  定义的主子空间的初始配置（显示为红色），以及潜在点  $Z$  在数据空间中的投影（由  $ZW^{\mathrm{T}}$  给出，显示为青色）。(c)经过第一个M步骤后， $W$  已经在  $Z$  保持固定的情况下得到更新。(d)在随后的E步骤中， $Z$  的值已更新并给出了正交投影，并且  $W$  保持不变。(e)经过第二个M步骤之后的结果。(f)收敛后的解

![](img/a892034e3d12d323fc23a59252f1c67a6da1e01e3881aebcd98ab59533d971cc.jpg)

![](img/047fb59f0073975dfeda0ea9b6e4bd04fb6102b7649c06c67335c5f0270da51a.jpg)

### 16.3.3 因子分析的EM算法

我们可以使用最大似然法来确定因子分析模型中的参数  $\mu, W$  和  $\Psi$  。 $\mu$  的解再次由样本均值给出（参见16.2.4小节）。然而，与概率PCA不同的是， $W$  不再有闭式的最大似然解，只能通过迭代来找到。由于因子分析模型是一个潜变量模型，因此可以使用类似于概率PCA的EM算法来完成这项工作（见习题16.24）（Rubin and Thayer, 1982）。具体来说，E步骤方程为

$$
\mathbb {E} \left[ z _ {n} \right] = \boldsymbol {G} \boldsymbol {W} ^ {\mathrm {T}} \boldsymbol {\Psi} ^ {- 1} \left(\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}\right) \tag {16.72}
$$

$$
\mathbb {E} \left[ z _ {n} z _ {n} ^ {\mathrm {T}} \right] = G + \mathbb {E} \left[ z _ {n} \right] \mathbb {E} \left[ z _ {n} \right] ^ {\mathrm {T}} \tag {16.73}
$$

其中

$$
\boldsymbol {G} = \left(\boldsymbol {I} + \boldsymbol {W} ^ {\mathrm {T}} \boldsymbol {\Psi} ^ {- 1} \boldsymbol {W}\right) ^ {- 1} \tag {16.74}
$$

注意这是以一种涉及对  $M \times M$  而不是  $D \times D$  的矩阵（除了  $D \times D$  对角矩阵  $\Psi$ ，其逆在  $\mathcal{O}(D)$  步内可以轻松计算）求逆的形式来表示的，这很方便，因为通常情况下  $M \ll D$  （见习题16.25）。

同样，M步骤方程为

$$
\boldsymbol {W} _ {\text {n e w}} = \left[ \sum_ {n = 1} ^ {N} \left(\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}\right) \mathbb {E} \left[ \boldsymbol {z} _ {n} \right] ^ {\mathrm {T}} \right] \left[ \sum_ {n = 1} ^ {N} \mathbb {E} \left[ \boldsymbol {z} _ {n} \boldsymbol {z} _ {n} ^ {\mathrm {T}} \right] \right] ^ {- 1} \tag {16.75}
$$

$$
\boldsymbol {\Psi} _ {\text {n e w}} = \operatorname {d i a g} \left\{\boldsymbol {S} - \boldsymbol {W} _ {\text {n e w}} \frac {1}{N} \sum_ {n = 1} ^ {N} \mathbb {E} \left[ z _ {n} \right] \left(\boldsymbol {x} _ {n} - \bar {\boldsymbol {x}}\right) ^ {\mathrm {T}} \right\} \tag {16.76}
$$

其中diag运算符会将矩阵的所有非对角元素设置为零。

## 16.4 非线性潜变量模型

到目前为止，在本章中，我们关注的是基于从潜空间到数据空间的线性变换的潜变量模型。自然地，我们会问：是否可以利用深度神经网络的灵活性来表示更复杂的变换，同时利用深度神经网络的学习能力让得到的分布适配于一个数据集？考虑一个简单的向量变量  $z$  上的分布，例如高斯分布：

$$
p _ {z} (z) = \mathcal {N} (z | \mathbf {0}, I) \tag {16.77}
$$

假设我们使用一个由深度神经网络给出的函数  $x = g(z, w)$  来变换  $z$ ，其中  $w$  代表权重和偏置。 $z$  上的分布与神经网络的结合定义了  $x$  上的分布。从这样的模型中采样是直截了当的，因为我们可以从  $p_z(z)$  中生成样本，然后使用神经网络函数对每个样本进行变换，以获得对应的  $x$  的样本。这是一个高效的过程，因为它不涉及迭代。

为了从数据中学习  $g(z, w)$ ，考虑如何计算似然函数  $p(x \mid w)$ 。 $x$  上的分布由密度的变量变换公式给出：

$$
p _ {x} (x) = p _ {z} (z (x)) | \det  J (x) | \tag {16.78}
$$

其中  $J$  是偏导数的雅可比矩阵，其元素由下式给出：

$$
J _ {i j} (\boldsymbol {x}) = \frac {\partial z _ {i}}{\partial x _ {j}} \tag {16.79}
$$

要计算式（16.78）右侧的分布  $p_{z}(z(x))$  （对于给定的数据向量  $\pmb{x}$  ）和式（16.79）中同一  $\pmb{x}$  值的雅可比矩阵，我们需要使用神经网络函数的逆函数  $z = g^{-1}(x,w)$  。对于大多数神经网络来说，这个逆函数将不是良好定义的。例如，神经网络可能实现一种多对一的函数关系，在这种情况下，不同的输入值会被映射到相同的输出值，变量变换公式并不能给出良好定义的密度。此外，如果潜空间的维度与数据空间的维度不同，那么变换将是不可逆的。

解决这个问题的一种方法是限制函数  $g(z, w)$  是可逆的，这要求  $z$  和  $x$  具有相同的维度。我们将在介绍标准化流（normalizing flow）的技术时更详细地探索这种方法（参见第18章）。

### 16.4.1 非线性流形

要求潜空间和数据空间具有相同的维度是一个重大的限制。考虑这样一种情况： $z$  的维度是  $M$ ， $x$  的维度是  $D$ ，其中  $M < D$ 。在这种情况下， $x$  上的分布局限于维度为  $M$  的流形或子空间，如图16.11所示。许多机器学习应用中都出现了低维流形，例如在模拟自然图像的分布时。非线性潜变量模型（nonlinear latent variable model）在建模此类数据时可能非常有用，因为它们表达了强烈的归纳偏置，即数据并不“填满”数据空间，而是被限制在一个流形上，尽管这个流形的形状和维度通常是未知的。

![](img/4b74f08d7cff4ce482b0ad54d2c193b9d941bad94371b9883265c3f3c1bae25d.jpg)  
图16.11 使用一个带有参数向量  $\pmb{w}$  的神经网络所代表的非线性函数  $x = g(z, w)$ ，演示从二维潜空间  $z = (z_1, z_2)$  到三维数据空间  $x = (x_1, x_2, x_3)$  的一个映射

然而，这个框架将零概率密度赋予那些不完全位于流形上的任何数据向量，这对基于梯度的学习算法是一个问题，因为对于真实的数据集，似然函数在每一个数

据点都将为零，并且对于小的  $w$  变化将保持不变。为了解决这个问题，我们使用之前用于解决回归和分类问题的策略，定义了整个数据空间上的条件分布，其参数由神经网络的输出决定。例如，如果  $x$  是连续变量的一个向量，则可以选择条件分布为高斯分布：

$$
p (\boldsymbol {x} \mid \boldsymbol {z}, \boldsymbol {w}) = \mathcal {N} \left(\boldsymbol {x} \mid \boldsymbol {g} (\boldsymbol {z}, \boldsymbol {w}), \sigma^ {2} \boldsymbol {I}\right) \tag {16.80}
$$

其中神经网络  $g(z, w)$  的输出单元激活函数是线性的，并且  $g \in \mathbb{R}^D$  。生成式模型由  $z$  上的潜变量分布和  $x$  上的条件分布指定，并且可以用图16.12所示的简单图模型来表示。

![](img/0790857c7ef23c35f3eda43167ac0370a8672dfe5003460d5c70b7aacf2cefc6.jpg)  
图16.12 由式（16.77）和式（16.80）定义的联合分布  $p(x,z) = p(x|z)p(z)$  的图模型

注意，从这个分布中独立抽取样本不仅简单而且效率高。首先使用标准方法从高斯分布[式（16.77）]中抽取一个样本。然后将这个值作为神经网络的输入，得到输出值  $g(z, w)$  。最后从均值为  $g(z, w)$  、协方差为  $\sigma^2 I$  的高斯分布中抽取一个样本[式（16.80）]。重复上述过程以生成多个独立样本。

潜变量分布  $p(z)$  与条件分布  $p(x|z)$  的组合定义了数据空间上的边缘分布：

$$
p (\boldsymbol {x}) = \int p (z) p (\boldsymbol {x} \mid z) \mathrm {d} z \tag {16.81}
$$

我们可以用一个涉及一维潜空间和二维数据空间的简单例子来说明这一点，见图16.13。

![](img/3297b3d930cae08fa38e8d84aafd58cc14be747500813ef0c77ba782046f739b.jpg)  
(a)  
图16.13 一个涉及一维潜空间和二维数据空间的非线性潜变量模型的示意图。(a) 潜空间中的先验分布由零均值、单位方差的高斯分布给出。(b) 最左侧的三个图显示了不同  $z$  值下的高斯条件分布  $p(x|z)$  的示例，最右侧的图显示了边缘分布  $p(x)$  的示例。定义条件分布均值的非线性函数  $g(z)$  由  $g_1(z) = \sin(z)$  和  $g_2(z) = \cos(z)$  给出，条件分布的标准差由  $\sigma = 0.3$  给出[经Prince(2020)授权使用]

![](img/371dfb0e661b28f558648012d177e30ce6163da566bdbfab80d74080f2e68f9f.jpg)  
(b)

### 16.4.2 似然函数

我们已经看到从这个非线性潜变量模型中抽取样本是很容易的。假设我们希望通过最大化似然函数来拟合观测到的数据集。可以通过对  $z$  进行积分并使用概率的加和法则和乘积法则来得到似然值：

$$
\begin{array}{l} p (\boldsymbol {x} \mid \boldsymbol {w}) = \int p (\boldsymbol {x} \mid \boldsymbol {z}, \boldsymbol {w}) p (\boldsymbol {z}) \mathrm {d} z \tag {16.82} \\ = \int \mathcal {N} (x \mid g (z, w), \sigma^ {2} I) \mathcal {N} (z \mid 0, I) d z \\ \end{array}
$$

尽管积分内部的两个分布都是高斯分布，但由神经网络定义的高度非线性函数  $g(z, w)$  却使得积分在解析上是不可处理的。一种计算似然函数的方法是从潜空间分布中抽取样本，并用这些样本来近似式（16.82），从而得到

$$
p (\boldsymbol {x} \mid \boldsymbol {w}) \approx \frac {1}{K} \sum_ {i = 1} ^ {K} p \left(\boldsymbol {x} \mid \boldsymbol {z} _ {i}, \boldsymbol {w}\right) \tag {16.83}
$$

其中  $z_{i} \sim p(z)$  。这将  $z$  的分布表示为具有固定混合系数  $1 / K$  的高斯混合分布，当样本数量趋于无穷时，该方法给出了真实的似然函数。然而，实际训练中所需的  $K$  值通常会大得不切实际。为什么会这样？请考虑图16.14所示的三个手写数字图像，并假设图16.14(a)代表我们希望计算似然函数的向量  $x$  。如果一个训练好的模型生成了图16.14(b)，我们会认为这是一个比较差的模型，因为这个图像并不是数字“2”的一个很好的表示，所以应该被赋予一个更低的似然值。相反，图16.14(c)是通过将图16.14(a)中的数字向下和向右移动0.5个像素获得的，是数字“2”的一个很好的表示，因此应该被赋予一个更高的似然值。由于是高斯分布，似然函数与网络输出和数据向量  $x$  之间负平方距离的指数成正比。如果方差参数  $\sigma^2$  设置得足够小，使得图16.14(b)具有低似然值，那么图16.14(c)就会有更低的似然值。即使模型在生成数字方面做得很好，我们也必须抽取极大量的  $z$  样本，才能找到一个足够接近图16.14(a)的数字图像。因此，我们需要寻求更精细的技术来训练非线性潜变量模型，这些技术可以服务于实际应用。在概述这些技术之前，我们先简要讨论一些关于离散数据空间的考虑。

![](img/d22922fad8317b6cafbbee00b3680d613d3ecb47fbf7d201296e1ce2bec54f2b.jpg)  
(a)  
图16.14 三个手写数字图像，旨在说明为什么从潜空间采样以计算似然函数需要大量的样本。(a)原始图像；(b)损坏的图像，其中部分笔画被移除；(c)将原始图像向下和向右各移动0.5个像素的结果。尽管图(c)在外观上更接近图(a)，但图(b)在似然性方面更接近图(a)[经Doersch（2016）授权使用]

![](img/c2ef6cc72112443b9b529aac108150477d4e94e8414e6ce31270e773696587c0.jpg)  
(b)

![](img/76ac34dc089e999cbf439c51c7a910a093af7564110b24cb22d0e4308a2e9d12.jpg)  
(c)

### 16.4.3 离散数据

如果观测到的数据集包含独立的二进制变量，则可以使用以下形式的条件分布：

$$
p (\boldsymbol {x} \mid \boldsymbol {z}, \boldsymbol {w}) = \prod_ {i = 1} ^ {D} g _ {i} (\boldsymbol {z}, \boldsymbol {w}) ^ {x _ {i}} \left(1 - g _ {i} (\boldsymbol {z}, \boldsymbol {w})\right) ^ {1 - x _ {i}} \tag {16.84}
$$

其中  $g_{i}(z,w) = \sigma (a_{i}(z,w))$  是输出单元  $i$  的激活值，激活函数  $\sigma (\cdot)$  由逻辑斯谛s sigmoid函数给出，  $a_{i}(z,w)$  是输出单元  $i$  的预激活值。类似地，对于独热编码（one-hot encoded）的分类变量，可以使用多项式分布：

$$
p (\boldsymbol {x} \mid \boldsymbol {z}, \boldsymbol {w}) = \prod_ {i = 1} ^ {D} g _ {i} (\boldsymbol {z}, \boldsymbol {w}) ^ {x _ {i}} \tag {16.85}
$$

其中，softmax激活函数为

$$
g _ {i} (z, w) = \frac {\exp \left(a _ {i} (z , w)\right)}{\sum_ {j} \exp \left(a _ {j} (z , w)\right)} \tag {16.86}
$$

我们还可以通过构造相关条件分布的乘积来考虑离散变量和连续变量的组合。

在实际应用中，连续变量通常以离散值表示，例如在图像中，红色、绿色和蓝色通道的强度可能取自8位二进制数字所能表示的数值集  $\{0, \dots, 255\}$  。当我们采用基于深度神经网络的高度灵活模型时，这可能会导致一个问题：如果密度坍塌到一个或多个离散值上，似然函数就有可能变为零，导致异常解。这个问题可以通过使用一种称为去量化（dequantization）的技术来解决，该技术涉及向变量添加噪声，这些噪声在连续离散值之间的区域内通常是均匀分布的，如图16.15所示。训练集的去量化就是用随机抽取的样本替换每个观测值，这些样本来自与特定离散值相关联的连续分布。采用这种方法，模型找到异常解的可能性就会降低。

![](img/9977e3209a28742c43d5b04238a27cb30af332bc4e0246da1a0c78f0006b219e.jpg)  
(a)  
图16.15 去量化的示意图。(a)单个变量上的离散分布；(b)相关的去量化连续分布

![](img/79c57e19019bf5a85eda213dfeb327ed935af8cb59497ee0e63209caef89ca58.jpg)  
(b)

### 16.4.4 构建生成式模型的4种方法

我们已经看到，基于深度神经网络的非线性潜变量模型提供了一个高度灵活的框架来构建生成式模型。由于神经网络变换的普适性，这些模型原则上能够以高精度近

似任何期望的分布。此外，一旦训练好，这些模型还可以使用一种高效、非迭代的过程从分布中生成样本。然而，我们也面临着一些与训练此类模型相关的挑战，迫切需要找到比线性模型更复杂的方法。业内已经提出了不少方法，每种方法都有自身的优势和局限性，它们大致可以分为以下4种。

（1）基于生成对抗网络（Generative Adversarial Network，GAN）（见第17章）放宽“网络映射必须是可逆的”这一要求，允许潜空间的维度低于数据空间的维度。这里也放弃了似然函数的概念，引入了第二个神经网络，其功能是为生成网络提供训练信号。由于缺乏明确定义的似然函数，训练过程可能不稳定，但一旦训练完成，就很容易从模型中生成样本，并且所得结果的质量也高。  
（2）变分自编码器（Variational AutoEncoder, VAE）（见第19章）也使用了第二个神经网络，其作用是近似潜变量的后验分布，从而允许计算似然函数的近似值。与GAN相比，其能使训练更为稳健，而且能直接从训练好的模型中采样，但是要获得高质量的结果可能更困难。  
（3）在标准化流（normalizing flow）（见第18章）中，将潜空间的维度设置为等于数据空间的维度，然后修改神经网络，使其变得可逆。“网络必须是可逆的”这一要求限制了其函数形式，但也使得无须近似就能计算似然函数，而且能进行高效采样。  
（4）扩散模型（diffusion model）（见第20章）使用一个网络学习如何通过一系列去噪步骤将先验分布的样本转换为数据分布的样本。这在许多应用中带来了先进的性能，缺点是由于多次经历网络的去噪过程，采样成本可能较高。

## 习题

16.1（ $\star \star$ ）使用归纳法证明，将数据投影到  $M$  维子空间以最大化投影数据的方差所定义的线性投影是由数据协方差矩阵  $S$  [由式（16.3）给出]的  $M$  个特征向量定义的，对应于  $M$  个最大特征值。在16.1节中，这个结果已经在  $M = 1$  的情况下得到证明。假设该结果对某个一般的  $M$  值成立，证明它因此对  $M + 1$  也成立。为此，首先将投影数据的方差关于定义新方向的向量  $\pmb{u}_M + 1$  的导数设置为零。这应该在满足约束条件的情况下完成，即  $\pmb{u}_M + 1$  与现有向量  $\pmb{u}_1, \dots, \pmb{u}_M$  正交，并且已归一化至单位长度。使用拉格朗日乘子保证这些约束，然后利用向量  $\pmb{u}_1, \dots, \pmb{u}_M$  的正交性质证明新向量  $\pmb{u}_M + 1$  是  $S$  的一个特征向量。最后证明如果选择与第  $M + 1$  个特征值相对应的特征向量（特征值按降序排列），则方差达到最大。  
16.2（ $\star \star$ ）证明在满足正交性质[式（16.7)]的情况下，PCA误差度量  $J$  [由式（16.15）给出]的最小值是在  $\pmb{u}_i$  为数据协方差矩阵  $\pmb{S}$  的特征向量时获得的。为此，引入一个拉格朗日乘子矩阵  $\pmb{H}$ ，每个约束对应一个拉格朗日乘子，因此修改后的误差度量  $\tilde{J}$  用矩阵表示为

$$
\widetilde {J} = \operatorname {t r} \left[ \widehat {\boldsymbol {U}} ^ {\mathrm {T}} \boldsymbol {S} \widehat {\boldsymbol {U}} \right] + \operatorname {t r} \left[ \boldsymbol {H} \left(\boldsymbol {I} - \widehat {\boldsymbol {U}} ^ {\mathrm {T}} \widehat {\boldsymbol {U}}\right) \right] \tag {16.87}
$$

其中  $\widehat{\pmb{U}}$  是一个  $D\times (D - M)$  大小的矩阵，其列由  $\pmb{u}_i$  给出。对  $\widehat{\pmb{U}}$  求  $\widetilde{J}$  的最小值并证明解满足  $S\widehat{U} = \widehat{U} H$  。

显然，一个可能的解是， $\widehat{U}$  的列是  $S$  的特征向量。在这种情况下， $H$  是一个包含对应特征值的对角矩阵。假设  $H$  是一个对称矩阵，并且通过使用特征向量将其展开，证明  $S \widehat{U} = \widehat{U} H$  的通用解（对于给出的  $\widetilde{J}$  值）与  $\widehat{U}$  的列是  $S$  的特征向量时的具体解相同。因为这些解都是等效的，所以选择特征向量解是很方便的。

16.3（ $\star$ ）假设式（16.30）定义的特征向量  $\pmb{\nu}_{i}$  具有单位长度，验证这些特征向量已归一化至单位长度。  
16.4（ $\star$ ）假设在概率PCA模型中将零均值、单位协方差的潜空间分布[式（16.31）]替换为一般形式的高斯分布  $\mathcal{N}(z|m,\Sigma)$  。通过重新定义模型的参数，证明这种替换会使得任意有效的  $m$  和  $\Sigma$  都无法改变观测变量边缘分布  $p(x)$  的形式。  
16.5（ $\star \star$ ）假设  $x$  是一个  $D$  维的随机变量，具有高斯分布  $\mathcal{N}(x|\mu, \Sigma)$ ，并考虑由  $y = Ax + b$  给出的另一个  $M$  维随机变量，其中  $A$  是一个  $M \times D$  大小的矩阵。证明  $y$  也具有高斯分布，并找出其均值和协方差。讨论此高斯分布在  $M < D$ 、 $M = D$  和  $M > D$  情况下的形式。  
16.6（ $\star \star$ ）利用式（2.122）和式（2.123）给出的一般分布的均值和协方差，推导概率PCA模型中边缘分布  $p(x)$  的结果[式（16.35)]。  
16.7（★）绘制16.2节描述的概率PCA模型的有向图，并观测变量  $x$  的各个组成部分明确显示为单独的节点。验证概率PCA模型具有与11.2.3小节讨论的朴素贝叶斯模型相同的独立结构。  
16.8（ $\star \star$ ）利用式（3.100），证明概率PCA模型的后验分布  $p(z|x)$  可以由式（16.43）给出。  
16.9（ $\star$ ）验证对概率PCA模型的对数似然函数[式（16.44）]关于参数  $\pmb{\mu}$  进行最大化会得到结果  $\pmb{\mu}_{ML} = \overline{\pmb{x}}$  ，其中  $\overline{\mathbf{x}}$  是数据向量的均值。  
16.10（ $\star \star$ ）通过计算对数似然函数[式（16.44）]对参数  $\pmb{\mu}$  的二阶导数，证明驻点  $\pmb{\mu}_{ML} = \overline{\pmb{x}}$  代表唯一的最大值。  
16.11（ $\star \star$ ）证明当  $\sigma^2 \to 0$  时，概率PCA模型的后验均值相当于对主子空间的正交投影，这与传统PCA模型的结果一致。  
16.12（ $\star \star$ ）对于  $\sigma^2 > 0$ ，证明概率PCA模型中的后验均值会相对于正交投影向原点移动。  
16.13（ $\star \star$ ）证明在概率PCA下，根据传统PCA的最小二乘投影成本，数据点的最优重建由下式给出：

$$
\tilde {\boldsymbol {x}} = \boldsymbol {W} _ {\mathrm {M L}} \left(\boldsymbol {W} _ {\mathrm {M L}} ^ {\mathrm {T}} \boldsymbol {W} _ {\mathrm {M L}}\right) ^ {- 1} \boldsymbol {M} \mathbb {E} [ \boldsymbol {z} \mid \boldsymbol {x} ] \tag {16.88}
$$

16.14（ $\star$ ）概率PCA模型中协方差矩阵的独立参数数量由式（16.52）给出。证明对于  $M = D - 1$  ，独立参数的数量与一般高斯分布中的相同；而对于  $M = 0$  ，独立参数的数量与各向同性协方差高斯分布相同。

16.15（ $\star$ ）推导出16.2.4小节描述的因子分析模型中独立参数数量的计算表达式。

16.16（ $\star \star$ ）证明16.2.4小节描述的因子分析模型不随潜空间坐标旋转而变化。

16.17（ $\star \star$ ）考虑一个线性-高斯潜变量模型，它具有潜空间分布  $p(z) = \mathcal{N}(x|\mathbf{0},I)$  和观测变量的条件分布  $p(x|z) = \mathcal{N}(x|Wz + \mu, \Phi)$ ，其中  $\Phi$  是任意对称且正定的噪声协方差矩阵。对数据变量进行非奇异线性变换  $x \to Ax$ ，其中  $A$  是一个  $D \times D$  矩阵。如果  $\mu_{\mathrm{ML}}$ ， $W_{\mathrm{ML}}$  和  $\Phi_{\mathrm{ML}}$  代表原始未变换数据所对应的最大似然解，证明  $A\mu_{\mathrm{ML}}$ 、 $AW_{\mathrm{ML}}$  和  $A\Phi_{\mathrm{ML}}A^{\mathrm{T}}$  代表变换后数据所对应的最大似然解。最后证明在以下两种情况下模型的形式得以保留：（i） $A$  是对角矩阵且  $\Phi$  也是对角矩阵，这对应于因子分析。变换后的  $\Phi$  仍然是对角矩阵，因此因子分析在各成分矩阵的变换下具有协变性。（ii） $A$  是正交矩阵且  $\Phi$  与单位矩阵成比例， $\Phi = \sigma^2 I$ ，这对应于概率 PCA。变换后的  $\Phi$  仍与单位矩阵成比例，因此概率 PCA 在数据空间的轴旋转下具有协变性，就像传统 PCA 一样。

16.18（ $\star$ ）验证连续潜变量模型对数似然函数可以写成两项之和[式（16.57）]，这两项分别由式（16.58）和式（16.59）定义。这可以通过使用概率的乘积法则来完成，公式如下：

$$
p (\boldsymbol {x}, \boldsymbol {z} | \boldsymbol {w}) = p (\boldsymbol {z} | \boldsymbol {x}, \boldsymbol {w}) p (\boldsymbol {x} | \boldsymbol {w}) \tag {16.89}
$$

然后将  $p(\pmb {x},\pmb {z}|\pmb {w})$  代入式（16.58）即可。

16.19（ $\star$ ）证明对于一组独立同分布数据，证据下界（ELBO）可以采用式（16.63）的形式。  
16.20（ $\star \star$ ）绘制一个有向图，以表示概率PCA模型的离散混合模型，其中的每个概率PCA模型都有自己的  $W$  、  $\pmb{\mu}$  和  $\sigma^2$  值。然后绘制一个修改后的有向图，这些参数值在该有向图所代表的混合模型的各部分之间是共享的。  
16.21（ $\star \star$ ）通过最大化式（16.65）给出的完整数据对数似然函数的期望来推导概率PCA模型的M步骤方程[式（16.68)]和[式（16.69)]。  
16.22（ $\star \star \star$ ）概率PCA的一个优点是，它可以应用于某些值缺失的数据集，前提是这些值是随机缺失的。在这种情况下，推导出最大化概率PCA模型似然函数的EM算法。注意， $\{z_{n}\}$  和向量  $\{x_{n}\}$  的缺失数据值现在都是潜变量。证明在所有数据值都观测到的特殊情况下，这可以简化为推导概率PCA的EM算法（见16.3.2小节）。  
16.23（ $\star \star$ ）设  $W$  是一个  $D \times M$  矩阵，其列定义了嵌入  $D$  维数据空间的  $M$  维线性子空间，并且  $\pmb{\mu}$  是一个  $D$  维向量。给定一个数据集  $\{x_{n}\}$ ，其中  $n = 1, \dots, N$ ，我们可以使用一组  $M$  维向量  $\{z_{n}\}$  的线性映射来近似数据点，使得  $x_{n}$  能被  $Wz_{n} + \pmb{\mu}$  近

似。相关的平方和重建成本由下式给出：

$$
J = \sum_ {n = 1} ^ {N} \left\| \boldsymbol {x} _ {n} - \boldsymbol {\mu} - \boldsymbol {W} \boldsymbol {z} _ {n} \right\| ^ {2} \tag {16.90}
$$

首先证明关于  $\pmb{\mu}$  最小化  $J$  会得到一个相似的表达式，其中  $x_{n}$  和  $z_{n}$  分别替换为零均值变量  $x_{n} - \overline{x}$  和  $Z_{n} - \overline{Z}$ ，其中  $x$  和  $z$  表示样本均值。然后证明在  $\pmb{W}$  保持固定的情况下关于  $z_{n}$  最小化  $J$  会产生 PCA EM 算法的 E 步骤 [式（16.70）]，并且在  $\{z_{n}\}$  保持固定的情况下关于  $\pmb{W}$  最小化  $J$  会产生 PCA EM 算法的 M 步骤 [式（16.71）]。

16.24（ $\star \star$ ）推导因子分析EM算法的E步骤方程[式（16.72）和式（16.73）]。注意，从习题16.26的结果可知，参数  $\pmb{\mu}$  可以用样本均值  $\pmb{x}$  替换。  
16.25（ $\star \star$ ）给出因子分析模型的完整数据对数似然函数的期望的表达式，从而推导出相应的M步骤方程[式（16.75）和式（16.76)]。  
16.26（ $\star \star$ ）通过考虑二阶导数，证明16.2.4小节讨论的因子分析模型的对数似然函数相对于参数  $\pmb{\mu}$  的唯一驻点是由式（16.1）定义的样本均值给出的。此外，证明这个驻点是一个最大值。
