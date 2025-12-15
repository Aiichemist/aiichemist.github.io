# 第14章 采样

![](img/2e5bee53ce05f46c36aaf04875db7c57be73316e5181b18b63ebc093567c66f9.jpg)

在深度学习领域，我们经常需要从概率分布  $p(z)$  中生成  $z$  的样本。这里的  $z$  可能是一个标量，分布可能是单变量高斯分布；或者  $z$  可能是一幅高分辨率的图像，而  $p(z)$  可能是由深度神经网络定义的生成式模型。这种生成样本的过程称为采样（sampling），也称蒙特卡洛采样（Monte Carlo sampling）。对于许多简单的分布，我们可以使用数值技术直接生成合适的样本。而对于更复杂的分布，包括那些隐式定义的分布，我们则需要使用更复杂的方法。我们把每个实例化的值称为一个样本，而不像传统统计学那样把一组值称为一个样本。

在本章中，我们将重点关注与深度学习最相关的采样方面的内容。关于蒙特卡洛采样方法的更通用的介绍，可以参考Gilks, Richardson, and Spiegelhalter（1996）以及Robert and Casella（1999）。

## 14.1 基本采样

在本节中，我们将介绍一系列相对简单的采样策略，用于从给定分布中生成随机样本。由于样本是由计算机算法生成的，它们是伪随机（pseudo-random）的，也就是说，它们将使用确定性算法来运算，但仍必须通过适当的随机性测试。这里我们假设已经存在一个确定性算法，它能生成在(0,1)区间均匀分布的伪随机数，事实上，大多数软件平台都内置了这样的功能。

### 14.1.1 期望

虽然在某些应用中我们可以直接使用样本本身，但在更多情况下，我们的目标是计算某个概率分布的期望。假设我们希望找到函数  $f(z)$  关于概率分布  $p(z)$  的期望。在这里， $z$  的分量可能包括离散变量、连续变量或它们两者的组合。对于连续变量，期望定义为

$$
\mathbb {E} [ f ] = \int f (z) p (z) \mathrm {d} z \tag {14.1}
$$

其中，积分在离散变量的情况下被求和替代。图14.1展示了单个连续变量的情况。我们在此假设这样的期望过于复杂，无法使用解析方法准确计算，这时候就需要用采样方法近似计算。采样方法的基本思路是获得从分布  $p(z)$  中独立地抽取的一组样本  $\boldsymbol{z}^{(l)}$  （其中  $l = 1,\dots ,L$ ），使得期望（14.1）可以通过有限和的形式来近似：

$$
\bar {f} = \frac {1}{L} \sum_ {l = 1} ^ {L} f \left(z ^ {(l)}\right) \tag {14.2}
$$

![](img/ab8771c3282fe5ab4640d8d1b588adf7b4ee9ebd9062218c52afdaf225ebb11f.jpg)  
图14.1 一个函数  $f(z)$ ，我们需要计算它关于概率分布  $p(z)$  的期望（示意图）

如果样本  $z^{(l)}$  是从概率分布  $p(z)$  中抽取的，则  $\mathbb{E}[\overline{f}] = \mathbb{E}[f(z)]$  ，因此估计量  $\overline{f}$  具有正确的均值（见习题14.1）。我们也可以把它改成

$$
\mathbb {E} [ f (z) ] \simeq \frac {1}{L} \sum_ {l = 1} ^ {L} f \left(z ^ {(l)}\right) \tag {14.3}
$$

其中符号  $\simeq$  表示右侧是左侧的无偏估计量，也就是说，当对噪声分布进行平均时，两侧是相等的（见习题14.2）。估计量［式（14.2）］的方差由下式给出：

$$
\operatorname {v a r} [ \bar {f} ] = \frac {1}{L} \mathbb {E} \left[ (f - \mathbb {E} [ f ]) ^ {2} \right] \tag {14.4}
$$

该方差是函数  $f(z)$  在概率分布  $p(z)$  下的方差。注意，这种方差会随着  $L$  的增加而线性减小，并且不依赖于  $z$  的维度，从原则上讲，使用相对较小的样本集合  $\{z^{(l)}\}$  就有可能获得高精度。然而，问题在于样本集合  $\{z^{(l)}\}$  中的样本可能不是独立的，所以有效样本量可能比表面上的样本量要小得多。另外，在图14.1中，如果  $f(z)$  在  $p(z)$  较大的区域很小，在  $p(z)$  较小的地方反而很大，则期望可能被小概率区域主导，这意味着为了达到足够的准确度，可能需要较大的样本量。

### 14.1.2 标准分布

现在假设我们已经有了一个均匀分布随机数的来源，我们讨论如何从简单的非均匀分布中生成随机数。假设  $z$  在区间（0,1）上均匀分布，并且我们使用某个函数  $g(\cdot)$  来变换  $z$  的值，使得  $y = g(z)$  。 $y$  的分布将由下式决定：

$$
p (y) = p (z) \left| \frac {\mathrm {d} z}{\mathrm {d} y} \right| \tag {14.5}
$$

在这里， $p(z) = 1$ 。我们的目标是选择合适的函数  $g(z)$ ，让生成的  $y$  具有某个特定的期望分布  $p(y)$ 。对式（14.5）进行积分，可以得到

$$
z = \int_ {- \infty} ^ {y} p (\hat {y}) \equiv h (y) \mathrm {d} \hat {y} \tag {14.6}
$$

这是  $p(y)$  的不定积分。  $y = h^{-1}(z)$  ，所以我们需要使用一个变换函数，该函数是目标分布不定积分的反函数，如图14.2所示。

![](img/dac9311b3e006c011ccd936e7ac6a6e17e6f1cc1f9ae29a0d9dba623c0eb353c.jpg)  
图14.2 生成非均匀分布随机数的变换方法的几何解释。  $h(y)$  是期望的目标分布  $p(y)$  的不定积分。如果用  $y = h^{-1}(z)$  变换均匀分布的随机变量  $z$ ，那么得到的变量  $y$  将服从  $p(y)$  分布

以指数分布为例

$$
p (y) = \lambda \exp (- \lambda y) \tag {14.7}
$$

其中  $0 \leqslant y < \infty$  。在这种情况下，式（14.6）中的积分的下限是 0，因此

$h(y) = 1 - \exp (-\lambda y)$  。如果我们使用  $y = -\lambda^{-1}\ln (1 - z)$  变换均匀分布的随机变量  $z$  ，那么 $y$  将服从指数分布。

可以应用变量变换方法的另一种分布是柯西分布：

$$
p (y) = \frac {1}{\pi} \frac {1}{1 + y ^ {2}} \tag {14.8}
$$

在这种情况下，不定积分的反函数可以用正切函数表示（见习题14.4）。

将该方法推广到多元变量的情形，需要引入变量变换的雅可比行列式，于是有

$$
p \left(y _ {1}, \dots , y _ {M}\right) = p \left(z _ {1}, \dots , z _ {M}\right) \left| \frac {\partial \left(z _ {1} , \cdots , z _ {M}\right)}{\partial \left(y _ {1} , \cdots , y _ {M}\right)} \right| \tag {14.9}
$$

作为变换方法的最后一个例子，我们介绍用于生成高斯分布样本的Box-Muller方法。首先，我们生成一对均匀分布的随机数  $z_{1}, z_{2} \in (-1, 1)$ ，这可以通过对在区间  $(0, 1)$

![](img/527308863ea2d0003191f516d697d8306c3c028ccf584c81a039a8c5720af17c.jpg)  
图14.3 Box-Muller方法用于生成高斯分布的随机数，为此，首先在单位圆内生成均匀分布的样本

上均匀分布的变量  $z$  进行  $z \rightarrow 2z - 1$  线性变换来实现。接下来丢弃所有不满足条件  $z_1^2 + z_2^2 \leqslant 1$  的数对，这将产生一个在单位圆内均匀分布的点集，且其概率密度为  $p(z_1, z_2) = 1 / \pi$  如图14.3所示。

最后，对于每对  $z_{1}, z_{2}$ ，我们计算量

$$
y _ {1} = z _ {1} \left(\frac {- 2 \ln r ^ {2}}{r ^ {2}}\right) ^ {1 / 2} \tag {14.10}
$$

$$
y _ {2} = z _ {2} \left(\frac {- 2 \ln r ^ {2}}{r ^ {2}}\right) ^ {1 / 2} \tag {14.11}
$$

其中  $r^2 = z_1^2 +z_2^2$  （见习题14.5）。  $y_{1}$  和  $y_{2}$  的联合概率密度函数为

$$
\begin{array}{l} p \left(y _ {1}, y _ {2}\right) = p \left(z _ {1}, z _ {2}\right) \left| \frac {\partial \left(z _ {1} , z _ {2}\right)}{\partial \left(y _ {1} , y _ {2}\right)} \right| \tag {14.12} \\ = \left[ \frac {1}{\sqrt {2 \pi}} \exp \left(- y _ {1} ^ {2} / 2\right) \right] \left[ \frac {1}{\sqrt {2 \pi}} \exp \left(- y _ {2} ^ {2} / 2\right) \right] \\ \end{array}
$$

因此， $y_{1}$  和  $y_{2}$  是独立的，并且每个变量都服从均值为零、方差为1的高斯分布。

如果  $y$  服从均值为零、方差为单位方差的高斯分布，那么  $\sigma y + \mu$  将服从均值为  $\mu$  、方差为  $\sigma^2$  的高斯分布。为了生成具有多元高斯分布的向量值变量，假设多元高斯分布的均值为  $\mu$  、协方差矩阵为  $\Sigma$  ，我们可以利用楚列斯基分解，其形式为  $\boldsymbol{\Sigma} = \boldsymbol{L}\boldsymbol{L}^{\mathrm{T}}$  (Deisenroth, Faisal, and Ong, 2020)。然后，如果  $z$  是一个随机向量，其分量是独立的且具有零均值、单位方差的高斯分布（见习题14.6），那么  $y = \mu + Lz$  将具有均值为  $\mu$

协方差矩阵为  $\pmb{\Sigma}$  的高斯分布。

显然，变换技术的成功依赖于计算所需分布的不定积分与求得其反函数的能力。这样的操作只对几种简单分布可行，因此我们必须转向替代方法，寻找更通用的策略。这里我们考虑两种替代方法：拒绝采样（rejection sampling）和重要性采样（importance sampling）。尽管这两种方法受限于单变量分布，并不能直接应用于多维的复杂问题，但它们是更通用策略的重要组成部分。

### 14.1.3 拒绝采样

拒绝采样框架允许我们从相对复杂的分布中采样，但有一定的限制条件。首先考虑单变量分布，然后考虑扩展到多维的情形。

假设我们想要从一个分布  $p(z)$  中进行采样，该分布不属于目前为止考虑过的简单标准分布，并且直接从  $p(z)$  中采样很困难。此外，假设我们能够轻易地计算任意给定值  $z$  的  $p(z)$ ，但计算结果需要乘以归一化常数  $Z_{p}$  的倒数才能得到真实的  $p(z)$ ，即

$$
p (z) = \frac {1}{Z _ {p}} \tilde {p} (z) \tag {14.13}
$$

其中  $\tilde{p}(z)$  很容易计算，但  $Z_{p}$  是未知的。

要应用拒绝采样，我们需要一些更简单的分布  $q(z)$ ，它们有时称为提议分布（proposal distribution），我们可以容易地从中抽取样本。接下来引入常数  $k$ ，其值的选择需要满足对于所有的  $z$  都有  $k q(z) \geqslant \tilde{p}(z)$ 。函数  $k q(z)$  称为比较函数，图 14.4 针对一元分布的情况对它进行了说明。拒绝采样的每一步都需要生成两个随机数。首先，我们从分布  $q(z)$  中生成一个随机数  $z_0$ 。接下来，我们从区间  $[0, k q(z_0)]$  上的均匀分布中

生成另一个随机数  $u_{0}$  。这对随机数在函数  $kq(z)$  的曲线下具有均匀分布。最后，如果  $u_{0} > \tilde{p}(z_{0})$  ，则拒绝该样本，否则保留  $u_{0}$  。因此，这对随机数如果位于图14.4中的灰色阴影区域内，则会被拒绝。剩余的随机数对在  $\tilde{p}(z)$  曲线下具有均匀分布（见习题14.7），因此相应的  $z$  值服从我们期望的分布  $p(z)$  。

原始的  $z$  值是从分布  $q(z)$  中生成的，然后这些样本以  $\tilde{p}(z) / kq(z)$  的概率得以接受，因此接受样本的概率由下式给出：

![](img/8c782a87bfcd37a22e1e0860011faa600bed4f50c5cd7bf25a51b78ba05a0442.jpg)  
图14.4 在拒绝采样中，样本是从一个简单的分布  $q(z)$  中抽取的，并且它们如果落在未归一化分布  $\tilde{p}(z)$  和缩放分布  $kq(z)$  之间的灰色区域内，则被拒绝。最终的样本是根据  $p(z)$  分布的， $p(z)$  是  $\tilde{p}(z)$  的归一化版本

$$
\begin{array}{l} p (\text {a c c e p t}) = \int \left\{\tilde {p} (z) / k q (z) \right\} q (z) d z \tag {14.14} \\ = \frac {1}{k} \int \tilde {p} (z) d z \\ \end{array}
$$

因此，这种方法拒绝的点的比例取决于未归一化分布  $\tilde{p}(z)$  曲线下的面积与缩放分布  $kq(z)$  曲线下的面积的比值。由此可见，常数  $k$  应尽可能小，同时满足对于任意  $z$ ,  $kq(z)$  处处都不小于  $\tilde{p}(z)$  的限制条件。

为了演示拒绝采样的用法，考虑从伽马分布中采样的任务：

$$
\operatorname {G a m} (z \mid a, b) = \frac {b ^ {a} z ^ {a - 1} \exp (- b z)}{\Gamma (a)} \tag {14.15}
$$

![](img/017b0402309dda860036001ea8ed89123314d60edf54d167358195b48d07d73f.jpg)  
图14.5式（14.15）给出的伽马分布为绿色曲线，缩放后的柯西提议分布为红色曲线。可以通过从柯西分布中采样，然后应用拒绝采样来获得伽马分布的样本

当  $a > 1$  ，伽马分布具有钟形形状，如图14.5所示。因此，合适的提议分布是柯西分布［式（14.8）]，因为它也是钟形的，而且我们可以使用前面讨论的变换方法从中采样。我们需要稍微扩展一下柯西分布，以确保其在任何位置的值都不小于伽马分布的值。这可以通过使用  $z = b\tan y + c$  变换均匀分布的随机变量  $y$  来实现，这会产生服从下式分布的随机数（见习题14.8）：

$$
q (z) = \frac {k}{1 + (z - c) ^ {2} / b ^ {2}} \tag {14.16}
$$

通过设置  $c = a - 1$  、  $b^{2} = 2a - 1$  ，并选择尽可能小的常数  $k$  来满足  $kq(z) \geqslant \tilde{p}(z)$  的要求，可以获得最小的拒绝率。最终得到的比较函数如图14.5所示。

### 14.1.4 适应性拒绝采样

许多时候，我们希望使用拒绝采样方法却很难确定包络分布  $q(z)$  的合适解析形

![](img/e0cc648d3810b00fab8cce2abb35dffbc01158c54753be5d2136db823cc04550.jpg)  
图14.6 在拒绝采样中，如果一个分布是对数凹的，则可以使用我们在一组网格点计算得到的切线来构造一个包络函数。如果一个样本被拒绝，就将它添加到网格点集合中，并用于完善包络分布

式。一种替代方法是基于分布  $p(z)$  的测量值构造包络函数（Gilks and Wild, 1992）。当  $p(z)$  是对数凹函数时，即  $\ln p(z)$  的导数是  $z$  的非增函数时，构造合适的包络函数尤其简单。构造的包络函数的说明见图14.6。

首先，在一些初始网格点上计算  $\ln p(z)$  及其梯度，然后利用得到的切线交点来构造包络函数。接下来，从包络分布中抽取一个样本值（见习题14.10）。这一步很简单，因为包络分布的对数是一系列线性函数，因此包络分布本身是由以下形式的分段指数分布构成的：

$$
q (z) = k _ {i} \lambda_ {i} \exp \left\{- \lambda_ {i} \left(z - z _ {i - 1}\right) \right\}, \quad z _ {i - 1} <   z \leqslant z _ {i} \tag {14.17}
$$

一旦抽取了一个样本，就可以应用常规的拒绝标准。样本如果被接受，它将是期望分布的一个抽取。样本如果被拒绝，它就被纳入网格点集合中，用于计算新的切线，并由此完善包络函数。随着网格点数量的增加，包络函数会越来越逼近期望分布  $p(z)$  的更好近似，并且样本被拒绝的概率就会降低。

Gilks 提出了旨在避免计算导数的自适应拒绝采样（Gilks, 1992）。自适应拒绝采样框架也可以扩展到非对数凹分布，只需要在每个拒绝采样步骤之后跟随一个Metropolis-Hastings 步骤（将在 14.2.3 小节中讨论），从而产生自适应拒绝Metropolis采样（Gilks, Best, and Tan, 1995）。

为了使拒绝采样具有实用价值，我们要求比较函数与目标分布尽可能接近，以便将拒绝率保持在最低限度。下面看看当我们试图在高维空间中使用拒绝采样时会发生什么。我们考虑一个有点儿故意设计的问题，希望从一个均值为零、协方差为  $\sigma_p^2 I$  的高斯分布中采样（其中  $I$  是单位矩阵），而提议分布本身是一个均值为零、协方差为  $\sigma_p^2 I$  的高斯分布。显然，必须有  $\sigma_q^2 \geqslant \sigma_p^2$  以确保存在一个常数  $k$  使得  $kq(z) \geqslant p(z)$  。在  $D$  维空间中， $k$  的最优值由  $k = \left(\sigma_q / \sigma_p\right)^D$  给出，参见图14.7中  $D = 1$  的情形。样本接受

率将是  $p(z)$  和  $kq(z)$  曲线下面积的比值，因为这两个分布都是归一化的，所以结果为  $1 / k$  。样本接受率会随着维度的增大呈指数下降。即使  $\sigma_q$  仅比  $\sigma_p$  高出  $1\%$ ，对于  $D = 1000$ ，接受率大约为  $1 / 20000$  。在这个例子中，比较函数接近目标分布。目标分布可能是多模态且具有尖锐峰值的，要找到一个好的提议分布和比较函数将极其困难。此外，接受率随维度呈指数下降是拒绝采样的一个普遍特性。尽管拒绝采样在一维或两维空间中可能是一种有用的技术，但它不适用于高维问题。然而，它可以作为在高维空间采样的更复杂算法中的一个子程序发挥作用。

![](img/276de10407234d8e535fc75095903e7e6f416641cc34aee51ad1abb0b5633ff7.jpg)  
图14.7 用来突出拒绝采样局限性的示例。可通过使用提议分布  $q(z)$  进行拒绝采样，并从高斯分布  $p(z)$  中抽取样本，该分布用绿色曲线表示，其缩放版本  $kq(z)$  用红色曲线表示，提议分布  $q(z)$  也是高斯分布

### 14.1.5 重要性采样

我们希望从复杂概率分布中进行采样的一个原因是为了计算形如式（14.1）的期望。重要性采样技术提供了一个直接近似期望的框架，但它本身并不提供从分布  $p(z)$  中抽取样本的方法。

式（14.2）给出的有限和近似期望值，依赖于能够从分布  $p(z)$  中抽取样本。然而，假设直接从  $p(z)$  中采样是不现实的，但对于任意给定的  $z$  值，我们能够容易地计算出  $p(z)$  的值。一种简单的计算期望方法是将  $z$  空间离散化为均匀网格，并将期望表示成求和的形式：

![](img/b2998c0d132113ae3a6a81e0ebbd94571ae596adcede1da94c4ad6542f876222.jpg)  
图14.8 重要性采样解决了这样一个问题：当我们很难直接从分布  $p(z)$  中采样时，如何计算函数  $f(z)$  关于该分布的期望。此方法从更易采样的分布  $q(z)$  中抽取样本  $\{z^{(l)}\}$ ，并且在求和计算中为每个样本项赋予权重  $p\left(z^{(l)}\right) / q\left(z^{(l)}\right)$

$$
\mathbb {E} [ f ] \approx \sum_ {l = 1} ^ {L} p (z ^ {(l)}) f (z ^ {(l)}) \tag {14.18}
$$

这种方法的一个明显问题是，求和项的数量会随着  $z$  的维度的增加呈指数增长。此外，正如我们已经注意到的，我们感兴趣的概率分布通常会将它们的大部分概率密度限制在  $z$  空间的相对较小的区域内，所以在高维问题中，均匀采样将非常低效，因为只有很少一部分样本会对求和做出显著贡献。我们真正希望的是  $p(z)$  较大的区域，或理想情况下，从  $p(z)f(z)$  较大的区域中选择样本点。

与拒绝采样一样，重要性采样也基于易于抽取样本的提议分布  $q(z)$ ，如图14.8所示。我们可以利用从 $q(z)$  中抽取的样本集  $\{z^{(l)}\}$ ，将期望表达为样本上的有限和形式：

$$
\begin{array}{l} \mathbb {E} [ f ] = \int f (z) p (z) \mathrm {d} z \\ = \int f (z) \frac {p (z)}{q (z)} q (z) \mathrm {d} z \tag {14.19} \\ \approx \frac {1}{L} \sum_ {l = 1} ^ {L} \frac {p (z ^ {(l)})}{q (z ^ {(l)})} f (z ^ {(l)}) \\ \end{array}
$$

量  $r_{l} = p\left(z^{(l)}\right) / q\left(z^{(l)}\right)$  称为重要性权重（importance weights），用于纠正从错误分布中采样而引入的偏差。请注意，与拒绝采样不同，这里生成的所有样本都会被保留。

通常分布  $p(z)$  的计算只能达到一个归一化常数的精度，所以  $p(z) = \tilde{p}(z) / Z_p$ ，其中  $\tilde{p}(z)$  可以轻易计算出来，而  $Z_p$  是未知的。

类似地，我们可能希望使用的重要性采样分布  $q(z) = \tilde{q} (z) / Z_{q}$  也具有同样的性质：

$$
\begin{array}{l} \mathbb {E} [ f ] = \int f (z) p (z) \mathrm {d} z \\ = \frac {Z _ {q}}{Z _ {p}} \int f (z) \frac {\tilde {p} (z)}{\tilde {q} (z)} q (z) \mathrm {d} z \tag {14.20} \\ \approx \frac {Z _ {q}}{Z _ {p}} \frac {1}{L} \sum_ {l = 1} ^ {L} \tilde {r} _ {l} f \left(\boldsymbol {z} ^ {(l)}\right) \\ \end{array}
$$

其中  $\tilde{r}_l = \tilde{p}\left(\boldsymbol{z}^{(l)}\right) / \tilde{q}\left(\boldsymbol{z}^{(l)}\right)$  。我们可以使用相同的样本集来计算比值  $Z_{p} / Z_{q}$ ，结果为

$$
\begin{array}{l} \frac {Z _ {p}}{Z _ {q}} = \frac {1}{Z _ {q}} \int \tilde {p} (z) \mathrm {d} z = \int \frac {\tilde {p} (z)}{\tilde {q} (z)} q (z) \mathrm {d} z \tag {14.21} \\ \approx \frac {1}{L} \sum_ {l = 1} ^ {L} \tilde {r} _ {l} \\ \end{array}
$$

因此，式（14.20）中的期望可用加权和来表示：

$$
\mathbb {E} [ f ] \approx \sum_ {l = 1} ^ {L} w _ {l} f (z ^ {(l)}) \tag {14.22}
$$

其中

$$
w _ {l} = \frac {\tilde {r} _ {l}}{\sum_ {m} \tilde {r} _ {m}} = \frac {\tilde {p} \left(z ^ {(l)}\right) / q \left(z ^ {(l)}\right)}{\sum_ {m} \tilde {p} \left(z ^ {(m)}\right) / q \left(z ^ {(m)}\right)} \tag {14.23}
$$

请注意，  $\{w_{l}\}$  都是非负数且和为1。

与拒绝采样一样，重要性采样的成功严重依赖于采样分布  $q(z)$  与目标分布  $p(z)$  的匹配程度。如果  $p(z)f(z)$  变化剧烈并且其大部分概率密度集中在  $z$  空间的相对较小区域内（这种情况很常见），则一组重要性权重  $\{r_i\}$  可能会受少数几个较大权重的支配，剩余的权重则相对无关紧要。因此，有效样本的数量可能比表面上的样本数  $L$  少得多。如果没有任何样本落在  $p(z)f(z)$  较大的区域内，问题将更为严重。在那种情况下，即使  $r_1$  和  $r_i f(z^{(l)})$  的表面方差可能很小，对期望的估计也可能产生严重错误。因此，重要性采样的一个主要缺点是，它可能产生任意错误的结果，并且没有诊断指标。这也突出了采样分布  $q(z)$  的一个关键要求，即在  $p(z)$  可能显著的区域内，它的值不应该太小或为零。

### 14.1.6 采样-重要性-重采样

14.1.3 小节讨论的拒绝采样部分地依赖于成功确定一个适当的  $k$  值。对于分布  $p(z)$  和  $q(z)$  的各种组合，想要确定一个适当的  $k$  值是不切实际的，因为当  $k$  值大到能保证目标分布被包络的时候，对应的样本接受率会小到没有实用价值。

与拒绝采样一样，采样-重要性-重采样方案也使用一个采样分布  $q(z)$ ，但避免了确定  $k$  值的要求。该方案分两个阶段。在第一阶段，从  $q(z)$  中抽取  $L$  个样本  $z^{(1)}, \dots, z^{(L)}$  。然后在第二阶段，使用式（14.23）构建权重  $(w_{1}, \dots, w_{L})$  。最后从离散分布  $(z^{(1)}, \dots, z^{(L)})$  中抽取第二组  $L$  个样本，其被抽中的概率由权重  $(w_{1}, \dots, w_{L})$  给出。

得到的  $L$  个样本只是近似按照  $p(z)$  分布的，但随着  $L$  趋于无穷大，该分布将变得正确。为了理解这一点，考虑单变量情况，并注意重采样值的累积分布由下式给出：

$$
\begin{array}{l} p (z \leqslant a) = \sum_ {l: z ^ {(l)} \leqslant a} w _ {l} \\ = \frac {\sum_ {l} I \left(z ^ {(l)} \leqslant a\right) \tilde {p} \left(z ^ {(l)}\right) / q \left(z ^ {(l)}\right)}{\sum_ {l} \tilde {p} \left(z ^ {(l)}\right) / q \left(z ^ {(l)}\right)} \tag {14.24} \\ \end{array}
$$

其中  $I(\cdot)$  是指示函数（如果其参数为真，则函数值为1，否则为0）。使  $L$  趋于无穷大，并假设分布的适当的正则性，我们就可以用根据原始采样分布  $q(z)$  加权的积分来替换求和：

$$
\begin{array}{l} p (z \leqslant a) = \frac {\int I (z \leqslant a) \{\tilde {p} (z) / q (z) \} q (z) \mathrm {d} z}{\int \{\tilde {p} (z) / q (z) \} q (z) \mathrm {d} z} \\ = \frac {\int I (z \leqslant a) \tilde {p} (z) \mathrm {d} z}{\int \tilde {p} (z) \mathrm {d} z} \tag {14.25} \\ \simeq \int I (z \leqslant a) p (z) \mathrm {d} z \\ \end{array}
$$

这就是  $p(z)$  的累积分布函数。同样，我们可以看到  $p(z)$  的归一化并不是必需的。

对于有限值  $L$  和给定的初始样本集，重采样值只能近似地从目标分布中抽取。与拒绝采样一样，随着采样分布  $q(z)$  更接近目标分布  $p(z)$ ，近似结果的效果也会越来越好。当  $q(z) = p(z)$  时，初始样本  $\left(z^{(1)},\dots ,z^{(L)}\right)$  具有目标分布，且权重  $w_{n} = 1 / L$ ，因此重采样值也具有目标分布。

如果需要关于分布  $p(z)$  的矩，则可以直接使用原始样本连同权重来计算，因为

$$
\begin{array}{l} \mathbb {E} [ f (z) ] = \int f (z) p (z) \mathrm {d} z \\ = \frac {\int f (z) [ \tilde {p} (z) / q (z) ] q (z) \mathrm {d} z}{\int [ \tilde {p} (z) / q (z) ] q (z) \mathrm {d} z} \tag {14.26} \\ \approx \sum_ {l = 1} ^ {L} w _ {l} f (z _ {l}) \\ \end{array}
$$

## 14.2 马尔可夫链蒙特卡洛采样

在上一节中，我们讨论了用于计算函数期望的拒绝采样和重要性采样策略，并且我们看到它们在高维空间中有严重局限性。本节将讨论一种非常通用和强大的采样策略，称为马尔可夫链蒙特卡洛采样，它允许从多种分布中进行采样并且可以很好地扩展到高维样本空间。马尔可夫链蒙特卡洛采样起源于物理学（Metropolis and Ulam, 1949），直至20世纪80年代末期才开始在统计学领域产生显著影响。

与拒绝采样和重要性采样一样，我们依然从提议分布中采样。然而这一次，我们

保持当前状态  $z^{(\tau)}$  的记录，提议分布  $q\left(z \mid z^{(\tau)}\right)$  以当前状态为条件，因此样本序列  $z^{(1)}, z^{(2)}, \cdots$  形成一个马尔可夫链（参见14.2.2小节）。同样，尽管  $Z_{p}$  的值可能是未知的，如果我们将  $p(z)$  写成  $p(z) = \tilde{p}(z) / Z_{p}$ ，我们假设对于任意给定的  $z$  值， $\tilde{p}(z)$  可以很容易计算出来。我们的提议分布足够简单，可以直接从中抽取样本。在算法的每一个迭代中，我们从提议分布中生成一个候选样本  $z^{*}$ ，然后根据一个适当的准则来接受该样本。

### 14.2.1 Metropolis算法

在基础的Metropolis算法（Metropolis et al., 1953）中，我们假设提议分布是对称的，即对于所有的  $z_A$  和  $z_B$  值，有  $q(z_A|z_B) = q(z_B|z_A)$  。然后候选样本以如下概率被接受：

$$
A \left(z ^ {\star}, z ^ {(\tau)}\right) = \min  \left(1, \frac {\tilde {p} \left(z ^ {\star}\right)}{\tilde {p} \left(z ^ {(\tau)}\right)}\right) \tag {14.27}
$$

这可以通过选择一个在单位区间  $(0,1)$  上均匀分布的随机数  $u$  来实现。如果  $A\left(z^{\star},z^{(\tau)}\right) > u$  ，则接受该样本。请注意，如果从  $z^{(\tau)}$  到  $z^{\star}$  的步骤导致  $p(z)$  的值增加，那么候选样本一定会被保留。

如果候选样本被接受，则  $z^{(\tau +1)} = z^{\star}$  ，否则候选样本  $z^{\star}$  将被丢弃。将  $z^{(\tau +1)}$  设置为 $z^{(\tau)}$  ，并从分布  $q\left(z|z^{(\tau +1)}\right)$  中抽取另一个候选样本。这与拒绝采样形成了鲜明对比，在拒绝采样中，被拒绝的样本将直接被丢弃。在Metropolis算法中，当一个候选样本被

拒绝时，它的前一个样本将包含在最终的样本列表中，导致样本有多个副本。当然，在实际应用中，每个保留的样本只会有一个副本，并附带一个整数权重因子，用于记录该状态出现的次数。正如我们将看到的，如果对于任何的  $z_A$  和  $z_B$  值， $q(z_A | z_B)$  都是正值（这是一个充分但非必要条件），则  $z^{(\tau)}$  的分布趋向于  $p(z)$  （当  $\tau \to \infty$  时）。然而，应该强调的是，序列  $z^{(1)}, z^{(2)}, \dots$  不是来自  $p(z)$  的独立样本集，因为连续的样本高度相关。如果希望获得独立的样本，那么我们可以丢弃大部分序列，每次只保留第  $M$  个样本。对于足够大的  $M$ ，保留的样本将在实际意义上是独立的。

算法14.1对Metropolis采样做了总结。图14.9显示了一个简单的示例，它使用

![](img/e149c4c05550c8563e736a295f993f2a2a38527042c7e0f3fe25f57ce3c0e6ec.jpg)  
图14.9 使用Metropolis算法从高斯分布中采样的简单示意图，该高斯分布的一个标准差等高线是用椭圆显示的。提议分布是一个各向同性的高斯分布，其标准差为0.2。接受的候选样本以绿线显示，拒绝的候选样本以红线显示。共生成150个候选样本，其中43个被拒绝

Metropolis算法从二维高斯分布中进行采样，其中提议分布是各向同性的高斯分布。

算法14.1: Metropolis采样  
```latex
Input: Unnormalized distribution  $\widetilde{p} (z)$  Proposal distribution  $q(z|\widehat{z})$  Initial state  $z^{(0)}$  Number of iterations  $T$    
Output:  $z\sim \widetilde{p} (z)$ $z_{\mathrm{prev}}\gets z^{(0)}$    
//迭代地进行消息传递  
for  $\tau \in \{1,\dots ,T\}$  do  
 $z^{\star}\sim q(z|z_{\mathrm{prev}})$  //从提议分布中采样 $u\sim \mathcal{U}(0,1)$  //从均匀分布中采样if  $\widetilde{p} (z^{\star}) / \widetilde{p} (z_{\mathrm{prev}}) > u$  then $|z_{\mathrm{prev}}\gets z^{\star}\quad /\quad z^{(\tau)} = z^{\star}$  else $|z_{\mathrm{prev}}\gets z_{\mathrm{prev}}\quad /\quad z^{(\tau)} = z^{(\tau -1)}$  end if  
end for  
return  $z_{\mathrm{prev}} / / z^{(T)}$
```

我们观察一个具体例子（即随机游走）的性质，来深入了解马尔可夫链蒙特卡洛算法的本质。考虑一个由整数构成的状态空间  $z$  ，已知概率

$$
p \left(z ^ {(\tau + 1)} = z ^ {(\tau)}\right) = 0. 5 \tag {14.28}
$$

$$
p \left(z ^ {(\tau + 1)} = z ^ {(\tau)} + 1\right) = 0. 2 5 \tag {14.29}
$$

$$
p \left(z ^ {(\tau + 1)} = z ^ {(\tau)} - 1\right) = 0. 2 5 \tag {14.30}
$$

其中  $z^{(\tau)}$  表示在时间步  $\tau$  的状态。如果初始状态是  $z^{(0)} = 0$  ，那么根据对称性，在时间步  $\tau$  的预期状态也将是零，即  $\mathbb{E}\left[z^{(\tau)}\right] = 0$  ，同样也容易看出  $\mathbb{E}\left[\left(z^{(\tau)}\right)^2\right] = \tau /2$  （见习题14.11）。因此，在时间步  $\tau$  之后，随机游走平均而言仅行进了与  $\sqrt{\tau}$  成比例的距离。这种平方根依赖性是随机游走行为的典型特征，表明随机游走在探索状态空间时效率非常低。正如我们将看到的，设计马尔可夫链蒙特卡洛采样算法的一个核心目标就是避免随机游走行为。

### 14.2.2 马尔可夫链

在更详细地讨论马尔可夫链蒙特卡洛采样之前，我们先研究一下马尔可夫链的一般性质。特别地，我们想知道马尔可夫链在何种情况下会收敛到期望的分布。一阶马尔可夫链定义为随机变量序列  $z^{(1)},\dots ,z^{(M)}$  ，使得对于  $m\in \{1,\dots ,M - 1\}$  ，下面的条件独

立性成立：

$$
p \left(z ^ {(m + 1)} \mid z ^ {(1)}, \dots , z ^ {(m)}\right) = p \left(z ^ {(m + 1)} \mid z ^ {(m)}\right) \tag {14.31}
$$

这可以用一个链式的有向图模型来表示（见图11.29）。然后，我们可以通过给出初始变量的概率分布  $p\left(z^{(0)}\right)$  和以转移概率（transition probabilities）形式表达的后续变量条件分布  $T_{m}\left(z^{(m)},z^{(m + 1)}\right)\equiv p\left(z^{(m + 1)}\mid z^{(m)}\right)$ ，即可完整定义马尔可夫链。如果转移概率对所有  $m$  都相同，则称马尔可夫链是齐次的或同质（homogeneous）的。

链中的某个特定变量的边缘概率可以用马尔可夫链中前一个变量的边缘概率表示为

$$
p \left(z ^ {(m + 1)}\right) = \int p \left(z ^ {(m + 1)} \mid z ^ {(m)}\right) p \left(z ^ {(m)}\right) d z ^ {(m)} \tag {14.32}
$$

其中的积分对于离散变量来说可以替换为求和。如果一个分布在马尔可夫链的每个步骤都保持该分布不变，那么该分布对于这个马尔可夫链来说就是不变的或者说稳定的。因此，对于具有转移概率  $T(z', z)$  的均匀马尔可夫链来说，如果满足以下条件，则分布  $p(z)$  是不变的：

$$
p ^ {\star} (z) = \int T \left(z ^ {\prime}, z\right) p ^ {\star} \left(z ^ {\prime}\right) \mathrm {d} z ^ {\prime} \tag {14.33}
$$

请注意，给定的马尔可夫链可能有多个不变分布。例如，如果转移概率由恒等变换给出，那么任何分布都将是不变的。

确保目标分布  $p(z)$  是不变的一个充分（但非必要）条件是选择满足详细平衡（detailed balance）性质的转移概率，该性质由下式定义

$$
p ^ {\star} (z) T \left(z, z ^ {\prime}\right) = p ^ {\star} \left(z ^ {\prime}\right) T \left(z ^ {\prime}, z\right) \tag {14.34}
$$

对于特定的分布  $p^{\star}(z)$  ，容易看出，如果转移概率满足对于特定分布的详细平衡性质，则保持该分布不变，因为

$$
\begin{array}{l} \int p ^ {\star} \left(z ^ {\prime}\right) T \left(z ^ {\prime}, z\right) d z ^ {\prime} = \int p ^ {\star} (z) T \left(z, z ^ {\prime}\right) d z ^ {\prime} (14.35) \\ = p ^ {\star} (z) \int p \left(z ^ {\prime} \mid z\right) d z ^ {\prime} (14.36) \\ = p ^ {\star} (z) (14.37) \\ \end{array}
$$

满足详细平衡性质的马尔可夫链被称为可逆的（reversible）。

我们的目标是使用马尔可夫链从给定分布中采样。如果能建立一个马尔可夫链使得目标分布是不变的，就可以达到目的。然而，我们还必须要求不管初始分布  $p\left(z^{(0)}\right)$  的选择如何，当  $m \to \infty$  时，分布  $p\left(z^{(m)}\right)$  都收敛到所需的不变分布  $p^{\star}(z)$  。这个性质称

为遍历性（ergodicity），此时不变分布称为平衡（equilibrium）分布。显然，一个遍历（ergodic）的马尔可夫链只能有一个平衡分布。可以证明，一个同质马尔可夫链是遍历的，只需要对不变分布和转移概率施加较弱的约束条件（Neal, 1993）。

在实践中，我们经常利用一组“基础”转移  $B_{1},\dots ,B_{K}$  来构造转移概率。这可以通过混合分布的形式来实现：

$$
T \left(z ^ {\prime}, z\right) = \sum_ {k = 1} ^ {K} \alpha_ {k} B _ {k} \left(z ^ {\prime}, z\right) \tag {14.38}
$$

混合系数  $\alpha_{1},\dots ,\alpha_{K}$  必须满足  $\alpha_{k}\geqslant 0$  且  $\sum_{k}\alpha_{k} = 1$  。“基础”转移可以通过连续应用结合起来，于是有

$$
T \left(z ^ {\prime}, z\right) = \sum_ {z _ {1}} \dots \sum_ {z _ {n - 1}} B _ {1} \left(z ^ {\prime}, z _ {1}\right) \dots B _ {K - 1} \left(z _ {K - 2}, z _ {K - 1}\right) B _ {K} \left(z _ {K - 1}, z\right) \tag {14.39}
$$

如果一个分布对每个基础转移都是不变的，那么很明显它也会对式（14.38）或式（14.39）给出的  $T(z',z)$  保持不变。对于混合分布[式（14.38）]，如果每个基础转移都满足详细平衡条件，那么复合转移概率  $T$  也将满足详细平衡条件。这对于使用式（14.39）构造的转移概率并不成立，尽管通过对基础转移的应用顺序进行对称化（即 $B_{1},B_{2},\dots ,B_{K},B_{K},\dots ,B_{2},B_{1}$ ），可以恢复详细平衡。使用复合转移概率的一个常见例子是让每个基础转移仅改变变量的一个子集。

### 14.2.3 Metropolis-Hastings算法

前面我们介绍了基础的Metropolis算法，但没有实践证明它能从所需分布中进行采样。在给出证明之前，我们首先讨论一种更泛化的形式，称为Metropolis-Hastings算法（Hastings,1970），该算法适用于提议分布不再是其参数的对称函数的情况。特别是在算法的第  $\tau$  步，当前状态为  $z^{(\tau)}$  ，首先从分布  $q_{k}\left(z\mid z^{(\tau)}\right)$  中抽取一个样本  $z^{\star}$  ，然后以概率  $A_{k}\left(z^{\star},z^{(\tau)}\right)$  接受它，其中

$$
A _ {k} \left(z ^ {\star}, z ^ {(\tau)}\right) = \min  \left(1, \frac {\tilde {p} \left(z ^ {\star}\right) q _ {k} \left(z ^ {(\tau)} \mid z ^ {\star}\right)}{\tilde {p} \left(z ^ {(\tau)}\right) q _ {k} \left(z ^ {\star} \mid z ^ {(\tau)}\right)}\right) \tag {14.40}
$$

这里的  $k$  标记了正在考虑的可能转换集合的成员。同样，计算接受准则不需要知道概率分布  $p(z) = \tilde{p}(z) / Z_p$  中的归一化常数  $Z_p$  。对于对称的提议分布，Metropolis-Hastings 准则 [式（14.40）] 可简化为标准的 Metropolis 准则 [式（14.27）]。算法 14.2 对 Metropolis-Hastings 采样做了总结。

算法14.2：Metropolis-Hastings采样  
```latex
Input: Unnormalized distribution  $\widetilde{p} (z)$  Proposal distributions  $\{q_k(z|\widehat{z}):k\in 1,\dots ,K\}$  Mapping from iteration index to distribution index  $M(\cdot)$  Initial state  $z^{(0)}$  Number of iterations  $T$    
Output:  $z\sim \widetilde{p} (z)$ $z_{\mathrm{prev}}\gets z^{(0)}$    
//迭代地进行消息传递  
for  $\tau \in \{1,\dots ,T\}$  do  
 $k\gets M(\tau)$  //从多次迭代中获取分布索引  
 $z^{\star}\sim q_{k}(z|z_{\mathrm{prev}})$  //从提议分布中采样  
 $u\sim \mathcal{U}(0,1)$  //从均匀分布中采样  
if  $\widetilde{p} (z^{\star})q(z_{\mathrm{prev}}|z^{\star}) / \widetilde{p} (z_{\mathrm{prev}})q(z^{\star}|z_{\mathrm{prev}}) > u$  then  
 $|z_{\mathrm{prev}}\gets z^{\star} / / z^{(\tau)} = z^{\star}$  else  
 $|z_{\mathrm{prev}}\gets z_{\mathrm{prev}} / / z^{(\tau)} = z^{(\tau -1)}$  end if  
end for  
return  $z_{\mathrm{prev}} / / z^{(T)}$
```

我们可以通过证明式（14.34）定义的详细平衡条件得以满足，从而进一步证明  $p(z)$  是由Metropolis-Hastings算法所定义的马尔可夫链的不变分布。利用（14.40），我们有

$$
\begin{array}{l} p (z) q _ {k} \left(\mathbf {z} ^ {\prime} \mid \mathbf {z}\right) A _ {k} \left(\mathbf {z} ^ {\prime}, z\right) = \min  \left(p (z) q _ {k} \left(z ^ {\prime} \mid z\right), p \left(z ^ {\prime}\right) q _ {k} \left(z \mid z ^ {\prime}\right)\right) \\ = \min  \left(p \left(z ^ {\prime}\right) q _ {k} \left(z \mid z ^ {\prime}\right), p (z) q _ {k} \left(z ^ {\prime} \mid z\right)\right) \tag {14.41} \\ = p \left(z ^ {\prime}\right) q _ {k} \left(z \mid z ^ {\prime}\right) A _ {k} \left(z, z ^ {\prime}\right) \\ \end{array}
$$

提议分布的具体选择对算法的性能有显著影响。对于连续状态空间，一个常见的选择是以当前状态为中心的高斯分布，在确定此分布的方差时有一个重要的权衡——如果方差较小，则接受转换的比例将会很高，但通过状态空间的过程表现为缓慢的随机游走，导致相关时间较长。然而，如果方差较大，则拒绝率将会很高，因为在我们考虑的复杂问题中，许多被提出的步骤将止于概率  $p(z)$  较低的状态。考虑一个有强相关性分量的多变量分布  $p(z)$ ，如图14.10所示。提议分布的尺度  $\rho$  应尽可能大，同时不至于引起高拒绝率。这表明  $\rho$  应该与最小的长度尺度  $\sigma_{\mathrm{min}}$  同一个数量级。随后系统通过随机游走沿着更延长的方

![](img/62f398bed1ba243af5206144521501b857e3cb70053e469e10c8134fd209d617.jpg)  
图14.10 使用各向同性的提议分布（蓝色圆圈）从相关多元高斯分布（红色椭圆）中进行Metropolis-Hastings采样的示意图，后者在不同方向上具有完全不同的标准差。为了保持较低的拒绝率，提议分布的尺度  $\sigma$  应该与最小标准差  $\sigma_{\mathrm{min}}$  的数量级相当，这导致了随机游走行为，其中大致独立的状态之间分隔的步数是  $(\sigma_{\mathrm{max}} / \sigma_{\mathrm{min}})^2$  阶的，其中  $\sigma_{\mathrm{max}}$  是最大的标准差

向探索分布，因此到达一个与原始状态或多或少独立的状态所需的步长是  $\left(\sigma_{\mathrm{max}} / \sigma_{\mathrm{min}}\right)^2$  数量级的。实际上，在二维问题中，随着  $\rho$  增加而导致的拒绝率增加已被接受转换的更大步长抵消。更一般地说，对于多元高斯分布，获得独立样本所需的步数和  $\left(\sigma_{\mathrm{max}} / \sigma_2\right)^2$  成正比，其中  $\sigma_{2}$  是第二小的标准差（Neal,1993）。撇开这些细节不谈，如果分布变化的长度尺度在不同方向上差异很大，那么Metropolis-Hastings算法可能会收敛得非常慢。

### 14.2.4 吉布斯采样

吉布斯采样（Geman and Geman, 1984）是一种简单且使用广泛的采样策略，可以看作Metropolis-Hastings算法的一个特例。考虑分布  $p(z) = p\left(z_{1}, \dots, z_{M}\right)$ ，我们希望从中采样，假设我们已经为马尔可夫链选择了一些初始状态。吉布斯采样过程中的每一步都包含用我们从一个变量的条件分布中抽取的值替换该变量的值，该条件分布是以其余变量的值为条件的。因此，可通过从分布  $p\left(z_{i} \mid z_{\backslash i}\right)$  中采样来替换  $z_{i}$ ，其中  $z_{i}$  表示  $z$  的第  $i$  个分量，  $z_{\backslash i}$  表示  $\{z_{1}, \dots, z_{M}\}$  但其中不包括  $z_{i}$  。这个过程要么按某种特定顺序循环遍历变量，要么在每一步随机从某些分布中选择要更新的变量。

例如，假设我们有一个关于3个变量的分布  $p\left(z_1, z_2, z_3\right)$ ，在时间步  $\tau$ ，我们选定了值  $z_1^{(\tau)}$ 、 $z_2^{(\tau)}$  和  $z_3^{(\tau)}$ 。首先将  $z_1^{(\tau)}$  替换为通过从如下条件分布中采样获得的新值  $z_1^{(\tau+1)}$ 。

$$
p \left(z _ {1} \mid z _ {2} ^ {(\tau)}, z _ {3} ^ {(\tau)}\right) \tag {14.42}
$$

然后将  $z_2^{(\tau)}$  替换为通过从如下条件分布中采样获得的新值  $z_2^{(\tau +1)}$  。

$$
p \left(z _ {2} \mid z _ {1} ^ {(\tau + 1)}, z _ {3} ^ {(\tau)}\right) \tag {14.43}
$$

这样新的  $z_{1}$  值便可立即在后续的采样步骤中使用。用我们从如下条件分布中抽取的样本  $z_{3}^{(r + 1)}$  更新  $z_{3}$  。

$$
p \left(z _ {3} \mid z _ {1} ^ {(\tau + 1)}, z _ {2} ^ {(\tau + 1)}\right) \tag {14.44}
$$

以此类推，依次循环遍历这3个变量。算法14.3对吉布斯采样做了总结。

算法14.3：吉布斯采样
```
Input: Initial values  $\{z_i:i\in 1,\dots ,M\}$  Conditional distributions  $\{p(z_i|\{z_{j\neq i}\}):i\in 1,\dots ,M\}$  Number of iterations  $T$

Output: Final values  $\{z_i : i \in 1, \dots, M\}$

for  $\tau \in \{1,\dots ,T\}$  do for  $i\in \{1,\dots ,M\}$  do  $|z_{i}\sim p(z_{i}|\{z_{j\neq i}\})$  end for

end for

return  $\{z_{i}:i\in 1,\dots ,M\}$
```

为了证明该过程从目标分布中采样，我们首先注意到分布  $p(z)$  是吉布斯采样步骤中的一个单独的不变量，因此它也是整个马尔可夫链的不变量。这是因为当我们从  $p\left(z_{i} \mid z_{\backslash i}\right)$  中采样时，边缘分布  $p\left(z_{\backslash i}\right)$  显然是不变的，因为  $z_{\backslash i}$  的值没有发生改变。同样，定义的每一步采样都从正确的条件分布  $p\left(z_{i} \mid z_{\backslash i}\right)$  中采样。因为这些条件分布和边缘分布共同指定了联合分布，所以我们可以看到联合分布本身是不变的。

确保能从正确分布中进行吉布斯采样的第二个条件是要具有遍历性（ergodic）。具有遍历性的一个充分条件是条件分布不存在零概率区域。如果是这种情况，那么  $z$  空间中的任意一点都可以通过有限步骤到达其他任意一点，每一步都包括更新每个分量变量中的一个。如果不满足这个要求，即某些条件分布存在零概率区域，那么遍历性（如果适用）必须被显式地证明。

为了完整地描述该算法，还需要指定初始状态的分布，尽管在多次迭代后抽取的样本实际上独立于此分布。当然，从马尔可夫链中连续抽取的样本将高度相关，因此要获得几乎独立的样本，就需要对序列进行子样本采样。

我们可以将吉布斯采样过程看作是Metropolis-Hastings算法的一个特定实例。考虑一个涉及变量  $z_{k}$  的Metropolis-Hastings采样步骤，其余变量  $z_{\backslash k}$  保持不变，从  $\pmb{z}$  到  $\pmb{z}^{\star}$  的转移概率由  $q_{k}(z^{\star}|z) = p(z_{k}^{\star}|z_{\backslash k})$  给出。注意  $z_{\backslash k}^{\star} = z_{\backslash k}$ ，因为这些分量在采样步骤中未发生改变。同样， $p(z) = p(z_k|z_{\backslash k})p(z_{\backslash k})$ 。因此，Metropolis-Hastings算法[式（14.40）]中确定接受概率的因子由下式给出：

$$
A \left(z ^ {\star}, z\right) = \frac {p \left(z ^ {\star}\right) q _ {k} \left(z \mid z ^ {\star}\right)}{p (z) q _ {k} \left(z ^ {\star} \mid z\right)} = \frac {p \left(z _ {k} ^ {\star} \mid z _ {\backslash k} ^ {\star}\right) p \left(z _ {\backslash k} ^ {\star}\right) p \left(z _ {k} \mid z _ {\backslash k} ^ {\star}\right)}{p \left(z _ {k} \mid z _ {\backslash k}\right) p \left(z _ {\backslash k}\right) p \left(z _ {k} ^ {\star} \mid z _ {\backslash k}\right)} = 1 \tag {14.45}
$$

其中  $z_{\backslash k}^{*} = z_{\backslash k}$  。因此，Metropolis-Hastings采样步骤总是可以接受。

与Mebropolis算法类似，我们可以通过研究吉布斯采样在特定分布（例如高斯分布）上的应用来深入了解其行为。考虑图14.11所示的两个变量中相关高斯分布，其条件分布的宽度为  $l$  ，边缘分布的宽度为 $L$  。典型的步长由条件分布决定，并且将是 $l$  阶的。由于状态根据随机游走演化，因此为了获得来自分布的独立样本，所需的步数将是  $(L / l)^2 l$  阶的。当然，如果高斯分布是不相关的，那么吉布斯采样过程将是最优的。对于这个简单问题，我们可以旋转坐标系，使得新变量不相关。但是，在实际应用中，我们通常不可能找到这样的转换。

![](img/1294530dcecd1158b4f381cc6faea60123dab91be96a26a0dd7848835103d52b.jpg)  
图14.11 交替更新两个变量的吉布斯采样示意图，这两个变量的分布是相关高斯分布。步长由条件分布（绿色曲线）的标准差控制，为  $\mathcal{O}(l)$  ，这导致在联合分布（红色椭圆）拉长方向上的进展缓慢。从分布中获得一个独立样本所需的步数是  $\mathcal{O}\left((L = l)^2\right)$

减少吉布斯采样中随机游走行为的一种方法称为超松弛（over-relaxation）（Adler, 1981）。在它的原始形式中，它适用于条件分布是高斯分布的问题，就是说，“条件分布为高斯分布”所涵盖的分布类型比“多元高斯分布”更广泛。例如，非高斯分布  $p(z,y) \propto \exp \left(-z^2 y^2\right)$  具有条件高斯分布。在吉布斯采样的每一步，特定分量  $z_{i}$  具有均值为  $\mu_{i}$  、方差为  $\sigma_{i}^{2}$  的条件分布。在超松弛框架中，  $z_{i}$  的值被替换为

$$
z _ {i} ^ {\prime} = \mu_ {i} + \alpha_ {i} \left(z _ {i} - \mu_ {i}\right) + \sigma_ {i} \left(1 - \alpha_ {i} ^ {2}\right) ^ {1 / 2} v \tag {14.46}
$$

其中  $\nu$  是一个有零均值和单位方差的高斯随机变量， $\alpha$  是一个参数，满足  $-1 < \alpha < 1$ 。当  $\alpha = 0$  时，该方法等同于标准的吉布斯采样；当  $\alpha < 0$  时，该步骤偏向于均值的相反侧。这一步骤保持了目标分布的不变性，因为如果  $z_{i}$  有均值  $\mu_{i}$  和方差  $\sigma_{i}^{2}$ ，那么  $z_{i}'$  也将如此（见习题14.14）。超松弛的作用是在变量高度相关时，鼓励状态空间中的定向运动。有序超松弛框架（ordered over-relaxation）（Neal, 1999）将这种方法推广到了非高斯分布。

吉布斯采样的实用性取决于从条件分布  $p(z_k|z_{\backslash k})$  中采样的难度。对于使用有向图模型表示的概率分布，单个节点的条件分布仅依赖于相应马尔可夫毯中的变量，如图14.12所示。对于有向图，各个节点的基于其父节点的条件分布的广泛选择将导致吉布斯采样的条件分布是对数凹的。因此，14.1.4小节讨论的自适应拒绝采样方法为从有

![](img/e471db845d25f0006fcf034dbe758c5c152dbeaf2089f7adef7e22e520b7f962.jpg)  
图14.12 吉布斯采样需要从一个变量  $z$  的条件分布中抽取样本，该条件分布是基于其余变量的。对于有向图模型，这个条件分布只是马尔可夫毯中节点状态的函数，马尔可夫毯在图中以蓝色阴影显示，它包括了父节点、子节点以及共同父节点

向图中进行蒙特卡洛采样提供了一个具有广泛适用性的框架。

由于基本的吉布斯采样技术一次只考虑一个变量，因此连续样本之间存在强依赖关系。在另一种极端情况下，如果我们能够直接从联合分布中采样（实际上这种操作很难实现），则连续样本将是独立的。我们可以采用一种折中策略来改进简单的吉布斯采样，即连续地从变量组而不是单个变量中采样。这可以通过分块吉布斯（blocking Gibbs）采样算法实现，该算法选择变量块（不一定是互斥的），然后依次对每个块中的变量进行联合采样，条件是其他所有变量保持其当前值（Jensen, Kong, and Kjaerulff, 1995）。

### 14.2.5 祖先采样

对于许多模型来说，联合分布  $p(z)$  可以方便地用图模型的术语来表示。对于一个没有观测变量的有向图，可以直接使用祖先采样（ancestral sampling）方法从联合分布中采样。联合分布由下式指定：

$$
p (z) = \prod_ {i = 1} ^ {M} p \left(z _ {i} | \mathrm {p a} (i)\right) \tag {14.47}
$$

其中  $z_{i}$  是与节点  $i$  关联的一组变量，  $\mathrm{pa}(i)$  表示与节点  $i$  的父节点关联的一组变量。为了获得联合分布的一个样本，我们按照  $z_{1},\dots ,z_{M}$  的顺序遍历这组变量，并从条件分布  $p(z_i|\mathrm{pa}(i))$  中采样。这总是可能的，因为在每个步骤，所有父变量都会被实例化。遍历图一次后，我们将获得联合分布的一个样本。这里假设我们可以从每个节点的各个条件分布中采样。

考虑这样一个有向图，其部分节点组成了证据集  $\varepsilon$  ，并被观测值实例化过。原则上可以扩展上述过程，至少对于代表离散变量的节点，我们可以给出以下逻辑采样方法（Henrion, 1988），它可以看作重要性采样的一种特殊情况（见14.1.5小节）。在每一步，当为一个观测值已知的变量  $z_{i}$  获得一个采样值时，对采样值与观测值进行比较，如果它们一致，则保留采样值，并继续依次处理下一个变量。然而，如果它们不一致，则丢弃到目前为止的整个样本，重新从图的第一个节点开始。这种算法可以正确地从后验分布中采样，因为它仅仅相当于从隐变量和数据变量的联合分布中抽取样本，然后丢弃那些与观测数据不一致的样本（在发现一个不一致的值后就不再从联合分布中继续采样，从而稍微提高了效率）。然而，随着观测变量数量的增加和这些变量可以取的状态数量的增加，从后验分布中接受一个样本的整体概率迅速降低，因此这种方法在实践中很少使用。

这种方法的改进版称为似然加权采样（likelihood weighted sampling）（Fung and Chang, 1990; Shachter and Peot, 1990），它基于祖先采样并结合了重要性采样。对于每一个变量，如果该变量在证据集中，那么把它设置为其实例化的值。如果它不在证据集中，则从条件分布  $p\big(z_i \mid \mathrm{pa}(i)\big)$  中采样，其中条件变量设置为其当前采样的值。随后产生的样本  $z$  的权重由下式给出（见习题 14.15）：

$$
r (z) = \prod_ {z _ {i} \notin \varepsilon} \frac {p \left(z _ {i} | \mathrm {p a} (i)\right)}{p \left(z _ {i} | \mathrm {p a} (i)\right)} \prod_ {z _ {i} \in \varepsilon} \frac {p \left(z _ {i} | \mathrm {p a} (i)\right)}{1} = \prod_ {z _ {i} \in \varepsilon} p \left(z _ {i} | \mathrm {p a} (i)\right) \tag {14.48}
$$

这种方法可以进一步使用自重要性采样（self-importance sampling）（Shachter and Peot, 1990）扩展，其中重要性采样分布不断更新，以反映当前估计的后验分布。

## 14.3 郎之万采样

Metropolis-Hastings 算法通过使用提议分布创建候选样本的马尔可夫链，并使用式（14.40）所示的准则接受或拒绝它们，从而从概率分布中抽取样本。这可能相对低效，因为提议分布通常是一个简单且固定的分布，它可以在数据空间的任何方向上生成更新，从而导致随机游走的行为。

我们已经看到，在训练神经网络时，利用关于模型可学习参数的对数似然梯度来最大化似然函数是极其有利的。类似地，我们可以引入利用关于数据向量的概率密度梯度的马尔可夫链采样算法，以便优先向概率较大的区域移动。这种技术称

为哈密顿蒙特卡洛（Hamiltonian Monte Carlo）采样，也称混合蒙特卡洛（hybrid Monte Carlo）采样。该采样方法使用了Metropolis接受准则（Duane et al., 1987; Bishop, 2006）。在这里，我们研究一种在深度学习中使用广泛的新的采样方法——朗之万采样。尽管它避免了使用Metropolis接受准则，但我们必须仔细设计算法，以确保产生的结果样本是无偏的。在基于能量函数的机器学习模型中，朗之万采样发挥了至关重要的作用。

### 14.3.1 基于能量的模型

许多生成式模型可以表示为条件概率分布  $p(\boldsymbol{x}|\boldsymbol{w})$ ，其中  $\boldsymbol{x}$  是数据向量， $\boldsymbol{w}$  代表可学习参数的向量。这样的模型可以通过最大化相对于训练集定义的对应似然函数来进行训练。然而，为了表示一个有效的概率分布，模型必须满足如下要求：

$$
\int p (\boldsymbol {x} \mid \boldsymbol {w}) p (\boldsymbol {x}) \mathrm {d} \boldsymbol {x} = 1 \tag {14.49}
$$

这样的归一化要求大大限制了适用模型的形式。在不考虑归一化约束的情况下，我们可以考虑一个更广泛的模型类别，称为基于能量的模型（LeCun et al., 2006）。假设我们有一个称为能量函数的函数  $E(x, w)$ ，它是其参数的实值函数，但没有其他约束。指数  $\exp\{-E(x, w)\}$  是一个非负数，因此可以看作未归一化的  $x$  上的概率分布。这里在指数中引入负号只是一个惯例，它意味着能量的较高值对应于概率的较低值。然后我们可以使用下式定义一个归一化的分布（见习题14.16）：

$$
p (\boldsymbol {x} | \boldsymbol {w}) = \frac {1}{Z (\boldsymbol {w})} \exp \{- E (\boldsymbol {x}, \boldsymbol {w}) \} \tag {14.50}
$$

其中归一化常数  $Z(w)$  称为配分函数（partition function），它由下式定义：

$$
Z (\boldsymbol {w}) = \int \exp \{- E (\boldsymbol {x}, \boldsymbol {w}) \} d \boldsymbol {x} \tag {14.51}
$$

能量函数通常使用深度神经网络进行建模，输入向量为  $\pmb{x}$ ，标量输出为  $E(x, w)$ ，其中  $w$  代表网络中权重和偏置的集合。

注意配分函数依赖于  $\pmb{w}$ ，这给训练带来了一些问题。例如，一组独立同分布数据 $\mathcal{D} = \left(x_1,\dots ,x_N\right)$  的对数似然函数具有以下形式：

$$
\ln p (\mathcal {D} \mid \boldsymbol {w}) = - \sum_ {n = 1} ^ {N} E \left(\boldsymbol {x} _ {n}, \boldsymbol {w}\right) - N \ln Z (\boldsymbol {w}) \tag {14.52}
$$

要计算  $\ln p(\mathcal{D}|\boldsymbol {w})$  关于  $\pmb{\psi}$  的梯度，我们需要知道  $Z(w)$  的形式。然而，对于具有许多选择的能量函数  $E(x,w)$  ，计算配分函数［式（14.51）］是不切实际的，因为这需要对整个  $\pmb{x}$  空间进行积分（或对离散变量求和）。“基于能量的模型”指的就是这种积分难以处理的模型。然而请注意，概率模型可以视为基于能量的模型的特例，因此本书讨论的许多模型可以视为基于能量的模型。基于能量的模型的最大优势是，它们在

不需要归一化的情况下具有灵活性。相应的缺点是，由于归一化常数未知，它们可能更难以训练。

### 14.3.2 最大化似然

我们已经开发了各种近似方法来训练基于能量的模型，而无须计算配分函数（Song and Kingma, 2021）。本小节探讨基于马尔可夫链蒙特卡洛采样的技术。另一种方法称为分数匹配，我们将在扩散模型的背景下进行讨论（见第20章）。

我们已经看到，对于基于能量的模型，由于存在未知的配分函数  $Z(w)$ ，似然函数不能显式求值。然而，我们可以利用蒙特卡洛采样方法，来计算对数似然函数关于模型参数的梯度。一旦以任何方式训练了基于能量的模型，就可以使用采样方法从模型中抽取样本了，比如使用蒙特卡洛采样方法。

使用式（14.50），对于一个基于能量的模型，对数似然函数关于模型参数的梯度可以写成以下形式：

$$
\nabla_ {w} \ln p (\boldsymbol {x} \mid \boldsymbol {w}) = - \nabla_ {w} E (\boldsymbol {x}, \boldsymbol {w}) - \nabla_ {w} \ln Z (\boldsymbol {w}) \tag {14.53}
$$

这是针对单个数据点  $\pmb{x}$  的似然函数，但实际上我们希望最大化训练数据集上的似然函数，该数据集是从某个未知分布  $p_{\mathcal{D}}(\pmb {x})$  中抽样得到的。如果我们假设数据点是独立同分布的，那么我们可以考虑与  $p_{\mathcal{D}}(\pmb {x})$  相关的似然对数期望的梯度，其由下式给出：

$$
\mathbb {E} _ {\boldsymbol {x} \sim p _ {\mathcal {D}}} \left[ \nabla_ {\boldsymbol {w}} \ln p (\boldsymbol {x} \mid \boldsymbol {w}) \right] = - \mathbb {E} _ {\boldsymbol {x} \sim p _ {\mathcal {D}}} \left[ \nabla_ {\boldsymbol {w}} E (\boldsymbol {x}, \boldsymbol {w}) \right] - \nabla_ {\boldsymbol {w}} \ln Z (\boldsymbol {w}) \tag {14.54}
$$

这里我们利用了  $-\nabla_{\boldsymbol{w}}\ln Z(\boldsymbol{w})$  不依赖于  $\pmb{x}$  的性质，因而可以把它移到期望之外。假设配分函数  $Z(\pmb{w})$  是未知的，但我们可以利用式（14.51）并重组得到

$$
- \nabla_ {w} \ln Z (w) = \int \left\{\nabla_ {w} E (x, w) \right\} p (x | w) d x \tag {14.55}
$$

式（14.55）右侧的部分对应于模型分布  $p(\boldsymbol{x}|\boldsymbol{w})$  的期望，它由下式给出（见习题14.18）：

$$
\int \left\{\nabla_ {w} E (\boldsymbol {x}, \boldsymbol {w}) \right\} p (\boldsymbol {x} \mid \boldsymbol {w}) d \boldsymbol {x} = \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {M}} \left[ \nabla_ {w} E (\boldsymbol {x}, \boldsymbol {w}) \right] \tag {14.56}
$$

结合式（14.54）～式（14.56），我们得到

$$
\nabla_ {w} \mathbb {E} _ {x \sim p _ {\mathcal {D}}} [ \ln p (\boldsymbol {x} \mid \boldsymbol {w}) ] = - \mathbb {E} _ {x \sim p _ {\mathcal {D}}} \left[ \nabla_ {w} E (\boldsymbol {x}, \boldsymbol {w}) \right] + \mathbb {E} _ {x \sim p _ {\mathcal {M}} (\boldsymbol {x})} \left[ \nabla_ {w} E (\boldsymbol {x}, \boldsymbol {w}) \right] \tag {14.57}
$$

图14.13对这个结果做了说明，详细解释如下。我们的目标是找到参数  $\pmb{w}$  的值，让似然函数的值最大化。为此，我们朝梯度  $\nabla_{\pmb{w}}\ln p(\pmb{x}|\pmb{w})$  的方向对  $\pmb{w}$  进行了微小调整。从式（14.57）中可以看出，这个梯度的期望值可以分解为两部分，且这两部分符号相反。式（14.57）右侧第一项的作用是减小  $E(x,\mathbf{w})$  ，因而会增大模型定义的数据点  $\pmb{x}$  的概率密度，这里的  $\pmb{x}$  是从  $p_{D}(x)$  中抽取的。式（14.57）右侧第二项的作用则是增大

$E(x, w)$ ，从而降低由模型定义的来自模型本身的数据点的概率密度。在模型密度超过训练数据密度的区域，净效果将是增加能量，因此降低了概率密度。相反，在训练数据密度超过模型密度的区域，净效果将是降低能量，因此增大了概率密度。正如我们所希望的那样，这两项一起将概率密度从训练数据密度低的区域移向了训练数据密度高的区域。当模型分布与数据分布相匹配时，这两项将在幅度上相等，此时式（14.57）左侧的梯度就等于零。

![](img/70c3911faaf5674b3606950e6e7379ea29321773c39249994c6c98c779f7af3b.jpg)  
图14.13通过最大化似然对基于能量的模型进行训练的示意图，其中显示了能量函数  $E(x, w)$  （绿色）及相关的模型分布  $p_{\mathcal{M}}(x)$  和真实数据分布  $p_{D}(x)$  。利用式（14.57）增加期望的对数似然，会在与模型样本（用蓝点表示）对应的点处推高能量函数，并在与数据集样本（用红点表示）对应的点处拉低能量函数

### 14.3.3 朗之万动力学

当把式（14.57）作为一种实际的训练方法应用时，我们需要近似计算式（14.57）右侧的两项。对于任何给定的  $x$  值，我们可以使用自动微分来评估  $\nabla_{\boldsymbol{w}}E(\boldsymbol{x},\boldsymbol{w})$  。对于式（14.57）中的第一项，我们可以使用训练集来估计关于  $x$  的期望：

$$
\mathbb {E} _ {\boldsymbol {x} \sim p _ {D}} \left[ \nabla_ {\boldsymbol {w}} E (\boldsymbol {x}, \boldsymbol {w}) \right] \approx \frac {1}{N} \sum_ {n = 1} ^ {N} \nabla_ {\boldsymbol {w}} E \left(\boldsymbol {x} _ {n}, \boldsymbol {w}\right) \tag {14.58}
$$

式（14.57）中的第二项更具挑战性，因为我们需要从能量函数定义的模型分布中抽取样本，而相应的配分函数难以处理。这可以使用马尔可夫链蒙特卡洛采样来完成。另一种受欢迎的采样方法称为随机梯度朗之万动力学，简称朗之万采样（Parisi, 1981; Welling and Teh, 2011）。这一项仅通过得分函数（score function）依赖于分布  $p(\pmb{x}|\pmb{w})$ ，得分函数定义为对数似然函数关于数据向量  $\pmb{x}$  的梯度，并由下式给出：

$$
\mathrm {s} (\boldsymbol {x}, \boldsymbol {w}) = \nabla_ {\boldsymbol {x}} \ln p (\boldsymbol {x} \mid \boldsymbol {w}) \tag {14.59}
$$

值得强调的是，这是针对数据点  $x$  的梯度，而非通常意义上针对可学习参数  $\pmb{w}$  的梯度。将式（14.50）代入式（14.59），我们可以得到

$$
\mathrm {s} (\boldsymbol {x}, \boldsymbol {w}) = - \nabla_ {\boldsymbol {x}} E (\boldsymbol {x}, \boldsymbol {w}) \tag {14.60}
$$

可以看到，配分函数不再出现，因为它与  $x$  无关。

让我们从一个先验分布中开始抽取一个初始值  $x^{(0)}$  ，然后执行以下马尔可夫链步骤：

$$
\boldsymbol {x} ^ {(\tau + 1)} = \boldsymbol {x} ^ {(\tau)} + \eta \nabla_ {\boldsymbol {x}} \ln p (\boldsymbol {x} ^ {(\tau)}, \boldsymbol {w}) + \sqrt {2 \eta} \varepsilon^ {(\tau)}, \quad \tau \in 1, \dots , \mathcal {T} \tag {14.61}
$$

其中  $\varepsilon^{(\tau)}\sim \mathcal{N}(\mathbf{0},\mathbf{I})$  是从零均值、单位协方差的高斯分布中独立抽取的样本，参数 $\eta$  控制步长的大小。朗之万方程的每次迭代都会沿着对数似然的梯度方向迈出一步，然后添加高斯噪声。可以证明，当  $\eta \to 0$  和  $\varGamma\rightarrow\infty$  时，  $z^{(r)}$  的值是从分布  $p(x)$  中独立抽取的样本。算法14.4对朗之万采样做了总结。

我们可以通过重复这个过程来生成一组样本  $\{x_{1},\dots ,x_{M}\}$  ，然后使用下式来近似计算式（14.57）中的第二项：

$$
\mathbb {E} _ {\boldsymbol {x} \sim p _ {\mathcal {M}} (\boldsymbol {x})} \left[ \nabla_ {\boldsymbol {w}} E (\boldsymbol {x}, \boldsymbol {w}) \right] \approx \frac {1}{M} \sum_ {m = 1} ^ {M} \nabla_ {\boldsymbol {w}} E \left(\boldsymbol {x} _ {m}, \boldsymbol {w}\right) \tag {14.62}
$$

运行长的马尔可夫链来生成独立样本的计算成本可能很高，所以我们需要考虑实际的近似方法。有一种方法称为对比散度（contrastive divergence）（Hinton, 2002）。在这里，用来评估式（14.62）的样本是通过从训练数据点  $x_{n}$  开始的蒙特卡洛链获得的。如果蒙特卡洛链运行了大量步骤，那么得出的值基本上就是来自模型分布的无偏样本。相反，对比散度方法建议只运行几步的蒙特卡洛链，甚至只有一步，这在计算上成本要低得多。相应的结果样本将远非无偏样本，并且会靠近数据流形。因此，使用梯度下降的效果将是仅在数据流形的邻域内塑造能量曲面，从而塑造概率密度。这对于诸如分类的任务可能是有效的，但预计在学习生成式模型方面效果会较差。

算法14.4：朗之万采样


Input: Initial value  $\pmb{x}^{(0)}$  Probability density  $p(\pmb{x},\pmb{w})$  Learning rate parameter  $\eta$  Number of iterations  $T$


Output: Final value  $x^{(T)}$

$$
\overline {{\boldsymbol {x} \leftarrow \boldsymbol {x} _ {0}}}
$$

for  $\tau \in \{1,\dots ,T\}$  do


$\begin{array}{rl} & {\varepsilon \sim \mathcal{N}(\varepsilon |\mathbf{0},\pmb {I})}\\ & {\pmb {x}\gets \pmb {x} + \eta \nabla_{\pmb{x}}\ln p(\pmb {x},\pmb {w}) + \sqrt{2\eta}\varepsilon} \end{array}$


end for

return  $\pmb{x}$  //最终值  $x^{(T)}$

## 习题

14.1（ $\star \star$ ）证明式（14.2）定义的  $\overline{f}$  是无偏估计，换句话说，其右边项的期望等于  $\mathbb{E}[f(z)]$ 。  
14.2（ $\star$ ）证明式（14.2）定义的  $\overline{f}$  的方差由式（14.4）给出。  
14.3（ $\star$ ）假设  $z$  是在区间  $(0, 1)$  上均匀分布的随机变量，使用  $y = h^{(-1)}(z)$  转换  $z$ ，其中  $h(y)$  由式（14.6）给出。证明  $y$  具有分布  $p(y)$ 。  
14.4（ $\star \star$ ）给定一个在区间  $(0, 1)$  上均匀分布的随机变量  $z$ ，找到一个转换  $y = f(z)$ ，使得  $y$  具有式（14.8）给出的柯西分布。  
14.5（ $\star \star$ ）假设变量  $z_{1}$  和  $z_{2}$  在单位圆上均匀分布，如图 14.3 所示，进行式（14.10）和式（14.11）给出的变量变换。证明  $(y_{1}, y_{2})$  具有式（14.12）所示的联合分布。  
14.6（ $\star \star$ ）假设  $z$  是一个具有零均值和单位协方差矩阵的  $D$  维高斯分布的随机变量，并且假设正定对称矩阵  $\pmb{\Sigma}$  具有楚列斯基分解  $\pmb{\Sigma} = \pmb{L}\pmb{L}^{\mathrm{T}}$ ，其中  $\pmb{L}$  是一个下三角矩阵（即主对角线以上的元素为零）。证明变量  $\pmb{y} = \pmb{\mu} + \pmb{L}\pmb{z}$  服从均值为  $\pmb{\mu}$  和协方差为  $\pmb{\Sigma}$  的高斯分布。这提供了一种利用服从零均值、单位方差的一元高斯样本来生成服从一般多元高斯分布样本的技术。  
14.7（ $\star \star$ ）证明拒绝采样确实是从期望的分布  $p(z)$  中抽取样本。假设提议分布是  $q(z)$ 。证明样本值  $z$  被接受的概率由  $\tilde{p}(z) / kq(z)$  给出，其中  $\tilde{p}$  是与  $p(z)$  成比例的任何未归一化的分布，并且常数  $k$  设置为能够确保  $kq(z) \geqslant \tilde{p}(z)$  的最小值。注意，抽取值  $z$  的概率等于从  $q(z)$  中抽取该值的概率乘以给定已经抽取了该值的情况下接受该值的概率。利用这一点，结合概率的加法和乘法规则，写出关于  $z$  的归一化分布形式，并证明它等于  $p(z)$ 。  
14.8（ $\star$ ）假设变量  $z$  在区间  $[0, 1]$  上均匀分布。证明变量  $y = b \tan(z) + c$  具有式（14.16）给出的柯西分布。  
14.9（ $\star \star$ ）使用连续性和归一化的要求，确定自适应拒绝采样中包络分布[式（14.17）]的系数  $k_{i}$ 。  
14.10（ $\star \star$ ）通过使用14.1.2小节讨论的从单一指数分布中采样的技术，设计一种算法，用来从式（14.17）定义的分段指数分布中采样。  
14.11（ $\star$ ）证明式（14.28）～式（14.30）定义的整数上的简单随机游走具有性质  $\mathbb{E}\left[\left(z^{(\tau)}\right)^2\right] = \mathbb{E}\left[\left(z^{(\tau - 1)}\right)^2\right] + 1 / 2$ ，并因此通过归纳得出  $\mathbb{E}\left[\left(z^{(\tau)}\right)^2\right] = \tau / 2$ 。  
14.12（ $\star \star$ ）证明14.2.4小节讨论的吉布斯采样满足式（14.34）定义的详细平衡性质。  
14.13（ $\star$ ）考虑图14.14所示的分布。讨论标准的吉布斯采样是不是遍历的，并由此判断是否能正确地从此分布中采样。

![](img/206210f9214549778e79299071209842d9cbdfc4d23ced4d52764cd0e544a36d.jpg)  
图14.14 变量  $z_{1}$  和  $z_{2}$  上的概率分布在阴影区域是均匀的，在其他区域为零

14.14（ $\star$ ）验证超松弛更新[式（14.46）]，其中  $z_{i}$  具有均值  $\mu_{i}$  和方差  $\sigma_{i}$ ， $\nu$  具有零均值和单位方差， $z_{i}^{\prime}$  具有均值  $\mu_{i}$  和方差  $\sigma_{i}^{2}$ 。  
14.15（ $\star$ ）证明在有向图的似然加权采样中，重要性采样权重由式（14.48）给出。  
14.16（ $\star$ ）证明只要  $Z(w)$  满足式（14.51），式（14.50）所示的分布关于  $\pmb{x}$  就是归一化的。  
14.17（ $\star \star$ ）利用式（14.50），证明基于能量的模型的对数似然函数的梯度可以写成式（14.52）的形式。  
14.18（ $\star \star$ ）利用式（14.54）～式（14.56），证明基于能量的模型的对数似然函数的梯度可以写成式（14.57）的形式。
