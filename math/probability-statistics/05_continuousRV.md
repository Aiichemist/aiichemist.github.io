# 连续随机变量

## 一元连续随机变量
如果随机变量 $X$ 的所有可能取值不可以逐个列举出来，而是整个数轴或数轴上某一区 间内的任一点，我们就称 $X$ 为连续随机变量。

### 概率密度函数：积分
离散随机变量对应的数学工具为求和 $Σ$，连续随机变量对应积分 $∫$。对于 连续随机变量 $X$，如果存在非负函数 $f_X(x)$ 使得：
$$\Pr(X\in B)=\int\limits_Bf_X(x)dx$$
则称函数 $f_X(x)$ 为 $X$ 的概率密度函数 (probability density function, PDF)。

特别地，如图 1 所示，当 $B$ 为区间 $[a, b]$ 时，随机变量 $X$ 的概率对应定积分：
$$\Pr(a\le X\le B)=\int_a^bf_X(x)dx$$
![](/img/Pasted%20image%2020230907230009.png ':size=70%')

### 概率密度非负，面积为1
概率密度函数 $f_X(x)$ 必须是非负 $f_X(x) ≥ 0$，且满足：
$$\Pr(-\infty\lt X\lt+\infty)=\int_{-\infty}^{+\infty}f_X(x)dx=1$$
上式常简写为：
$$\int\limits_xf_X(x)dx=1$$
如图 2 所示，从图像上来看，$f_X(x)$ 曲线和整个横轴包围区域的面积为 $1$，这也是归一化。换句 话说，一个函数要想能当做概率密度函数来用先要满足非负、面积为 $1$ 这两个条件。

![](/img/Pasted%20image%2020230907230224.png ':size=70%')

### 单点集合：概率密度非负，但是概率为0
利用数值积分方法，$X$ 的取值范围在 $[a, a + Δ]$ 对应的概率为：
$$\Pr(a\le X\le a+\Delta)=\int_{a}^{a+\Delta}f_X(x)dx\approx f_X(a)\Delta$$

当 $Δ → 0$ 时，$\Pr(a ≤ X ≤ a + Δ) → 0$。

也就是说，对于单点集合，$X = a$ 的概率为 $0$：
$$\Pr(X=a)=\int_a^af_X(x)dx=0$$
即便概率密度 $f_X(a)$ 大于 $0$。

### 区间端点
因此，对于连续随机变量 $X$，区间端点对概率计算不起任何作用，因此以下四个概率值等价：
$$\Pr(a\le X\le b)=\Pr(a\lt X\le b)=\Pr(a\le X\lt b)=\Pr(a\lt X\lt b)$$

### 概率密度值可以大于1
再次强调 $f_X(x)$ 并不是概率，而是概率密度，因此 $f_X(x)$ 可大于 $1$。

比如，图 3 所示在 $[0, 0.5]$ 区间上连续均匀分布的概率密度函数 $f_X(x)$。很明显，$f_X(x)$ 的最大值 为 $2$，但是长方形的面积仍为 $1$：
$$\begin{split}
\Pr(-\infty\lt X\lt +\infty)
&=\int_{-\infty}^{0}f_X(x)dx
+\int_0^{0.5}f_X(x)dx
+\int_{0.5}^{+\infty}f_X(x)dx\\
&=0+\int_{0}^{0.5}2dx+0\\
&=2x\big|_0^{0.5}=1\\
\end{split}$$
反复强调，图 3 中的 $2$ 不是概率值，而是概率密度。对于一元随机变量，概率密度函数在 一定区间内积分结果才是概率值。概率密度虽然不是概率值，但也量化“可能性”。
![](/img/Pasted%20image%2020230907231011.png ':size=70%')

### 累积分布函数
本书前文介绍，给定一元离散随机变量 $X$ 的概率质量函数 $p_X(x)$，求解其 CDF 时，用的是累加 $Σ$。

以图 4 (a) 为例，对于一元连续随机变量 $X$，求累积分布函数 CDF $F_X(x)$ 用的是积分，也就是求面积：
$$F_X(x)=\Pr(X\le x)=\int_{-\infty}^xf_X(t)dt$$
图 4 (a) 中 $f_X(x)$ 图形的面积对应概率值，而图 4 (b) 中 $F_X(x)$ 的高度对应概率值。

随机变量 $X$ 在 $[a, b]$ 区间对应的概率可以用 CDF $F_X(x)$ 计算：
$$\Pr(a\le X\le b)=F_X(b)-F_X(a)$$
再次强调，对于一元连续随机变量，PDF 是概率密度，CDF 是概率。
![](/img/Pasted%20image%2020230907231309.png ':size=70%')

## 期望、方差和标准差
### 期望值
连续随机变量 $X$ 期望定义如下：
$$E(X)=\int_{-\infty}^{+\infty}x\cdot\underbrace{f_X(x)dx}_{Weight}$$
上式也相当于加权平均。其中，$f_X(x)$ 相当于是“权重”。显然，$f_X(x)$ 非负，但是 $x$ 取值可正可 负。这也就是说，$E(X)$ 可正可负。

上式常简写为：
$$E(X)=\int\limits_{x}x\cdot f_X(x)dx$$
权重当然满足$\int\limits_xx\cdot f_X(x)dx$

### 连续均匀分布
如图 5 所示，如果随机变量 $X$ 在 $[a, b]$ 上服从连续均匀分布 (continuous uniform distribution)， $X$ 的概率密度函数为：
$$f_X(x)=\begin{cases}
\dfrac{1}{b-a}&\text{for }a\le x\le b\\
0 &\text{for }x\lt a\text{ or }x\gt b\\
\end{cases}$$
![](/img/Pasted%20image%2020230907231758.png ':size=70%')

$X$ 的期望值为：
$$E(X)=\int_a^bx\cdot\dfrac{1}{b-a}dx=\frac{1}{b-a}\frac{x^2}{2}\bigg|_a^b=\dfrac{1}{b-a}\dfrac{b^2-a^2}{2}=\frac{a+b}{2}$$
随机变量 $X$ 的取值在 $[a, b]$ 变化，对应的概率密度变化用 $f_X(x)$ 刻画。而求得的期望值 $E(X)$ 则是一个标量，这相当于总结归纳，也是降维。

几何角度来看，如图 5 所示，计算 $X$ 的期望值相当于找到一块均质木板的质心在长度方向上的位置。

### 方差
连续随机变量 $X$ 方差的定义为：
$$var(X)=E\left[\left(X-E(X)\right)^2\right]=\int\limits_x\left(\underbrace{x-E(X)}_{Deviation}\right)^2\cdot\underbrace{f_X(x)}_{Weight}dx$$
同样，连续随机变量 $X$ 的方差也满足如下计算技巧：
$$var(X)=E\left((X-E(X))^2\right)=E(X^2)-(E(X))^2$$
其中，
$$E(X^2)=\int\limits_xx^2\cdot f_X(x)dx$$
### 数值积分
如图 6 所示，随机变量 $X$ 在 $[0, 1]$ 上为均匀分布。我们可以很容易通过积分得到期望值、方差。但是，并不是所有的概率密度函数都有解析式；此外，即便概率密度函数有解析式，也不代表我们能计算得到积分的解析解，比如高斯函数。 

如图 7 所示，这就需要用到**数值积分** (numerical integration)。当然，我们还可以用**蒙特卡洛模拟** (Monte Carlo simulation) 估算面积。
![](/img/Pasted%20image%2020230907232453.png ':size=70%')
![](/img/Pasted%20image%2020230907232458.png ':size=70%')

## 二元连续随机变量
假设同一个试验中，有两个连续随机变量 $X$ 和 $Y$，非负二元函数 $f_{X,Y}(x,y)$ 为 $(X, Y)$ 的**联合概率密度函数** (joint probability density function 或 joint PDF)。

对于一元连续随机变量，积分得到的面积对应概率。而二元随机变量计算概率的工具是二重积分，从图像上来看，二重积分得到的体积对应概率。

如图 8 所示，给定积分区域 $A = {(x, y) | a < x < b, c < y < d}$，概率 $\Pr((X, Y) ∈ A)$ 对应的二重积 分为：
$$\underbrace{\Pr((X,Y)\in A)}_{Probability}=\int_c^d\int_a^b\underbrace{f_{X,Y}(x,y)}dxdy$$
![](/img/Pasted%20image%2020230907232735.png ':size=70%')

### 体积为1：样本空间概率为1
如果积分区域为整个平面，二重积分的结果为1：
$$\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}\underbrace{f_{X,Y}(x,y)}_{\text{Joint PDF}}dxdy=1$$
也就是说，图 8 中 $f_{X,Y}(x,y)$ 曲面和水平面围成几何形状的体积为 $1$，代表样本空间的概率为 $1$。上式本质上也是“穷举法”。

### 累积概率密度CDF
二元累积概率函数 CDF $F_{X,Y}(x,y)$ 定义为：
$$\underbrace{F_{X,Y}(x,y)}_{Probability}=\Pr(X\lt x,Y\lt y)=\int_{-\infty}^y\int_{-\infty}^x\underbrace{f_{X,Y}(s,t)}_{\text{Joint PDF}}dsdt$$
图9 所示等高线为某个二元累积概率函数 $F_{X,Y}(x,y)$。图 9 还绘制了两条边缘 CDF 曲线。
![](/img/Pasted%20image%2020230908084628.png ':size=70%')

## 边缘概率：二元PDF偏积分
图 10 所示为二元概率密度函数 $f_{X,Y}(x,y)$ 曲面和边缘概率曲线的关系。
![](/img/Pasted%20image%2020230908084723.png ':size=70%')

### 边缘概率密度函数$f_X(x)$
如图 11 所示，连续随机变量 $X$ 的边缘概率密度函数 $f_X(x)$ 可以通过 $f_{X,Y}(x,y)$ 对 $y$“偏积分”得到：
$$\underbrace{f_X(x)}_{Marginal}=\overbrace{\int_{-\infty}^{+\infty}\underbrace{f_{X,Y}(x,y)}_{Joint}dy}^{\text{Eliminate y}}$$
上式，相当于消去 (降维、压扁、折叠) 变量 y，这和离散随机变量的“偏求和”类似。
![](/img/Pasted%20image%2020230908085017.png ':size=70%')
上式可以简写为：
$$\underbrace{f_X(x)}_{Marginal}=\overbrace{\int\limits_y\underbrace{f_{X,Y}(x,y)}_{Joint}dy}^{\text{Eliminate y}}$$
注意，$f_X(x)$ 还是概率密度函数，而不是概率。也就是说，$f_{X,Y}(x,y)$ 二重积分得到概率， $f_{X,Y}(x,y)$ “偏积分” 得到的还是概率密度函数。

图 12 比较 $f_{X,Y}(x,y = c)$ 和 $f_X(x)$ 曲线。当 $y = c$ 取不同值时，我们可以看到 $f_{X,Y}(x,y)$ 和 $f_X(x)$ 曲线 形状不同。当 $y = c$ 时，$f_{X,Y}(x,y = c)$ 不是一元连续随机变量 PDF；原因就是面积不为 $1$。但是经过 归一化之后，它们就变成了一元随机变量 PDF。这个归一化的工具就是“贝叶斯定理”。

![](/img/Pasted%20image%2020230908085307.png ':size=70%')
![](/img/Pasted%20image%2020230908085316.png ':size=70%')

### 体密度 vs 面密度 vs 线密度
几何上来看，如图 13 所示，$f_{X,Y,Z}(x,y,z)$ 相当于“体密度”，$f_{X,Y}(x,y)$ 相当于“面密度”，$f_X(x)$ 相当于“线密度”。而概率值就相当于质量。

用白话说，体密度就是“铁块”的密度，计算铁块质量时会用到“体积 × 体密度”。 

面密度就是“铁皮”的密度。铁皮厚度太薄，不便测量。计算铁皮质量时，我们用“面积 × 面密度”。 

线密度对应“铁丝”的密度。关心铁丝横截面面积没有意义，实践中铁丝粗细有特定标准、型号。计算铁丝质量时，我们用“长度 × 线密度”。
![](/img/Pasted%20image%2020230908085452.png ':size=70%')

## 条件概率：引入贝叶斯定理
给定 $X = x$ 条件下，且 $f_X(x) > 0$，条件概率密度函数 $f_{Y|X}(y|x)$ 可以通过下式求得：
$$\underbrace{f_{X|Y}(x|y)}_{Conditional}=\frac{\overbrace{f_{X,Y}(x,y)}^{Joint}}{\underbrace{f_Y(y)}_{Marginal}}$$
再次强调，上式中，边缘 $f_Y(y)$ 也是概率密度。

图15 中 $f_{X,Y}(x,y = −1)$ 曲线代表 $Y = −1$ 时 $(X, Y)$ 联合概率密度函数。

$f_{X,Y}(x,y = −1)$ 对 $x$ 在 $(−∞, + ∞)$ 积分的结果为边缘概率概率密度 $f_Y(y = −1)$。也就是说，$f_{X,Y}(x,y = −1)$ 曲线面积为边缘概率密度 $f_Y(y = −1)$。

下一步，$f_{X,Y}(x,y = −1)$ 经过 $f_Y(y = −1)$ 缩放得到条件概率曲线 $f_{X|Y}(x|y = −1)$。

注意，$f_{X|Y}(x|y = −1)$ 和横轴围成图形的面积为 $1$，这代表 $Y = −1$ 这个新的样本空间概率为 $1$。
![](/img/Pasted%20image%2020230908091031.png ':size=70%')

图 16 比较 $f_X(x)$ 和 $y$ 取不同值时条件概率密度函数 $f_{X|Y}(x|y)$ 图像。将这些曲线投影到同一个平面，得到图 17。注意，图 17 中所有曲线和横轴围成图形的面积都是 $1$。
![](/img/Pasted%20image%2020230908091127.png ':size=70%')
![](/img/Pasted%20image%2020230908091135.png ':size=70%')

### 联合概率、边缘概率、条件概率
根据贝叶斯定理，联合概率、边缘概率、条件概率三者关系为：
$$\underbrace{f_{X,Y}(x,y)}_{Joint}
=\underbrace{f_{X|Y}(x|y)}_{Conditional}\underbrace{f_Y(y)}_{Marginal}=\underbrace{f_{Y|X}(y|x)}_{Conditional}\underbrace{f_X(x)}_{Marginal}$$
连续随机变量 $X$ 的边缘分布概率密度函数 $f_X(x)$ 可以通过下式获得：
$$\underbrace{f_X(x)}_{Marginal}=\int_{-\infty}^{+\infty}\underbrace{f_{X,Y}(x,y)}_{Joint}dy=\int_{-\infty}^{+\infty}\underbrace{f_{X|Y}(x|t)}_{Conditional}\underbrace{f_Y(t)}_{Marginal}dt$$

### 独立性：比较条件概率和边缘概率
如果连续随机变量$X$和$Y$独立，下式成立
$$f_{X|Y}(x|y)=f_X(x)$$
图 20 所示为 $X$ 和 $Y$ 独立，条件概率密度函数 $f_{X|Y}(x|y)$ 和边缘概率密度函数 $f_X(x)$ 之间关系。可以发现条件概率 $f_{X|Y}(x|y)$ 的曲线和 $Y$ 的取值无关。条件概率 $f_{X|Y}(x|y)$ 的曲线形状和边缘概率 $f_X(x)$ 完全一致。这和图 16 完全不同。
![](/img/Pasted%20image%2020230908092055.png ':size=70%')
![](/img/Pasted%20image%2020230908094846.png ':size=70%')

### 独立：联合概率
对于两个连续随机变量 $X$ 和 $Y$，如果两者独立，则联合概率密度函数 $f_{X,Y}(x,y)$ 为边缘概率密度 函数 $f_X(x)$ 和 $f_Y(y)$ 的乘积：
$$f_{X,Y}(x,y)=f_X(x)f_Y(y)$$
图 22 所示为连续随机变量 $X$ 和 $Y$ 独立，联合概率 $f_{X,Y}(x,y)$ 曲面。图 23 所示为联合概率 $f_{X,Y}(x,y)$ 平面等高线。
![](/img/Pasted%20image%2020230908092421.png ':size=70%')
![](/img/Pasted%20image%2020230908092431.png ':size=70%')

## 以鸢尾花数据为例：不考虑分类标签
本章以下两节还是用鸢尾花数据集花萼长度 ($X_1$)、花萼宽度 ($X_2$)、分类标签 ($Y$) 为例，讲解本章前文介绍连续随机变量主要知识点。图 24 所示为不考虑分类时，鸢尾花样本数据花萼长度、花萼宽度散点图。
![](/img/Pasted%20image%2020230908092710.png ':size=70%')
![](/img/Pasted%20image%2020230908092729.png ':size=70%')

### 概率密度估计—>联合概率密度函数
基于**高斯核密度估计** (kernel density estimation, KDE)，可以得到如图 25 所示联合概率密度 函数 $f_{X_1,X_2}(x_1,x_2)$。暖色系对应较大的概率密度值，也就是说鸢尾花样本分布更为密集。

核密度估计的基本思想是，通过在每个数据点处放置一个核函数 (如高斯核函数)，以此来估计概率密度函数。这样，在整个数据集上使用核函数后，我们可以获得一条连续的概率密度曲线，该曲线可以用来估计各种统计量，如均值和方差。

再次强调，图 25 仅仅代表 $f_{X_1,X_2}(x_1,x_2)$ 的一种估计。即便采用相同的 KDE，使用不同的核函数、改变算法参数会导致 $f_{X_1,X_2}(x_1,x_2)$ 曲面形状变化。

![](/img/Pasted%20image%2020230908093028.png ':size=70%')

### 联合概率密度函数$f_{X_1,X_2}(x_1,x_2)$的剖面线
$f_{X_1,X_2}(x_1, x_2)$ 本质上是个二元函数。

如图 26 所示，当固定 $x_1$ 取值时，$f_{X_1,X_2}(x_1 = c, x_2)$ 代表一条曲线。将一系列类似曲线投影到竖直平面得到图 26 (b)。图 26 (b)，这些直线和整个水平轴围成的面积就是边缘概率 $f_{X_1}(x_1 = c)$。而计算面积的数学工具就是“偏积分”。

![](/img/Pasted%20image%2020230908093546.png ':size=70%')

图 27 所示为固定 $x_2$ 时，概率密度函数$f_{X_1,X_2}(x_1,x_2)$ 随 $x_1$ 变化。图 26 (b) 中直线和整个水平轴围成的面积对应边缘概率 $f_{X_2}(x_2 = c)$。
![](/img/Pasted%20image%2020230908093919.png ':size=70%')

### 花萼长度边缘 PDF $f_{X_1}(x_1)$：偏积分
图28 所示为求解花萼长度边缘概率密度函数 $f_{X_1}(x_1)$ 的过程：
$$\underbrace{f_{X_1}(x_1)}_{Marginal}=\int\limits_{x_2}\underbrace{f_{X_1,X_2}(x_1,x_2)}_{Joint}dx_2$$
图 28 中彩色阴影面积对应边缘概率，即 $f_{X_1}(x_1)$ 曲线特定一点的高度。再次强调，$f_{X_1}(x_1)$ 本身也是概率密度，不是概率值。$f_{X_1}(x_1)$ 再积分可以得到概率。 

如图 28 (b) 所示，$f_{X_1}(x_1)$ 曲线和整个横轴围成图形的面积为 $1$。大家可以试着用数值积分计算 期望值 $E(X_1)$。

![](/img/Pasted%20image%2020230908094413.png ':size=70%')
![](/img/Pasted%20image%2020230908094818.png ':size=70%')

### 联合 PDF vs 边缘 PDF
图30 所示为联合 PDF 和边缘 PDF 之间关系。图中联合概率密度函数 $f_{X_1,X_2}(x_1,x_2)$ 采用高斯 KDE 估计得到。图 30 中的 $f_{X_1,X_2}(x_1,x_2)$ 比较精准地捕捉到了鸢尾花样本数据的分布特征。
![](/img/Pasted%20image%2020230908095910.png ':size=70%')

### 假设独立
如果假设 $X_1$ 和 $X_2$ 独立，联合概率密度 $f_{X_1,X_2}(x_1,x_2)$ 可通过下式计算得到：
$$f_{X_1,X_2}(x_1,x_2)=f_{X_1}(x_1)\cdot f_{X_2}(x_2)$$
图 31 所示为假设 $X_1$ 和 $X_2$ 独立时 $f_{X_1,X_2}(x_1,x_2)$ 的平面等高线和边缘 PDF 之间关系。

比较鸢尾花样本数据分布和假设 $X_1$ 和 $X_2$ 独立时估算得到的 $f_{X_1,X_2}(x_1,x_2)$ 等高线，很遗憾地发现图 31 这个联合概率密度函数 $f_{X_1,X_2}(x_1,x_2)$ 没有合理反映样本数据分布，尽管图 30 和图 31 边缘概率完全一致。
![](/img/Pasted%20image%2020230908100204.png ':size=70%')

### 给定花萼长度，花萼宽度的条件 PDF $f_{X_2 | X_1}(x_2 | x_1)$
如图 32 所示，利用贝叶斯定理，条件概率密度 $f_{X_2 | X_1}(x_2 | x_1)$ 可以通过下式计算：
$$\underbrace{f_{X_2|X_1}(x_2|x_1)}_{Conditional}=\dfrac{\overbrace{f_{X_1,X_2}(x_1,x_2)}^{Joint}}{\underbrace{f_{X_1}(x_1)}_{Marginal}}$$
注意，上式中 $f_{X_1}(x_1) > 0$。上式分母中的边缘概率 $f_{X_1}(x_1)$ 起到归一化作用。

如图 32 (b) 所示，经过归一化的条件概率曲线围成的面积变为 $1$，即：
$$\int\limits_{x_2}\underbrace{f_{X_2|X_1}(x_2|x_1)}_{Conditional}dx_2=\int\limits_{x_2}\dfrac{\overbrace{f_{X_1,X_2}(x_1,x_2)}^{Joint}}{\underbrace{f_{X_1}(x_1)}_{Marginal}}dx_2=\dfrac{\int\limits_{x_2}f_{X_1,X_2}(x_1,x_2)dx_2}{f_{X_1}(x_1)}=\dfrac{f_{X_1}(x_1)}{f_{X_1}(x_1)}=1$$

将不同位置的条件 PDF $f_{X_2 | X_1}(x_2 | x_1)$ 曲线投影到平面得到图 33。图 33 (b) 中每条曲线和横轴围成面积都是 1。请仔细比较图 26 和图 33。此外，$f_{X_2 | X_1}(x_2 | x_1)$ 本身也是一个二元函数。图 34 所 示为 $f_{X_2 | X_1}(x_2 | x_1)$ 三维等高线和平面等高线。

![](/img/Pasted%20image%2020230908101807.png ':size=70%')
![](/img/Pasted%20image%2020230908101816.png ':size=70%')
![](/img/Pasted%20image%2020230908101835.png ':size=70%')
![](/img/Pasted%20image%2020230908102423.png ':size=70%')
![](/img/Pasted%20image%2020230908102434.png ':size=70%')
![](/img/Pasted%20image%2020230908102445.png ':size=70%')

## 以鸢尾花数据为例：考虑分类标签
本节将以鸢尾花标签为条件讨论条件概率。图 38 所示为考虑分类标签的鸢尾花数据散点图。
![](/img/Pasted%20image%2020230908102510.png ':size=70%')
### 给定分类标签 $Y = C_1 (setosa)$
图 39 所示为给定分类标签 $Y = C_1 (setosa)$ 条件下，条件概率 $f_{X_1,X_2 | Y}(x_1, x_2 | y = C_1)$ 平面等高线 和条件边缘概率密度曲线。

$f_{X_1,X_2 | Y}(x_1, x_2 | y = C_1)$ 曲面和整个水平面围成体积为 $1$，也就是说：
$$\int\limits_{x_2}\int\limits_{x_1}\underbrace{f_{X_1,X_2|Y}(x_1,x_2|C_1)}_{\text{Conditional PDF}}dx_1dx_2=1$$
用 KDE 估算 $f_{X_1,X_2 | Y}(x_1, x_2 | y = C_1)$ 时，我们仅仅考虑标签为 $C_1$ 的数据。同理，估算条件边缘概率曲线 $f_{X_1 | Y}(x_1 | y = C_1)$、$f_{X_2 | Y}(x_2 | y = C_1)$ 时，我们也不考虑其他标签数据。

图39 中，$f_{X_1 | Y}(x_1 | y = C_1)$、$f_{X_2 | Y}(x_2 | y = C_1)$ 分别和 $x_1$、$x_2$ 围成的面积也是 $1$，即：
$$\begin{split}
\int\limits_{x_1}\underbrace{f_{X_1|Y}(x_1|C_1)}_{\text{Conditional PDF}}dx_1=1\\
\int\limits_{x_2}\underbrace{f_{X_2|Y}(x_2|C_1)}_{\text{Conditional PDF}}dx_2=1\\
\end{split}$$
![](/img/Pasted%20image%2020230908103242.png ':size=70%')

### 给定分类标签 $Y = C_2 (versicolor)$
图 40 所示为，给定分类标签 $Y = C_2 (versicolor)$，条件概率 $f_{X_1,X_2 | Y}(x_1, x_2 | y = C_2)$ 平面等高线和条件边缘概率密度曲线。
![](/img/Pasted%20image%2020230908103402.png ':size=70%')

### 给定分类标签 $Y = C_3 (virginica)$
图 41 所示为，给定分类标签 $Y = C3 (virginica)$，条件概率 $f_{X_1,X_2 | Y}(x_1, x_2 | y = C_3)$ 平面等高线和条件边缘概率密度曲线。
![](/img/Pasted%20image%2020230908103541.png ':size=70%')

### 全概率定理：穷举法
如图 42 所示，利用全概率定理，三幅条件概率等高线叠加可以得到联合概率密度，即：
$$\begin{split}
f_{X_1,X_2}(x_1,x_2)
&=f_{X_1,X_2|Y}(x_1,x_2|y=C_1)p_Y(C_1)\\
&+f_{X_1,X_2|Y}(x_1,x_2|y=C_2)p_Y(C_2)\\
&+f_{X_1,X_2|Y}(x_1,x_2|y=C_3)p_Y(C_3)\\
\end{split}$$

![](/img/Pasted%20image%2020230908114111.png ':size=70%')

### 给定$X_1$和$X_2$，$Y$的条件概率：后验概率
根据贝叶斯定理，当 $f_{X_1,X_2}(x_1,x_2) > 0$ 时，**后验** (posterior) PDF $f_{Y|X_1,X_2}(C_k | x_1,x_2)$ 可以根据下式计算得到：
$$\overbrace{f_{Y|X_1,X_2}(C_k | x_1,x_2)}^{Posterior}=\dfrac{\overbrace{f_{X_1,X_2,Y}(x_1,x_2,C_k)}^{Joint}}{\underbrace{f_{X_1,X_2}(x_1,x_2)}_{Evidence}}$$
从分类角度来看，这相当于已知某个样本鸢尾花花萼长度和花萼宽度，该样本对应不同分类的概率。

### 假设条件独立
如图 43 所示，如果假设条件独立，$f_{X_1,X_2 | Y}(x_1, x_2 | y = C_1)$ 可以通过下式计算得到：
$$\underbrace{f_{X_1,X_2|Y}(x_1,x_2|y=C_1)}_{\text{Conditional joint}}=\underbrace{f_{X_1|Y}(x_1|y=C_1)}_{\text{Conditional marginal}}\cdot\underbrace{f_{X_2|Y}(x_2|y=C_1)}_{\text{Conditional marginal}}$$
同理我们可以计算得到 $f_{X_1,X_2|Y}(x_1,x_2|y=C_2)$、$f_{X_1,X_2|Y}(x_1,x_2|y=C_3)$，具体如图 44、图 45 所示。
![](/img/Pasted%20image%2020230908115018.png ':size=70%')
![](/img/Pasted%20image%2020230908115042.png ':size=70%')
![](/img/Pasted%20image%2020230908115059.png ':size=70%')

如图 46 所示，并利用全概率定理，我们也可以估算 $f_{X_1,X_2}(x_1, x_2)$：
$$\begin{split}
f_{X_1,X_2}(x_1, x_2)
&=f_{X_1,X_2|Y}(x_1,x_2|y=C_1)p_Y(C_1)\\
&+f_{X_1,X_2|Y}(x_1,x_2|y=C_2)p_Y(C_2)\\
&+f_{X_1,X_2|Y}(x_1,x_2|y=C_3)p_Y(C_3)\\
&=f_{X_1|Y}(x_1|y=C_1)
f_{X_2|Y}(x_2|y=C_1)
p_Y(C_1)+\\
&+f_{X_1|Y}(x_1|y=C_2)
f_{X_2|Y}(x_2|y=C_2)
p_Y(C_2)+\\
&+f_{X_1|Y}(x_1|y=C_3)
f_{X_2|Y}(x_2|y=C_3)
p_Y(C_3)\\
\end{split}$$
这是**朴素贝叶斯分类器** (Naive Bayes classifier) 的重要技术细节之一。

![](/img/Pasted%20image%2020230908115610.png ':size=70%')

最后，比较离散和连续随机变量
![](/img/Pasted%20image%2020230908115830.png ':size=70%')
![](/img/Pasted%20image%2020230908115754.png ':size=70%')








