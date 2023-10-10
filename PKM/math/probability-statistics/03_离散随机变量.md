# 离散随机变量
## 随机：天地不仁，以万物为刍狗

### 随机试验
**随机试验** (random experiment) 是指在相同条件下对某个随机现象进行的大量重复观测。随机试验需要满足如下条件：
- **可重复**，在相同条件下试验可以重复进行；
- **样本空间明确**，每次试验的可能结果不止一个，并且能事先明确试验的所有可能结果；
- **单次试验结果不确定**，进行一次试验之前不能确定哪一个结果会出现，但必然出现样本空间 中的一个。

### 两种随机变量：离散、连续
**随机变量** (random variable) 是指在一次试验中可能出现不同取值的量，其取值由随机事件的结果决定。随机变量可以看做一个函数，它将样本数值赋给试验结果。换句话说，它是试验样本空间到实数集合的函数。比如上一章为了方便表达“抛三枚色子试验”中三枚色子各自点数，我们定义了 X1、X2、X3，它们都是随机变量。 

随机变量分为两种——**离散** (discrete)、**连续** (continuous)。 

如果随机变量的所有取值能够一一列举出来，可以是为有限个或可数无穷个，这种随机变量被称作**离散随机变量** (discrete random variable)。 比如，投一枚硬币结果正面为 1、反面为 0。掷一枚色子得到的点数为 1、2、3、4、5、6中 的一个值。再比如，鸢尾花的标签有三种——setosa (C1)、versicolour (C2)、virginica (C3)。

与之相对的是，**连续随机变量** (continuous random variable)。连续随机变量取值可能对应全部实数，或者数轴上某一区间。比如，温度、人的身高体重都是连续随机变量。再比如，鸢尾花花萼长度、花萼宽度、花瓣长度、花瓣宽度也都可以视作连续随机变量。

### 两种概率分布函数
研究随机变量取值的统计规律是概率论重要目的之一。概率分布函数是对统计规律的简化和抽象。图 2 比较两种概率分布函数——**概率质量函数 PMF**、**概率密度函数 PDF**。

白话来说，概率质量函数 PMF、概率密度函数 PDF 就是两种对样本空间概率为 1“切片、切 块”、“切丝、切条”的不同方法。
![](/upload/Pasted%20image%2020230905171039.png ':size=70%')

### 概率质量函数PMF
**概率质量函数** (probability mass function, PMF) 是离散随机变量在特定取值上的 概率。

$p_X(x)$ 本身就是“概率值”，因此计算离散随机变量 $X$ 取不同值时的概率，使用求和运算。 

$p_X(x)$ 对应的数学运算符是 $Σ$。

### 随机变量的函数
$X$ 为一个随机变量，对 $X$ 进行函数变换，可以得到其他的随机变量 $Y$：
$$Y=h(X)$$
特别地，如果 $h()$ 为线性函数，从 $X$ 到 $Y$ 进行的是线性变换，比如：
$$Y=h(X)=aX+b$$
![](/upload/Pasted%20image%2020230905171439.png ':size=70%')
![](/upload/Pasted%20image%2020230905171458.png ':size=70%')
![](/upload/Pasted%20image%2020230905171518.png ':size=70%')
![](/upload/Pasted%20image%2020230905171532.png ':size=70%')

### 归一律
一元离散随机变量 $X$ 的概率质量函数 $p_X(x)$ 有如下重要性质：
$$\sum_xp_X(x)=1,\quad0\le p_X(x)\le 1$$
上式实际上就是“穷举法”，即遍历所有 $X$ 取值，将它们的概率值求和，结果为 $1$。“穷举法”也 叫归一律。

### 概率密度函数PDF
与 PMF 相对的是**概率密度函数** (probability density function, PDF)。PDF 对应连续随机变量， 本书用小写斜体字母 $f$ 表达 PDF，比如连续随机变量 $X$ 的概率密度函数记做 $f_X(x)$。

当连续随机变量取不同值时，概率密度函数 $f_X(x)$ 用积分方式得到概率值。因此，$f_X(x)$ 对应的 数学运算符是积分符号 $∫$。

注意，联合概率密度函数 $f_{X_1,X_2,X_3}(x_1,x_2,x_3)$ “偏积分”结果还是概率密度。$f_{X_1,X_2,X_3}(x_1,x_2,x_3)$ 三 重积分结果才是概率值。

值得反复强调的是，PMF 本身就是概率，对应的数学工具为 $Σ$ 求和。PDF 积分后才可能是概率，对应的数学工具为 $∫$ 积分。

一元连续随机变量 $X$ 的概率密度函数 $f_X(x)$ 也有如下重要性质：
$$\int_{-\infty}^{+\infty}f_X(x)\mathrm{d}x=1,\quad f_X(x)\ge0$$
上式也相当于是“穷举法”。

概率质量函数 PMF、概率密度函数 PDF 是特殊的函数。特殊之处在于它们的输入为随机变量 的取值，输出为概率质量、概率密度。但是，本质上，它们又都是函数。所以，我们可以把函数 的分析工具用在概率质量函数 PMF、概率密度函数 PDF 上。

## 期望值：随机变量的可能取值加权平均

### 期望值
离散随机变量 $X$ 有 $n$ 个取值 ${x^{(1)}, x^{(2)}, …, x^{(n)}}$，$X$ 的期望 (expectation)，也叫期望值 (expected value)，$E(X)$ 为：
$$\begin{split}
\mathop{E(X)}_{Scalar}
&=\mu_X
=x^{(1)}p_X\left(x^{(1)}\right)
+x^{(2)}p_X\left(x^{(2)}\right)
+\cdots
+x^{(n)}p_X\left(x^{(n)}\right)\\
&=\sum_{i=1}^nx^{(i)}\cdot\underbrace{p_X(x^{(i)})}_{Weight}
\end{split}$$
上式相当于加权平均数，边缘 PMF $p_X(x)$ 代表权重。

运算符 $E()$ 把随机变量一系列取值转化成了一个标量数值，这相当于降维。如图 7 所示，从矩 阵乘法角度，计算期望值 $E(X)$ 相当于将 $X$ 这个维度折叠。
![](/upload/Pasted%20image%2020230905172557.png ':size=70%')

为了方便，常把上式简写作：
$$E(X)=\sum_xx\cdot p_X(x)$$
$\sum\limits_x(\cdot)$代表对 $x$ 的遍历求和，也就是穷举。求加权平均值时，权重之和为 $1$，也就是说边缘 PMF $p_X(x)$ 满足 $\sum\limits_xp_X(x)=1$ 。特别是对于多元随机变量，我们也经常把期望值 (均值) 叫做**质心** (centroid)。

以下几个有关期望的性质：
$$\begin{split}
&E(aX)=aE(X)\\
&E(X+Y)=E(X)+E(Y)\\
\end{split}$$
如果 $X$ 和 $Y$ 独立：
$$E(XY)=E(X)E(Y)$$
此外，
$$\begin{split}
E(\sum_{i=1}^na_iX_i)
&=\sum_{i=1}^na_iE(X_i)\\
&=\begin{bmatrix}a_1 & a_2 & \cdots & a_n\end{bmatrix}
\begin{bmatrix}
E(X_1)\\
E(X_2)\\
\vdots\\
E(X_n)\\
\end{bmatrix}
\end{split}$$

## 方差：随机变量离期望距离平方的平均值
### 方差
随机变量 $X$ 另外一个重要特征是**方差** (variance)，记做 $var(X)$。对于离散随机变量 $X$，方差用 来度量 $X$ 和数学期望 $E(X)$ 之间的偏离程度。具体定义为：
$$var(X)
=\overbrace{E\left[\left(\underbrace{X-E(X)}_{Deviation}\right)^2\right]}^{Expectation}
=\sum_x\left(\underbrace{x-E(X)}_{Demean}\right)^2\cdot\underbrace{p_X(x)}_{Weight}$$
上式中$x-E(X)$ 代表以期望值 $E(X)$ 为参照，样本点 $x$ 的偏离量。

如图 9 所示， $X−E(X)$ 代表**去均值** (demean)，也叫**中心化** (centralize)。
![](/upload/Pasted%20image%2020230905173804.png ':size=70%')

观察上式，容易发现方差实际上是 $(X-E(X))^2$ 的期望值。上式就是求 $(x-E(X))^2$ 的加权平均数，权重为 $p_X(x)$。

从几何角度，$(X-E(X))^2$ 代表以 $|X-E(X)|$ 为边长的正方形的面积。而对于离散随机变量，$p_X(x)$ 就是权重，体现不同样本重要性。

### 技巧：方差计算
方差有个简便算法：
$$var(X)=\underbrace{E(X^2)}_{\text{Expectation of }X^2}-\underbrace{E(X)^2}_{\text{Square of }E(X)}$$
其中，$E(X^2)$ 为：
$$\underbrace{E(X^2)}_{\text{Expectation of }X^2}=\sum_{x}x^2\cdot\underbrace{p_X(x)}_{Weight}$$

### 几何意义
方差度量离散程度，本质上来说是“自己”和“自己”比较的产物。前一个“自己”是 $X$ 每个样本， 后一个“自己”是代表 $X$ 整体位置的期望值 $E(X)$。 

如图 10 所示，方差 $var(X)$ 代表样本以**质心** (centroid) 为基准的离散程度。
![](/upload/Pasted%20image%2020230905174511.png ':size=70%')
计算方差 $var(X)$ 有 $E(X^2)$ 和 $-E(X)^2$ 两部分。

$E(X^2)$ 度量 $X$ 样本以**原点** (origin) 为基准的离散程度。

$E(X)^2$ 则代表 $X$ 整体，即 $E(X)$，相对于原点的离散程度。$-E(X)^2$中的“负号”代表将基准从原点移到质心。

特别地，当 $X$ 的质心位于原点，即 $E(X) = 0$ 时，$var(X)$ 为：
$$var(X)=E(X^2)$$

### 标准差
**标准差** (standard deviation) 是方差的平方根：
$$std(X)=\sigma_X=\sqrt{var(X)}$$
方差既然可以用来度量“离散程度”，为什么我们还需要标准差？ 

简单来说，标准差 $σ_X$、期望值 $E(X)$、随机变量 $X$ 为同一量纲。比如，鸢尾花花萼长度 $X$ 的单 位是 cm，期望值 $E(X)$ 的单位也是 $cm$，而 $σ_X$ 的单位也对应是 $cm$。但是，$var(X)$ 的量纲是 $cm^2$。

### 需要注意的性质
请注意以下方差性质：
$$\begin{split}
&var(a)=0\\
&var(X+a)=var(X)\\
&var(aX)=a^2var(X)\\
&var(aX+b)=a^2var(X)\\
&var(X+Y)=var(X)+var(Y)+2cov(X,Y)\\
\end{split}$$
其中 $cov(X,Y)$ 为随机变量 $X$ 和 $Y$ 的协方差

请注意以下标准差性质：
$$\begin{split}
&\sigma(a)=0\\
&\sigma(X+a)=\sigma(X)\\
&\sigma(aX)=|a|\sigma(X)\\
&\sigma(aX+b)=|a|\sigma(X)\\
&\sigma(X+Y)=\sqrt{\sigma^2(X)+\sigma^2(Y)+2\rho(X,Y)\sigma(X)\sigma(Y)}\\
\end{split}$$

### 汇总
折叠、总结、汇总、降维、压扁 … 本章及后文会用这些字眼形容期望值、方差、标准差。这是因为，计算期望值、方差、标准差时，我们不再关注随机变量样本具体取值，**而是在乎某种方式的汇总** (aggregation)。 

期望值、方差、标准差将“数组”转化成特定标量值。因此，这个特定维度相当于被折叠、总结、汇总、降维、压扁 … 对于多元随机变量，我们可以选择某个、某几个维度上完成汇总计算。

如果汇总的形式为期望，它相当于找到随机变量整体的“位置”。如果汇总的形式为方差、标准差，两者都度量随机变量“离散”程度。

其他常用的汇总形式还包括：**计数** (count)、**求和** (sum)、**四分位** (quartile)、**百分位**(percentile)、**最大值** (maximum)、**最小值** (minimum)、**中位数** (median)、**众数** (mode)、**偏度**、**峰度**等等。

## 累积分布函数CDF：累加
对于离散随机变量，**累积分布函数** (Cumulative Distribution Function, CDF) 对应概率质量函数 的求和。

对于离散随机变量 **X**，累积分布函数 **F_X(x)** 的定义为：
$$F_X(x)=\Pr(X\le x)=\sum_{t\le x}p_X(t)$$
上式相当于累加概念，累加从 $X$ 最小样本值开始并截止于 $X = x$。 
离散随机变量 $X$ 的取值范围为 $a\lt x\le b$ 时，对应的概率可以利用 CDF 计算：
$$\Pr(a\lt x\le b)=F_X(b)-F_X(a)$$
![](/upload/Pasted%20image%2020230906113154.png ':size=70%')
注意，对于离散随机变量，**区间端点的开闭影响结果**。
$$\Pr(1\lt X\le 3)=\frac{1}{3}\quad \Pr(1\le X\le 3)=\frac{1}{2}$$
对于连续随机变量，就没有区间端点的麻烦了。

## 二元离散随机变量
假设同一个试验中，有两个离散随机变量 $X$ 和 $Y$。二元随机变量 $(X, Y)$ 概率取值可以用联合 概率质量函数 (joint Probability Mass Function, joint PMF) $p_{X,Y}(x, y)$ 刻画。

概率质量函数 $p_{X,Y}(x, y)$ 代表事件 ${X = x, Y = y}$ 发生的联合概率：
$$\underbrace{p_{X,Y}(x,y)}_{Joint}=\Pr(X=x,Y=y)=\Pr(X=x\cap Y=y)$$
再次强调，对于二元离散随机变量，$p_{X,Y}(x, y)$ 本身就是概率值。

图 12 所示为二元离散随机变量 $(X, Y)$ 的样本空间 $Ω$，空间中共有 $81$ 个点。从函数角度来看， $p_{X,Y}(x, y)$ 是个二元函数。因此，我们可以用二元函数的分析方法来讨论 $p_{X,Y}(x, y)$。
![](/upload/Pasted%20image%2020230906113558.png ':size=70%')

### 取值
图3 所示为二元联合概率质量函数 $p_{X,Y}(x, y)$ 的取值表格。图 13 同时用热图来可视化 $p_{X,Y}(x, y)$。

二元联合概率质量函数 $p_{X,Y}(x, y)$ 也有一条重要的性质：
$$\sum_x\sum_y\underbrace{p_{X,Y}(x,y)}_{Joint}=\sum_y\sum_x\underbrace{p_{X,Y}(x,y)}_{Joint}=1,\quad 0\le p_{X,Y}(x,y)\le 1$$
也就是说，图 13 这幅热图中所有数值 (概率，概率质量) 求和的结果为 1，和求和顺序无关。
![](/upload/Pasted%20image%2020230906113816.png ':size=70%')

### 火柴梗图
二元联合概率质量函数 $p_{X,Y}(x, y)$ 长成什么样子呢？

火柴梗图最适合可视化概率质量函数，如图 14 所示。 

注意，为了展示火柴梗图分别沿 $X$、$Y$ 方向变化趋势，图 14 将火柴梗散点连线。一般情况，火柴梗图不存在连线。
![](/upload/Pasted%20image%2020230906113923.png ':size=70%')
![](/upload/Pasted%20image%2020230906113934.png ':size=70%')

## 协方差、相关性系数
### 协方差
二元离散随机变量 $(X, Y)$ 的协方差定义为：
$$cov(X,Y)=E\left(\left(X-E(X)\right)\left(Y-E(Y)\right)\right)$$
如果 $(X, Y)$ 的概率质量函数为 $p_{X,Y}(x, y)$，$X$ 的取值为 $x^{(i)}(i = 1, 2, …, n)$，$Y$ 的取值为 $y^{(j)} (j = 1, 2, …, m)$。上式可以展开写成：
$$\begin{split}cov(X,Y)
&=E((X-E(X))(Y-E(Y)))\\
&=\sum_{i=1}^n\sum_{j=1}^mp_{X,Y}(x^{(i)},y^{(j)})\left(x^{(i)}-E(X)\right)\left(y^{(j)}-E(Y)\right)
\end{split}$$
其中，
$$E(X)=\sum_xx\cdot p_X(x),\quad E(Y)=\sum_yy\cdot p_Y(y)$$
上式常简写为：
$$cov(X,Y)=\sum_x\sum_yp_{X,Y}(x,y)(x-E(X))(y-E(Y))$$
类似方差，协方差运算也有如下技巧：
$$\begin{split}
cov(X,Y)&=E(XY)-E(X)E(Y)\\
&=\sum_x\sum_yx\cdot yp_{X,Y}(x,y)-\left(\sum_xx\cdot p_X(x)\right)\cdot\left(\sum_yy\cdot p_Y(y)\right)
\end{split}$$

### 相关性
$(X,Y)$相关性的定义为：
$$\rho_{X,Y}=\frac{cov(X,Y)}{\sigma_X\sigma_Y}$$
展开得到：
$$\rho_{X,Y}=\frac{E(XY)-E(X)E(Y)}{\sqrt{E(X^2)-E(X)^2}\sqrt{E(Y^2)-E(Y)^2}}$$
相关性的取值范围 $[−1, 1]$。相对协方差，相关性更适合横向比较。

### 协方差性质：
请注意一下协方差性质
$$\begin{split}
cov(X,a)&=0\\
cov(X,X)&=var(X)\\
cov(X,Y)&=cov(Y,X)\\
cov(aX,bY)&=abcov(X,Y)\\
cov(X+a,Y+b)&=cov(X,Y)\\
cov(aX+bY,Z)&=acov(X,Z)+bcov(Y,Z)\\
cov(aX+bY,cW+dV)&=accov(X,W)+adcov(X,V)+bccov(Y,W)+bdcov(Y,V)\\
\end{split}$$

此外，方差和协方差的关系：
$$var(\sum_{i=1}^na_iX_i)=\sum_ia_i^2var(X_i)+2\sum_{i,j:i<j}a_ia_jcov(X_i,X_j)=\sum_{i,j}a_ia_jcov(X_1,X_2$$
特别地，当 $n = 2$ 时，上式可以写成：
$$var\left(\sum_{i=1}^na_iX_i\right)=\sum_ia_i^2var(X_i)+2\sum_{i,j:i<j}a_ia_jcov(X_i,X_j)=\sum_{i,j}a_ia_jcov(X_i,X_j)$$
上式可以写成如下矩阵乘法运算：
$$var(\sum_{i=1}^na_iX_i)
=\underbrace{\begin{bmatrix}
a_1\\
a_2\\
\vdots\\
a_n
\end{bmatrix}^\mathrm{T}}_{\vec{a}}
\underbrace{\begin{bmatrix}
cov(X_1,X_1) & cov(X_1,X_2) & \cdots & cov(X_1,X_n)\\
cov(X_2,X_1) & cov(X_2,X_2) & \cdots & cov(X_2,X_n)\\
\vdots & \vdots & \ddots & \vdots\\
cov(X_n,X_1) & cov(X_n,X_2) & \cdots & cov(X_n,X_n)\\
\end{bmatrix}}_{\Sigma}
\underbrace{\begin{bmatrix}
a_1\\
a_2\\
\vdots\\
a_n
\end{bmatrix}}_{\vec{a}}
=\vec{a}^T\Sigma\vec{a}$$

### 几何视角
对于如下等式：
$$var(X+Y)=var(X)+var(Y)+2cov(X,Y)$$
即，
$$\sigma_{X+Y}^2=\sigma_X^2+\sigma_Y^2+2\rho_{X,Y}\sigma_X\sigma_Y$$
$σ_X、σ_Y、σ_{X + Y}$ 相当于三角形的三个边，,$\rho_{X,Y}$ 相当 $σ_X、σ_Y$ 于夹角的余弦值。如图 15 所示，当,$\rho_{X,Y}$ 取不同值时，三角形呈现不同的形态。

特别地，如果, $\rho_{X,Y}=0$ ，三角形为直角三角形，满足：
$$\sigma_{X+Y}^2=\sigma_X^2+\sigma_Y^2$$
![](/upload/Pasted%20image%2020230906191805.png ':size=70%')

## 边缘概率：偏求和，相当于降维
**边缘概率** (marginal probability) 是某个事件发生的概率，而与其它事件无关。对于离散随机变 量来说，利用全概率定理，也就是穷举法，我们可以把联合概率结果中不需要的那些事件全部合 并。合并的过程叫做**边缘化** (marginalization)。

### 边缘概率$p_X(x)$
根据全概率公式，对于二元联合概率质量函数 $p_{X,Y}(x, y)$，求解边缘概率 $p_X(x)$ 相当于利用“偏 求和”消去 $y$：
$$\underbrace{p_X(x)}_{Marginal}=\sum_y\underbrace{p_{X,Y}(x,y)}_{Joint}$$
也就是说，在 $X = x$ 取值条件下，$p_{X,Y}(x, y)$ 对所有 $y$ 的求和。 

从函数角度来看，$p_{X,Y}(x, y)$ 是个二元函数，$p_X(x)$ 是个一元函数。 

从矩阵运算角度来看，$p_{X,Y}(x, y)$ 代表矩阵，矩阵沿 $Y$ 方向求和，折叠得到行向量 $p_X(x)$。行向 量 $p_X(x)$ 进一步求和结果为标量 $1$，对应样本空间概率。反向来看，概率 $1$ 沿 $X$ 和 $Y$ 展开，相当于 “切片、切丝”。

### 几何视角：叠加
显然，边缘分布 $p_X(x)$ 和 $p_Y(y)$ 本身也是概率质量函数。从图像上来看，$p_X(x)$ 相当于 $p_{X,Y}(x, y)$ 中 $y$ 在取不同值时对应的火柴梗图叠加得到，具体如图 17 所示。同理，图 18 所示为边缘分布 $p_Y(y)$ 求解过程。
![](/upload/Pasted%20image%2020230906192217.png ':size=70%')
![](/upload/Pasted%20image%2020230906192229.png ':size=70%')

## 条件概率：引入贝叶斯定理
### 联合概率—>条件概率
利用贝叶斯定理，条件概率 $p_{X|Y}(x|y)$ 可以用联合概率 $p_{X,Y}(x,y)$ 除以边缘概率 $p_Y(y)$ 得到：
$$\underbrace{p_{X|Y}(x|y)}_{Conditional}=\frac{\overbrace{p_{X,Y}(x,y)}^{Joint}}{\underbrace{p_Y(y)}_{Marginal}}$$
从函数角度来看，$p_{X|Y}(x|y)$ 本质上也是个二元函数。首先，$p{X|Y}(x | y)$ 显然随着 $X = x$ 变化。虽 然 $Y = y$ 为条件，但是这个条件也可以变动。$Y = y$ 变动就会导致概率质量函数 $p_{X|Y}(x | y)$ 变化。 

从矩阵运算角度来看，$p_{X,Y}(x,y)$ 相当于矩阵，$p_Y(y)$ 相当于列向量。两者相除用到广播原则 (broadcasting)。得到的条件概率 $p_{X|Y}(x|y)$ 也是个矩阵，形状和 $p_{X,Y}(x,y)$ 一 致。 

$p_{X|Y}(x|y)$ 对 $x$ 求和等于 $1$：
$$\sum_xp_{X|Y}(x|y)=1$$
也就是说，$p_{X|Y}(x|y)$ 矩阵的每一行求和结果为 $1$。也就是说，每一行代表一个不同的“样本空 间”。

注意，上式的结果实际上是一维数组，$\sum\limits_x()$ 完成 $X$ 方向压缩，但是 $Y$ 这个维度没有被压缩。 换个视角来看，条件概率的“条件”就是“新的样本空间”，这个新的样本空间对应概率为 $1$。

### 举个例子
如图 19 所示，$Y = 2$ 时，边缘概率 $p_Y(Y = 2)$ 可以通过求和得到：
$$p_Y(2)=\sum_xp_{X,Y}(x,2)$$
$p_Y(2)$ 为一定值。给定 $Y = 2$ 作为条件时，条件概率 $p_{X|Y}(x|2)$ 通过下式得到：
$$\underbrace{p_{X|Y}(x|2)}_{Conditional}
=\frac{\overbrace{p_{X,Y}(x,2)}^{Joint}}{\underbrace{p_Y(2)}_{Marginal}}$$
观察图 19，发现 $p_{X,Y}(x,2)$ 到 $p_{X|Y}(x|2)$ 相当于曲线缩放过程。
![](/upload/Pasted%20image%2020230906193032.png ':size=70%')

进一步，条件概率 $p_{X|Y}(x|2)$ 对 $x$ 求和得到 $1$：
$$\sum_xp_{X|Y}(x|2)=\dfrac{\sum\limits_xp_{X,Y}(x,2)}{p_Y(2)}=\dfrac{p_Y(2)}{p_Y(2)}=1$$
$p_{X,Y}(x,2)$ 到 $p_{X|Y}(x|2)$ 是一个归一化 (normalization) 过程。也就是说，上式分母中的 $p_Y(y)$ 是一个归一化系数。这样，满足了归一化条件，$p_{X|Y}(x|2)$ 就“摇身一变”成了概率质量函数。

引入贝叶斯定理，边缘概率 $p_X(x)$ 相当于是条件概率的加权平均：
$$\underbrace{p_X(x)}_{Marginal}=\sum_y\underbrace{p_{X,Y}(x,y)}_{Joint}=\sum_y\underbrace{p_{X|Y}(x|y)}_{Conditional}\underbrace{p_Y(y)}_{Marginal}$$

### 条件概率—>联合概率
相反，条件概率 $p_{X|Y}(x|y)$ 到联合概率 $p_{X,Y}(x,y)$ 相当于，以边缘概率 $p_Y(y)$ 作为系数缩放 $p_{X|Y}(x|y)$ 的过程：
$$\underbrace{p_{X,Y}(x,y)}_{Joint}=\underbrace{p_{X|Y}(x|y)}_{Conditional}\underbrace{p_Y(y)}_{Marginal}$$

## 独立性：条件概率等于边缘概率
### 独立
如果两个离散变量 $X$ 和 $Y$ 独立，条件概率 $p_{X|Y}(x|y)$ 等于边缘概率 $p_X(x)$，下式成立：
$$\underbrace{p_{X|Y}(x|y)}_{Conditional}=\underbrace{p_X(x)}_{Marginal}$$
如图 21 所示，$X$ 和 $Y$ 独立，不管 $y$ 取任何值 $(0 ~ 8)$，$p_X(x)$ 的形状和 $p_{X|Y}(x|y)$ 相同。
![](/upload/Pasted%20image%2020230906194632.png ':size=70%')

上式等价于下式：
$$\underbrace{p_{Y|X}(y|x)}_{Conditional}=\underbrace{p_Y(y)}_{Marginal}$$

同理，如图 22 所示，$X$ 和 $Y$ 独立时，$p_Y(y)$ 的形状和 $p_{Y|X}(y|x)$ 相同。这恰恰说明，$X$ 的取值和 $Y$ 无关，也就是为什么条件概率 $p_{Y|X}(y|x)$ 的形状不受 $X = x$ 影响，都和 $pY(y)$ 相同。
![](/upload/Pasted%20image%2020230906194805.png ':size=70%')

### 独立：计算联合概率
另外一个角度，如果离散随机变量 $X$ 和 $Y$ 独立，联合概率 $p_{X,Y}(x,y)$ 等于 $p_Y(y)$ 和 $p_X(x)$ 两个边 缘概率质量函数 PMF 乘积：
$$\underbrace{p_{X,Y}(x,y)}_{Joint}=\underbrace{p_Y(y)}_{Marginal}+\underbrace{p_X(x)}_{Marginal}$$
从向量角度来看，把 $p_Y(y)$ 和 $p_X(x)$ 看成是两个向量，上式相当于 $p_Y(y)$ 和 $p_X(x)$ 的张量积。
![](/upload/Pasted%20image%2020230906195036.png ':size=70%')

### 不独立
图 24 给出另一个联合概率 $p_{X,Y}(x, y)$ 的图像。
![](/upload/Pasted%20image%2020230906195121.png ':size=70%')
如果 $X$ 和 $Y$ 不独立，如果 $p_Y(y) > 0$，条件概率 $p_{X|Y}(x|y)$ 公式如下：
$$\underbrace{p_{X|Y}(x|y)}_{Conditional}=\frac{\overbrace{p_{X,Y}(x,y)}^{Joint}}{\underbrace{p_Y(y)}_{Marginal}}=\frac{\overbrace{p_{X,Y}(x,y)}^{Joint}}{\sum\limits_xp_{X,Y}(x,y)}$$
如图 25 所示，当 $X$ 和 $Y$ 不独立，条件概率 $p_{X|Y}(x|y)$ 不同于边缘概率 $p_X(x)$。
![](/upload/Pasted%20image%2020230906195419.png ':size=70%')

## 再谈概率1：展开、折叠
### 偏求和：压扁
几何上，$p_{X_1,X_2,X_3}(x_1, x_2, x_3)$ 可以视作一个三维立方体。而偏求和是个降维过程，把立方体在不同维度上压扁。

如图 48 所示，$p_{X_1,X_2,X_3}(x_1, x_2, x_3)$ 在 $x_1$ 上偏求和，压扁得到 $p_{X_2,X_3}(x_2, x_3)$：
$$p_{X_2,X_3}(x_2,x_3)=\sum_{x_1}p_{X_1,X_2,X_3}(x_1,x_2,x_3)$$
如图 48 所示，$p_{X_2,X_3}(x_2, x_3)$ 代表一个二维平面，相当于一个矩阵。

而 $p_{X_2,X_3}(x_2, x_3)$ 进一步沿着 $x_2$ 折叠便得到边缘概率质量函数 $p_{X_3}(x_3)$：
$$p_{X_3}(x_3)=\sum_{x_2}p_{X_2,X_3}(x_2,x_3)=\sum_{x_2}\sum_{x_1}p_{X_1,X_2,X_3}(x_1,x_2,x_3)$$
而 $p_{X_3}(x_3)$ 相当于一个向量。

沿着哪个方向求和，就相当于完成了这个维度上数据的合并。这个维度因此便消失。
![](/upload/Pasted%20image%2020230906195910.png ':size=70%')
![](/upload/Pasted%20image%2020230906195925.png ':size=70%')

### 条件概率：切片
如图 50 所示，条件概率 $p_{X_1,X_2|X_3}(x_1, x_2|c)$ 相当于在 $X_3 = c$ 处切了一片，只考虑切片上的概率分 布情况，而不考虑整个立方体的概率分布。

也就是说，$X_3 = c$ 对应的切片是条件概率 $p_{X_1,X_2|X_3}(x_1, x_2 | c)$ 的样本空间。
![](/upload/Pasted%20image%2020230906200206.png ':size=70%')

计算条件概率时，首先将切片上的联合概率求和得到 $p_{X_3}(c)$：
$$p_{X_3}(c)=\sum_{x_2}\sum_{x_1}p_{X_1,X_2,X_3}(x_1,x_2,c)$$
然后，用联合概率除以 $p_{X_3}(c)$ 得到条件概率 $p_{X_1,X_2 | X_3}(x_1, x_2 | c)$：
$$p_{X_1,X_2|X_3}(x_1,x_2|c)=\frac{p_{X_1,X_2,X_3}(x_1,x_2,c)}{p_{X_3}(c)}$$



