# 一、概念
## 1.1 导数
>[!note] **导数：** 设$y=f(x)$定义在区间$I$上，让自变量在$x=x_0$处加一个增量$\Delta x$，其中$x_0\in I$，$x_0+\Delta x\in I$，则可得函数的增量$\Delta y=f(x_0+\Delta x)-f(x_0)$。若函数增量$\Delta y$与自变量增量$\Delta x$的比值在$\Delta x\to 0$时的极限存在，则称函数$y=f(x)$在$x_0$处可导，并称这个极限为$y=f(x)$在点$x_0$处的**导数**，记作$f'(x)$
>即$f'(x)=\lim\limits_{\Delta x\to 0}\dfrac{\Delta y}{\Delta x}=\lim\limits_{\Delta x\to 0}\dfrac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$。

下面三句话等价：

1.  $y=f(x)$在 $x_0$ 处可导。

2.  $y=f(x)$在 $x_0$ 处导数存在。

3.  $f'(x)=A$（$A$为有限数）

单侧导数分为左导数和右导数。

$f'_-(x)=\lim\limits_{\Delta x\to 0^-}\dfrac{\Delta y}{\Delta x}=\lim\limits_{\Delta x\to 0}\dfrac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$

$f'_+(x)=\lim\limits_{\Delta x\to 0^+}\dfrac{\Delta y}{\Delta x}=\lim\limits_{\Delta x\to 0}\dfrac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$

所以$f(x)$在$x_0$处可导的充要条件是其**左导数和右导数存在且相等**。

若$f(x)$在$x_0$的左右，如$y=\vert x\vert$在$0$的左右出现了单侧的不同的切线，那这个$x_0$就是一个**角点**，该**角点处不可导**。

若$f(x)$在$x_0$处导数为无穷，如$y=x^{\frac{1}{3}}$在$0$处利用导数的极限定义计算得到为正无穷，那么该点的导数为**无穷导数**，在考研中被认为是不存在的。

>[!tip] **定理：** 若$f(x)$为可导的偶函数，则$f'(x)$为奇函数，若$f(x)$为可导的奇函数，则$f'(x)$为偶函数。
>该证明是准备部分的定理。
>
>证明：首先已知$f(-x)=f(x)$，证明$f'(-x)=-f'(x)$。
>$\begin{aligned}
    >f'(-x) &=\lim\limits_{\Delta x\to 0}\dfrac{f(-x+\Delta x)-f(-x)}{\Delta x} \\
    >& =\lim\limits_{\Delta x\to 0}\dfrac{f(x+(-\Delta x))}{\Delta x} \\
    >& =-\lim\limits_{-\Delta x\to 0}\dfrac{f(x+(-\Delta x))}{-\Delta x} \\
    >& =-f'(x)
>\end{aligned}$
>
>同理得证$f(-x)=-f(x)\Rightarrow f'(-x)=f'(x)$。

>[!tip] **定理：** $f(x)$为可导的周期为$T$的周期函数，则$f'(x)$也是以$T$为周期的周期函数。
>
>证明：已知$f(x+T)=f(x)$，求证$f'(x+T)=f'(x)$。
>
>$\therefore f'(x+T)=\lim\limits_{\Delta x\to 0}\dfrac{f(x+T+\Delta x)-f(x+T)}{\Delta x}$
>
>$=\lim\limits_{\Delta x\to 0}\dfrac{f(x+\Delta x)-f(x)}{\Delta x}=f'(x)$。

> **例题：** 设$f(x)$是二阶可导的以2为周期的奇函数，且$f(\dfrac{1}{2})>0$，$f'(\dfrac{1}{2})>0$，比较$f(-\dfrac{1}{2})$、$f'(\dfrac{3}{2})$、$f''(0)$的大小。
>
>解：$\because f(x)$为二阶奇函数，$\therefore f(x)\text{奇函数}\Rightarrow f'(x)\text{偶函数}\Rightarrow f''(x)\text{奇函数}\Rightarrow f''(0)=0$。
>
>$\therefore f(-\dfrac{1}{2})=-f(\dfrac{1}{2})<0$。
>
>$\because f(x)T=2\Rightarrow f'(x)T=2$，$\therefore f'(\dfrac{3}{2})=f'(\dfrac{3}{2}-2)=f'(-\dfrac{1}{2})=f'(\dfrac{1}{2})>0$。
>
>$\therefore f'(\dfrac{3}{2})>f''(0)>f(-\dfrac{1}{2})$。

> **例题：** $\left(x^\alpha\right)'=\alpha x^{\alpha-1}(x>0)$。

>解：$\lim\limits_{\Delta x\to 0}\dfrac{f(x+\Delta x)-f(x)}{\Delta x}=\lim\limits_{\Delta x\to 0}\dfrac{\left(x+\Delta x\right)^\alpha-x^\alpha}{\Delta x}$
>
>$=\lim\limits_{\Delta x\to 0}\dfrac{x^\alpha\left[\left(1+\dfrac{\Delta x}{x}\right)^\alpha-1\right]}{\Delta x}=\lim\limits_{\Delta x\to 0}\dfrac{x^\alpha\cdot\alpha\cdot\dfrac{\Delta x}{x}}{\Delta x}=\alpha x^{\alpha-1}$

## 1.2 导数的几何意义

导数$f'(x_0)$在几何上就是曲线$y=f(x)$在点$(x_0,f(x_0))$处切线的斜率。

切线方程：$y-y_0=f'(x_0)(x-x_0)$。

法线方程：$y-y_0=-\dfrac{1}{f'(x_0)}(x-x_0)$。

## 1.3 高阶导数定义

>[!note] **高阶导数：** $f^{(n)}(x_0)=\lim\limits_{\Delta x\to 0}\dfrac{f^{(n-1)}(x_0+\Delta x)-f^{(n-1)}(x_0)}{\Delta x}$
>其中 $n\geqslant 2$ 且 $n\in N^+$，$f^{(n-1)}(x)$ 在 $x_0$ 的某领域内有定义，$x_0+\Delta x$ 也在该邻域内。

若$f^{(n)}(x)$在区间$I$上连续，称$f(x)$在$I$上$n$阶连续可导。

-   $(e^x)^{(n)}=e^x$。

-   $(\sin x)^{(n)}=\sin(x+n\dfrac{\pi}{2})$。

-   $(\cos x)^{(n)}=\cos(x+n\dfrac{\pi}{2})$。

-   $(\ln(1+x))^{(n)}=(-1)^{n-1}\dfrac{(n-1)!}{(1+x)^n}$。
>[!note] **注：**
>1. 如果 $f(x)$ 在点 $x_0$ 处有二阶导数，则 $f(x)$ 在 $x_0$ 的某个邻域内有一阶导数且 $f'(x)$ 在 $x_0$ 处连续
>
>2. 如果 $f(x)$ 在点 $x_0$ 处有 $n$ 阶导数，则 $f(x)$ 在 $x_0$ 的某个邻域内有 $1~(n-1)$ 阶的各阶导数

## 1.4函数微分

### 1.4.1定义
![](/img/Pasted%20image%2020240226202148.png)

有一个边长为 $x$ 的正方形，变化了 $\Delta x$，其面积$\Delta S=(x+\Delta x)^2-x^2=2x\Delta x+(\Delta x)^2$

当 $\Delta x\to 0$ 时，将这个变化定义为 $2x\cdot\Delta x+o(\Delta x)$，前项为**线性主部**，后面为**误差**。这个就是$S$的微分。

增量 $\Delta y=f(x_0+\Delta)-f(x_0)=A\Delta x+o(\Delta x)$，这个 $A\Delta x$ 定义为 $\textrm{d}y$，叫做 $y$ 的微分。

$\therefore \textrm{d}y\vert_{x=x_0}=A\Delta x=y'(x_0)\cdot\Delta x=y'(x_0)\cdot\textrm{d}x$。

由此，**可导必可微**，**可微必可导**。

所以==可微就是用简单线性取代复杂线性，如图用直线取替代曲线。微分就是瞬时改变量，而导数就是瞬时改变速率==。

# 二、计算
## 2.1 基本求导公式
### 2.1.1 对幂指函数
![](/img/Pasted%20image%2020240226202221.png)
### 2.1.2 三角与反三角函数
![](/img/Pasted%20image%2020240226202238.png)
### 2.1.3 双曲与反双曲函数

-   双曲正弦：$\textrm{sinh}\,x=\textrm{sh}\,x=\dfrac{e^{x}-e^{-x}}{2}$。

-   双曲余弦：$\textrm{cosh}\,x=\textrm{ch}\,x=\dfrac{e^{x}+e^{-x}}{2}$。

-   双曲正切：$\textrm{tanh}\,x=\textrm{th}\,x=\dfrac{\textrm{sinh}\,x}{\textrm{cosh}\,x}=\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$。

-   双曲余切：$\textrm{coth}\,x=\dfrac{\textrm{cosh}\,x}{\textrm{sinh}\,x}=\dfrac{e^{x}+e^{-x}}{e^{x}-e^{-x}}$。

-   双曲正割：$\textrm{sech}\,x=\dfrac{1}{\textrm{cosh}\,x}=\dfrac{2}{e^{x}+e^{-x}}$。

-   双曲余割：$\textrm{csch}\,x=\dfrac{1}{\textrm{sinh}\,x}=\dfrac{2}{e^{x}-e^{-x}}$。

-   反双曲正弦：$\textrm{arcsinh}\,x=\ln\left(x+\sqrt{x^2+1}\right)$。

-   反双曲余弦：$\textrm{arccosh}\,x=\ln\left(x+\sqrt{x^2-1}\right)$。

-   反双曲正切：$\textrm{arctanh}\,x=\dfrac{1}{2}\ln\left(\dfrac{1+x}{1-x}\right)$。

![](/img/Pasted%20image%2020240226202253.png)
## 2.2 四则运算
>[!note] **求导四则运算：**
>若函数可导：
>1.  和差的导数：$$[u(x)\pm v(x)]'=u'(x)\pm v'(x)$$
>
>2.  积的导数：$$\begin{split}&[u(x)v(x)]'=u'(x)v(x)+u(x)v'(x)\\
&[u(x)v(x)w(x)]'=u'(x)v(x)w(x)+u(x)v'(x)w(x)+u(x)v(x)w'(x)\end{split}$$
>
>3.  商的导数：$$\left[\dfrac{u(x)}{v(x)}\right]'=\dfrac{u'(x)v(x)-u(x)v'(x)}{[v(x)]^2},\quad v(x)\neq 0$$
>---
>证明 $(uv)'=u'v+uv'$ 。
>
>证明：令$f(x)=u(x)v(x)$。
>
>$$\begin{split}(u\cdot v)'
>&=f'(x)=\lim\limits_{\Delta x\to 0}\dfrac{f(x+\Delta x)-f(x)}{\Delta x}=\lim\limits_{\Delta x\to 0}\dfrac{u(x+\Delta x)v(x+\Delta x)-u(x)v(x)}{\Delta x}\\
>&=\lim\limits_{\Delta x\to 0}\dfrac{u(x+\Delta x)v(x+\Delta x)-u(x)v(x+\Delta x)+u(x)v(x+\Delta x)-u(x)v(x)}{\Delta x}\\
>&=\lim\limits_{\Delta x\to 0}\dfrac{u(x+\Delta x)-u(x)}{\Delta x}v(x+\Delta x) +\lim\limits_{\Delta x\to 0}\dfrac{v(x+\Delta x)-v(x)}{\Delta x}u(x)\\
>&=u'(x)v(x)+v'(x)u(x)\\
>\end{split}$$

## 2.3 复合函数的导数

>[!tip] **定理：** $u=g(x)$在$x$可导，$y=f(u)$在$u=g(x)$处可导，则$\{f[g(x)]\}'=f'[g(x)]g'(x)$。

> **例题：** 设$f(x)=\prod\limits_{n=1}^{100}\left(\tan\dfrac{\pi x^n}{4}-n\right)$，则$f'(1)$为？
>
>解：原式=$\left(\tan\dfrac{\pi x}{4}-1\right)\left(\tan\dfrac{\pi x^2}{4}-2\right)\cdots\left(\tan\dfrac{\pi x^{100}}{4}-100\right)$
>
>令$\left(\tan\dfrac{\pi x^2}{4}-2\right)\cdots\left(\tan\dfrac{\pi x^{100}}{4}-100\right)=g(x)$
>
>$\therefore f(x)=\left(\tan\dfrac{\pi x}{4}-1\right)\cdot g(x)$
>
>$\therefore f'(x)=\sec^2\dfrac{\pi x}{4}\cdot\dfrac{\pi}{4}\cdot g(x)+\left(\tan\dfrac{\pi x}{4}-1\right)\cdot g'(x)$
>
>$\therefore$根据导数的四则运算，需要导数的乘积为每一项求导乘以其他不求导项的和，而 $\tan\dfrac{\pi x}{4}-1$ 当 $x=1$ 时为0，只要它不求导，其他的项都必然是0，所以原式的后面的结果都是0
>
>$\therefore f'(1)=f'(x)\vert_{x=1}=\dfrac{\pi}{2}\cdot g(1)+0\cdot g'(x)=\dfrac{\pi}{2}\cdot g(1)=\dfrac{\pi}{2}(-1)(-2)\cdots(-99)=-\dfrac{\pi}{2}\cdot 99!$

## 2.4 微分形式不变性

>[!tip] 设$y=f(u)$可微，$u=g(x)$可微，则$y=f(g(x))$可微，且$\textrm{d}y=y'_{x}\textrm{d}x=y'_{u}\textrm{d}u$
>即对哪个变量求导都是一样的，即$\textrm{d}\{f\,[g(x)]\}=f\,'[g(x)]g'(x)\textrm{d}x$。

>[!tip] 一阶微分形式不变性指：$\textrm{d}f\,(\varsigma)=f\,'(\varsigma)\textrm{d}\varsigma$，无论$\varsigma$是什么（类似导数的链式求导法则）。

> **例题：** 设$y=e^{\sin(\ln x)}$，求$\textrm{d}y$。
>
>解：$\because y=e^{\sin(\ln x)} \therefore$
>$\begin{aligned}
    >\textrm{d}y &=\textrm{d}e^{\sin(\ln x)} \\
    >& =e^{\sin(\ln x)}\cdot\textrm{d}(\sin(\ln x)) \\
    >& =e^{\sin(\ln x)}\cdot\cos(\ln x)\cdot\textrm{d}\ln x \\
    >& =e^{\sin(\ln x)}\cdot\cos(\ln x)\cdot\dfrac{1}{x}\textrm{d}x
>\end{aligned}$

## 2.5 分段函数的导数

>[!note] **分段函数的导数：**
>设$f(x)=\left\{
    >\begin{array}{lcl}
        >f_1(x), & & x\geqslant x_0 \\
        >f_2(x), & & x<x_0 \\
    >\end{array}
>\right.$
>
>**在分段点用定义**：
>
>判断$f'_+(x_0)=\lim\limits_{x\to x_0^+}\dfrac{f_1(x)-f(x_0)}{x-x_0}$==$\overset{?}{=}$==$\lim\limits_{x\to x_0^-}\dfrac{f_2(x)-f(x_0)}{x-x_0}$
>
>如果相等就挖去这个点，否则就包含这个点
>
>**非分段点使用导数公式求导**：
>
>$x>x_0,f'(x)=f_1'(x),x<0，f'(x)=f_2'(x)$

## 2.6 反函数导数

>[!tip] **定理：** $y=f(x)$可导，且$f'(x)\neq 0$，则存在反函数$x=\varphi(y)$，且$\dfrac{\textrm{d}x}{\textrm{d}y}=\dfrac{1}{\dfrac{\textrm{d}y}{\textrm{d}x}}$，即$\varphi'(x)=\dfrac{1}{f'(x)}$。
>$y=f(x)$可导，且$f'(x)\neq 0$就是指严格单调，而严格单调必有反函数。

> **例题：** 求$y=\arcsin x,x\in(-1,1)$与$y=\arctan x$的导数。
>
>解：首先反三角函数就是三角函数的反函数。
>
>求 $y=\arcsin x$，即 $x=\sin y$。
>
>$\therefore\dfrac{\textrm{d}\arcsin x}{\textrm{d}x}=\dfrac{1}{\dfrac{\textrm{d}\sin y}{\textrm{d}y}}=\dfrac{1}{\cos y}=\dfrac{1}{\sqrt{1-\sin^2y}}=\dfrac{1}{\sqrt{1-x^2}}$。
>
>求 $y=\arctan x$，即 $x=\tan y$。
>
>$\therefore\dfrac{\textrm{d}\arctan x}{\textrm{d}x}=\dfrac{1}{\dfrac{\textrm{d}\tan y}{\textrm{d}y}}=\dfrac{1}{\sec^2y}=\dfrac{1}{1+\tan^2y}=\dfrac{1}{1+x^2}$。

>[!tip] **定理：** 二阶反函数导数
>
>$f''(x)=y''_{xx}=\dfrac{\textrm{d}\left(\dfrac{\textrm{d}y}{\textrm{d}x}\right)}{\textrm{d}x}=\dfrac{\textrm{d}^2y}{\textrm{d}x^2}=\dfrac{\textrm{d}\left(\dfrac{1}{\varphi'(y)}\right)}{\textrm{d}x}=\dfrac{\textrm{d}\left(\dfrac{1}{\varphi'(y)}\right)}{\textrm{d}y}\cdot\dfrac{\textrm{d}y}{\textrm{d}x}$
>
>$=-\dfrac{x_{yy}''}{(x_y')^2}\cdot\dfrac{1}{x_y'}=-\dfrac{x_{yy}''}{(x_y')^3}$

>[!warning] **注意：** 其中$\textrm{d}x\cdot\textrm{d}x=(\textrm{d}x)^2=\textrm{d}x^2$ 称为**微分的幂**，而$\textrm{d}(x^2)$叫**幂的微分**。

> **例题：** 设$y=f(x)$的反函数是$x=\varphi(y)$，且$f(x)=\int_1^{2x}e^{t^2}\textrm{d}t+1$，求$\varphi''(1)$。
>
>解：$\because y=f(x)$
>
>$\therefore x=\varphi(y)$，$x_{yy}''=\varphi''(y)=-\dfrac{y_{xx}''}{(y_x')^3}=-\dfrac{f''(x)}{[f'(x)]^3}$。
>
>其中根据变限积分求导公式：
>
>$f'(x)=2e^{4x^2}$，$f''(x)=2e^{4x^2}\cdot 8x=16xe^{4x^2}$。
>
>又$y=1\Rightarrow x=\dfrac{1}{2}\Rightarrow\varphi''(1)=-\dfrac{f''\left(\dfrac{1}{2}\right)}{\left[f'\left(\dfrac{1}{2}\right)\right]^3}=-\dfrac{1}{e^2}$。

## 2.7 隐函数求导法

设函数$y=y(x)$由方程$F(x,y)=0$确定的可导函数，则方程两边对自变量$x$求导，（$y=y(x)$就是将$y$看作中间变量）得到一个关于$y'$的方程。解该方程就可以得出$y'$。

> **例题：** 设$y=y(x)$是由方程$\sin(xy)=\ln\dfrac{x+e}{y}+1$确定的隐函数，求$y'(0)$。
>
>解：$y(0)=e^2$
>
>两边求导：
>
>$\begin{aligned}
>\sin(xy) &=\ln(x+e)-\ln(y)+1 \\
>\cos(xy)(y+xy') &=\dfrac{1}{x+e}-\dfrac{y'}{y} \\
>e^2&=\dfrac{1}{e}-\dfrac{y'(0)}{e^2} \\
>y'(0) & =e-e^4
>\end{aligned}$

## 2.8 参数方程函数导数

>[!tip] **定理：** 设函数$y=y(x)$由参数方程$\left\{\begin{array}{l}x=\varphi(t) \\y=\psi(t)\end{array}\}\right.$确定，其中$t$为参数，且$\varphi(t)\psi(t)$对于$t$都可导，$\varphi(t)\neq 0$，则：
>一阶导数：$\dfrac{\textrm{d}y}{\textrm{d}x}=\dfrac{\textrm{d}y/\textrm{d}t}{\textrm{d}x/\textrm{d}t}=\dfrac{\psi'(t)}{\varphi'(t)}=u(t)$。
>
>二阶导数：$\dfrac{\textrm{d}^2y}{\textrm{d}x^2}=\dfrac{\textrm{d}\left(\dfrac{\textrm{d}y}{\textrm{d}x}\right)}{\textrm{d}x}=\dfrac{\textrm{d}\left(\dfrac{\textrm{d}y}{\textrm{d}x}\right)/\textrm{d}t}{\textrm{d}x/\textrm{d}t}=\dfrac{\textrm{d}u/\textrm{d}t}{\textrm{d}x/\textrm{d}t}=\dfrac{u'_t}{x'_t}$

> **例题：** 设$y=y(x)$由方程$\left\{\begin{array}{l}x=\sin t \\y=t\sin t+\cos \end{array}\right.$（$t$为参数）确定，求$\dfrac{\textrm{d}^2y}{\textrm{d}x^2}\vert_{t=\frac{\pi}{4}}$。
>
>解：求参数方程的二阶导数首先就要求出其一阶导数：
>
>$\dfrac{\textrm{d}y}{\textrm{d}x}=\dfrac{y_t'}{x_t'}=\dfrac{t\cos t}{\cos t}=t$。
>
>$\therefore\dfrac{\textrm{d}^2y}{\textrm{d}x^2}=\dfrac{\textrm{d}\left(\dfrac{\textrm{d}y}{\textrm{d}x}\right)}{\textrm{d}x}=\dfrac{t_t'}{(\sin t)_t'}=\dfrac{1}{\cos t}$
>
>$\therefore \sqrt{2}$。

当所求是极坐标方程时，可以使用$x=\rho(\theta)\cos\theta$和$y=\rho(\theta)\sin\theta$进行转换为参数方程然后进行求导。
## 2.9 幂指函数求导法

**非常重要**

对于==$u(x)^{v(x)}(u(x)>0,u(x)\neq 1)$== ，除了**对数求导法**外还可以使用**指数函数** $$u(x)^{v(x)}=e^{v(x)\ln u(x)}$$
然后求导得到：
$$\begin{split}
[u(x)^{v(x)}]'
&=[e^{v(x)\ln u(x)}]'\\
&=u(x)^{v(x)}\left[v'(x)\ln u(x)+v(x)\cdot\dfrac{u'(x)}{u(x)}\right]\\
\end{split}$$

> **例题：** 求$y=x^x(x>0)$的导数。
>
>解：$\because x^x=e^{x\ln x}$，$\therefore (x^x)'=(e^{x\ln x})'=x^x\cdot(\ln x+1)$。

> **例题：** 求解$y=x^{\frac{1}{x}}(x>0)$的整数最大值。
>
>解：$\because y=x^{\frac{1}{x}}=e^{\frac{1}{x}\ln x}$。
>
>$\therefore y'=\left(x^{\frac{1}{x}}\right)=\left(e^{\frac{1}{x}\ln x}\right)'=x^{\frac{1}{x}}\cdot\dfrac{1-\ln x}{x^2}$。
>
>令导数结果为0，因为$x^{\frac{1}{x}}$与$x^2$在$x>0$时都不为0，所以只有一个驻点$x=e$。
>
>$0<x<e$时$1-\ln x$大于0，所以导数大于0，函数在该区间增。相反$x>e$时函数在区间减。
>
>研究驻点左侧情况，求对应的极限：$e^{\lim\limits_{x\to 0^+}\frac{\ln x}{x}}=e^{-\infty}\to 0$。
>
>研究驻点右侧情况，求对应的极限：$e^{\lim\limits_{x\to+\infty}\frac{\ln x}{x}}=e^0\to 1$。
![](/img/Pasted%20image%2020240226202006.png)
>
>所以必然在$\sqrt{2}$与$\sqrt[3]{3}$两点取得整数最大值，而全部六次方后$\sqrt{2}^6=8<\sqrt[3]{3}=9$，所以$\sqrt[3]{3}$为最大整数解。

## 2.10 高阶导数

>[!tip] **定理：** 设$u,v$都是$n$阶可导，则：
>
>1. $(u\pm v)^{(n)}=u^{(n)}\pm v^{(n)}$。
>
>2. 莱布尼兹公式：$(uv)^{(n)}=\sum_{k=0}^nC_n^ku^{(n-k)}v^{(k)}$。

### 2.10.1 归纳法

即依次求导得出规律。

>[!note] $(a^x)^n=a^x(\ln a)^{(n)}$
>如 $y=2^x$，则 $y'=2^x\ln 2$，$y''=2^x(\ln 2)^2\cdots$ 
>
>得到 ==$y^{(n)}=2^x(\ln 2)^n,n\in N$==

> **例题：** 求$\sin x$的$n$阶导数。
>
>解：$\because \sin x'=\cos x$而不断求导会发现正负号会++--++--地变化而难以归纳为公式，所以需要另想办法。
>
>使用诱导公式：
>
>$y'=\cos x=\sin(x+\dfrac{\pi}{2})$
>
>$y''=\cos(x+\dfrac{\pi}{2})=\sin(x+\dfrac{\pi}{2}+\dfrac{\pi}{2})$
>
>$\cdots$
>
>$y^{(n)}=\sin(x+\dfrac{\pi}{2}\cdot n)$

### 2.10.2 莱布尼茨公式

>[!tip] **定理：** 设$u=u(x)$，$v=v(x)$均$n$阶可导，则$$(uv)^{(n)}=\sum_{k=0}^nC_n^ku^{(n-k)}v^{(k)}$$
>
>展开：$(uv)^{(n)}=C_n^0u^{(n)}v^{(0)}+C_n^1u^{(n-1)}v'+\cdots+C_n^nu^{(0)}v^{(n)}$。
>
>莱布尼兹公式里的系数与考研数学准备章节的因式分解公式的二次项公式的系数一致，可以使用杨辉三角形来记忆：
![](/img/Pasted%20image%2020240226201921.png)

> **例题：** 已知函数$y=e^x\cos x$，求$y^{(4)}$
>
>解：根据莱布尼兹公式：
>
>$(e^x\cos x)^{(4)}$
>
>$=C_4^0e^x\cos x+C_4^1e^x(-\sin x)+C_4^2e^x(-\cos x)+C_4^3e^x(\sin x)+C_4^4e^x(\cos x)$
>
>$=e^x\cos x+4e^x(-\sin x)+6e^x(-\cos x)+4e^x\sin x+e^x\cos x$
>
>$=-4e^x\cos x$

### 2.10.3 泰勒展开式
任何一个无穷阶可导的函数都可写成：
$$y=f(x)=\sum_{i=0}^{\infty}\dfrac{f^{(n)}(x_0)}{n!}(x-x_0)^n$$
或者
$$y=f(x)=\sum_{i=0}^{\infty}\dfrac{f^{(n)}(0)}{n!}x^n$$
题目给出一个具体的无穷阶可导函数 $y=f(x)$ ，可以通过已知公式展开为幂级数

>[!tip] 泰勒展开的唯一性：
>无论 $f(x)$ 由何种方式展开，其泰勒展开式具有唯一性

于是我们通过比较公式的系数，获得 $f^{(n)}(x_0)$或$f^{(n)}(0)$

> 设 $f(x)=x^2\ln(1-x)$，则当 $n\ge3$ 时，$f^{(n)}(0)=$
>利用泰勒公式展开，有：
>$$f(x)=x^2\ln(1-x)=x^2\cdot\sum_{m=1}^{\infty}(-1)^{m-1}\cdot\dfrac{(-1)^mx^m}{m}=-\sum_{m=1}^{\infty}\dfrac{x^{m+2}}{m}=-\sum_{m=0}^{\infty}\dfrac{x^{m+3}}{m+1}$$
>又由于$f(x)=\sum\limits_{m=0}^{\infty}\dfrac{f^{(n)}(0)}{n!}x^n$，根据函数展开式的唯一性，比较系数，有 $n=m+3$ ，所以：
>$$\dfrac{f^{(n)}(0)}{n!}=-\dfrac{1}{m+1}$$
>即 $f^{(n)}(0)=-\dfrac{n!}{n-2}$

# 三、几何应用

## 3.1 单调性与极值

>[!note] **极值：** 若$\exists\,\delta>0$，使
>$\forall x\in U(x_0,\delta)$恒有$f(x)\geqslant f(x_0)$，则$f(x)$在$x_0$取**极小值**。
>
>$\forall x\in U(x_0,\delta)$恒有$f(x)\leqslant f(x_0)$，则$f(x)$在$x_0$取**极大值**。

>[!tip] 常函数处处为极值

>[!tip] **定理：** （极值必要条件）
>若$f(x)$在$x_0$处可导，且$x_0$处取得极值，则$f'(x_0)=0$。

>[!tip] **定理：** （极值第一充分条件）
>若$f(x)$在 $x=x_0$ 处**连续**，且在$\mathring{U}(x_0,\delta)$内**可导**
>1.  若$x<x_0$时，$f'(x)\gt 0$，$x>x_0$时$f'(x)\lt 0$，则$x_0$取得极大值。
>
>2. 若$x>x_0$时，$f'(x)\gt 0$，$x<x_0$时$f'(x)\lt 0$，则$x_0$取得极小值。
>3.  若$f'(x)$在$x_0$处不变号，则无极值点。

>[!tip] **定理：** （极值第二充分条件）
>若$f(x)$在$x=x_0$处**二阶可导**，且$f'(x_0)=0$，$f''(x_0)\neq 0$：
>
>1.  当$f''(x_0)<0$，则$f(x)$在$x_0$取极大值。
>
>2.  当$f''(x_0)>0$，则$f(x)$在$x_0$取极小值。

>[!tip] **定理：** （极值第三充分条件）
>若$f(x)$在$x=x_0$处$n(n\geqslant2)$阶可导，且$f'(x_0)=f''(x_0)=\cdots=f^{(n-1)}(x_0)=0$，$f^{(n)}(x_0)\neq0$，则：
>
>1.  当$n$为**偶数**时$f(x)$在$x_0$取得极值。
>	1. 当$f^{(n)}(x_0)<0$，则$f(x)$在$x_0$取极大值，
>	2. 当$f^{(n)}(x_0)>0$，则$f(x)$在$x_0$取极小值。
>
>2.  当$n$为奇数时$f(x)$在$x_0$处无极值。

## 3.2 凹凸性与拐点

>[!note] 若函数$f(x)$在区间$I$上连续，且对$I$上任意两点$x_1,x_2$恒有：
>
>1.  $f(\dfrac{x_1+x_2}{2})<\dfrac{f(x_1)+f(x_2)}{2}$，则$f(x)$在$I$上凹。
>
>2.  $f(\dfrac{x_1+x_2}{2})>\dfrac{f(x_1)+f(x_2)}{2}$，则$f(x)$在$I$上凸。

而当凹凸性发生改变的点就是**拐点**。

>[!tip] **定理：** 
>
>1.  函数$f(x)$在区间$[a,b]$上连续，在$(a,b)$内二阶可导。
>
>1.  若$(a,b)$内$f''(x)>0$，则$f(x)$在$[a,b]$上凹。
>
>1.  若$(a,b)$内$f'(x)<0$，则$f(x)$在$[a,b]$上凸。

>[!tip] **定理：** （拐点必要条件）
>若$f(x)$在$x_0$处可导，且$x_0$处取得极值，则$f'(x_0)=0$。

>[!note] 某点为函数的拐点有两种情况：
>1. 该点二阶导数为0
>
>2. 该点二阶导数不存在

>[!tip] **定理：** （拐点第一充分条件）
>若$f(x)$在 $x=x_0$ 处连续，且在$\mathring{U}(x_0,\delta)$内**二阶导数存在**，且在该点的左右邻域内$f''(x)$**变号**，则点$(x_0,f(x_0))$为曲线的拐点

>[!tip] **定理：** （极值第二充分条件）
>若$f(x)$在$x=x_0$的某邻域内**三阶可导**，且$f''(x_0)=0$，$f'''(x_0)\neq 0$，则点$(x_0,f(x_0))$为曲线的拐点

>[!tip] **定理：** （极值第三充分条件）
>若$f(x)$在$x_0$处$n(n\geqslant2)$阶可导，且$f'(x_0)=f''(x_0)=\cdots=f^{(n-1)}(x_0)=0$，$f^{(n)}(x_0)\neq0$，则当$n$为**奇数**时，点$(x_0,f(x_0))$为曲线的拐点

## 3.3 极值点与拐点的重要结论

>[!tip] 曲线的可导点不可同时为极值点和拐点；曲线的不可导点可同时为极值点和拐点

>[!tip] 设多项式函数 $f(x)=(x-a)^ng(x)(n\gt1)$，且$g(a)\neq0$，则
>- 当 $n$ 为偶数时，$x=a$ 是 $f(x)$ 的极值点
>
>- 当 $n$ 为奇数时，点$(a,0)$ 是 $f(x)$ 的拐点

>[!tip] 设多项式函数 $f(x)=(x-a_1)^{n_1}(x-a_2)^{n_2}\cdots(x-a_k)^{n_k}$，其中 $n_i$ 是正整数，$a_i$是实数且$a_i$两两不等，$i=1,2,\cdots,k$，
>
>记$k_1$为$n_i$**为1**的个数，$k_2$为$n_i\gt1$且$n_i$为**偶数**的个数，$k_3$为$n_i\gt1$且$n_i$为**奇数**的个数，
>
>则$f(x)$的极值点个数为$k_1+2k_2+k_3-1$，拐点个数$k_1+2k_2+3k_3-2$
## 3.4 渐近线

-   若$\lim\limits_{x\to x_0}f(x)=\infty$，那么$x=x_0$就是**垂直渐近线**。

-   若$\lim\limits_{x\to\infty}f(x)=A$，那么$y=A$就是**水平渐近线**。

-   若$\begin{cases}\lim\limits_{x\to\infty}\dfrac{f(x)}{x}=a\\\lim\limits_{x\to\infty}[f(x)-ax]=b\end{cases}$ 那么 $y=ax+b$ 就是**斜渐近线**。

考的比较多的是斜渐进性，计算较复杂，如果能写成$y=f(x)=ax+b+o(x)$，$o(x)$为$x\to\infty$的高阶无穷小，则能快速得到斜渐进线。

>[!note] 寻找渐近线的顺序：垂直渐近线、水平渐近线、斜渐近线
>
>1. 先找无定义点、区间端点、分段函数分段点：判断这些点的函数是否为$\infty$，若是则为垂直渐近线
>
>2. 然后判断当$x\to\infty$时函数极限是否为常数，若是则为水平渐近线
>3. 最后计算$\lim\limits_{x\to\infty}\dfrac{f(x)}{x}$ 与 $\lim\limits_{x\to\infty}[f(x)-ax]$ 是否均存在，若有一个不存在则无斜渐近线

## 3.5 函数最值

### 3.5.1 连续函数**闭区间**最值

1.  求出$f(x)$在$(a,b)$内的**驻点**和**不可导的点**$x_1,x_2\cdots,x_n$ 并求出函数值$f(x_1),f(x_2)\cdots,f(x_n)$

2.  求出**端点值**$f(a),f(b)$。

3.  比较求出最值。

### 3.5.2 连续函数**开区间**最值

1.  求出$f(x)$在$(a,b)$内的**驻点**和**不可导的点**$x_1,x_2\cdots,x_n$并求出函数值$f(x_1),f(x_2)\cdots,f(x_n)$

2.  求出**端点单侧极限**：$\lim\limits_{x\to a^+}f(x)$ 与 $\lim\limits_{x\to b^-}f(x)$ 其中 $a,b$ 可以为 $\infty$

3.  比较求出最值。

>[!tip] 驻点：函数一阶导数为零的点

## 3.6 函数图像绘制
### 3.6.1 给出一般函数 $f(x)$

1.  确定函数定义域，并考察其**奇偶性**与**周期性**。

2.  求出**一阶导数**与**二阶导数**，并计算**导数为0**与**不存在**的点。

3.  根据导数判断**单调性**与**凹凸性**，并求出**极值**与**拐点**。

4.  求出渐近线。

5.  确定另外的特殊点。

### 3.6.2 给出参数方程

1. 描点法

2. 化为直角坐标方程/极坐标方程

## 3.7 曲率与曲率半径

### 3.7.1 曲率

>[!note] **曲率：** 表明曲线在某一点的弯曲程度的数值，针对曲线上某个点的切线方向角对弧长的转动率，通过微分来定义，表明曲线偏离直线的程度。曲率越大，表示曲线的弯曲程度越大。

曲率的倒数就是**曲率半径**。

两点切线改变角相同时，弯曲程度与两点之间的弧长度成反比。

两点之间的弧长度相同时，弯曲程度与两点切线改变角成正比。

$y-y_0$平均曲率：$\hat{k}=\dfrac{\vert\Delta\alpha\vert}{\vert\Delta s\vert}$。

$y$ 曲率：$k=\lim\limits_{\Delta x\to 0}\left\lvert\dfrac{\Delta\alpha}{\Delta s}\right\rvert=\left\lvert\dfrac{\textrm{d}\alpha}{\textrm{d}s}\right\rvert$（$\alpha$为$y$处切线与$x$轴所成角）。

需要对曲率公式进行化简，得到$s$与$\alpha$关于$x$的表示。根据弧微分的定义：$\textrm{d}s=\sqrt{1+f'^2(x)}\textrm{d}x$。

而对于$\alpha$：$\tan\alpha=y'=f'(x)$。

两边对$x$求导：$\sec^2\alpha\cdot\dfrac{\textrm{d}\alpha}{\textrm{d}x}=y''=f''(x)$。

又$\because\sec^2\alpha=1+\tan^2\alpha=1+y'^2$。

$\therefore\dfrac{d\alpha}{dx}=\dfrac{y''}{1+y'^2}\Rightarrow d\alpha=\dfrac{y''}{1+y'^2}dx$。

$\therefore k=\left\lvert\dfrac{\textrm{d}\alpha}{\textrm{d}s}\right\rvert=\dfrac{\vert y''\vert}{[1+(y')^2]^{\frac{3}{2}}}$。

对于参数方程，$k=\dfrac{\vert y''x'-y'x''\vert}{\left[(x')^2+(y')^2\right]^{\frac{3}{2}}}$

### 3.7.2 曲率半径

$\odot\,O$为函数$L$在点$X$处的曲率圆，该圆与$L$在$X$处相切，切线为$T$。

该点的曲率半径为$R$，其中$R=\dfrac{1}{K}$。

# 四、中值定理、微分等式与微分不等式

## 4.1 函数中值定理

都假定$f(x)$在$[a,b]$上连续。

### 4.1.1 有界与最值定理

>[!tip] **有界与最值定理：** $m\leqslant f(x)\leqslant M$，其中$m$，$M$分别为$f(x)$在$[a,b]$上的最大值和最小值。
### 4.1.2 介值定理

>[!tip] **介值定理：** 当$m\leqslant\mu\leqslant M$，存在$\xi\in[a,b]$，使得$f(\xi)=\mu$。

### 4.1.3 平均值定理

>[!tip] **平均值定理：** 当$a<x_1<x_2<\cdots<x_n<b$时，在$[x_1,x_n]$内至少存在一点$\xi$，使得$f(\xi)=\dfrac{f(x_1)+f(x_2)+\cdots+f(x_n)}{n}$。
>证明：已知$f(x)$在$[x_1,x_n]$上连续，根据有界与最值定理，$m\leqslant f(x)\leqslant M$。
>
>即$m\leqslant f(x_1)\leqslant M$、$m\leqslant f(x_2)\leqslant M$......$m\leqslant f(x_n)\leqslant M$。
>
>将这些式子全部相加，得到$nm\leqslant f(x_1)+f(x_2)+\cdots+f(x_n)\leqslant nM$。
>
>所以$m\leqslant\dfrac{f(x_1)+f(x_2)+\cdots+f(x_n)}{n}\leqslant m$。
>
>由介值定理，可知存在$\xi\in[a,b]$使得$f(\xi)=\dfrac{f(x_1)+f(x_2)+\cdots+f(x_n)}{n}$。

### 4.1.4 零点定理

>[!tip] **定理：** 当$f(a)\cdot f(b)<0$时，存在$\xi\in(a,b)$，使得$f(\xi)=0$。

## 4.2 微分中值定理

四个定理都是**建立局部与整体的关系**，**利用导数控制函数**，反之不能使用函数控制导数。

$\text{罗尔定理}\xrightleftharpoons[\text{特例：}f(a)=f(b)]{\text{泛化：任意端点值}}\text{拉格朗日中值定理}\xrightleftharpoons[\text{特例：}F(x)=x]{\text{泛化：参数方程}}\text{柯西中值定理}$

### 4.2.1 罗尔定理

#### 4.2.1.1定义
![](/img/Pasted%20image%2020240226202346.png)
>[!note] **极值：** 若 $\exists\delta>0$，使 $\forall x\in U(x_0,\delta)$ 恒有 $f(x)\geqslant f(x_0)$，则 $f(x)$ 在 $x_0$ 处取极小值，恒有 $f(x)\leqslant f(x_0)$，则 $f(x)$ 在 $x_0$ 处取极大值。

>[!tip] **费马引理：** 若 $f(x)$ 在 $x_0$ 处取得极值，且 $f(x)$ 在 $x_0$ 处可导，则 $f'(x_0)=0$

>[!tip] **罗尔定理：**
>1.  $f(x)$在$[a,b]$上连续。
>
>2.  $f(x)$在$(a,b)$内可导。
>
>3.  $f(a)=f(b)$。
>
>则$\exists\,\xi\in(a,b)$，使得$f'(\xi)=0$。

#### 4.2.1.2 推广

-   设$f(x)$在$(a,b)$内可导，$\lim\limits_{x\to a^+}f(x)=\lim\limits_{x\to b^-}f(x)=A$，则在$(a,b)$内至少存在一点$\xi$，使得$f'(\xi)=0$。

-   设$f(x)$在$(a,b)$内可导，$\lim\limits_{x\to a^+}f(x)=\lim\limits_{x\to b^-}f(x)=\pm\infty$，则在$(a,b)$内至少存在一点$\xi$，使得$f'(\xi)=0$。

-   设$f(x)$在$(a,+\infty)$内可导，$\lim\limits_{x\to a^+}f(x)=\lim\limits_{x\to+\infty}f(x)=A$，则在$(a,+\infty)$内至少存在一点$\xi$，使得$f'(\xi)=0$。

-   设$f(x)$在$(-\infty,+\infty)$内可导，$\lim\limits_{x\to-\infty}f(x)=\lim\limits_{x\to+\infty}f(x)=A$，则在$(-\infty,+\infty)$内至少存在一点$\xi$，使得$f'(\xi)=0$。

### 4.2.2 拉格朗日中值定理
>[!tip] **拉格朗日中值定理：**
>1.  $f(x)$在$[a,b]$上连续。
>
>2.  $f(x)$在$(a,b)$内可导。
>
>则$\exists\,\xi\in(a,b)$，使得$f(b)-f(a)=f'(\xi)(b-a)$。

>[!tip] **拉格朗日中值定理的几何意义：** 若连续曲线$y=f(x)$的弧 $\overset{\LARGE{\frown}}{AB}$ 上除端点外处处具有不垂直于 $x$ 轴的切线，则这弧上至少有一点 $C$ 使曲线在该点处的切线平行于弦 $AB$。

其中$f(b)-f(a)=f'[a+\theta(b-a)](b-a)(0<\theta<1)$，$\because f'(\xi)=f'[a+(\xi-a)]=f'[a+\dfrac{\xi-a}{b-a}(b-a)]$。

>[!note] **有限增量公式：** $\Delta y=f(x_0+\Delta x)-f(x_0)=f'[x_0+\theta\Delta x]\Delta x(0<\theta<1)$。
>有限增量公式中的$\Delta x$不一定很小，这个是一个增量的准确公式。即将增量$\Delta y$用$\Delta x$和该线段上某点的导数来表示，与微分值不同的是这个是个准确值而不是近似值，但是不好用，因为$\theta$未知。

>[!tip] **推论：** $f(x)$ 在 $I$ 上连续且可导，则 $I$ 上 $f(x)=C\Leftrightarrow f'(x)\equiv 0$。

> **例题：** 证明$x>0$时，$\dfrac{x}{1+x}<\ln(1+x)<x$。
>证明：令$f(x)=\ln x$，又$\ln 1=0$，$\therefore\ln(1+x)=\ln(1+x)-\ln 1$。
>
>根据拉格朗日中值定理$\ln(1+x)-\ln 1=f'(\xi)x=\dfrac{x}{\xi}(1<\xi<1+x)$
>
>又$\dfrac{x}{1+x}<\dfrac{x}{\xi}<x$，$\therefore$得证。

### 4.2.3 柯西中值定理
>[!tip] **柯西中值定理：**
>1.  $f(x)$与$F(x)$在$[a,b]$上连续。
>
>2.  $f(x)$与$F(x)$在$(a,b)$内可导，且$\forall x\in(a,b)$，$F'(x)\neq 0$。
>
>则$\exists\,\xi\in(a,b)$，使得$\dfrac{f(b)-f(a)}{F(b)-F(a)}=\dfrac{f'(\xi)}{F'(\xi)}$。

### 4.2.4 泰勒公式

#### 4.2.4.1 佩亚诺余项

设$f(x)$在$x_0$处$n$阶可微，则:
$$f(x)=\sum\limits_{k=0}^n\dfrac{f^{(k)}(x_0)}{k!}(x-x_0)^k+o((x-x_0)^n)$$这个就是带佩亚诺余项的泰勒公式。

其中 $f(x)=\sum\limits_{k=0}^n\dfrac{f^{(k)}(x_0)}{k!}(x-x_0)^k$就是$f(x)$在$x_0$处的$n$次泰勒多项式，$o((x-x_0)^n)$就是函数的佩亚诺余项。

>[!note] 缺点：
>1.  只给出余项的定性描述，不能进行定量分析。
>
>2.  适用范围小。

#### 4.2.4.2 拉格朗日余项

设$f(x)$在 $x_0$ 处 $n+1$ 阶可微，$x_0\in I$则$\forall x\in I$，$\exists\,\xi\in I(\xi\in(x_0,x))$使得
$$f(x)=\sum\limits_{k=0}^n\dfrac{f^{(k)}(x_0)}{k!}(x-x_0)^k+\dfrac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$$
这个就是带拉格朗日余项的泰勒公式。

其中 $R_n(x)=\dfrac{f^{(n+1)}(\xi)}{(n+1)!}(x-x_0)^{n+1}$就是函数的拉格朗日余项。

根据拉格朗日中值定理推广的方式：$R_n(x)=\dfrac{f^{(n+1)}[x_0+\theta(x-x_0)]}{(n+1)!}(x-x_0)^{n+1}(\theta\in(0,1))$。

若$\vert f^{(n+1)}(x)\vert\leqslant M$，则$\vert R_n(x)\vert=\dfrac{\vert f^{(n+1)}(\xi)\vert}{(n+1)!}\vert x-x_0\vert^{n+1}\leqslant\dfrac{M}{(n+1)!}\vert x-x_0\vert^{n-1}$。

>[!note] 特点：
>
>1.  进行定量研究。
>
>2.  可以进行整体的研究。
>
>3.  计算量较大。

## 4.3 微分等式
### 4.3.1 零点定理（证明根的存在性）
若 $f(x)$ 在 $[a,b]$ 上连续，且$f(a)f(b)\lt 0$，则 $f(x)=0$ 在 $(a,b)$ 内至少有一个根
### 4.3.2 单调性（证明根的唯一性）
若 $f(x)$ 在 $(a,b)$ 内单调，则 $f(x)=0$ 在 $(a,b)$ 内至多有一个根，这里区间 $(a,b)$ 可以是有限区间也可以是无穷区间
### 4.3.3 罗尔定理及其推论
>[!tip] **罗尔定理：**
>1.  $f(x)$在$[a,b]$上连续。
>
>2.  $f(x)$在$(a,b)$内可导。
>
>3.  $f(a)=f(b)$。
>
>则$\exists\,\xi\in(a,b)$，使得$f'(\xi)=0$。

### 4.3.4 实系数奇次方程至少有一个实根

>[!tip] 任何实系数奇次方程$x^{2n+1}+a_1x^{2n}+\cdots+a_{2n}x+a_{2n+1}=0$至少有一个实根
>
>证明：设
>$f(x)=x^{2n+1}+a_1x^{2n}+\cdots+a_{2n}x+a_{2n+1}$
>则$\lim\limits_{x\to+\infty}f(x)=+\infty$，$\lim\limits_{x\to-\infty}f(x)=-\infty$，由$f(x)$的连续性及推广的零点定理，知存在 $\xi\in(-\infty,+\infty)$，使$f(\xi)=0$，即 $f(x)=1$ 至少有一个实根

## 4.4 微分不等式

### 4.4.1 用函数性态（包括单调性、凹凸性和最值等）证明不等式
1. 若有 $f'(x)\ge0,a\lt x\lt b$，则有$f(a)\le f(x)\le f(b)$
2. 若有 $f''(x)\ge0,a\lt x\lt b$，则有$f'(a)\le f'(x)\le f'(b)$
	1. 当 $f'(a)\gt0$ 时，$f'(x)\gt0\Rightarrow f(x)$ 单调递增
	2. 当 $f'(a)\lt0$ 时，$f'(x)\lt0\Rightarrow f(x)$ 单调递减
3. 设 $f(x)$ 在 $I$ 内连续，且有唯一的极值点 $x_0$，则$\begin{cases}当x_0为极大值点，即为I内的最大值点，由f(x_0)\ge f(x)\\当x_0为极小值点，即为I内的最小值点，由f(x_0)\le f(x)\end{cases}$，其中$x\in I$
### 4.4.2 用常数变量化证明不等式
如果欲证的不等式中都是常数，则可以将其中一个或者几个常数变量化，再利用上面所述的导数工具去证明

### 4.4.3 用中值定理证明不等式
主要用拉格朗日中值定理或者泰勒公式

# 五、物理应用与经济应用
## 5.1 物理应用

考的可能性不大。

如$v=\lim\limits_{\Delta t\to0}\dfrac{\Delta s}{\Delta t}=s'(t)$，加速度$a(t)=\lim\limits_{\Delta t\to0}\dfrac{\Delta v}{\Delta t}=v'(t)=s''(t)$。

## 5.2 相关变化律

这个部分在书上主要是跟隐函数共同出现。

相关变化率含有一个最终的自变量$t$，$xy$都是关于$t$的函数。即隐函数$\dfrac{\textrm{d}y}{\textrm{d}t}=\dfrac{\textrm{d}y}{\textrm{d}x}\dfrac{\textrm{d}x}{\textrm{d}t}$。

> **例题：** 已知动点$P$在曲线$y=x^3$上运动，记坐标原点与点$P$之间的距离为$l$。若点$P$的横坐标对事件的变化率为常数$v_0$，则当$P$运动到点$(1,1)$时，求$l$对时间的变化率。
>
>解：求$l$对时间的变化率就是求$\dfrac{\textrm{d}l}{\textrm{d}t}$，即求$\dfrac{\textrm{d}l}{\textrm{d}x}\dfrac{\textrm{d}x}{\textrm{d}t}$，且已知$\dfrac{\textrm{d}x}{\textrm{d}t}=v_0$。
>
>又$l=\sqrt{x^2+y^2}=\sqrt{x^2+x^6}=x\sqrt{1+x^4}$。
>
>$\therefore\dfrac{\textrm{d}l}{\textrm{d}x}=\dfrac{1+3x^4}{\sqrt{1+x^4}}$，所以$\dfrac{\textrm{d}l}{\textrm{d}t}=\dfrac{1+3x^4}{\sqrt{1+x^4}}v_0$，代入$(1,1)$得到：$2\sqrt{2}v_0$。

