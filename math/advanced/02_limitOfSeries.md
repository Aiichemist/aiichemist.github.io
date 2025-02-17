## 1. 1 数列的概念
### 子列
从数列 $\{a_n\}:a_1,a_2,\cdots,a_n,\cdots$ 中选取**无穷多项**，并按原来的先后顺序组成新的数列，称新数列为原数列的子列，记为：
$$\{a_{n_k}\}:a_{n_1},a_{n_2},\cdots,a_{n_k},\cdots$$
其中下标 $n_1,n_2,\cdots,n_k,\cdots$ 为正整数
### 等差数列
1. 通项公式 $a_n=a_1+(n-1)d$
2. 前 $n$ 项的和 $S_n=\dfrac{n}{2}(a_1+a_n)$
### 等比数列
1. 通项公式 $a_n=a_1q^{n-1}$
2. 前 $n$ 项的和 $S_n=\begin{cases}na_1,&q=1\\\dfrac{a_1(1-q^n)}{1-q},&q\neq 1\end{cases}$
### 单调数列
### 有界数列
### 常见数列前 n 项和
$$\sum_{i=1}^Nk=1+2+3+\cdots+n=\dfrac{n(n+1)}{2}$$
$$\sum_{i=1}^Nk^2=1^2+2^2+3^2+\cdots+n^2=\dfrac{n(n+1)(2n+1)}{6}$$
$$\sum_{i=1}^n\dfrac{1}{k(k+1)}=\dfrac{1}{1\times2}+\dfrac{1}{2\times3}+\dfrac{1}{3\times4}+\cdots+\dfrac{1}{n(n+1)}=\dfrac{n}{n+1}$$
## 1.2 数列极限的定义

### 极限定义

>[!note] **数列极限：**设$\{x_n\}$为一数列，若存在常数$a$，对于不论任意小的$\xi>0$，总存在正整数$N$，使$n>N$时，$\vert x_n-a\vert<\xi$恒成立，则常数$a$为数列$\{x_n\}$的**极限**，或$\{x_n\}$**收敛**于$a$，记为：$\lim\limits_{x\to\infty}x_n=a$或$x_n\to a(n\to\infty)$。

>[!note] $\xi-N$语言：$\lim\limits_{x\to\infty}x_n=a$$\Leftrightarrow$==$\forall\xi>0,\exists N\in N_+$，当$n>N$时，恒有$\vert x_n-a\vert<\xi$==。

如果不存在该数$a$，则称数列$x_n$**发散**。

即无论给出多么小的$\xi$，总可以找到一项从该项之后函数值与极限值之间的差小于$\xi$，即更接近这个极限值而不是其他任何值，所以该数列趋向于极限值。

### 极限证明
>[!note] 如何证明函数极限
>
>令$x_n$为通项，$a$为极限值，$\xi$为任意正数。
>
>1.  写出$\vert x_n-a|<\xi$。
>2. 反解出项数$n<g(\xi)$。
>3.  取$N=[g(\xi)]+1$，所以令$n>N$就可以证明。

>[!note] **例题：** 用定义证明$\lim\limits_{x\to\infty}\left[1+\dfrac{(-1)^n}{n}\right]=1$
>
>证明：计算距离：$\left\vert 1+\dfrac{(-1)^n}{n}-1\right\vert=\left\vert\dfrac{(-1)^n}{n}\right\vert<\xi$。
>
>解得到：$\dfrac{1}{n}<\xi$，反解为$n>\dfrac{1}{\xi}$。
>
>取整：$N=\left[\dfrac{1}{\xi}\right]+1$。
>
>$\therefore\forall\xi>0$，当$n>N$时，就有$n>\dfrac{1}{\xi}$，使得$\left\vert 1+\dfrac{(-1)^n}{n}-1\right\vert=\left\vert\dfrac{(-1)^n}{n}\right\vert<\xi$。
>
>$\therefore$证明完毕。

>[!note] **例题：** 用定义证明$\lim\limits_{n\to\infty}q^n=0$（$q$为常数且$\vert q\vert<1$）。
>
>证明：$\vert q^n-0\vert<\xi$。
>
>即$\vert q^n\vert<\xi$，
>
>取对数进行反解$n\ln\vert q\vert<\ln\xi$，
>
>又因为$\vert q\vert<1$，
>
>所以$\ln\vert q\vert<0$，
>
>所以得到$n>\dfrac{\ln\xi}{\ln\vert q\vert}$。（若 $\xi>1$ 则 $n$ 就是负数，这样条件必然成立）
>
>取$N=\left[\dfrac{\ln\xi}{\ln\vert q\vert}\right]+1$。
>
>$\therefore$当$n>N$时，必然$n>\dfrac{\ln\xi}{\ln\vert q\vert}$，有$\vert q^n-0\vert<\xi$。
>
>故$\lim\limits_{n\to\infty}q^n=0$。

### 子数列

>[!note] **子数列：**从数列$\{a_n\}:a_1,a_2,\cdots,a_n,\cdots$中选取**无穷多项**并**按原来顺序**组成的新数列就称为原数列的子列，记为$\{a_{n_k}\}:a_{n_1},a_{n_2},\cdots,a_{n_k},\cdots$。

若$n_k$分别取奇数和偶数，则得到**奇数项数列**与**偶数项数列**。

>[!tip] **定理：** 若数列$\{a_n\}$收敛，则其任何子列$\{a_{n_k}\}$也收敛，且极限值相同。

所以对于其变式我们用到更多：

1.  若一个数列$\{a_n\}$能**找到一个发散的子列**，那该数列发散。

2.  若一个数列$\{a_n\}$能找到**两个极限值不同**的收敛子列，那么这个数列发散。

3.  若一个数列$\{a_n\}$收敛，则其奇数子列与偶数子列都收敛于同一个值。

>[!note] **例如：** 对于数列$\{(-1)^n\}$，能找到其奇数子列收敛于-1，偶数子列收敛于1，所以收敛值不同，原数列发散。

### 数列绝对值

>[!tip] **定理：** 若$\lim\limits_{x\to\infty}a_n=A$，则$\lim\limits_{x\to\infty}\vert a_n\vert=\vert A\vert$。
>
>证明：$\because\lim\limits_{n\to\infty}a_n=A\Leftrightarrow\forall\xi>0,\exists N>0,\text{当}n>N$，恒有$\vert a_n-A\vert<\xi$。
>
>又由重要不等式$\vert\vert a\vert-\vert b\vert\vert\leqslant\vert a-b\vert$，所以$\vert\vert a_n\vert-\vert A\vert\vert\leqslant\xi$。
>
>所以恒成立，证明完毕。

从这个题推出：$\lim\limits_{n\to\infty}a_n=0\Leftrightarrow\lim\limits_{n\to\infty}\vert a_n\vert=0$。所以**如果我们以后需要证明某一数列极限为0，可以证明==数列绝对值极限为0==，而数列绝对值绝对是大于等于0的，所以由夹逼准则，其中小的一头已经固定为0了，所以只用==找另一个偏大的数列夹逼==所证明数列就可以了。**


## 1.3 收敛数列的性质

### 1.3.1 唯一性

>[!tip] 定理：若数列$\{x_n\}$收敛于$A$，则$A$是唯一的。
>
>证明：设$\lim\limits_{n\to\infty}a_n=A$且$\lim\limits_{n\to\infty}a_n=B$且$A\neq B$。
>
>不如设$A>B$。任意取$\xi=\dfrac{A-B}{2}>0$。
>
>$\because\lim\limits_{n\to\infty}a_n=A$
>
>$\therefore\exists N_1>0$，当$n>N_1$时，$\vert a_n-A\vert<\dfrac{A-B}{2}$。
>
>得到$\dfrac{A+B}{2}<a_n<\dfrac{3A-B}{2}$并设为式子一。
>
>又$\because\lim\limits_{n\to\infty}a_n=B$
>
>$\therefore\exists N_2>0$，当$n>N_2$时，$\vert a_n-B\vert<\dfrac{A-B}{2}$。
>
>得到$\dfrac{3A-B}{2}<a_n<\dfrac{A+B}{2}$并设为式子二。
>
>取$N=\max\{N_1,N_2\}$，当$n>N$时，式子一二同时成立，
>
>而$A\neq B$，则这两个式子不可能同时成立，矛盾。
>
>同理$A<B$时也矛盾，所以$A\neq B$矛盾。

### 1.3.2 有界性

>[!tip] 定理：若数列$\{x_n\}$极限存在，则数列有界。
>
>即$\lim\limits_{n\to\infty}a_n=A$，则存在$M>0$，使得$\vert a_n\vert\leqslant M$。
>
>证明：由极限定义，取$\xi=1$。
>
>$\because\lim\limits_{n\to\infty}a_n=A$
>
>$\therefore\exists N>0$，当$n>N$时，$\vert a_n-A\vert<1$。
>
>$\because\text{重要不等式}\,\vert\vert a_n\vert-\vert A\vert\vert\leqslant\vert a_n-A\vert$
>
>$\therefore n>N$时，$\vert\vert a_n\vert-\vert A\vert\vert<1\Rightarrow\vert a_n\vert<1+\vert A\vert$
>
>取$M=\max\{\vert a_1\vert,\vert a_2\vert,\cdots,\vert a_N\vert,1+\vert A\vert\}$
>
>$\forall n$，有$\vert a_n\vert\leqslant M$

所以**数列极限存在则数列有界**。

但是**数列有界不一定极限存在**，如$1+(-1)^n$。

### 1.3.3 保号性

较重要。也称为**脱帽法**。

>[!tip] [定理]**脱帽法**。若数列$\{a_n\}$存在极限$\lim\limits_{n\to\infty}a_n=A\neq 0$，则存在正整数$N$，当$n>N$时$a_n$都与$A$同号。

简单来说，**就是极限大于0，后面一部分数列大于0**，极限小于0，后面一部分数列小于0。

>[!tip] [推论]**戴帽法**：若数列$\{a_n\}$从某项开始$a_n\geqslant B$，且$\lim\limits_{n\to\infty}a_n=A$，则$A\geqslant B$。**这里一定要带等号。**
>
>证明：设$A>0$，取$\xi=\dfrac{A}{2}>0$。
>
>$\because\lim\limits_{n\to\infty}a_n=A$
>
>$\therefore\exists N>0$，当$n>N$时，$\vert a_n-A\vert<\dfrac{A}{2}\Rightarrow a_n>\dfrac{A}{2}>0$
>
>同理得证极限值小于0的情况。

>[!tip] 总结
>**脱帽严格不等**：
>$\begin{cases}\lim x_n>0\Rightarrow x_n>0\\
>\lim x_n<0\Rightarrow x_n<0\end{cases}$
>
>**带帽非严格不等**：$\begin{cases}x_n \ge 0 \Rightarrow \lim x_n\ge 0\\
>x_n\le 0 \Rightarrow \lim x_n\le0\end{cases}$

## 1.4 数列极限运算法则
>[!note] 数列极限运算法则
>若$\lim\limits_{n\to\infty}x_n=a$，$\lim\limits_{n\to\infty}y_n=b$则：
>1.  $\lim\limits_{n\to\infty}x_n\pm y_n=a\pm b$。
>2.  $\lim\limits_{n\to\infty}(x_ny_n)=\lim\limits_{n\to\infty}x_n\lim\limits_{n\to\infty}y_n=ab$。
>3.  $\lim\limits_{n\to\infty}\dfrac{x_n}{y_n}=\dfrac{\lim\limits_{n\to\infty}x_n}{\lim\limits_{n\to\infty}y_n}=\dfrac{a}{b}(b\neq 0)$。

>[!note] **例题：** 若$\lim\limits_{n\to\infty}(a_n+b_n)=1$且$\lim\limits_{n\to\infty}(a_n-b_n)=3$，计算$\lim\limits_{n\to\infty}a_n$与$\lim\limits_{n\to\infty}b_n$。
>
>解：首先是不能通过运算法则第一条将两个条件直接加减的，因为不能保证两个极限是否都存在。
>
>所以必须先令$u_n=a_n+b_n$，$v_n=a_n-b_n$，所以$\lim\limits_{n\to\infty}u_n=1$，$\lim\limits_{n\to\infty}v_n=3$。
>
>因为这两个极限都存在，所以可以进行运算。
>
>相加得到$\lim\limits_{n\to\infty}(u_n+v_n)=2\lim\limits_{n\to\infty}a_n=4$。
>
>所以得到$\lim\limits_{n\to\infty}a_n=2$。同理$\lim\limits_{n\to\infty}(u_n-v_n)$得到$\lim\limits_{n\to\infty}b_n=-1$。

## 1.5 海涅定理（归结原则）

>[!tip] **定理：** 设$f(x)$在$\mathring{U}(x_0,\delta)$内有定义，则$\lim\limits_{x\to x_0}f(x)=A$存在$\Leftrightarrow$对任何$\mathring{U}(x_0,\delta)$内以$x_0$为极限的数列$\{x_n\}(x_n\neq x_0)$，极限$\lim\limits_{n\to\infty}f(x_n)=A$存在。

海涅定理用来连接数列极限与函数极限。在极限存在下他们可以相互转换。

>[!tip] **常考：**
>1. 当 $x\to 0$ 时，取 $x_n=\dfrac{1}{n}$ ，即若 $\lim\limits_{x\to 0}f(x)=A$ ，则 $\lim\limits_{n\to\infty}\left(\dfrac{1}{n}\right)=A$
>
>2. ==当 $x\to\infty$ 时，取 $x_n=n$ ，即若 $\lim\limits_{x\to+\infty}f(x)=A$ ，则 $\lim\limits_{n\to\infty}f(n)=A$==
>
>3. 当 $x\to a$ 时，且 $x_n\neq a$ ，若 $\lim\limits_{x\to a}f(x)=A$ ，则 $\lim\limits_{n\to\infty}f(x_n)=A$

>[!note] **例题：** 求$\lim\limits_{n\to\infty}\left(n\tan\dfrac{1}{n}\right)^{n^2}$（$n\in N^+$）。
>
>解：首先将式子由数列极限变为函数极限，并取$x=\dfrac{1}{n}$：$\lim\limits_{x\to 0}\left(\dfrac{\tan x}{x}\right)^{\frac{1}{x^2}}$。
>
>又$u^v=e^{v\ln u}$，对式子取指数$\therefore =e^{\lim\limits_{x\to 0}\frac{1}{x^2}\ln\frac{\tan x}{x}}$
>
>又在$x\to 0$下使用等价无穷小$\ln (1+x)\sim x$，
>
>$\therefore \ln(1+g(x))\sim g(x),g(x)\to 0$。
>
>而在$x\to 0$时，根据等价无穷小$\tan x\sim x$，
>
>所以$\dfrac{\tan x}{x}$趋于1，不满足趋于0的条件。
>
>所以正好将$\ln\dfrac{\tan x}{x}$变形$\ln\left(1+\dfrac{\tan x}{x}-1\right)$。
>
>$\therefore \ln\left(1+\dfrac{\tan x}{x}-1\right)\sim\dfrac{\tan x}{x}-1$，$\dfrac{\tan x}{x}-1\to 0$。
>
>又根据泰勒展开$\tan x-x=x+\dfrac{x^3}{3}+o(x^3)-x-0\cdot x^3=\dfrac{x^3}{3}$。
>
>$\therefore e^{\lim\limits_{x\to 0}\frac{1}{x^2}\ln\frac{\tan x}{x}}=e^{\lim\limits_{x\to 0}\frac{1}{x^2}\frac{\tan x-x}{x}}=e^{\lim\limits_{x\to 0}\frac{1}{x^2}\cdot\frac{x^2}{3}}=e^{\frac{1}{3}}$
>
>根据海涅定理：取$x=\dfrac{1}{n},n\to\infty$，$\lim\limits_{n\to\infty}\left(n\tan\dfrac{1}{n}\right)^{n^2}=e^{\frac{1}{3}}$。

## 1.6 夹逼准则

>[!tip] [定理] **数列的夹逼准则**
>1.  $y_n\leqslant x_n\leqslant z_n (n=1,2,3,\cdots)$。
>
>2.  $\lim\limits_{n\to\infty}y_n=a,\lim\limits_{n\to\infty}z_n=a$。(**也可以是同号无穷大**)
>
>则$\lim\limits_{n\to\infty}x_n=a$。
> 
>---
>
>证明：由于$\lim\limits_{n\to\infty}y_n=a,\lim\limits_{n\to\infty}z_n=a$。
>
>则$\forall\xi>0$，$\exists N\in N_+$，当$n>N$时，$\vert y_n-a|<\xi$，$\vert z_n-a|<\xi$。
>
>$\therefore a-\xi<y_n<a+\xi$，$a-\xi<z_n<a+\xi$。
>
>$\therefore a-\xi<y_n\leqslant x_n\leqslant z_n<a+\xi$。
>
>$\therefore\vert x_n-a\vert<\xi$。

>[!note] **例题：** 求极限$\lim\limits_{n\to\infty}\left(\dfrac{n}{n^2+1}+\dfrac{n}{n^2+2}+\cdots+\dfrac{n}{n^2+n}\right)$。
>
>解：使用夹逼准则：$\dfrac{n^2}{n^2+n}<\sum_{i=1}^n\dfrac{n}{n^2+i}<\dfrac{n^2}{n^2+1}$。
>
>又$\lim\limits_{n\to\infty}\dfrac{n^2}{n^2+1}=\lim\limits_{n\to\infty}\dfrac{n^2/n^2}{n^2/n^2+1/n^2}=\lim\limits_{n\to\infty}\dfrac{1}{1+\dfrac{1}{n^2}}=1$。
>
>且$\lim\limits_{n\to\infty}\dfrac{n^2}{n^2+n}=\lim\limits_{n\to\infty}\dfrac{1}{1+\dfrac{1}{n}}=1$。
>
>由夹逼准则，原式的极限为1。

数列的夹逼准则下不等式的证明往往要使用到**放缩法**，对于分式的放缩主要在于**分母的放缩**，不变分子，分母变小原式变大，分母变大原式变小。然后**分子分母除以最高项**得到逼向0的极限。

### 1.6.1 放缩的常用方法
>[!note] **利用简单的放大与缩小**
>$$\begin{cases}
>n\cdot u_{\min}\le u_1+u_2+\cdots+u_m\le n\cdot u_{\max}\\
>1\cdot u_{\max}\le u_1+u_2+\cdots+u_m\le n\cdot u_{\max}(u_i\ge 0)\\
>\end{cases}$$

>[!note] **利用重要不等式**
>1. 设 $a,b$ 为实数，则$$\begin{cases}
>|a\pm b|\le|a|+|b|\\
>||a|-|b||\le|a-b|\\
>\end{cases}$$
>>[!tip] 可以将上述不等式拓展为 $n$ 个实数的情形，即
>>$$|a_1\pm a_2\pm \cdots\pm a_n|\le|a_1|+|a_2|+\cdots+|a_n|$$
>1. $$\begin{cases}\sqrt{ab}\le\dfrac{a+b}{2}\le\sqrt{\dfrac{a^2+b^2}{2}}\qquad\qquad(a,b\ge 0)\\\sqrt[3]{abc}\le\dfrac{a+b+c}{3}\le\sqrt{\dfrac{a^2+b^2+c^2}{3}}\quad(a,b,c\ge 0)\\\end{cases}$$
>>[!tip] $|ab|\le\dfrac{a^2+b^2}{2}$
>>
>> 例如，若 $u_n\gt 0$，则 $\dfrac{u_n}{n}=u_n\cdot\dfrac{1}{n}\le\dfrac{u_n^2+\dfrac{1}{n^2}}{2}$
>
>3. 设 $a\ge b\le 0$，则 $\begin{cases}当 m\gt 0 时，a^m\ge b^m\\当 m\lt 0 时，a^m\le b^m\end{cases}$
>
>4. 若 $0\lt a\lt x\lt b$ ， $0\lt c\lt y\lt d$ ，则$\dfrac{c}{b}\lt\dfrac{y}{x}\lt\dfrac{d}{a}$
>
>5. $\sin x\lt x\lt \tan x\left(0\lt x\lt\dfrac{\pi}{2}\right)$
>
>6. $\sin x\lt x(x\gt 0)$
>
>7. 当 $0\lt x\lt\dfrac{\pi}{4}$ 时，$x\lt\tan x\lt\dfrac{4}{\pi}x$ 
>
>8. 当 $0\lt x\lt\dfrac{\pi}{2}$ 时，$\sin x\gt \dfrac{2}{\pi}x$ 
>
>9. $\arctan x\le x\le \arcsin x(0\le x\le 1)$
>
>10. $e^x\ge x+1(\forall x)$
>
>11. $x-1\ge\ln x(x\gt0)$
>
>12.$\dfrac{1}{1+x}\lt\ln\left(1+\dfrac{1}{x}\right)\lt\dfrac{1}{x}(x\gt 0)$ 或 $\dfrac{x}{1+x}\lt\ln(1+x)\lt x(x\gt 0)$ 

>[!note] **利用闭区间上连续函数必有最大值与最小值**

>[!note] **利用压缩映射原理：**
>**原理一：** 对数列 $\{x_n\}$，若存在常数 $k(0\lt k\lt 1)$，使得 $|x_{n+1}-a|\le k|x_n-a|,n=1,2,\cdots$，则 $\{x_n\}$ 收敛于 $a$
>
>**原理二：** 对数列 $\{x_n\}$，若 $x_{n+1}=f(x_n),n=1,2,\cdots,f(x)$ 可导， $a$ 是 $f(x)=x$ 的唯一解，且 $\forall x\in R$，有 $|f'(x)|\le k\lt1$，则 $\{x_n\}$ 收敛于 $a$

>[!tip] **结论：** 当 $0\lt a\lt b$ 时， $\lim\limits_{n\to\infty}(a^{-n}+b^{-n})^{\frac{1}{n}}=\lim\limits_{n\to\infty}\sqrt[n]{\left(\dfrac{1}{a}\right)^n+\left(\dfrac{1}{b}\right)^n}=\dfrac{1}{a}(取较大值)$

## 1.7 单调有界准则
>[!tip] **定理：** **单调有界数列必有极限**，即若$\{x_n\}$单调增加（减少）且有上界（下界），则极限存在。
>
>该部分需要证明两个地方：
>1.  **数列单调**：$x_{n+1}-x_n$与0的关系，或$\dfrac{x_{n+1}}{x_n}$与1的关系。
>
>2.  **有界**：$\vert x_n\vert\leqslant M$是否存在。
>
>见到**递推式（迭代式）**$a_{n+1}=f(a_n)$，一般都要用单调有界准则。单调性通过**减或除**进行计算，有界性通过**不等式**来计算。

>[!note] **例题：** 已知$a_1=a>0$，证明$a_{n+1}=\dfrac{1}{2}\left(a_n+\dfrac{2}{a_n}\right)$的极限存在并求出。
>
>解：$\because a_1=a>0$，且递推式中没有负数与减的操作，所以$a_n>0$。
>
>由重要不等式$\dfrac{a+b}{2}\geqslant\sqrt{ab}$，所以$a_{n+1}=\dfrac{1}{2}\left(a_n+\dfrac{2}{a_n}\right)\geqslant\sqrt{a_n\cdot\dfrac{2}{a_n}})=\sqrt{2}$
>
>$\therefore$数列$\{a_n\}$有下界$\sqrt{2}$。
>
>又$a_{n+1}-a_n=\dfrac{2-a_n^2}{2a_n}$，且由上面证明已知$a_n^2\geqslant\sqrt{2}$，所以该式子小于等于0。
>
>$\therefore a_{n+1}\leqslant a_n$，得到数列单调减少。
>
>由单调有界准则，$\lim\limits_{n\to\infty}a_n$存在并记为$A$。
>
>将$A$代入递推式并两边求极限：$A=\dfrac{1}{2}(A+\dfrac{2}{A})$，得到$A=\pm\sqrt{2}$。
>
>又因为保号性，数列下界为$\sqrt{2}$，所以$A=\sqrt{2}$。

>[!note] **例题：** 求证$x_{n+1}=\sin x_n$极限存在，$0<x_1<\pi$。
>
>解：由三角函数中的不等式$\sin x<x$。
>
>当$n=1$，$\because 0<x_1<\pi$，$\therefore 0<\sin x_1<1$，$\therefore 0<x_2=sin x_1<x<\pi$。
>
>假设$0<x_n=\sin x_{n-1}<\pi$。
>
>$\therefore 0<x_{n+1}=\sin x_n<x_n<\pi$。
>
>故$\{x_n\}\searrow$且有下界0。
>
>$\therefore\lim\limits_{n\to\infty}x_n$存在，并记为$A$。
>
>对两边取极限：$A=\sin A$，所以$A=0$。
>
>$\therefore\lim\limits_{n\to\infty}x_n=0$。

>[!note] **例题：** 证明$a_n=\dfrac{1}{1^2}+\dfrac{1}{2^2}+\cdots+\dfrac{1}{n^2}$存在极限。
>
>证明：因为是递推式，所以一般使用**单调有界准则**。
>
>$a_{n+1}=\dfrac{1}{1^2}+\dfrac{1}{2^2}+\cdots+\dfrac{1}{n^2}+\dfrac{1}{(n+1)^2}$。
>
>$\Rightarrow a_{n+1}-a_n=\dfrac{1}{(n+2)^2}>0\Rightarrow\{a_n\}\nearrow$
>
>$\begin{aligned}a_n & =\dfrac{1}{1\cdot 1}+\dfrac{1}{2\cdot 2}+\cdots+\dfrac{1}{n\cdot n} \\
>    & \text{裂项相消} \\
>    < & 1+\dfrac{1}{1\cdot 2}+\cdots+\dfrac{1}{(n-1)\cdot(n)} \\
>    = & 1+(1-\dfrac{1}{2})+(\dfrac{1}{2}-\dfrac{1}{3})+\cdots+(\dfrac{1}{n-1}-\dfrac{1}{n}) \\
>    = & 2-\dfrac{1}{n} \\
>    < & 2 \text{ （上界）}
>\end{aligned}$
>
>单调增且有上界，所以必然有极限。

## 1.8 证明数列单调性的常用方法
### 1.8.1 作差作商
### 1.8.2 利用数学归纳法
### 1.8.3 利用重要不等式
### 1.8.4 邻项差同号则单调
$x_{n}-x_{n-1}$ 与 $x_{n-1}-x_{n-2}$ 同号，则 $\{x_n\}$ 单调

### 1.8.5 利用结论
>[!note] **结论：**
>对 $x_{n+1}=f(x_n)(n-1,2,\cdots),x_n\in I.$
>1. ==若 $f'(x)\gt0,x\in I$，则数列 $\{x_n\}$单调==，
>且$\begin{cases}当 x_2\gt x_1时，数列\{x_n\}单调增加\\\\当 x_2\lt x_1时，数列\{x_n\}单调增少\end{cases}$
>
>2. ==若 $f'(x)\lt0,x\in I$，则数列 $\{x_n\}$不单调==


