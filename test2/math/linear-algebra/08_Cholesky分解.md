
# Cholesky分解：适用于正定矩阵

(Cholesky decomposition, Cholesky factorization）

  

Cholesky 分解 (Cholesky decomposition) 是 LU 分解的特例，适用于**正定矩阵**

  

### 正定矩阵

  

在$\vec{x}$为非零列向量$(\vec{x}\neq 0)$条件下，如果方阵$A$满足：

$$x^TAx\gt 0$$

则称方阵$A$为**正定矩阵**，正定矩阵的特征值均为正

  

正定矩阵都是对称方阵

![](/upload/Pasted%20image%2020230802172917.png ':size=70%')

![](/upload/Pasted%20image%2020230802172933.png ':size=70%')

![](/upload/Pasted%20image%2020230802172953.png ':size=70%')

  
  

Cholesky 分解把矩阵分解为一个下三角矩阵以及它的转置矩阵的乘积：

$$A=LL^T$$

![](/upload/Pasted%20image%2020230729211524.png ':size=70%')

利用上三角矩阵 $R$，Cholesky 分解也可以写成：

$$A=R^TR$$

  

### LDL分解

(lower-diagonal-lower transposed decomposition, LDL/LDLT decomposition)

Cholesky分解可以进一步拓展为**LDL分解**

$$A=LDL^T=LD^{\frac{1}{2}}(D^{\frac{1}{2}})^TL^T=LD^{\frac{1}{2}}(LD^{\frac{1}{2}})^T$$

其中，L 为下三角矩阵，但是对角线元素均为 1；D 为对角矩阵，起到缩放作用；几何角度来看，

L 的作用就是“剪切”。也就是说，矩阵 A 被分解成“剪切 → 缩放 → 剪切

![](/upload/Pasted%20image%2020230729211511.png ':size=70%')

  

令，

$$B=D^{\frac{1}{2}}$$

则，

$$A=LB(LB)^T$$

$LB$相当于$A$的平方根

用上三角矩阵$R$替换$L^T$,则，

$$A=R^TBBR=(BR)^TBR$$

### 几何角度：开合

给定如$2\times 2$矩阵$P$，它的对角线元素为$1$，非对角线元素为余弦值$\cos\theta_{1,2}:$

$$P=\begin{bmatrix}1 & \cos\theta_{1,2}\\\cos\theta_{1,2} & 1\end{bmatrix}$$

对矩阵 $P$ 进行 Cholesky 分解可以得到：

$$P=LL^T=\underbrace{\begin{bmatrix}1 & 0\\\cos\theta_{1,2} & \sin\theta_{1,2}\end{bmatrix}}_L\underbrace{\begin{bmatrix}1 & \cos\theta_{1,2}\\0 & \sin\theta_{1,2}\end{bmatrix}}_{L^T}=\begin{bmatrix}1 & \cos\theta_{1,2}\\\cos\theta_{1,2} & 1\end{bmatrix}$$

利用上三角矩阵$R$，还可以写成：

$$P=R^TR=\underbrace{\begin{bmatrix}1 & 0\\\cos\theta_{1,2} & \sin\theta_{1,2}\end{bmatrix}}_{R^T}\underbrace{\begin{bmatrix}1 & \cos\theta_{1,2}\\0 & \sin\theta_{1,2}\end{bmatrix}}_R$$

将$R$写成：

$$R=\begin{bmatrix}1 & \cos\theta_{1,2}\\0 & \sin\theta_{1,2}\end{bmatrix}=\begin{bmatrix}\vec{r}_1 & \vec{r}_2\end{bmatrix}$$

在平面直角坐标系中，$\vec{e}_1$ 和 $\vec{e}_2$ 分别代表水平和竖直正方向的单位向量，$[\vec{e}_1, \vec{e}_2]$ 是$R^2$空间的标准正交基。$R$分别乘 $\vec{e}_1$ 和 $\vec{e}_2$，得到 $\vec{r}_1$ 和 $\vec{r}_2$：

$$\begin{split}\vec{r}_1&=R\vec{e}_1=\begin{bmatrix}1 & \cos\theta_{1,2}\\0 & \sin\theta_{1,2}\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}1\\0\end{bmatrix}\\\vec{r}_2&=R\vec{e}_2=\begin{bmatrix}1 & \cos\theta_{1,2}\\0 & \sin\theta_{1,2}\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix}=\begin{bmatrix}\cos\theta_{1,2}\\\sin\theta_{1,2}\end{bmatrix}\end{split}$$

很容易判断 $r_1$ 和 $r_2$ 均为单位向量。

而向量 $r_1 和 $r_2$ 夹角余弦值正是 $\cos\theta_{1,2}$：

$$\cos\theta=\frac{\vec{r}_1\cdot\vec{r}_2}{\Vert\vec{r}_1\Vert\Vert\vec{r}_2\Vert}=\cos\theta_{1,2}$$

从几何角度来讲，$R$ 相当于把原本正交的 $[\vec{e}_1, \vec{e}_2]$ 标准正交基转化成具有一定夹角的 $[\vec{r}_1, \vec{r}_2]$ 非正交基，且$\vec{e}_1=\vec{r}_1$，相当于**锚定**

![](/upload/Pasted%20image%2020230802174827.png ':size=70%')

![](/upload/Pasted%20image%2020230802174858.png ':size=70%')

计算中 $R$ 的行列式值：

$$\vert R\vert=\begin{vmatrix}1 & \cos\theta_{1,2}\\0 & \sin\theta_{1,2}\end{vmatrix}=\sin\theta_{1,2}$$

![](/upload/Pasted%20image%2020230802175022.png ':size=70%')

$P=\begin{bmatrix}1 & \cos\theta_{1,2}\\\cos\theta_{1,2} & 1\end{bmatrix}$相当于指定了目标向量夹角的余弦值

$P=\begin{bmatrix}1 & \cos\theta_{1,2} & \cos\theta_{1,3}\\\cos\theta_{1,2} & 1 & \cos\theta_{2,3}\\\cos\theta_{1,3} & \cos\theta_{2,3} & 1\end{bmatrix}$指定了目标向量两两"相对夹角"余弦值$\cos\theta_{1,2}、\cos\theta_{1,3}、\cos\theta_{2,3}$。即$\vec{r}_1$ 和 $\vec{r}_2$ 的相对夹角余弦值为$\cos\theta_{1,2}$，$\vec{r}_1$ 和 $\vec{r}_3$ 的相对夹角余弦值为 $\cos\theta_{1,3}$，$\vec{r}_2$ 和 $\vec{r}_3$ 的相对夹角余弦值为 $\cos\theta_{2,3}$。我们想要找到空间中满足这个条件的三个单位向量

  
  

### 几何变换：缩放—>开合

给定$\Sigma$具体形式如下：

$$\Sigma=\begin{bmatrix}a_2 & a\cdot b\cdot \cos\theta_{1,2}\\a\cdot b\cdot \cos\theta_{1,2} & b^2\end{bmatrix}$$

其中$a$和$b$都是正数

先把$\Sigma$写成：

$$\Sigma=\underbrace{\begin{bmatrix}a\\& b\end{bmatrix}}_S\underbrace{\begin{bmatrix}1 & \cos\theta_{1,2}\\\cos\theta_{1,2} & 1\end{bmatrix}}_P\underbrace{\begin{bmatrix}a\\&b\end{bmatrix}}_S$$

将$\Sigma=(RS)^T(RS)$代入：

$$\Sigma=(RS)^T(RS)=\underbrace{\begin{bmatrix}a&0\\ 0&b\end{bmatrix}}_S\underbrace{\begin{bmatrix}1 & 0\\\cos\theta_{1,2} & \sin\theta_{1,2}\end{bmatrix}}_{R^T}\underbrace{\begin{bmatrix}1 & \cos\theta_{1,2}\\0 & \sin\theta_{1,2}\end{bmatrix}}_R\underbrace{\begin{bmatrix}a&0\\0&b\end{bmatrix}}_S$$

上式相当于对$\Sigma$直接进行Cholesky分解的结果

将 $RS (S 先、R 后)$ 作用在在 $\vec{e}_1$ 和 $\vec{e}_2$ 上，得到 $\vec{x}_1$ 和 $\vec{x}_2$：

$$\begin{split}\vec{x}_1&=RS\vec{e}_1=\begin{bmatrix}1 & \cos\theta_{1,2}\\0 & \sin\theta_{1,2}\end{bmatrix}\begin{bmatrix}a&0\\0&b\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}=a\begin{bmatrix}1\\0\end{bmatrix}\\\vec{x}_2&=RS\vec{e}_2=\begin{bmatrix}1 & \cos\theta_{1,2}\\0 & \sin\theta_{1,2}\end{bmatrix}\begin{bmatrix}a&0\\0&b\end{bmatrix}\begin{bmatrix}0\\1\end{bmatrix}=b\begin{bmatrix}\cos\theta_{1,2}\\\sin\theta_{1,2}\end{bmatrix}\end{split}$$

这相当于，对 $\vec{e}1$ 和 $\vec{e}2$ 先缩放 (S)，再开合 (R)。

![](/upload/Pasted%20image%2020230802180418.png ':size=70%')

$$\cos\theta=\frac{\vec{x}_1\cdot\vec{x}_2}{\Vert\vec{x}_1\Vert\Vert\vec{x}_2\Vert}=\frac{a\cdot b\cdot \cos\theta_{1,2}}{a\cdot b}=\cos\theta_{1,2}$$

发现向量 $\vec{x}_1$ 和 $\vec{x}_2$ 夹角等同于向量 $\vec{r}_1$ 和 $\vec{r}_2$ 夹角。

  

### 相似度矩阵

矩阵$X$的格拉姆矩阵$G$：

$$G=X^TX=\begin{bmatrix}\vec{x}_1^T\\\vec{x}_2^T\\\vdots\\\vec{x}_D^T\end{bmatrix}\begin{bmatrix}\vec{x}_1 & \vec{x}_2 & \cdots & \vec{x}_D\end{bmatrix}=\begin{bmatrix}\vec{x}_1^T\vec{x}_1 & \vec{x}_1^T\vec{x}_2 & \cdots & \vec{x}_1^T\vec{x}_D\\\vec{x}_2^T\vec{x}_1 & \vec{x}_2^T\vec{x}_2 & \cdots & \vec{x}_2^T\vec{x}_D\\\vdots & \vdots & \ddots & \vdots\\\vec{x}_D^T\vec{x}_1 & \vec{x}_D^T\vec{x}_2 & \cdots & \vec{x}_D^T\vec{x}_D\end{bmatrix}$$

![](/upload/Pasted%20image%2020230802193354.png ':size=70%')

对$G$进行Cholesky分解得到：

$$G=R_G^TR_G$$

将$R_G$写成一排列向量：

$$R_G=\begin{bmatrix}\vec{r}_{G,1} & \vec{r}_{G,2} & \cdots & \vec{r}_{G,D}\end{bmatrix}$$

代入$G=R_G^TR_G$得到：

$$G=\begin{bmatrix}\vec{r}_{G,1}^T \\ \vec{r}_{G,2}^T \\ \vdots \\ \vec{r}_{G,D}^T\end{bmatrix}\begin{bmatrix}\vec{r}_{G,1} & \vec{r}_{G,2} & \cdots & \vec{r}_{G,D}\end{bmatrix}=\begin{bmatrix}\vec{r}_{G,1}^T\vec{r}_{G,1} & \vec{r}_{G,1}^T\vec{r}_{G,2} & \cdots & \vec{r}_{G,1}^T\vec{r}_{G,D}\\\vec{r}_{G,2}^T\vec{r}_{G,1} & \vec{r}_{G,2}^T\vec{r}_{G,2} & \cdots & \vec{r}_{G,1}^T\vec{r}_{G,D}\\ \vdots & \vdots & \ddots & \vdots \\ \vec{r}_{G,D}^T\vec{r}_{G,1} & \vec{r}_{G,D}^T\vec{r}_{G,2} & \cdots & \vec{r}_{G,D}^T\vec{r}_{G,D}\end{bmatrix}$$

![](/upload/Pasted%20image%2020230802193405.png ':size=70%')

以向量夹角余弦形式展开$G$中向量积：

$$G=\begin{bmatrix}\Vert\vec{x}_1\Vert\Vert\vec{x}_1\Vert\cos\theta_{1,1} & \Vert\vec{x}_1\Vert\Vert\vec{x}_2\Vert\cos\theta_{1,2} & \cdots &\Vert\vec{x}_1\Vert\Vert\vec{x}_D\Vert\cos\theta_{1,D}\\ \Vert\vec{x}_2\Vert\Vert\vec{x}_1\Vert\cos\theta_{2,1} & \Vert\vec{x}_2\Vert\Vert\vec{x}_2\Vert\cos\theta_{2,2} & \cdots & \Vert\vec{x}_2\Vert\Vert\vec{x}_D\Vert\cos\theta_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ \Vert\vec{x}_D\Vert\Vert\vec{x}_1\Vert\cos\theta_{D,1} & \Vert\vec{x}_D\Vert\Vert\vec{x}_2\Vert\cos\theta_{D,2} & \cdots &\Vert\vec{x}_D\Vert\Vert\vec{x}_D\Vert\cos\theta_{D,D}\end{bmatrix}$$

定义缩放矩阵$S$：

$$S=\begin{bmatrix}\Vert\vec{x}_1\Vert\\&\Vert\vec{x}_2\Vert\\&&\ddots\\&&&\Vert\vec{x}_D\Vert\end{bmatrix}$$

对$G$左右分别乘上$S$的逆，得到$C$：

$$C=S^{-1}GS^{-1}=\begin{bmatrix} \frac{\vec{x}_1\cdot\vec{x}_1}{\Vert\vec{x}_1\Vert\Vert\vec{x}_1\Vert} &
\frac{\vec{x}_1\cdot\vec{x}_2}{\Vert\vec{x}_1\Vert\Vert\vec{x}_2\Vert} & \cdots &
\frac{\vec{x}_1\cdot\vec{x}_D}{\Vert\vec{x}_1\Vert\Vert\vec{x}_D\Vert} \\
\frac{\vec{x}_2\cdot\vec{x}_1}{\Vert\vec{x}_2\Vert\Vert\vec{x}_1\Vert} &
\frac{\vec{x}_2\cdot\vec{x}_2}{\Vert\vec{x}_2\Vert\Vert\vec{x}_2\Vert} & \cdots &
\frac{\vec{x}_2\cdot\vec{x}_D}{\Vert\vec{x}_2\Vert\Vert\vec{x}_D\Vert} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\vec{x}_D\cdot\vec{x}_1}{\Vert\vec{x}_D\Vert\Vert\vec{x}_1\Vert} &
\frac{\vec{x}_D\cdot\vec{x}_2}{\Vert\vec{x}_D\Vert\Vert\vec{x}_2\Vert} & \cdots &
\frac{\vec{x}_D\cdot\vec{x}_D}{\Vert\vec{x}_D\Vert\Vert\vec{x}_D\Vert} \\
\end{bmatrix} $$

矩阵$C$中元素就是向量两两夹角余弦值

  

矩阵$C$即为**余弦相似度矩阵**

  

$$C=\begin{bmatrix}1 & \cos\theta_{1,2} & \cdots &\cos\theta_{1,D}\\ \cos\theta_{2,1} & 1 & \cdots & \cos\theta_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ \cos\theta_{D,1} & \cos\theta_{D,2} & \cdots &1\end{bmatrix}$$

对$C$进行Cholesky分解得到：

$$C=LL^T=R^TR$$

将$R$写成$[\vec{r}_1,\vec{r}_2,\ldots,\vec{r}_D]$,$C$可以写成：

$$C=R^TR=\begin{bmatrix}\vec{r}_{1}^T \\ \vec{r}_{2}^T \\ \vdots \\ \vec{r}_{D}^T\end{bmatrix}\begin{bmatrix}\vec{r}_{1} & \vec{r}_{2} & \cdots & \vec{r}_{D}\end{bmatrix}=\begin{bmatrix}1 & \cos\theta_{1,2} & \cdots &\cos\theta_{1,D}\\ \cos\theta_{2,1} & 1 & \cdots & \cos\theta_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ \cos\theta_{D,1} & \cos\theta_{D,2} & \cdots &1\end{bmatrix}$$

鸢尾花数据矩阵的格拉姆矩阵 $G$，先转化成相似度矩阵 $C$，再转化成角度矩阵。角度越小说明特征越相似。

![](/upload/Pasted%20image%2020230802195649.png ':size=70%')