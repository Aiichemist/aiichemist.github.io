# 向量
## 余弦相似度
$$k(\vec{x},\vec{q})=\frac{\vec{x}·\vec{q}}{\Vert\vec{x}\Vert\Vert\vec{q}\Vert}=\frac{\vec{x}^\mathrm{T}\vec{q}}{\Vert\vec{x}\Vert\Vert\vec{q}\Vert}$$
## 余弦距离
$$d(\vec{x},\vec{q})=1-k(\vec{x},\vec{q})=1-\frac{\vec{x}·\vec{q}}{\Vert\vec{x}\Vert\Vert\vec{q}\Vert}$$
## 向量积（叉积）
$$\vec{a} \times\vec{b}$$
$$\Vert \vec{a}\times\vec{b}\Vert=\Vert\vec{a}\Vert\Vert\vec{b}\Vert\sin\theta$$
## 逐项积（阿达玛乘积）
$$\vec{a}\odot\vec{b}=\begin{bmatrix}a_1b_1 & a_2b_2& \cdots a_nb_n\end{bmatrix}^\mathrm{T}$$
## 张量积（克罗内克积）
$$\begin{split}\vec{a}\otimes\vec{b}=\begin{bmatrix}a_1\\a_2\\\vdots\\a_n\end{bmatrix}_{n\times1}\otimes\begin{bmatrix}b_1\\b_2\\\vdots\\b_m\end{bmatrix}_{m\times1}&=\vec{a}\vec{b}^\mathrm{T}=\begin{bmatrix}a_1\\a_2\\\vdots\\a_n\end{bmatrix}\begin{bmatrix}b_1\\b_2\\\vdots\\b_m\end{bmatrix}^\mathrm{T} \\&=\begin{bmatrix}a_1b_1 & a_1b_2 & \cdots & a_1b_n\\a_2b_1 & a_2b_2 &  \cdots & a_2b_m\\\vdots & \vdots  & \ddots & \vdots \\a_nb_1 & a_nb_2 & \cdots & a_nb_m \end{bmatrix}_{n\times m}\end{split}$$
克罗内克积性质：
$$A\otimes(B+C)=A\otimes B+A\otimes C$$
$$(B+C)\otimes A=B\otimes A+C\otimes A$$
$$(kA)\otimes B=A\otimes (kB)=k(A\otimes B)$$
$$(A\otimes B)\otimes C=A\otimes (B\otimes C)$$
$$A\otimes \boldsymbol{0}=\boldsymbol{0}\otimes A=\boldsymbol{0}$$
## Lp范数
$$\Vert\vec{x}\Vert_p=(\vert x_1\vert^p+\vert x_2 \vert^p+\cdots \vert x_D\vert^p)^{\frac{1}{p}}=(\sum_{j=1}^{D}\vert x_j\vert^p)^\frac{1}{p}$$
$给定列向量\vec{x}=\begin{bmatrix}x_1 & x_2\end{bmatrix}^\mathrm{T},当p一定时，将上式写成二元函数 f(x_1, x_2):$
$$f(x_1,x_2)=(\vert x_1\vert^p+\vert x_2\vert^p)^\frac{1}{p}$$
$p=1时，f(x_1, x_2)函数的等高线为旋转正方形：(曼哈顿距离)$
$$f(x_1,x_2)=\vert x_1\vert+\vert x_2\vert$$
$p=2时，f(x_1, x_2) 函数的等高线为正圆：（欧几里得距离）$
$$f(x_1,x_2)=\sqrt{x_1^2+x_2^2}$$
$p=+\infty时，f(x_1, x_2) 函数等高线为正方形：（切比雪夫距离）$
$$f(x_1,x_2)=max(\vert x_1\vert,\vert x_2\vert)$$
![](/upload/Pasted%20image%2020230705104155.png ':size=40%')

| 距离度量 | 定义 | 平面直角坐标系中等距线 |
| ----- | ----- | ----- |
| 欧氏距离 | $\sqrt{(\vec{x}-\vec{q})^T(\vec{x}-\vec{q})}$ | ![](/upload/Pasted%20image%2020230705104920.png ':size=30%') |
| 标准化欧氏距离 | $\sqrt{(\vec{x}-\vec{q})^TD^{-1}(\vec{x}-\vec{q})}$ | ![](/upload/Pasted%20image%2020230705105517.png ':size=30%') |
| 马氏距离 | $\sqrt{(\vec{x}-\vec{q})^T\sum^{-1}(\vec{x}-\vec{q})}$ | ![](/upload/Pasted%20image%2020230705105727.png ':size=30%') |
| 曼哈顿距离 | $\Vert \vec{x}-\vec{q}\Vert_1$ | ![](/upload/Pasted%20image%2020230705110441.png ':size=30%') |
| 切比雪夫距离 | $\Vert \vec{x}-\vec{q}\Vert_\infty$ | ![](/upload/Pasted%20image%2020230705110455.png ':size=30%') |
| 闵氏距离 | $\Vert\vec{x}-\vec{q}\Vert_p$ | ![](/upload/Pasted%20image%2020230705110509.png ':size=70%') |

## 三角不等式（闵可夫斯基不等式）
$$\Vert \vec{x}+\vec{y}\Vert_p\le\Vert\vec{x}\Vert_p+\Vert\vec{y}\Vert_p$$

## 高斯核函数
**二元高斯函数:**
$$f(x_1,x_2)=\exp(-\gamma(x_1^2+x_2^2))$$
![](/upload/Pasted%20image%2020230705110729.png ':size=70%')
**高斯核函数(径向基核函数)：**
$$\kappa_{RBF}(\vec{x},\vec{q})=\exp(-\gamma\Vert\vec{x}-\vec{q}\Vert_2^2)=\exp(-\gamma\Vert\vec{x}-\vec{q}\Vert^2)$$
$$\kappa_{RBF}(\vec{x},\vec{q})=\exp(-\frac{\Vert\vec{x}-\vec{q}\Vert^2}{2\sigma^2})$$