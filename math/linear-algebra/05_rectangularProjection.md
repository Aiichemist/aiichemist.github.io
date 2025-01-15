
# 正交投影

![](/img/Pasted%20image%2020230725155453.png ':size=70%')

## 标量投影

$\vec{x}在\vec{v}方向上的标量投影为向量\vec{z}的长度(向量模)，记作s$

$$\vec{z}=s\frac{\vec{v}}{\Vert\vec{v}\Vert}$$

即：

$$(\vec{x}-\vec{z})^T\vec{v}=0$$

$$(\vec{x}-s\frac{\vec{v}}{\Vert\vec{v}\Vert})^T\vec{v}=0$$

得：

$$s=\frac{\vec{x}^T\vec{v}}{\Vert\vec{v}\Vert}$$

$注意，\vec{x}和\vec{v}为等行数列向量$

$特别地，如果\vec{v}本身就是单位向量，则：$

$$s=\vec{x}^T\vec{v}=\vec{v}^T\vec{x}=\vec{x}\cdot\vec{v}=\vec{v}\cdot\vec{x}=\langle\vec{x},\vec{v}\rangle$$

## 向量投影

$$proj_{\vec{v}}(\vec{x})=s\frac{\vec{v}}{\Vert\vec{v}\Vert}=\frac{\vec{x}\cdot\vec{v}}{\vec{v}\cdot\vec{v}}\vec{v}=\frac{\langle\vec{x},\vec{v}\rangle}{\langle\vec{v},\vec{v}\rangle}\vec{v}$$

$特别地，如果\vec{v}为单位向量，\vec{x}在\vec{v}方向上的向量投影则可以写成：$

$$proj_{\vec{v}}(\vec{x})=(\vec{x}^T\vec{v})\vec{v}=(\vec{x}\cdot\vec{v})\vec{v}=(\vec{v}\cdot\vec{x})\vec{v}=\langle\vec{x},\vec{v}\rangle\vec{v}$$

## 投影矩阵

$$proj_{\vec{v}}(\vec{x})=(\vec{v}^T\vec{x})\vec{v}=\vec{v}(\vec{v}^T\vec{x})=\vec{v}\vec{v}^T\vec{x}=(\vec{v}\otimes\vec{v})\vec{x}$$

$称\vec{v}\otimes\vec{v}为$**投影矩阵**

  

## 正交矩阵

满足下式的方阵V为**正交矩阵**：

$$V^TV=I$$

V为方阵是前提

## 规范正交基性质

#### 向量长度不变

$$\begin{split}\Vert V^T\vec{x}\Vert_2^2&=V^T\vec{x}\cdot V^T\vec{x}=(V^T\vec{x})^T(V^T\vec{x})=\vec{x}^TV^TV\vec{x}\\&=\vec{x}^TI\vec{x}=\vec{x}^T\vec{x}=\vec{x}\cdot\vec{x}=\Vert x\Vert_2^2\end{split}$$

#### 夹角不变

$$\frac{\vec{z}_i\cdot\vec{z}_j}{\Vert \vec{z}_i\Vert\Vert \vec{z}_j\Vert}=\frac{\vec{z}_i\cdot\vec{z}_j}{\Vert \vec{x}_i\Vert\Vert \vec{x}_j\Vert}=\frac{V^T\vec{x}_i\cdot V^T\vec{x}_j}{\Vert \vec{x}_i\Vert\Vert \vec{x}_j\Vert}=\frac{(V^T\vec{x}_i)V^T\vec{x}_j}{\Vert \vec{x}_i\Vert\Vert \vec{x}_j\Vert}=\frac{\vec{x}_i^T\vec{x}_j}{\Vert \vec{x}_i\Vert\Vert \vec{x}_j\Vert}=\frac{\vec{x}_i\cdot\vec{x}_j}{\Vert \vec{x}_i\Vert\Vert \vec{x}_j\Vert}$$

#### 行列式值

$$(det(V))^2=det(V^T)det(V)=det(V^TV)=det(I)=1$$

  

## 豪斯霍尔德矩阵

$$H=I-2\vec{v}\otimes \vec{v}$$

豪斯霍尔德反射：

![](/img/Pasted%20image%2020230726124748)

$$\vec{z}=2\vec{p}-\vec{x}=2(\vec{\tau}\otimes \vec{\tau})\vec{x}-\vec{x}=(2\vec{\tau}\otimes\vec{\tau}-I)\vec{x}=\underbrace{(I-2\vec{v}\otimes \vec{v})}_H\vec{x}$$

  

## 格拉姆-斯密特正交化

正交化过程如下：

$$\begin{split}\vec{\eta}_1&=\vec{x}_1\\\vec{\eta}_2&=\vec{x}_2-proj_{\vec{\eta}_1}(\vec{x}_2)\\\vec{\eta}_3&=\vec{x}_3-proj_{\vec{\eta}_1}(\vec{x}_3)-proj_{\vec{\eta}_2}(\vec{x}_3)\\\ldots\\\vec{\eta}_D&=\vec{x}_D-\sum_{j=1}^{D-1}proj_{\vec{\eta}_j}(\vec{x}_D)\end{split}$$

$其中\vec{\eta}_2为\vec{\eta}_1的$**正交补**

然后单位化：

$$\vec{q}_1=\frac{\vec{\eta}_1}{\Vert\vec{\eta}_1\Vert},\vec{q}_2=\frac{\vec{\eta}_2}{\Vert\vec{\eta}_2\Vert},\vec{q}_3=\frac{\vec{\eta}_3}{\Vert\vec{\eta}_3\Vert},\cdots,\vec{q}_D=\frac{\vec{\eta}_D}{\Vert\vec{\eta}_D\Vert}$$

## 一元线性回归

![](/img/Pasted%20image%2020230726130639)

$$\hat{y}=\vec{x}(\vec{x}^T\vec{x})^{-1}\vec{x}^T\vec{y}$$

$残差\varepsilon=\vec{y}-\hat{y}$

## 二元线性回归

![](/img/Pasted%20image%2020230726131833)

$$\hat{y}=b_1\vec{x}_1+b_2\vec{x}_2=\underbrace{\begin{bmatrix}\vec{x}_1&\vec{x}_2\end{bmatrix}}_X\begin{bmatrix}b_1\\b_2\end{bmatrix}=Xb$$

$$b=(X^TX)^{-1}X^Ty$$

$$\hat{y}=X(X^TX)^{-1}X^Ty$$

$X(X^TX)^{-1}X^T常被称作$**帽子矩阵**

## 多元线性回归

![](/img/Pasted%20image%2020230726131818)

$$\hat{y}=b_1\vec{x}_1+b_2\vec{x}_2+\cdots+b_D\vec{x}_D$$

$$\hat{y}=b_0\vec{1}+b_1\vec{x}_1+b_2\vec{x}_2+\cdots+b_D\vec{x}_D$$

$X=[x_0,x_1,x_2,\ldots,x_D]叫作设计矩阵,其中x_0=1$

## 多项式回归

例如一元三次多项式回归模型：

$$\hat{y}=b_0+b_1x+b_2x^2+b_3x^3$$

此时设计矩阵X为:

$$X=\begin{bmatrix}1 & x_1 & x_1^2 & x_1^3\\1 & x_2 & x_2^2 & x_2^3\\\vdots & \vdots & \vdots & \vdots\\1 & x_n & x_n^2 & x_n^3\end{bmatrix}_{n\times 4}$$

$$\vec{b}=\begin{bmatrix}b_0\\b_1\\b_2\\b_3\end{bmatrix}=(X^TX)^{-1}X^Ty$$