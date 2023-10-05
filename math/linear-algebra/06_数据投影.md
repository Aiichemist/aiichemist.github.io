# 数据投影

![](/upload/Pasted%20image%2020230726151135.png ':size=70%')

## 层层叠加

$$\begin{split}X&=XVV^T=X\begin{bmatrix}\vec{v}_1 & \vec{v}_2 & \cdots & \vec{v}_D\end{bmatrix}\begin{bmatrix}\vec{v}_1^T\\\vec{v}_2^T\\\vdots\\\vec{v}_D^T\end{bmatrix}\\&=\underbrace{X\vec{v}_1\vec{v}_1^T}_{X_1}+\underbrace{X\vec{v}_2\vec{v}_2^T}_{X_2}+\cdots+\underbrace{X\vec{v}_D\vec{v}_D^T}_{X_D}\end{split}$$

  
即，
$$X=X_1+X_2+\cdots+X_D$$

![](/upload/Pasted%20image%2020230726134055.png ':size=70%')

  

## 二次投影

取$X_j$中第$i$行行向量$\vec{x}_j^{(i)}$

$$\vec{x}_j^{(i)}=\vec{x}^{(i)}\vec{v}_j\vec{v}_j^T=\vec{z}_{i,j}\vec{v}_j^T$$

上式中$\vec{z}_{i,j}$就是$\vec{x}^{(i)}$正交投影到子空间$span(\vec{v}_j)$对应的坐标点，这是第一次投影

而$\vec{z}_{i,j}\vec{v}_j^T$得到的是$\vec{z}_{i,j}在R^D$的坐标点，这便是第二次投影

$\vec{x}^{(i)}\rightarrow\vec{z}_{i,j}$表示标量投影，$\vec{x}^{(i)}\rightarrow\vec{x}^{(i)}\vec{v}_j\vec{v}_j^T$表示向量投影

![](/upload/Pasted%20image%2020230726151429.png ':size=70%')

第一次投影由$A(5,2)$投到$\vec{v}_1$方向，为标量投影$\approx5.33\vec{v}$

第二次投影由$5.33\vec{v}_1$得到其在$span(\vec{e}_1,\vec{e}_2)$上的坐标$\approx\begin{bmatrix}4.616 & 2.665\end{bmatrix}$

得到$X_{150\times 2}$在$\vec{v}_1$二次投影结果$X_1$为：

$$X_1=X\vec{v}_1\otimes\vec{v}_1=X\vec{v}_1$$