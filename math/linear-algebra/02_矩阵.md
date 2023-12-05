# 矩阵

**方阵**：行列数相等的矩阵

  

**对称矩阵**：元素以主对角线镜像对称的方阵，其转置为本身$A=A^T$

  

**对角矩阵**：除主对角线以外的元素皆为0（不一定是方阵）

$$\boldsymbol{\Lambda}_{n\times n}=\begin{bmatrix}\lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n\end{bmatrix}$$

**n阶单位矩阵**：对角线上为1，其他为0的$n\times n$方阵

$$\boldsymbol{I}_{n\times n}=\begin{bmatrix}1 & 0 & \cdots & 0\\0 & 1 & \cdots & 0\\\vdots & \vdots & \ddots & \vdots\\0 & 0 & \cdots & 1\end{bmatrix}$$

**三角矩阵**：

$$上三角矩阵:\boldsymbol{U}_{n\times n}=\begin{bmatrix}u_{1,1} & u_{1,2} & \cdots & u_{1,n}\\0 & u_{2,2} & \cdots & u_{2,n}\\\vdots & \vdots & \ddots & \vdots\\0 & 0 & \cdots & u_{n,n}\end{bmatrix}$$

$$下三角矩阵：\boldsymbol{L}_{n\times n}=\begin{bmatrix}l_{1,1} & 0 & \cdots & 0\\l_{2,1} & l_{2,2} & \cdots & 0\\\vdots & \vdots & \ddots & \vdots\\l_{n,1} & l_{n,2} & \cdots & l_{n,n}\end{bmatrix}$$

  

## 矩阵运算

### 矩阵加减：

$$\boldsymbol{A}_{m\times n}+\boldsymbol{B}_{m\times n}=\begin{bmatrix}a_{1,1}+b_{1,1} & a_{1,2}+b_{1,2} & \cdots & a_{1,n}+b_{1,n}\\a_{2,1}+b_{2,1} & a_{2,2}+b_{2,2} & \cdots & a_{2,n}+b_{2,n}\\\vdots & \vdots & \ddots & \vdots\\a_{m,1}+b_{m,1} & a_{m,2}+b_{m,2} & \cdots & a_{m,n}+b_{m,n}\end{bmatrix}$$

### 零矩阵:

用$O$表示元素全为0的矩阵，零矩阵参与运算时，注意$O$的形状

### 矩阵标量乘法：

$$k\boldsymbol{X}=\begin{bmatrix}k\cdot x_{1,1} & k\cdot x_{1,2} & \cdots & k\cdot x_{1,D}\\k\cdot x_{2,1} & k\cdot x_{2,2} & \cdots & k\cdot x_{2,D}\\\vdots & \vdots & \ddots & \vdots\\k\cdot x_{n,1} & k\cdot x_{n,2} & \cdots & k\cdot x_{n,D}\end{bmatrix}$$

### 广播原则：

当两个数组的形状并不相同的时候，可以通过广播原则扩展数组来实现相加、相减等操作

  

### 矩阵乘法：

$$\boldsymbol{C}_{n\times m}=\boldsymbol{A}_{n\times D}\boldsymbol{B}_{D\times m}=\boldsymbol{A}_{n\times D}@\boldsymbol{B}_{D\times m}=\begin{bmatrix}c_{1,1} & c_{1,2} & \cdots & c_{1,m}\\c_{2,1} & c_{2,2} & \cdots & c_{2,m}\\\vdots & \vdots & \ddots & \vdots\\c_{n,1} & c_{n,2} & \cdots & c_{n,m}\end{bmatrix}$$

其中：

$$\boldsymbol{A}_{n\times D}=\begin{bmatrix}a_{1,1} & a_{1,2} & \cdots & a_{1,D}\\a_{2,1} & a_{2,2} & \cdots & a_{2,D}\\\vdots & \vdots & \ddots & \vdots\\a_{n,1} & a_{n,2} & \cdots & a_{n,D}\end{bmatrix},\boldsymbol{B}_{D\times m}=\begin{bmatrix}b_{1,1} & b_{1,2} & \cdots & b_{1,m}\\b_{2,1} & b_{2,2} & \cdots & b_{2,m}\\\vdots & \vdots & \ddots & \vdots\\b_{D,1} & b_{D,2} & \cdots & b_{D,m}\end{bmatrix}$$

  

### 矩阵的幂：

$$\boldsymbol{A}^0=\boldsymbol{I}$$

$$\boldsymbol{A}^1=\boldsymbol{A}$$

$$\boldsymbol{A}^2=\boldsymbol{A}\boldsymbol{A}$$

$$\boldsymbol{A}^{n+1}=\boldsymbol{A}^n\boldsymbol{A}$$

### 矩阵逆：

$方阵\boldsymbol{A}如果可逆，仅当存在矩阵\boldsymbol{B}使得:$

$$\boldsymbol{A}\boldsymbol{B}=\boldsymbol{B}\boldsymbol{A}=\boldsymbol{I}$$

$\boldsymbol{B}叫做矩阵\boldsymbol{A}的逆，一般记作\boldsymbol{A}^{-1}$

矩阵**可逆**也称**非奇异**

  

### 正交矩阵：

$$\boldsymbol{A}^T=\boldsymbol{A}^{-1}\Rightarrow \boldsymbol{A}^T\boldsymbol{A}=\boldsymbol{A}\boldsymbol{A}^T=\boldsymbol{I}$$

### 迹：

$n\times n矩阵A的迹为其主对角线元素之和：$

$$tr(\boldsymbol{A})=\sum_{i=1}^na_{i,i}=a_{1,1}+a_{2,2}+\cdots+a_{n,n}$$

### 逐项积（阿达玛乘积）

$$\boldsymbol{A}_{n\times D}\odot\boldsymbol{B}_{n\times D}=\begin{bmatrix}a_{1,1}b_{1,1} & a_{1,2}b_{1,2} & \cdots & a_{1,D}b_{1,D}\\a_{2,1}b_{2,1} & a_{2,2}b_{2,2} & \cdots & a_{2,D}b_{2,D}\\\vdots & \vdots & \ddots & \vdots\\a_{n,1}b_{n,1} & a_{n,2}b_{n,2} & \cdots & a_{n,D}b_{n,D}\end{bmatrix}_{n\times D}$$

### 线性方程组

$$\boldsymbol{Ax=b}$$

$$\underbrace{\begin{bmatrix}a_{1,1} & a_{1,2} & \cdots & a_{1,D}\\a_{2,1} & a_{2,2} & \cdots & a_{2,D}\\\vdots & \vdots & \ddots & \vdots\\a_{n,1} & a_{n,2} & \cdots & a_{n,D}\end{bmatrix}}_{\boldsymbol{A}_{n\times D}}\underbrace{\begin{bmatrix}x_1\\x_2\\\vdots \\x_D\end{bmatrix}}_{\boldsymbol{x}_{D\times1}}=\underbrace{\begin{bmatrix}b_1\\b_2\\\vdots\\b_n\end{bmatrix}}_{\boldsymbol{b}_{n\times1}}$$

恰定方程组：有唯一解$$\boldsymbol{Ax=b}\Rightarrow\boldsymbol{x=A^{-1}b}$$

欠定方程组：有无穷多解

超定方程组：解不存在

特别地，如果$\boldsymbol{A^TA}$可逆，$\boldsymbol{x}$可以通过下式求解：

$$\boldsymbol{Ax=b}\Rightarrow\boldsymbol{A^TAx=A^Tb}\Rightarrow\boldsymbol{x=\underbrace{(A^TA)^{-1}A^Tb}_{A^+}}$$

$(A^TA)^{-1}A^T$常被称作**广义逆**，或**伪逆**

注意，如果$A^TA$非满秩，则$A^TA$不可逆，则需要**摩尔-彭若斯广义逆**

  

### 线性组合

$$A_{n\times D}=\begin{bmatrix}\vec{a_1} & \vec{a_2} & \cdots \vec{a_D}\end{bmatrix}$$

$$\begin{bmatrix}\vec{a_1} & \vec{a_2} & \cdots & \vec{a_D}\end{bmatrix}_{1\times D}\begin{bmatrix}x_1\\x_2\\\vdots\\x_D\end{bmatrix}_{D\times 1}=\vec{b}_{n\times 1}$$

$$x_1\vec{a_1}+x_2\vec{a_2}+\cdots+x_D\vec{a_D}=\vec{b}_{n\times 1}$$

$$x_1\begin{bmatrix}a_{1,1}\\a_{2,1}\\\vdots\\a_{n,1}\end{bmatrix}+x_2\begin{bmatrix}a_{1,2}\\a_{2,2}\\\vdots\\a_{n,2}\end{bmatrix}+\cdots+x_D\begin{bmatrix}a_{1,D}\\a_{2,D}\\\vdots\\a_{n,D}\end{bmatrix}=\begin{bmatrix}b_{1}\\b_{2}\\\vdots\\b_{n}\end{bmatrix}$$

当$x_1、x_2\cdots x_D$取具体值时，上式表示**线性组合**

  

### 线性映射

不同实数空间之间的映射

### 线性变换

相同实数空间之间的映射

  

### 二次型

$$\boldsymbol{x}^TQ\boldsymbol{x}=q$$

  

其中，$Q$为对称阵，$q$为实数

$$\boldsymbol{x}=\begin{bmatrix}x_1\\x_2\\\vdots\\x_D\end{bmatrix},\boldsymbol{Q}=\begin{bmatrix}q_{1,1} & q_{1,2} & \cdots & q_{1,D}\\q_{2,1} & q_{2,2} & \cdots & q_{2,D}\\\vdots & \vdots & \ddots & \vdots\\q_{D,1} & q_{D,2} & \cdots & q_{D,D}\end{bmatrix}$$

$\boldsymbol{x}^TQ\boldsymbol{x}$类似$\boldsymbol{x}^T\boldsymbol{x}$，也表示某种“距离的平方”

$$\boldsymbol{x}^TQ\boldsymbol{x}=\sum_{i=1}^Dq_{i,i}x_i^2+\sum_{i=1}^D\sum_{j=1}^Dq_{i,j}x_{i}x_{j}=q,i\neq j$$

### 高斯分布

$$f_{X_1,X_2}(x_1,x_2)=\frac{1}{2\pi \sigma_1\sigma_2\sqrt{1-\rho_{1,2}^2}}\exp(-\frac{1}{2}(\overbrace{\frac{1}{(1-\rho_{1,2}^2)}((\frac{x_1-\mu_1}{\sigma_1})^2-2\rho_{1,2}(\frac{x_1-\mu_1}{\rho_1})(\frac{x_2-\mu_2}{\rho_2})+(\frac{x_2-\mu_2}{\sigma_2})^2)}^{Ellipse}))$$

  

### 幂等矩阵

方阵$\boldsymbol{A}$如果满足：

$$\boldsymbol{A}^2=\boldsymbol{A}$$

则称$\boldsymbol{A}$为**幂等矩阵**

  

### 对角阵：批量缩放

如果形状相同的方阵$\boldsymbol{A}$和$\boldsymbol{B}$都为对角阵，两者乘积还是一个对角阵：

$$\boldsymbol{A}_{D\times D}\boldsymbol{B}_{D\times D}=\begin{bmatrix}a_1 & & & \\ & a_2 & & \\ & & \ddots &\\ & & & a_D\end{bmatrix}\begin{bmatrix}b_1 & & & \\ & b_2 & & \\ & & \ddots &\\ & & & b_D\end{bmatrix}=\begin{bmatrix}a_1b_1 & & & \\ & a_2b_2 & & \\ & & \ddots &\\ & & & a_Db_D \end{bmatrix}$$

常采用$\boldsymbol{\Lambda}$和$\boldsymbol{S}$代表对角阵

  

### 右乘

矩阵$\boldsymbol{X}$乘$D\times D$对角方阵$\Lambda$：

$$\begin{split}\boldsymbol{X}_{n\times D}\boldsymbol{\Lambda}_{D\times D}=&\begin{bmatrix}\vec{x_1}&\vec{x_2}&\cdots & \vec{x_D}\end{bmatrix}\begin{bmatrix}\lambda_1 & 0 & \cdots & 0\\0 & \lambda_2 & \cdots & 0\\\vdots & \vdots &\ddots & \vdots\\0 & 0 & \cdots & \lambda_D\end{bmatrix}\\&=\begin{bmatrix}\lambda_1\vec{x_1} & \lambda_2\vec{x_2} & \cdots & \lambda_D \vec{x_D}\end{bmatrix}\end{split}$$

$\boldsymbol{\Lambda}$的对角线元素相当于缩放系数，分别对矩阵$\boldsymbol{X}$的每一**列**数值进行不同比例缩放

  

### 左乘

$n\times n$对角阵$Lambda$左乘矩阵$\boldsymbol{X}$：

$$\boldsymbol{\Lambda}_{n\times n}\boldsymbol{X}_{n\times D}=\begin{bmatrix}\lambda_1 & 0 & \cdots & 0\\0 & \lambda_2 & \cdots & 0\\\vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n\end{bmatrix}_{n\times n}\begin{bmatrix}\vec{x}^{(1)}\\\vec{x}^{(2)}\\\vdots\\\vec{x}^{(n)}\end{bmatrix}_{n\times 1}=\begin{bmatrix}\lambda_1\vec{x}^{(1)}\\\lambda_2\vec{x}^{(2)}\\\vdots \\ \lambda_n\vec{x}^{(n)}\end{bmatrix}_{n\times 1}$$

$\boldsymbol{\Lambda}$的对角线元素相当于缩放系数，分别对矩阵$\boldsymbol{X}$的每一**行**数值进行不同比例缩放

  

### 置换矩阵：调换元素顺序

行向量$\vec{a}$乘副对角矩阵，如果副对角线上元素都为1，得到左右翻转的行向量：

$$\begin{bmatrix}a_1 & a_2 & \cdots & a_D\end{bmatrix}_{1\times D}\begin{bmatrix}&&&1\\&&1&\\&\cdots&&\\1&&&\end{bmatrix}_{D\times D}=\begin{bmatrix}a_D & a_{D-1} & \cdots & a_1\end{bmatrix}$$

**置换矩阵**是由0和1组成的方阵。置换矩阵的每一行、每一列都恰好只有一个1，其余元素均为0

  

### 矩阵乘向量：映射到一维

$$\boldsymbol{X}_{n\times D}\boldsymbol{\vec{v}}_{D\times 1}=\boldsymbol{\vec{z}}_{n\times 1}$$

![](/upload/Pasted%20image%2020230720124026.png ':size=70%')

### 矩阵乘矩阵：映射到多维

![](/upload/Pasted%20image%2020230720123950.png ':size=70%')

  

### 格拉姆矩阵

(Gram matrix)

$$G=X^TX=\begin{bmatrix}\vec{x}_1^T\\\vec{x}_2^T\\\vdots\\\vec{x}_D^T\end{bmatrix}\begin{bmatrix}\vec{x}_1 & \vec{x}_2 & \cdots & \vec{x}_D\end{bmatrix}=\begin{bmatrix}\vec{x}_1^T\vec{x}_1 & \vec{x}_1^T\vec{x}_2 & \cdots & \vec{x}_1^T\vec{x}_D\\\vec{x}_2^T\vec{x}_1 & \vec{x}_2^T\vec{x}_2 & \cdots & \vec{x}_2^T\vec{x}_D\\\vdots & \vdots & \ddots & \vdots\\\vec{x}_D^T\vec{x}_1 & \vec{x}_D^T\vec{x}_2 & \cdots & \vec{x}_D^T\vec{x}_D\end{bmatrix}$$

$$(X^TX)_{i,j}=\vec{x}_i^T\vec{x}_j$$

格拉姆矩阵为对称矩阵：

$$G^T=(X^TX)^T=X^TX=G$$

### 爱因斯坦求和约定

  

### 分块矩阵

$$X_{n\times D}=\begin{bmatrix}X_{r\times q} & X_{r\times (D-q)}\\X_{(n-r)\times q} & X_{(n-r)\times(D-q)}\end{bmatrix}$$

转置：

$$A^T=\begin{bmatrix}A_{1,1}^T & A_{2,1}^T\\A_{1,2}^T & A_{2,2}^T\end{bmatrix}$$

标量乘法：

$$k\boldsymbol{A}=\begin{bmatrix}k\boldsymbol{A}_{1,1} & k\boldsymbol{A}_{1,2}\\k\boldsymbol{A}_{2,1} & k\boldsymbol{A}_{2,2}\end{bmatrix}$$

逆：

$$\begin{bmatrix}A & B\\C & D\end{bmatrix}^{-1}=\begin{bmatrix}(A-BD^{-1}C)^{-1} & -(A-BD^{-1}C)^{-1}BD^{-1}\\-D^{-1}C(A-BD^{-1}C)^{-1} & D^{-1}+D^{-1}C(A-BD^{-1}C)^{-1}BD^{-1}\end{bmatrix}$$