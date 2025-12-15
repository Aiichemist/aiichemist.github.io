# 附录

## 附录A 线性代数

附录A列出了一些有用的性质以及涉及矩阵和行列式的恒等式。我们假设读者已经熟悉了基本的线性代数知识，所以附录A并不是介绍这些概念的教程。对于某些结论，我们会说明如何进行证明，而在更复杂的情况下，建议感兴趣的读者参考与该主题有关的教科书。在所有情况下，我们假设矩阵的逆存在，并且假设矩阵的维度可以使公式得到正确定义。关于线性代数的更详细且全面的讨论可以参见Golub and Van Loan（1996），并且Lutkepohl（1996）给出了大量的矩阵性质，Magnus and Neudecker（1999）讨论了矩阵导数。

### A.1 矩阵恒等式

矩阵  $\pmb{A}$  由元素  $A_{ij}$  组成，其中  $i$  索引行， $j$  索引列。我们用  $\pmb{I}_N$  来表示  $N\times N$  恒等矩

阵（也称单位矩阵）。当维度上没有歧义的时候，我们可以直接用  $\pmb{I}$  代表单位矩阵。转置矩阵  $A^{\mathrm{T}}$  包含元素  $\left(A^{\mathrm{T}}\right)_{ij} = A_{ji}$  。从转置的定义来看，我们有

$$
\left(\boldsymbol {A} \boldsymbol {B}\right) ^ {\mathrm {T}} = \boldsymbol {B} ^ {\mathrm {T}} \boldsymbol {A} ^ {\mathrm {T}} \tag {A.1}
$$

对此我们可以通过写出索引来验证。  $A$  的逆矩阵记为  $A^{-1}$  ，满足

$$
\boldsymbol {A} \boldsymbol {A} ^ {- 1} = \boldsymbol {A} ^ {- 1} \boldsymbol {A} = \boldsymbol {I} \tag {A.2}
$$

因为  $ABB^{-1}A^{-1} = I$  ，所以有

$$
\left(\boldsymbol {A} \boldsymbol {B}\right) ^ {- 1} = \boldsymbol {B} ^ {- 1} \boldsymbol {A} ^ {- 1} \tag {A.3}
$$

并且有

$$
\left(\boldsymbol {A} ^ {\mathrm {T}}\right) ^ {- 1} = \left(\boldsymbol {A} ^ {- 1}\right) ^ {\mathrm {T}} \tag {A.4}
$$

这通过取式（A.2）的转置并应用式（A.1）可以很容易得证。

一个涉及矩阵的逆的有用恒等式如下：

$$
\left(\boldsymbol {P} ^ {- 1} + \boldsymbol {B} ^ {\mathrm {T}} \boldsymbol {R} ^ {- 1} \boldsymbol {B}\right) ^ {- 1} \boldsymbol {B} ^ {\mathrm {T}} \boldsymbol {R} ^ {- 1} = \boldsymbol {P} \boldsymbol {B} ^ {\mathrm {T}} \left(\boldsymbol {B} \boldsymbol {P} \boldsymbol {B} ^ {\mathrm {T}} + \boldsymbol {R}\right) ^ {- 1} \tag {A.5}
$$

这很容易通过对式（A.5）的两边右乘  $\left(\boldsymbol{BP}\boldsymbol{B}^{\mathrm{T}} + \boldsymbol{R}\right)$  得证。假设  $\pmb{P}$  的维度为  $N\times N$  、 $\pmb{R}$  的维度为  $M\times M$  ，则  $\pmb{B}$  的维度就是  $M\times N$  。如果  $M\ll N$  ，那么计算式（A.5）右边的值要比计算其左边的值开销小得多。有时也会出现如下特殊情况：

$$
\left(\boldsymbol {I} + \boldsymbol {A} \boldsymbol {B}\right) ^ {- 1} \boldsymbol {A} = \boldsymbol {A} \left(\boldsymbol {I} + \boldsymbol {B} \boldsymbol {A}\right) ^ {- 1} \tag {A.6}
$$

另一个有用的关于矩阵的逆的恒等式如下：

$$
\left(\boldsymbol {A} + \boldsymbol {B} \boldsymbol {D} ^ {- 1} \boldsymbol {C}\right) ^ {- 1} = \boldsymbol {A} ^ {- 1} - \boldsymbol {A} ^ {- 1} \boldsymbol {B} \left(\boldsymbol {D} + \boldsymbol {C A} ^ {- 1} \boldsymbol {B}\right) ^ {- 1} \boldsymbol {C A} ^ {- 1} \tag {A.7}
$$

这就是伍德伯里恒等式，对式（A.7）的两边同时乘以  $\left(A + BD^{-1}C\right)$  即可得证。这在某些情况下很有用，例如，当  $A$  很大且为对角矩阵时， $A$  的逆很容易求得，或者当  $B$  有很多行但列较少时（ $C$  则相反），式（A.7）右边的计算开销就比左边的计算开销小得多。

如果当且仅当所有  $\alpha_{n} = 0$  时， $\sum_{n}\alpha_{n}\pmb{a}_{n} = \mathbf{0}$  才成立，则称向量  $\{\pmb{a}_1,\dots ,\pmb{a}_N\}$  是线性无关（linearly independent）的。这意味着没有任何一个向量可以表示为其余向量的线性组合。矩阵的秩等于线性无关行的最大数量（也可等价地说，矩阵的秩等于线性无关列的最大数量）。

### A.2 迹和行列式

方阵有迹和行列式。矩阵  $\pmb{A}$  的迹  $\operatorname{tr}(A)$  定义为其主对角线上元素的和。通过写出

索引，我们可以得到

$$
\operatorname {t r} (\boldsymbol {A B}) = \operatorname {t r} (\boldsymbol {B A}) \tag {A.8}
$$

通过将式（A.8）应用于三个矩阵的乘积，可以得到

$$
\operatorname {t r} (\boldsymbol {A B C}) = \operatorname {t r} (\boldsymbol {C A B}) = \operatorname {t r} (\boldsymbol {B C A}) \tag {A.9}
$$

这就是迹运算的循环（cyclic）性质。显然，这一性质可以扩展到任意数量的矩阵的乘积。  $N\times N$  矩阵  $\pmb{A}$  的行列式  $|\pmb {A}|$  由下式定义：

$$
\left| \boldsymbol {A} \right| = \sum (\pm 1) A _ {1 i _ {1}} A _ {2 i _ {2}} \dots A _ {N i _ {N}} \tag {A.10}
$$

其中，行列式对每行和每列中元素的乘积求和，并根据排列  $i_1i_2\dots i_N$  为偶数还是奇数来确定系数是  $+1$  还是  $-1$  。注意  $|I| = 1$  ，且对角矩阵的行列式是由其主对角线上元素的乘积给出的。因此，对于  $2\times 2$  的矩阵，其行列式的形式如下：

$$
\left| \boldsymbol {A} \right| = \left| \begin{array}{l l} a _ {1 1} & a _ {1 2} \\ a _ {2 1} & a _ {2 2} \end{array} \right| = a _ {1 1} a _ {2 2} - a _ {1 2} a _ {2 1} \tag {A.11}
$$

两个矩阵乘积的行列式由下式给出：

$$
\left| \boldsymbol {A} \boldsymbol {B} \right| = \left| \boldsymbol {A} \right| \left| \boldsymbol {B} \right| \tag {A.12}
$$

式（A.12）可以从式（A.10）推导得出。同样，逆矩阵的行列式由下式给出：

$$
\left| \boldsymbol {A} ^ {- 1} \right| = \frac {1}{\left| \boldsymbol {A} \right|} \tag {A.13}
$$

式（A.13）可以通过取式（A.2）的行列式并应用式（A.12）推导得出。

如果  $\pmb{A}$  和  $\pmb{B}$  都是大小为  $N\times M$  的矩阵，则有

$$
\left| \boldsymbol {I} _ {N} + \boldsymbol {A} \boldsymbol {B} ^ {\mathrm {T}} \right| = \left| \boldsymbol {I} _ {M} + \boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {B} \right| \tag {A.14}
$$

一种有用的特殊情况如下：

$$
\left| \boldsymbol {I} _ {N} + \boldsymbol {a} \boldsymbol {b} ^ {\mathrm {T}} \right| = 1 + \boldsymbol {a} ^ {\mathrm {T}} \boldsymbol {b} \tag {A.15}
$$

其中  $\pmb{a}$  和  $\pmb{b}$  都是  $N$  维的列向量。

### A.3 矩阵导数

有时我们需要考虑向量和矩阵关于标量的导数。向量  $\pmb{a}$  关于标量  $x$  的导数也是一个向量，其分量为

$$
\left(\frac {\partial \boldsymbol {a}}{\partial x}\right) _ {i} = \frac {\partial a _ {i}}{\partial x} \tag {A.16}
$$

矩阵关于标量的导数也有类似的定义。标量  $x$  关于向量  $\pmb{a}$  的导数可以定义为

$$
\left(\frac {\partial x}{\partial \boldsymbol {a}}\right) _ {i} = \frac {\partial x}{\partial a _ {i}} \tag {A.17}
$$

类似地：

$$
\left(\frac {\partial \boldsymbol {a}}{\partial \boldsymbol {b}}\right) _ {i j} = \frac {\partial a _ {i}}{\partial b _ {j}} \tag {A.18}
$$

通过写出分量，易证

$$
\frac {\partial}{\partial \boldsymbol {x}} \left(\boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {a}\right) = \frac {\partial}{\partial \boldsymbol {x}} \left(\boldsymbol {a} ^ {\mathrm {T}} \boldsymbol {x}\right) = \boldsymbol {a} \tag {A.19}
$$

类似地：

$$
\frac {\partial}{\partial x} (\boldsymbol {A B}) = \frac {\partial \boldsymbol {A}}{\partial x} \boldsymbol {B} + \boldsymbol {A} \frac {\partial \boldsymbol {B}}{\partial x} \tag {A.20}
$$

矩阵的逆关于标量  $x$  的导数可以定义为

$$
\frac {\partial}{\partial x} \left(\boldsymbol {A} ^ {- 1}\right) = - \boldsymbol {A} ^ {- 1} \frac {\partial \boldsymbol {A}}{\partial x} \boldsymbol {A} ^ {- 1} \tag {A.21}
$$

这可以通过使用式（A.20）微分方程  $A^{-1}A = I$ ，然后右乘  $A^{-1}$  得证。

类似地：

$$
\frac {\partial}{\partial x} \ln | A | = \operatorname {t r} \left(A ^ {- 1} \frac {\partial A}{\partial x}\right) \tag {A.22}
$$

我们稍后会证明这一点。如果我们选择  $x$  作为  $A$  的元素之一，则有

$$
\frac {\partial}{\partial A _ {i j}} \operatorname {t r} (A B) = B _ {j i} \tag {A.23}
$$

其可以通过使用索引表示法写出矩阵得到。这个结论也可以写成如下更紧凑的形式：

$$
\frac {\partial}{\partial \boldsymbol {A}} \operatorname {t r} (\boldsymbol {A} \boldsymbol {B}) = \boldsymbol {B} ^ {\mathrm {T}} \tag {A.24}
$$

通过使用这种表示法，我们可以得到以下性质：

$$
\frac {\partial}{\partial \boldsymbol {A}} \operatorname {t r} \left(\boldsymbol {A} ^ {\mathrm {T}} \boldsymbol {B}\right) = \boldsymbol {B} \tag {A.25}
$$

$$
\frac {\partial}{\partial \boldsymbol {A}} \operatorname {t r} (\boldsymbol {A}) = \boldsymbol {I} \tag {A.26}
$$

$$
\frac {\partial}{\partial \boldsymbol {A}} \operatorname {t r} \left(\boldsymbol {A} \boldsymbol {B} \boldsymbol {A} ^ {\mathrm {T}}\right) = \boldsymbol {A} \left(\boldsymbol {B} + \boldsymbol {B} ^ {\mathrm {T}}\right) \tag {A.27}
$$

这些性质同样可以通过写出矩阵的索引得证。另外，我们还可得到

$$
\frac {\partial}{\partial \boldsymbol {A}} \ln | \boldsymbol {A} | = \left(\boldsymbol {A} ^ {- 1}\right) ^ {\mathrm {T}} \tag {A.28}
$$

式（A.28）可以由式（A.22）和式（A.24）推导得出。

### A.4 特征向量

对于大小为  $M \times M$  的方阵  $\mathbf{A}$ ，其特征向量方程定义为

$$
\boldsymbol {A} \boldsymbol {u} _ {i} = \lambda_ {i} \boldsymbol {u} _ {i} \tag {A.29}
$$

其中  $i = 1,\dots ,M,u_{i}$  是特征向量，  $\lambda_{i}$  是对应的特征值。这可以视为一组  $M$  个同时满足的齐次线性方程，有解的条件是

$$
\left| \boldsymbol {A} - \lambda_ {i} \boldsymbol {I} \right| = 0 \tag {A.30}
$$

这就是特征方程。因为这是一个关于  $\lambda_{i}$  的  $M$  阶多项式，所以它必须有  $M$  个解（尽管这些解不必都是不同的）。 $A$  的秩等于非零特征值的数量。

需要特别讨论的是对称矩阵，对称矩阵可能是协方差矩阵、核矩阵或黑塞矩阵。对称矩阵具有性质  $A_{ij} = A_{ji}$ ，或等价地有  $\mathbf{A}^{\mathrm{T}} = \mathbf{A}$  。对称矩阵的逆矩阵也是对称矩阵，这可以通过取  $\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$  的转置并利用  $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$  以及  $\mathbf{I}$  的对称性得证。

通常，矩阵的特征值是复数，但对于对称矩阵，其特征值  $\lambda_{i}$  是实数。为了证明这一点，可以首先对式（A.29）左乘  $\left(\pmb{u}_i^*\right)^{\mathrm{T}}$  （其中  $*$  表示复共轭），从而得到

$$
\left(\boldsymbol {u} _ {i} ^ {*}\right) ^ {\mathrm {T}} \boldsymbol {A} \boldsymbol {u} _ {i} = \lambda_ {i} \left(\boldsymbol {u} _ {i} ^ {*}\right) ^ {\mathrm {T}} \boldsymbol {u} _ {i} \tag {A.31}
$$

接下来取式（A.29）的复共轭并左乘  $\pmb{u}_i^{\mathrm{T}}$  ，从而得到

$$
\boldsymbol {u} _ {i} ^ {\mathrm {T}} \boldsymbol {A} \boldsymbol {u} _ {i} ^ {*} = \lambda_ {i} ^ {*} \boldsymbol {u} _ {i} ^ {\mathrm {T}} \boldsymbol {u} _ {i} ^ {*} \tag {A.32}
$$

其中使用了  $A^{*} = A$  ，因为我们仅考虑实矩阵  $\pmb{A}$  。对式（A.32）取转置并使用 $A^{\mathrm{T}} = A$  ，我们看到两个方程的左边相等，因此有  $\lambda_i^* = \lambda_i$  ，即  $\lambda_{i}$  必须是实数。

实对称矩阵的特征向量  $\pmb{u}_i$  可以选择为正交归一化的向量（即正交的且为单位长度的向量），从而有

$$
\boldsymbol {u} _ {i} ^ {\mathrm {T}} \boldsymbol {u} _ {j} = I _ {i j} \tag {A.33}
$$

其中  $I_{ij}$  是单位矩阵  $\pmb{I}$  的元素。为了说明这一点，首先将式（A.29）左乘  $\pmb{u}_j^{\mathrm{T}}$  ，从而得到第一个方程

$$
\boldsymbol {u} _ {j} ^ {\mathrm {T}} \boldsymbol {A} \boldsymbol {u} _ {i} = \lambda_ {i} \boldsymbol {u} _ {j} ^ {\mathrm {T}} \boldsymbol {u} _ {i} \tag {A.34}
$$

然后交换索引，从而得到第二个方程：

$$
\boldsymbol {u} _ {i} ^ {\mathrm {T}} \boldsymbol {A} \boldsymbol {u} _ {j} = \lambda_ {j} \boldsymbol {u} _ {i} ^ {\mathrm {T}} \boldsymbol {u} _ {j} \tag {A.35}
$$

取第二个方程的转置并利用对称性质  $A^{\mathrm{T}} = A$  ，最后将这两个方程相减，从而得到

$$
\left(\lambda_ {i} - \lambda_ {j}\right) \boldsymbol {u} _ {i} ^ {\mathrm {T}} \boldsymbol {u} _ {j} = 0 \tag {A.36}
$$

因此，对于  $\lambda_{i} \neq \lambda_{j}$ ，我们有  $\boldsymbol{u}_{i}^{\mathrm{T}}\boldsymbol{u}_{j} = 0$ ，即  $\boldsymbol{u}_{i}$  和  $\boldsymbol{u}_{j}$  是正交的。如果这两个特征值相等，那么任何线性组合  $\alpha \boldsymbol{u}_{i} + \beta \boldsymbol{u}_{j}$  都将是具有相同特征值的特征向量，因此我们可以任意选择一个线性组合，然后选择另一个线性组合并使其与第一个线性组合正交（可以证明退化的特征向量永远不是线性相关的）。因此，特征向量可以选择为正交向量，并通过归一化变为单位长度。由于有  $M$  个特征值，而对应的  $M$  个正交特征向量构成了一个完备集，因此任何  $M$  维向量都可以表示为特征向量的线性组合。

我们可以将特征向量  $\pmb{u}_j$  作为  $M\times M$  矩阵  $\pmb{U}$  的列，该矩阵由于正交归一性而满足

$$
\boldsymbol {U} ^ {\mathrm {T}} \boldsymbol {U} = \boldsymbol {I} \tag {A.37}
$$

这样的矩阵称为正交矩阵。有趣的是，这个矩阵的行也是正交的，所以有  $\mathbf{U}\mathbf{U}^{\mathrm{T}} = \mathbf{I}$  。为了证明这一点，请注意式（A.37）意味着  $\mathbf{U}^{\mathrm{T}}\mathbf{U}\mathbf{U}^{-1} = \mathbf{U}^{-1} = \mathbf{U}^{\mathrm{T}}$  ，从而有  $\mathbf{U}\mathbf{U}^{-1} = \mathbf{U}\mathbf{U}^{\mathrm{T}} = \mathbf{I}$  。使用式（A.12），我们还可以推导出  $|\mathbf{U}| = 1$  。

特征向量方程〔式（A.29）]可以用以下关于  $U$  的形式来表示：

$$
\boldsymbol {A} \boldsymbol {U} = \boldsymbol {U} \Lambda \tag {A.38}
$$

其中  $\Lambda$  是  $M\times M$  的对角矩阵，其对角元素由特征值  $\lambda_{i}$  给出。

假设我们通过使用正交矩阵  $\pmb{U}$  转换列向量  $\pmb{x}$  得到了一个新的向量：

$$
\tilde {\boldsymbol {x}} = \boldsymbol {U} \boldsymbol {x} \tag {A.39}
$$

则向量的长度保持不变，因为

$$
\tilde {\boldsymbol {x}} ^ {\mathrm {T}} \tilde {\boldsymbol {x}} = \boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {U} ^ {\mathrm {T}} \boldsymbol {U} \boldsymbol {x} = \boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {x} \tag {A.40}
$$

同样，任何两个此类向量之间的夹角也保持不变，因为

$$
\tilde {\boldsymbol {x}} ^ {\mathrm {T}} \tilde {\boldsymbol {y}} = \boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {U} ^ {\mathrm {T}} \boldsymbol {U} \boldsymbol {y} = \boldsymbol {x} ^ {\mathrm {T}} \boldsymbol {y} \tag {A.41}
$$

因此，向量乘以  $\pmb{U}$  可以解释为坐标系发生刚性旋转。

从式（A.38）可以看出

$$
\boldsymbol {U} ^ {\mathrm {T}} \boldsymbol {A} \boldsymbol {U} = \boldsymbol {\Lambda} \tag {A.42}
$$

又因为  $\pmb{A}$  是对角矩阵，我们可以说矩阵  $\pmb{A}$  被矩阵  $\pmb{U}$  对角化了。左乘  $\pmb{U}$  并右乘  $\pmb{U}^{\mathrm{T}}$

我们可以得到

$$
\boldsymbol {A} = \boldsymbol {U} \Lambda \boldsymbol {U} ^ {\mathrm {T}} \tag {A.43}
$$

取这个方程的逆并利用式（A.3）以及  $\pmb{U}^{-1} = \pmb{U}^{\mathrm{T}}$  ，可以得到

$$
\boldsymbol {A} ^ {- 1} = \boldsymbol {U} \boldsymbol {\Lambda} ^ {- 1} \boldsymbol {U} ^ {\mathrm {T}} \tag {A.44}
$$

以上两个方程也可以写成以下形式：

$$
\boldsymbol {A} = \sum_ {i = 1} ^ {M} \lambda_ {i} \boldsymbol {u} _ {i} \boldsymbol {u} _ {i} ^ {\mathrm {T}} \tag {A.45}
$$

$$
\boldsymbol {A} ^ {- 1} = \sum_ {i = 1} ^ {M} \frac {1}{\lambda_ {i}} \boldsymbol {u} _ {i} \boldsymbol {u} _ {i} ^ {\mathrm {T}} \tag {A.46}
$$

如果我们取式（A.43）的行列式并使用式（A.12），则可以得到

$$
\left| \boldsymbol {A} \right| = \prod_ {i = 1} ^ {M} \lambda_ {i} \tag {A.47}
$$

类似地，取式（A.43）的迹，并利用迹运算的循环性质［式（A.8）］以及  $\pmb{U}^{\mathrm{T}}\pmb {U} = \pmb{I}$  ，我们可以得到

$$
\operatorname {t r} (A) = \sum_ {i = 1} ^ {M} \lambda_ {i} \tag {A.48}
$$

作为一个练习，请读者利用式（A.33）、式（A.45）、式（A.46）和式（A.47）的结论验证式（A.22）。

如果对于所有非零向量  $\pmb{w}$ ，有  $\pmb{w}^{\mathrm{T}}\pmb{A}\pmb{w} > 0$ ，则称矩阵  $\pmb{A}$  为正定矩阵，记为  $A\succ 0$ 。等价地，正定矩阵对于其所有特征值，有  $\lambda_{i} > 0$ （这可以通过将  $\pmb{w}$  分别设置为每个特征向量来验证，并注意到任意向量都可以表示为特征向量的线性组合而看出）。注意，一个矩阵的所有元素均为正并不一定意味着该矩阵就是正定矩阵。例如，矩阵

$$
\left( \begin{array}{l l} 1 & 2 \\ 3 & 4 \end{array} \right) \tag {A.49}
$$

具有特征值  $\lambda_1\approx 5.37$  和  $\lambda_{2}\approx -0.37$  。如果对于所有的  $\pmb{w}$  ，有  $w^{\mathrm{T}}Aw\geqslant 0$  ，则称这个矩阵为半正定矩阵，记为  $A\succeq 0$  ，等价于  $\lambda_i\geqslant 0$  。

矩阵的条件数由下式给出：

$$
\mathrm {C N} = \left(\frac {\lambda_ {\max}}{\lambda_ {\min}}\right) ^ {1 / 2} \tag {A.50}
$$

其中  $\lambda_{\mathrm{max}}$  是最大的特征值，  $\lambda_{\mathrm{min}}$  是最小的特征值。

## 附录B 变分法

我们可以将函数  $y(x)$  视为如下运算：对于任何输入值  $x$ ，返回输出值  $y$  。同样，我们也可以将泛函  $F[y]$  视为接收函数  $y(x)$  并返回输出值  $F$  的运算。泛函的一个示例是在二维平面上计算绘制的曲线的长度，其中曲线的路径是由一个函数定义的。在机器学习中，一个广泛使用的泛函是连续变量  $x$  的熵  $H[x]$  。对于任意选择的概率密度函数  $p(x)$ ，它都会返回一个标量值以表示该密度下  $x$  的熵。因此，  $p(x)$  的熵也可以类似地写作  $H[p]$  。

传统微积分中的一个常见问题，就是找到一个  $x$  值以使函数  $y(x)$  最大化（或最小化）。类似地，在变分法中，我们也要寻求一个函数  $y(x)$ ，该函数能使泛函  $F[y]$  最大化（或最小化）。也就是说，在所有可能的函数中，我们希望找到能使泛函  $F[y]$  取最大值（或最小值）的特定函数。例如，我们可以使用变分法来证明两点之间直线最短，而最大熵分布是高斯分布。

如果不熟悉普通微积分的法则，则可以对变量  $x$  施加一个小的改变  $\varepsilon$  ，然后以  $\varepsilon$  的幂展开来计算传统导数  $\mathrm{dy} / \mathrm{dx}$  ，从而有

$$
y (x + \varepsilon) = y (x) + \frac {\mathrm {d} y}{\mathrm {d} x} \varepsilon + \mathcal {O} (\varepsilon^ {2}) \tag {B.1}
$$

最后取极限  $\varepsilon \to 0$  。类似地，对于具有多个变量的函数  $y(x_{1},\dots ,x_{D})$  ，相应的偏导数由下式定义：

$$
y \left(x _ {1} + \varepsilon_ {1}, \dots , x _ {D} + \varepsilon_ {D}\right) = y \left(x _ {1}, \dots , x _ {D}\right) + \sum_ {i = 1} ^ {D} \frac {\partial y}{\partial x _ {i}} \varepsilon_ {i} + \mathcal {O} \left(\varepsilon^ {2}\right) \tag {B.2}
$$

当需要考虑对函数  $y(x)$  进行微小改动  $\varepsilon \eta(x)$  会使泛函  $F[y]$  发生多大的改变时，就可以类似地定义泛函的导数，其中  $\eta(x)$  是  $x$  的任意函数，如图 B.1 所示。用  $\delta F / \delta y(x)$  表示  $F[y]$  相对于  $y(x)$  的泛函导数，并用以下关系式加以定义：

$$
F [ y (x) + \varepsilon \eta (x) ] = F [ y (x) ] + \varepsilon \int \frac {\delta F}{\delta y (x)} \eta (x) d x + \mathcal {O} \left(\varepsilon^ {2}\right) \tag {B.3}
$$

![](img/9a24811e611ed641a696f7c8eee60e8a7c7dff01498529b2862393ce8f877244.jpg)  
图B.1 泛函的导数可以用当  $y(x)$  变为 $y(x) + \varepsilon \eta (x)$  时，泛函  $F[y]$  的值如何变化来定义，其中  $(x)$  是  $x$  的任意函数

式（B.3）可以看作式（B.2）的自然扩展，其中  $F[y]$  依赖于一组连续的变量，即所有点  $x$  处  $y$  的值。如果要求泛函相对于函数  $y(x)$  的微小变化是平稳的，则有

$$
\int \frac {\delta F}{\delta y (x)} \eta (x) \mathrm {d} x = 0 \tag {B.4}
$$

由于式（B.4）必须对任意选择的  $\eta (x)$  都成立，

因此泛函的导数必须消除。为了说明这一点，选择一个扰动  $\eta(x)$ ，其在除了点  $\hat{x}$  的邻域之外的任意位置都为零。在这种情况下，泛函的导数在  $x = \hat{x}$  处必须为零。但是，由于对任意的  $\hat{x}$  都必须满足此条件，因此对于任意的  $x$ ，泛函的导数都必须消除。

考虑一个定义在函数  $G(y, y', x)$  的积分之上的泛函，它同时还依赖于  $y(x)$  及其导数  $y'(x)$ ，并且直接依赖于  $x$ ：

$$
F [ y ] = \int G (y (x), y ^ {\prime} (x), x) d x \tag {B.5}
$$

其中，假设  $y(x)$  的值在积分的边界处（可能在无穷大处）为定值。考虑函数  $y(x)$  的变动，我们可以得到

$$
F [ y (x) + \varepsilon \eta (x) ] = F [ y (x) ] + \varepsilon \int \left\{\frac {\partial G}{\partial y} \eta (x) + \frac {\partial G}{\partial y ^ {\prime}} \eta^ {\prime} (x) \right\} d x + \mathcal {O} (\varepsilon^ {2}) \tag {B.6}
$$

让我们以式（B.3）的形式来表示式（B.6）。为此，对式（B.6）右侧的第二项进行分部积分，并注意  $\eta(x)$  必须在积分的边界处消除（因为  $y(x)$  在边界处为定值），从而得到

$$
F [ y (x) + \varepsilon \eta (x) ] = F [ y (x) ] + \varepsilon \int \left\{\frac {\partial G}{\partial y} - \frac {d}{d x} \left(\frac {\partial G}{\partial y ^ {\prime}}\right) \right\} \eta (x) d x + \mathcal {O} (\varepsilon^ {2}) \tag {B.7}
$$

与式（B.3）进行比较，即可确定泛函的导数。

由于要求消除泛函的导数，因此有

$$
\frac {\partial G}{\partial y} - \frac {\mathrm {d}}{\mathrm {d} x} \left(\frac {\partial G}{\partial y ^ {\prime}}\right) = 0 \tag {B.8}
$$

这就是欧拉-拉格朗日（Euler-Lagrange）方程。例如，如果

$$
G = y (x) ^ {2} + \left(y ^ {\prime} (x)\right) ^ {2} \tag {B.9}
$$

则欧拉-拉格朗日方程的形式变为

$$
y (x) - \frac {\mathrm {d} ^ {2} y}{\mathrm {d} x ^ {2}} = 0 \tag {B.10}
$$

这个二阶微分方程可以利用  $y(x)$  的边界条件来求解。

考虑由积分定义的泛函，其积分采用  $G(y,x)$  的形式，并且不依赖于  $y(x)$  的导数。在这种情况下，只要  $x$  的所有值都满足  $\partial G / \partial y(x) = 0$ ，就能保证解的稳定性。

如果我们关于概率分布优化一个泛函，则需要保持概率的归一化约束。满足该要求最简单的方式就是使用拉格朗日乘子（参见附录C），其允许进行无约束的优化。

上述结果可以直接扩展到多维变量  $\pmb{x}$  上。有关变分法的更全面讨论，请参阅Sagan(1969)。

## 附录C 拉格朗日乘子

拉格朗日乘子也称拉格朗日乘子或未定乘子，用于在一个或多个约束条件下，寻找多变量函数的驻点。

考虑在与  $x_{1}, x_{2}$  有关的约束条件下找到函数  $f(x_{1}, x_{2})$  最大值的问题。该约束条件可以写成如下形式：

$$
g \left(x _ {1}, x _ {2}\right) = 0 \tag {C.1}
$$

一种方法是求解约束方程[式（C.1）]，从而将  $x_{2}$  表示为  $x_{1}$  的函数，形式为 $x_{2} = h(x_{1})$  。然后将其代入  $f(x_{1},x_{2})$  ，得到形式为  $f\big(x_1,h\big(x_1)\big)$  的仅关于  $x_{1}$  的函数。接下来，可以通过常用的微分方法找到关于  $x_{1}$  的最大值，得到驻点值  $x_{1}^{*}$  ，相应的  $x_{2}$  值由 $x_{2}^{*} = h\left(x_{1}^{*}\right)$  给出。

![](img/80266b909f745b9d830091295f31ccf9aa53fcfcae2f444136291b680ffaa6e0.jpg)  
图C.1 拉格朗日乘子法的几何解释。我们想要在约束条件  $g(x) = 0$  下最大化函数  $f(x)$  。如果  $\pmb{x}$  是  $D$  维的，则约束条件  $g(x) = 0$  对应一个维度为  $D - 1$  的子空间，如红色曲线所示。这个问题可以通过优化拉格朗日函数  $L(x, \lambda) = f(x) + \lambda g(x)$  来解决

这种方法存在的一个问题是，找到约束方程的一个解析解可能会很困难（该方程使得  $x_{2}$  可以表示为  $x_{1}$  的显式函数）。此外，这种方法区别对待  $x_{1}$  和  $x_{2}$ ，因而会破坏这些变量之间的自然对称性。

一种更优雅且通常更简单的方法是引入一个称为拉格朗日乘子的参数  $\lambda$  。我们将从几何角度来说明这一技术。考虑一个具有  $x_{1},\dots ,x_{D}$  分量的  $D$  维变量  $\pmb{x}$  。约束方程  $g(x) = 0$  表示  $\pmb{x}$  空间中的一个 $(D - 1)$  维曲面，如图C.1所示。

首先注意，在该约束曲面上的任何一点，约束函数的梯度  $\nabla g(\pmb{x})$  都与曲面正交。为了说明这一点，考虑一个位于该约束曲面上的点  $\pmb{x}$  以及另一个

同样位于该约束曲面上的附近点  $x + \varepsilon$  。如果我们在点  $x$  附近进行泰勒展开，则有

$$
g (\boldsymbol {x} + \varepsilon) \approx g (\boldsymbol {x}) + \varepsilon^ {\mathrm {T}} \nabla g (\boldsymbol {x}) \tag {C.2}
$$

因为点  $x$  和点  $x + \varepsilon$  都位于约束曲面上，所以我们有  $g(x) = g(x + \varepsilon)$ ，从而有  $\varepsilon^{\mathrm{T}}\nabla g(x) \approx 0$ 。当  $\| \varepsilon \| \to 0$  时，我们有  $\varepsilon^{\mathrm{T}}\nabla g(x) = 0$ ，并且因为  $\varepsilon$  与约束曲面  $g(x) = 0$  平行，所以我们看到  $\nabla g$  与约束曲面是垂直的。

接下来我们寻找约束曲面上使得  $f(x)$  最大化的点  $\pmb{x}^{\star}$  。这样的点满足  $\nabla f(\pmb {x})$  也与约束曲面垂直的性质，如图C.1所示，否则我们可以通过沿约束曲面移动一小段距离来增大  $f(x)$  的值。因此，  $\nabla f$  和  $\nabla g$  是平行（或反平行）的，并且必定存在一个参数 $\lambda$  ，使得

$$
\nabla f + \lambda \nabla g = 0 \tag {C.3}
$$

其中  $\lambda \neq 0$  称为拉格朗日乘子。注意， $\lambda$  可正可负。

至此，我们很容易引出下式定义的拉格朗日函数：

$$
L (\boldsymbol {x}, \lambda) \equiv f (\boldsymbol {x}) + \lambda g (\boldsymbol {x}) \tag {C.4}
$$

可通过设置  $\nabla_{x}L = 0$  得到带约束的驻点条件[式（C.3)]。此外，通过  $\partial L / \partial \lambda = 0$  可推出约束条件  $g(\pmb {x}) = 0$  。

为了找到在约束条件  $g(x) = 0$  下函数  $f(x)$  的最大值，首先定义式（C.4）所示的拉格朗日函数，然后寻找  $L(x, \lambda)$  关于  $x$  和  $\lambda$  的驻点。对于一个  $D$  维向量  $x$ ，这会给出  $D + 1$  个方程，用于确定驻点  $x^*$  以及  $\lambda$  的值。如果只对  $x^*$  感兴趣，则可以从驻点方程中消去  $\lambda$ ，而不需要计算  $\lambda$  的值（“未定乘子”由此得名）。

举个简单的例子，假设我们希望找到函数  $f(x_{1}, x_{2}) = 1 - x_{1}^{2} - x_{2}^{2}$  在约束条件  $g(x_{1}, x_{2}) = x_{1} + x_{2} - 1 = 0$  下的驻点，如图 C.2 所示。相应的拉格朗日函数由下式给出：

![](img/73bc3e006f1d2b3af2dc1963d92a267e841e484c61e0a0dd43ca802f6d3445b9.jpg)  
图C.2 使用拉格朗日乘子法的一个简单示例。在约束条件  $g(x_{1},x_{2}) = 0$  下最大化函数  $f(x_{1},x_{2}) = 1 - x_{1}^{2} - x_{2}^{2}$  其中  $g(x_{1},x_{2}) = x_{1} + x_{2} - 1$  。圆圈表示函数  $f(x_{1},x_{2})$  的等值线，对角线则代表约束曲面  $g(x_{1},x_{2}) = 0$

$$
L (x, \lambda) = 1 - x _ {1} ^ {2} - x _ {2} ^ {2} + \lambda (x _ {1} + x _ {2} - 1) \tag {C.5}
$$

这个拉格朗日函数关于  $x_{1}$  、  $x_{2}$  和  $\lambda$  的驻点条件给出了如下耦合方程：

$$
- 2 x _ {1} + \lambda = 0 \tag {C.6}
$$

$$
- 2 x _ {2} + \lambda = 0 \tag {C.7}
$$

$$
x _ {1} + x _ {2} - 1 = 0 \tag {C.8}
$$

解这些方程，可以得到驻点  $\left(x_{1}^{*}, x_{2}^{*}\right) = \left(1 / 2, 1 / 2\right)$ ，相应的拉格朗日乘子为  $\lambda = 1$ 。

到目前为止，我们已经讨论了在形如  $g(\pmb{x}) = 0$  的等式约束（equality constraint）下如何最大化函数的问题。接下来考虑在形如  $g(\pmb{x}) \geqslant 0$  的不等式约束（inequality constraint）下如何最大化  $f(\pmb{x})$  的问题，如图 C.3 所示。

根据带约束的驻点是否位于  $g(\pmb{x}) > 0$  的区域内，这个问题有两种可能的解。如果驻点位于  $g(\pmb{x}) > 0$  的区域内，则约束是非激活的（inactive）；而如果驻点位于边界  $g(\pmb{x}) = 0$  上，则约束是激活的（active）。在前一种情况下，函数  $g(\pmb{x})$  不起作用，因此驻点条件是  $\nabla f(\pmb{x}) = 0$  ，这对应于  $\lambda = 0$  的拉格朗日函数［式（C.4）］的驻点。在后一种情

![](img/266cc0213f3c3eebc5de8c28cbaadeba3a59485b522357ad79560cea25f2d9df.jpg)  
图C.3 在不等式约束  $g(x) \geqslant 0$  下最大化函数  $f(x)$  的问题示意图来求解：

况下，驻点位于边界上，类似于前面讨论的等式约束，对应于  $\lambda \neq 0$  的拉格朗日函数［式（C.4）］的驻点。可见拉格朗日乘子的符号是关键，因为只有当  $f(x)$  的梯度远离 $g(x) > 0$  的区域时，函数  $f(x)$  才会达到最大值，如图C.3所示。因此，当  $\lambda >0$  时，我们有  $\nabla f(\pmb {x}) = -\lambda \nabla g(\pmb {x})$  。

对于这两种情况，乘积  $\lambda g(x) = 0$  。因此，在  $g(x) \geqslant 0$  的不等式约束下最大化  $f(x)$  的问题，可通过在以下条件下关于  $x$  和  $\lambda$  优化拉格朗日函数[式（C.4）]

$$
g (x) \geqslant 0 \tag {C.9}
$$

$$
\lambda \geqslant 0 \tag {C.10}
$$

$$
\lambda g (\boldsymbol {x}) = 0 \tag {C.11}
$$

我们称它们为卡鲁什-库恩-塔克（Karush-Kuhn-Tucker, KKT）条件（Karush, 1939; Kuhn and Tucker, 1951）。

注意，如果我们希望在不等式约束  $g(x) \geqslant 0$  下最小化（而非最大化）函数  $f(x)$ ，则需要关于  $x$  最小化拉格朗日函数  $L(x, \lambda) = f(x) - \lambda g(x)$ ，这同样要求  $\lambda \geqslant 0$ 。

最后，拉格朗日乘子法可以很容易地扩展到具有多个等式约束和不等式约束的情况。假设我们希望在约束条件  $g_{j}(\pmb{x}) = 0 (j = 1, \dots, J)$  和  $h_{k}(\pmb{x}) \geqslant 0 (k = 1, \dots, K)$  下最大化  $f(\pmb{x})$ ，则可以引入拉格朗日乘子  $\{\lambda_j\}$  和  $\{\mu_k\}$ ，并优化如下拉格朗日函数：

$$
L \left(\boldsymbol {x}, \left\{\lambda_ {j} \right\}, \left\{\mu_ {k} \right\}\right) = f (\boldsymbol {x}) + \sum_ {j = 1} ^ {J} \lambda_ {j} g _ {j} (\boldsymbol {x}) + \sum_ {k = 1} ^ {K} \mu_ {k} h _ {k} (\boldsymbol {x}) \tag {C.12}
$$

其中  $\mu_k \geqslant 0$  且  $\mu_k h_k(x) = 0, k = 1, \dots, K$  。关于拉格朗日乘子法更详细的讨论，请参阅 Nocedal and Wright（1999）。

## 参考资料

Abramowitz, M., and I. A. Stegun. 1965. Handbook of Mathematical Functions. Dover.  
Adler, S. L. 1981. "Over-relaxation method for the Monte Carlo evaluation of the partition function for multiquadratic actions." Physical Review D 23:2901-2904.  
Aghajanyan, Armen, Bernie Huang, Candace Ross, Vladimir Karpukhin, Hu Xu, Naman Goyal, Dmytro Okhonko, et al. 2022. CM3: A Causal Masked Multimodal Model of the Internet. Technical report. arXiv:2201.07520.  
Aghajanyan, Armen, Luke Zettlemoyer, and Sonal Gupta. 2020. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. Technical report. arXiv:2012.13255.  
Ahn, J. H., and J. H. Oh. 2003. "A constrained EM algorithm for principal component analysis." Neural Computation 15 (1): 57-65.  
Alayrac, Jean-Baptiste, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, et al. 2022. Flamingo: a Visual Language Model for Few-Shot Learning. Technical report. arXiv:2204.14198.  
Amari, S., A. Cichocki, and H. H. Yang. 1996. "A new learning algorithm for blind

signal separation." In Advances in Neural Information Processing Systems, edited by D. S. Touretzky, M. C. Mozer, and M. E. Hasselmo, 8:757-763. MIT Press.  
Anderson, J. A., and E. Rosenfeld. 1988. Neurocomputing: Foundations of Research. MIT Press.  
Anderson, T. W. 1963. "Asymptotic Theory for Principal Component Analysis." Annals of Mathematical Statistics 34:122-148.  
Arjovsky, M., S. Chintala, and L. Bottou. 2017. Wasserstein GAN. Technical report. arXiv:1701.07875.  
Attias, H. 1999. "Independent factor analysis." Neural Computation 11 (4): 803-851.  
Austin, Jacob, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. 2021. "Structured Denoising Diffusion Models in Discrete State-Spaces." In Advances in Neural Information Processing Systems, 34:17981-17993.  
Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer Normalization. Technical report. arXiv:1607.06450.  
Bach, F. R., and M. I. Jordan. 2002. "Kernel Independent Component Analysis." Journal of Machine Learning Research

3:1-48.  
Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. 2015. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. Technical report. arXiv:1511.00561.  
Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural Machine Translation by Jointly Learning to Align and Translate. Technical report. arXiv:1409.0473.  
Baldi, P., and K. Hornik. 1989. "Neural networks and principal component analysis: learning from examples without local minima." Neural Networks 2 (1): 53-58.  
Balduzzi, David, Marcus Frean, Lennox Leary, JP Lewis, Kurt Wan-Duo Ma, and Brian McWilliams. 2017. The Shattered Gradients Problem: If resnets are the answer, then what is the question? Technical report. arXiv:1702.08591.  
Bartholomew, D J. 1987. Latent Variable Models and Factor Analysis. Charles Griffin.  
Basilevsky, Alexander. 1994. Statistical Factor Analysis and Related Methods: Theory and Applications. Wiley.  
Bather, J. 2000. Decision Theory: An Introduction to Dynamic Programming and Sequential Decisions. Wiley.  
Battaglia, Peter W., Jessica B. Hamrick, Victor Bapat, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, et al. 2018. Relational inductive biases, deep learning, and graph networks. Technical report. arXiv:1806.01261.

Baydin, A. G., B. A. Pearlmutter, A. A. Radul, and J. M. Siskind. 2018. "Automatic differentiation in machine learning: a survey." Journal of Machine Learning Research 18:1-43.  
Becker, S., and Y. LeCun. 1989. "Improving the convergence of back-propagation learning with second order methods." In Proceedings of the 1988 Connectionist Models Summer School, edited by D. Touretzky, G. E. Hinton, and T. J. Sejnowski, 29-37. Morgan Kaufmann.  
Belkin, Mikhail, Daniel Hsu, Siyuan Ma, and Soumik Mandal. 2019. "Reconciling modern machine-learning practice and the classical bias-variance trade-off." Proceedings of the National Academy of Sciences 116 (32): 15849-15854.  
Bell, A. J., and T. J. Sejnowski. 1995. "An information maximization approach to blind separation and blind deconvolution." Neural Computation 7 (6): 1129-1159.  
Bellman, R. 1961. Adaptive Control Processes: A Guided Tour. Princeton University Press.  
Bengio, Yoshua, Aaron Courville, and Pascal Vincent. 2012. Representation Learning: A Review and New Perspectives. Technical report. arXiv:1206.5538.  
Bengio, Yoshua, Nicholas Léonard, and Aaron Courville. 2013. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. Technical report. arXiv:1308.3432.  
Berger, J. O. 1985. Statistical Decision Theory and Bayesian Analysis. Second. Springer.  
Bernardo, J. M., and A. F. M. Smith. 1994.

Bayesian Theory. Wiley.  
Bishop, C. M. 1995a. "Regularization and Complexity Control in Feed-forward Networks." In Proceedings International Conference on Artificial Neural Networks ICANN'95, edited by F. Fougelman-Soulie and P. Gallinari, 1:141-148. EC2 et Cie.  
Bishop, Christopher M. 1992. "Exact Calculation of the Hessian Matrix for the Multilayer Perceptron." Neural Computation 4 (4): 494-501.  
Bishop, Christopher M. 1994. "Novelty Detection and Neural Network Validation." IEE Proceedings: Vision, Image and Signal Processing 141 (4): 217-222.  
Bishop, Christopher M. 1995b. Neural Networks for Pattern Recognition. Oxford University Press.  
Bishop, Christopher M. 1995c. "Training with noise is equivalent to Tikhonov regularization." Neural Computation 7 (1): 108-116.  
Bishop, Christopher M. 2006. Pattern Recognition and Machine Learning. Springer.  
Bommasani, Rishi, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, et al. 2021. On the Opportunities and Risks of Foundation Models. Technical report. arXiv:2108.07258.  
Bottou, L. 2010. "Large-scale machine learning with stochastic gradient descent." In Proceedings COMPSTAT 2010, 177-186. Springer.  
Bourlard, H., and Y. Kamp. 1988. "Autoassociation by multilayer perceptrons and singular value decomposition." Biological

Cybernetics 59:291-294.  
Breiman, L. 1996. "Bagging predictors." Machine Learning 26:123-140.  
Brinker, T. J., A. Hekler, A. H. Enk, C. Berking, S Haferkamp, A. Hauschild, M. Weichenthal, et al. 2019. "Deep neural networks are superior to dermatologists in melanoma image classification." European Journal of Cancer 119:11-17.  
Brock, Andrew, Jeff Donahue, and Karen Simonyan. 2018. "Large-Scale GAN Training for High Fidelity Natural Image Synthesis." In Proceedings of the International Conference Learning Representations (ICLR). ArXiv:1809.11096.  
Bronstein, Michael M., Joan Bruna, Taco Cohen, and Petar Velickovic. 2021. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. Technical report. arXiv:2104.13478.  
Bronstein, Michael M., Joan Bruna, Yann Le-Cun, Arthur Szlam, and Pierre Vandergheynst. 2017. "Geometric Deep Learning: Going Beyond Eulcidean Data." In IEEE Signal Processing Magazine, vol. 34. 4. IEEE, July.  
Broomhead, D. S., and D. Lowe. 1988. "Multivariable functional interpolation and adaptive networks." Complex Systems 2:321-355.  
Brown, Tom B., Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, et al. 2020. Language Models are Few-Shot Learners. Technical report. arXiv:2005.14165.  
Bubeck, Sébastien, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric

Horvitz, Ece Kamar, Peter Lee, et al. 2023. Sparks of Artificial General Intelligence: Early experiments with GPT-4. Technical report. arXiv:2303.12712.  
Cardoso, J-F. 1998. "Blind signal separation: statistical principles." Proceedings of the IEEE 9 (10): 2009-2025.  
Caruana, R. 1997. "Multitask learning." Machine Learning 28:41-75.  
Casella, G., and R. L. Berger. 2002. Statistical Inference. Second. Duxbury.  
Chan, K., T. Lee, and T. J. Sejnowski. 2003. "Variational Bayesian learning of ICA with missing data." Neural Computation 15 (8): 1991-2011.  
Chen, A. M., H. Lu, and R. Hecht-Nielsen. 1993. "On the geometry of feedforward neural network error surfaces." Neural Computation 5 (6): 910-927.  
Chen, Mark, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. 2020. "Generative Pretraining From Pixels." Proceedings of Machine Learning Research 119:1691-1703.  
Chen, R. T. Q., Rubanova Y, J. Bettencourt, and D. Duvenaud. 2018. Neural Ordinary Differential Equations. Technical report. arXiv:1806.07366.  
Chen, Ting, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A Simple Framework for Contrastive Learning of Visual Representations. Technical report. arXiv:2002.05709.  
Cho, Kyunghyun, Bart van Merrienboer, Caglar Gülc,ehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase Representations using

RNN Encoder-Decoder for Statistical Machine Translations. Technical report. arXiv:1406.1078.  
Choudrey, R. A., and S. J. Roberts. 2003. "Variational mixture of Bayesian independent component analyzers." Neural Computation 15 (1):213-252.  
Christiano, Paul, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. Technical report.arXiv:1706.03741.  
Collobert, R. 2004. "Large Scale Machine Learning." PhD diss., Université Paris VI.  
Comon, P., C. Jutten, and J. Herault. 1991. "Blind source separation, 2: problems statement." Signal Processing 24 (1): 11-20.  
Cover, T., and P. Hart. 1967. "Nearest neighbor pattern classification." IEEE Transactions on Information Theory IT-11:21-27.  
Cover, T. M., and J. A. Thomas. 1991. Elements of Information Theory. Wiley.  
Cox, R. T. 1946. "Probability, frequency and reasonable expectation." American Journal of Physics 14 (1): 1-13.  
Cybenko, G. 1989. "Approximation by superpositions of a sigmoidal function." Mathematics of Control, Signals and Systems 2:304-314.  
Dawid, A. P. 1979. "Conditional Independence in Statistical Theory (with discussion)." Journal of the Royal Statistical Society, Series B 4:1-31.  
Dawid, A. P. 1980. "Conditional Independence for Statistical Operations." Annals of Statistics 8:598-617.

Deisenroth, M. P., A. A. Faisal, and C. S. Ong. 2020. Mathematics for Machine Learning. Cambridge University Press.  
Dempster, A. P., N. M. Laird, and D. B. Rubin. 1977. "Maximum likelihood from incomplete data via the EM algorithm." Journal of the Royal Statistical Society, B 39 (1): 1-38.  
Deng, Jia, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. "ImageNet: A largescale hierarchical image database." In IEEE Conference on Computer Vision and Pattern Recognition.  
Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pretraining of Deep Bidirectional Transformers for Language Understanding. Technical report. arXiv:1810.04805.  
Dhariwal, Prafulla, and Alex Nichol. 2021. Diffusion Models Beat GANs on Image Synthesis. Technical report. arXiv:2105.05233.  
Dinh, Laurent, David Krueger, and Yoshua Bengio. 2014. NICE: Nonlinear Independent Components Estimation. Technical report. arXiv:1410.8516.  
Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. 2016. Density estimation using Real NVP. Technical report. arXiv:1605.08803.  
Dodge, Samuel, and Lina Karam. 2017. A Study and Comparison of Human and Deep Learning Recognition Performance Under Visual Distortions. Technical report. arXiv:1705.02498.  
Doersch, C. 2016. Tutorial on Variational Autoen-coders. Technical report. arXiv:1606.05908.

Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, et al. 2020. An Image is Worth  $16 \times 16$  Words: Transformers for Image Recognition at Scale. Technical report. arXiv:2010.11929.  
Duane, S., A. D. Kennedy, B. J. Pendleton, and D. Roweth. 1987. "Hybrid Monte Carlo." Physics Letters B 195 (2): 216-222.  
Duchi, J., E. Hazan, and Y. Singer. 2011. "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization." Journal of Machine Learning Research 12:2121-2159.  
Duda, R. O., and P. E. Hart. 1973. Pattern Classification and Scene Analysis. Wiley.  
Dufter, Philipp, Martin Schmitt, and Hinrich Schütze. 2021. Position Information in Transformers: An Overview. Technical report. arXiv:2102.11090.  
Dumoulin, Vincent, and Francesco Visin. 2016. A guide to convolution arithmetic for deep learning. Technical report. arXiv:1603.07285.  
Elliott, R. J., L. Aggoun, and J. B. Moore. 1995. Hidden Markov Models: Estimation and Control. Springer.  
Esser, Patrick, Robin Rombach, and Bjorn Ommer. 2020. Taming Transformers for High-Resolution Image Synthesis. Technical report. arXiv:2012.09841.  
Esteva, A., B. Kuprel, R. A. Novoa, J. Ko, S. M. Swetter, H. M. Blau, and S. Thrun. 2017. "Dermatologis-level classification of skin cancer with deep neural networks." Nature 542:115-118.

Everitt, B. S. 1984. An Introduction to Latent Variable Models. Chapman / Hall.  
Eykholt, Kevin, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi Kohno, and Dawn Song. 2018. "Robust Physical-World Attacks on Deep Learning Visual Classification." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).  
Fawcett, T. 2006. "An introduction to ROC analysis." Pattern Recognition Letters 27:861-874.  
Feller, W. 1966. An Introduction to Probability Theory and its Applications. Second. Vol. 2. Wiley.  
Fletcher, R. 1987. Practical Methods of Optimization. Second. Wiley.  
Forsyth, D. A., and J. Ponce. 2003. Computer Vision: A Modern Approach. Prentice Hall.  
Freund, Y., and R. E. Schapire. 1996. "Experiments with a new boosting algorithm." In Thirteenth International Conference on Machine Learning, edited by L. Saitta, 148-156. Morgan Kaufmann.  
Fukushima, K. 1980. "Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position." Biological Cybernetics 36:193-202.  
Funahashi, K. 1989. "On the approximate realization of continuous mappings by neural networks." Neural Networks 2 (3): 183-192.  
Fung, R., and K. C. Chang. 1990. "Weighting and Integrating Evidence for Stochastic Simulation in Bayesian Networks." In Uncertainty in Artificial Intelligence,

edited by P. P. Bonissone, M. Henrion, L. N. Kanal, and J. F. Lemmer, 5:208-219. Elsevier.  
Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. 2015. A Neural Algorithm of Artistic Style. Technical report. arXiv:1508.06576.  
Geman, S., and D. Geman. 1984. "Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images." IEEE PAMI 6 (1): 721-741.  
Gemmeke, Jort F., Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. 2017. "Audio Set: An ontology and human-labeled dataset for audio events." In Proc. IEEE ICASSP 2017. New Orleans, LA.  
Germain, Mathieu, Karol Gregor, Iain Murray, and Hugo Larochelle. 2015. MADE: Masked Autoencoder for Distribution Estimation. Technical report. arXiv:1502.03509.  
Gilks, W. R. 1992. "Derivative-free adaptive rejection sampling for Gibbs sampling." In Bayesian Statistics, edited by J. Bernardo, J. Berger, A. P. Dawid, and A. F. M. Smith, vol. 4. Oxford University Press.  
Gilks, W. R., N. G. Best, and K. K. C. Tan. 1995. "Adaptive rejection Metropolis sampling." Applied Statistics 44:455-472.  
Gilks, W. R., S. Richardson, and D. J. Spiegelhalter. 1996. Markov Chain Monte Carlo in Practice. Chapman / Hall.  
Gilks, W. R., and P. Wild. 1992. "Adaptive rejection sampling for Gibbs sampling." Applied Statistics 41:337-348.  
Gilmer, Justin, Samuel S. Schoenholz, Patrick

F. Riley, Oriol Vinyals, and George E. Dahl. 2017. Neural Message Passing for Quantum Chemistry. Technical report. arXiv:1704.01212.  
Girshick, Ross B. 2015. Fast R-CNN. Technical report. arXiv:1504.08083.  
Golub, G. H., and C. F. Van Loan. 1996. Matrix Computations. Third. John Hopkins University Press.  
Gong, Yuan, Yu-An Chung, and James R. Glass. 2021. AST: Audio Spectrogram Transformer. Technical report. arXiv:2104.01778.  
Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. MIT Press.  
Goodfellow, Ian J., Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative Adversarial Networks. Technical report. arXiv:1406.2661.  
Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. 2014. Explaining and Harnessing Adversarial Examples. Technical report. arXiv:1412.6572.  
Grathwohl, Will, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. 2018. FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. Technical report. arXiv:1810.01367.  
Griewank, A., and A Walther. 2008. Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation. Second. SIAM.  
Grosse, R. 2018. Automatic Differentiation. CSC321 Lecture 10. University of

Toronto.  
Gulrajani, I., F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville. 2017. Improved training of Wasserstein GANs. Technical report. arXiv:1704.00028.  
Gutmann, Michael, and Aapo Hyvarinen. 2010. "Noise-contrastive estimation: A new estimation principle for unnormalized statistical models." Journal of Machine Learning Research 9:297-304.  
Hamilton, W. L. 2020. Graph Representation Learning. Morgan / Claypool.  
Hartley, R., and A. Zisserman. 2004. Multiple View Geometry in Computer Vision. Second. Cambridge University Press.  
Hassibi, B., and D. G. Stork. 1993. "Second order derivatives for network pruning: optimal brain surgeon." In Proceedings International Conference on Neural Information Processing Systems (NeurIPS), edited by S. J. Hanson, J. D. Cowan, and C. L. Giles, 5:164-171. Morgan Kaufmann.  
Hastie, T., R. Tibshirani, and J. Friedman. 2009. The Elements of Statistical Learning. Second. Springer.  
Hastings, W. K. 1970. "Monte Carlo sampling methods using Markov chains and their applications." Biometrika 57:97-109.  
He, Kaiming, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dolkar, and Ross B. Girshick. 2021. Masked Autoencoders Are Scalable Vision Learners. Technical report. arXiv:2111.06377.  
He, Kaiming, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2019. Momentum

Contrast for Unsupervised Visual Representation Learning. Technical report. arXiv:1911.05722.  
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015a. Deep Residual Learning for Image Recognition. Technical report. arXiv:1512.03385.  
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015b. *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. Technical report. arXiv:1502.01852.  
Henrion, M. 1988. "Propagation of Uncertainty by Logic Sampling in Bayes' Networks." In Uncertainty in Artificial Intelligence, edited by J. F. Lemmer and L. N. Kanal, 2:149-164. North Holland.  
Higgins, I., L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinik, S. Mohamed, and A. Lerchner. 2017. "β-VAE: learning basic visual concepts with a constrained variational framework." In Proceedings of the International Conference Learning Representations (ICLR).  
Hinton, G. E. 2012. Neural Networks for Machine Learning. Lecture 6.5. Coursera Lectures.  
Hinton, G. E., M. Welling, Y. W. Teh, and S Osindero. 2001. "A new view of ICA." In Proceedings of the International Conference on Independent Component Analysis and Blind Signal Separation, vol. 3.  
Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. 2015. Distilling the Knowledge in a Neural Network. Technical report. arXiv:1503.02531.  
Hinton, Geoffrey E. 2002. "Training products of experts by minimizing contrastive

divergence." Neural Computation 14:1771-1800.  
Ho, Jonathan, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. Technical report. arXiv:2006.11239.  
Ho, Jonathan, Chitwan Sahara, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. 2021. Cascaded Diffusion Models for High Fidelity Image Generation. Technical report. arXiv:2106.15282.  
Hochreiter, S., and J. Schmidhuber. 1997. "Long short-term Memory." Neural Computation 9 (8): 1735-1780.  
Hojen-Sorensen, P. A., O. Winther, and L. K. Hansen. 2002. "Mean field approaches to in-dependent component analysis." Neural Computation 14 (4): 889-918.  
Holtzman, Ari, Jan Buys, Maxwell Forbes, and Yejin Choi. 2019. The Curious Case of Neural Text Degeneration. Technical report. arXiv:1904.09751.  
Hornik, K., M. Stinchcombe, and H. White. 1989. "Multilayer feedforward networks are universal approximators." Neural Networks 2 (5):359-366.  
Hospedales, Timothy, Antreas Antoniou, Paul Micaelli, and Amos Storkey. 2021. "Meta-learning in neural networks: A survey." IEEE Transactions on Pattern Analysis and Machine Intelligence 44 (9): 5149-5169.  
Hotelling, H. 1933. "Analysis of a complex of statistical variables into principal components." Journal of Educational Psychology 24:417-441.  
Hotelling, H. 1936. "Relations between two sets of variables." Biometrika 28:321-377.

Hu, Anthony, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. 2023. GAIA-1: A Generative World Model for Autonomous Driving. Technical report. arXiv:2309.17080.  
Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. Technical report. arXiv:2106.09685.  
Hubel, D. H., and T. N. Wiesel. 1959. "Receptive fields of single neurons in the cat's striate cortex." Journal of Physiology 148:574-591.  
Hyvarinen, A. 2005. "Estimation of Non- Normalized Statistical Models by Score Matching." Journal of Machine Learning Research 6:695-709.  
Hyvarinen, A., and E. Oja. 1997. "A fast fixed-point algorithm for independent component analysis." Neural Computation 9 (7): 1483-1492.  
Hyvärinen, Aapo, Jarmo Hurri, and Patrick O. Hoyer. 2009. Natural Image Statistics: A Probabilistic Approach to Early Computational Vision. Springer.  
Ioffe, S., and C. Szegedy. 2015. "Batch normalization." In Proceedings of the International Conference on Machine Learning (ICML), 448-456.  
Jacobs, R. A., M. I. Jordan, S. J. Nowlan, and G. E. Hinton. 1991. "Adaptive mixtures of local experts." Neural Computation 3 (1): 79-87.  
Jebara, T. 2004. Machine Learning: Discriminative and Generative. Kluwer.

Jensen, C., A. Kong, and U. Kjaerulff. 1995. "Blocking Gibbs sampling in very large probabilistic expert systems." International Journal of Human Computer Studies. Special Issue on Real-World Applications of Uncertain Reasoning. 42:647-666.  
Jolliffe, I. T. 2002. Principal Component Analysis. Second. Springer.  
Jumper, John, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, and Olaf Ronneberger. 2021. "Highly accurate protein structure prediction with AlphaFold." Nature 596:583-589.  
Jutten, C., and J. Herault. 1991. "Blind separation of sources, 1: An adaptive algorithm based on neuromimetic architecture." Signal Processing 24 (1): 1-10.  
Kaplan, Jared, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling Laws for Neural Language Models. Technical report. arXiv:2001.08361.  
Karras, Tero, Timo Aila, Samuli Laine, and Jaakko Lehtinen. 2017. Progressive Growing of GANs for Improved Quality, Stability, and Variation. Technical report. arXiv:1710.10196.  
Karush, W. 1939. "Minima of functions of several variables with inequalities as side constraints." Master's thesis, Department of Mathematics, University of Chicago.  
Khosla, Prannay, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu,

and Dilip Krishnan. 2020. Supervised Contrastive Learning. Technical report. arXiv:2004.11362.  
Kingma, D., and J. Ba. 2014. Adam: A method for stochastic optimization. Technical report. arXiv:1412.6980.  
Kingma, D. P., and M. Welling. 2013. "Auto-encoding variational Bayes." In Proceedings of the International Conference on Machine Learning (ICML). ArXiv:1312.6114.  
Kingma, Diederik P., and Max Welling. 2019. An Introduction to Variational Autoencoders. Technical report. arXiv:1906.02691.  
Kingma, Durk P, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. 2016. "Improved variational inference with inverse autoregressive flow." Advances in Neural Information Processing Systems 29.  
Kipf, Thomas N., and Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks. Technical report. arXiv:1609.02907.  
Kloeden, Peter E, and Eckhard Platen. 2013. Numerical solution of stochastic differential equations. Vol. 23. Stochastic Modelling and Applied Probability. Springer.  
Kobyzev, I., S. J. D. Prince, and M. A. Brubaker. 2019. "Normalizing flows: an introduction and review of current methods." IEEE Transactions on Pattern Analysis and Machine Intelligence 43 (11): 3964-3979.  
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. 2012. "Imagenet classification with deep convolutional

neural networks." In Advances in Neural Information Processing Systems, vol. 25.  
Kuhn, H. W., and A. W. Tucker. 1951. "Nonlinear programming." In Proceedings of the 2nd Berkeley Symposium on Mathematical Statistics and Probabilities, 481-492. University of California Press.  
Kullback, S., and R. A. Leibler. 1951. "On information and sufficiency." Annals of Mathematical Statistics 22 (1): 79-86.  
Kurkova, V., and P. C. Kainen. 1994. "Functionally Equivalent Feedforward Neural Networks." Neural Computation 6 (3): 543-558.  
Lasserre, J., Christopher M. Bishop, and T. Minka. 2006. "Principled hybrids of generative and discriminative models." In Proceedings 2006 IEEE Conference on Computer Vision and Pattern Recognition, New York.  
Lauritzen, S. L. 1996. Graphical Models. Oxford University Press.  
Lawley, D. N. 1953. "A Modified Method of Estimation in Factor Analysis and Some Large Sample Results." In Uppsala Symposium on Psychological Factor Analysis, 35-42. Number 3 in Nordisk Psykologi Monograph Series. Uppsala: Almqvist / Wiksell.  
Lazarsfeld, P. F., and N. W. Henry. 1968. Latent Structure Analysis. Houghton Mifflin.  
LeCun, Y., B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. 1989. "Backpropagation Applied to Handwritten ZIP Code Recognition." Neural Computation 1 (4): 541-551.

LeCun, Y., L. Bottou, Y. Bengio, and P. Haffner. 1998. "Gradient-Based Learning Applied to Document Recognition." Proceedings of the IEEE 86:2278-2324.  
LeCun, Y., J. S. Denker, and S. A. Solla. 1990. "Optimal Brain Damage." In Proceedings International Conference on Neural Information Processing Systems (NeurIPS), edited by D. S. Touretzky, 2:598-605. Morgan Kaufmann.  
LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. 2015. "Deep Learning." Nature 512:436-444.  
LeCun, Yann, Sumit Chopra, Raia Hadsell, Marc' Aurelio Ranzato, and Fu-Jie Huang. 2006. "A Tutorial on Energy-Based Learning." In Predicting Structured Data, edited by G. Bakir, T. Hofman, B. Schölkopf, A. Smola, and B. Taskar. MIT Press.  
Leen, T. K. 1995. "From data distributions to regularization in invariant learning." Neural Computation 7:974-981.  
Leshno, M., V. Y. Lin, A. Pinkus, and S. Schocken. 1993. "Multilayer feedforward networks with a polynomial activation function can approximate any function." Neural Networks 6:861-867.  
Li, Hao, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. 2017. Visualizing the Loss Landscape of Neural Nets. Technical report. arXiv:1712.09913.  
Li, Junnan, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. BLIP: Bootstrapping Language-Image Pretraining for Unified Vision-Language Understanding and Generation. Technical report. arXiv:2201.12086.

Lin, Min, Qiang Chen, and Shuicheng Yan. 2013. Network in Network. Technical report. arXiv:1312.4400.  
Lin, Tianyang, Yuxin Wang, Xiangyang Liu, and Xipeng Qiu. 2021. A Survey of Transformers. Technical report. arXiv:2106.04554.  
Lipman, Yaron, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2022. Flow Matching for Generative Modeling. Technical report arXiv:2210.02747.  
Liu, Pengfei, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2021. Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. Technical report. arXiv:2107.13586.  
Lloyd, S. P. 1982. "Least squares quantization in PCM." IEEE Transactions on Information Theory 28 (2): 129-137.  
Long, Jonathan, Evan Shelhamer, and  
Trevor Darrell. 2014. Fully Convolutional Networks for Semantic Segmentation. Technical report. arXiv:1411.4038.  
Luo, Calvin. 2022. Understanding Diffusion Models: A Unified Perspective. Technical report. arXiv:2208.11970.  
Lütkepohl, H. 1996. Handbook of Matrices. Wiley. MacKay, D. J. C. 1992. "A Practical Bayesian Framework for Back-propagation Networks." Neural Computation 4 (3): 448-472.  
MacKay, D. J. C. 2003. Information Theory, Inference and Learning Algorithms. Cambridge University Press.  
MacQueen, J. 1967. "Some methods for

classification and analysis of multivariate observations." In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, edited by L. M. LeCam and J. Neyman, I:281-297. University of California Press.  
Magnus, J. R., and H. Neudecker. 1999. Matrix Differential Calculus with Applications in Statistics and Econometrics. Wiley.  
Mallat, S. 1999. A Wavelet Tour of Signal Processing. Second. Academic Press.  
Mao, X., Q. Li, H. Xie, R. Lau, Z. Wang, and S. Smolley. 2016. Least Squares Generative Adversarial Networks. Technical report. arXiv:1611.04076.  
Mardia, K. V., and P. E. Jupp. 2000. Directional Statistics. Wiley.  
Martens, James, Ilya Sutskever, and Kevin Swersky. 2012. "Estimating the Hessian by Back-propagating Curvature." In Proceedings of the International Conference on Machine Learning (ICML). ArXiv:1206.6464.  
McCullagh, P., and J. A. Nelder. 1989. Generalized Linear Models. Second. Chapman / Hall.  
McCulloch, W. S., and W. Pitts. 1943. "A Logical Calculus of the Ideas Immanent in Nervous Activity." Reprinted in Anderson and Rosenfeld (1988), Bulletin of Mathematical Biophysics 5:115-133.  
McLachlan, G. J., and T. Krishnan. 1997. The EM Algorithm and its Extensions. Wiley.  
McLachlan, G. J., and D. Peel. 2000. Finite Mixture Models. Wiley.  
Meng, X. L., and D. B. Rubin. 1993.

"Maximum likelihood estimation via the ECM algorithm: a general framework." Biometrika 80:267-278.  
Mescheder, L., A. Geiger, and S. Nowozin. 2018. Which Training Methods for GANs do actually Converge? Technical report. arXiv:1801.04406.  
Metropolis, N., A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller. 1953. "Equation of State Calculations by Fast Computing Machines." Journal of Chemical Physics 21 (6): 1087-1092.  
Metropolis, N., and S. Ulam. 1949. "The Monte Carlo method." Journal of the American Statistical Association 44 (247): 335-341.  
Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient Estimation of Word Representations in Vector Space. Technical report. arXiv:1301.3781.  
Minsky, M. L., and S. A. Papert. 1969. Perceptrons. Expanded edition 1990. MIT Press.  
Mirza, M., and S. Osindero. 2014. Conditional Generative Adversarial Nets. Technical report. arXiv:1411.1784.  
Miskin, J. W., and D. J. C. MacKay. 2001. "Ensemble learning for blind source separation." In Independent Component Analysis: Principles and Practice, edited by S. J. Roberts and R. M. Everson. Cambridge University Press.  
Møller, M. 1993. "Efficient Training of Feed-Forward Neural Networks." PhD diss., Aarhus University, Denmark.  
Montúfar, G. F., R. Pascanu, K. Cho, and Y. Bengio. 2014. "On the number of

linear regions of deep neural networks." In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS). ArXiv:1402.1869.  
Mordvintsev, Alexander, Christopher Olah, and Mike Tyka. 2015. Inceptionism: Going Deeper into Neural Networks. Google AI blog.  
Murphy, Kevin P. 2022. Probabilistic Machine Learning: An introduction. MIT Press.probml.ai.  
Murphy, Kevin P. 2023. Probabilistic Machine Learning: Advanced Topics. MIT Press.  
Nakkiran, Preetum, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever. 2019. Deep Double Descent: Where Bigger Models and More Data Hurt. Technical report. arXiv:1912.02292.  
Neal, R. M. 1993. Probabilistic inference using Markov chain Monte Carlo methods. Technical report CRG-TR-93-1. Department of Computer Science, University of Toronto, Canada.  
Neal, R. M. 1999. "Suppressing random walks in Markov chain Monte Carlo using ordered over-relaxation." In Learning in Graphical Models, edited by Michael I. Jordan, 205-228. MIT Press.  
Neal, R. M., and G. E. Hinton. 1999. "A new view of the EM algorithm that justifies incremental and other variants." In Learning in Graphical Models, edited by M. I. Jordan, 355-368. MIT Press.  
Nelder, J. A., and R. W. M. Wedderburn. 1972. "Generalized linear models." Journal of the Royal Statistical Society, A

135:370-384.  
Nesterov, Y. 2004. Introductory Lectures on Convex Optimization: A Basic Course. Kluwer.  
Nichol, Alex, and Prafulla Dhariwal. 2021. Improved Denoising Diffusion Probabilistic Models. Technical report. arXiv:2102.09672.  
Nichol, Alex, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mc-Grew, Ilya Sutskever, and Mark Chen. 2021. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. Technical report. arXiv:2112.10741.  
Nocedal, J., and S. J. Wright. 1999. Numerical Optimization. Springer.  
Noh, Hyeonwoo, Seunghoon Hong, and Bohyung Han. 2015. Learning Deconvolution Network for Semantic Segmentation. Technical report. arXiv:1505.04366.  
Nowlan, S. J., and G. E. Hinton. 1992. "Simplifying neural networks by soft weight sharing." Neural Computation 4 (4): 473-493.  
Ogden, R. T. 1997. Essential Wavelets for Statistical Applications and Data Analysis. Birkhäuser.  
Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. 2016. Pixel Recurrent Neural Networks. Technical report. arXiv:1601.06759.  
Oord, Aaron van den, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. 2016. Conditional Image Generation with PixelCNN Decoders. Technical report.

arXiv:1606.05328.  
Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. 2018. Representation Learning with Contrastive Predictive Coding. Technical report. arXiv:1807.03748.  
Oord, Aaron van den, Oriol Vinyals, and Koray Kavukcuoglu. 2017. Neural Discrete Representation Learning. Technical report. arXiv:1711.00937.  
OpenAI. 2023. GPT-4 Technical Report. Technical report. arXiv:2303.08774.  
Opper, M., and O. Winther. 2000. "Gaussian processes and SVM: mean field theory and leave-one-out." In Advances in Large Margin Classifiers, edited by A. J. Smola, P. L. Bartlett, B. Scholkopf, and D. Shuurmans, 311-326. MIT Press.  
Papamakarios, G., T. Pavlakou, and Iain Murray. 2017. "Masked Autoregressive Flow for Density Estimation." In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS), vol. 30.  
Papamakarios, George, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. 2019. Normalizing Flows for Probabilistic Modeling and Inference. Technical report. arXiv:1912.02762.  
Parisi, Giorgio. 1981. "Correlation functions and computer simulations." *Nuclear Physics B* 180:378-384.  
Pearl, J. 1988. Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann.  
Pearlmutter, B. A. 1994. "Fast exact multiplication by the Hessian." Neural Computation 6 (1): 147-160.  
Pearlmutter, B. A., and L. C. Parra. 1997.

"Maximum likelihood source separation: a context-sensitive generalization of ICA." In Advances in Neural Information Processing Systems, edited by M. C. Mozer, M. I. Jordan, and T. Petsche, 9:613-619. MIT Press.  
Pearson, Karl. 1901. "On lines and planes of closest fit to systems of points in space." The London, Edinburgh and Dublin Philosophical Magazine and Journal of Science, Sixth Series 2:559-572.  
Phuong, Mary, and Marcus Hutter. 2022. Formal Algorithms for Transformers. Technical report. arXiv:2207.09238.  
Prince, Simon J.D. 2020. Variational autoencoders.  
Prince, Simon J.D. 2023. Understanding Deep Learning. MIT Press.  
Radford, A., L. Metz, and S. Chintala. 2015. Unsupervised representation learning with deep convolutional generative adversarial networks. Technical report. arXiv:1511.06434.  
Radford, Alec, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. Technical report. arXiv:2103.00020.  
Radford, Alec, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language Models are Unsupervised Multitask Learners. Technical report. OpenAI.  
Rakhimov, Ruslan, Denis Volkhonskiy, Alexey Artemov, Denis Zorin, and Evgeny Burnaev. 2020. Latent Video Transformer. Technical report. arXiv:2006.10704.

Ramachandran, P., B. Zoph, and Q. V. Le. 2017. Searching for Activation Functions. Technical report. arXiv:1710.05941v2.  
Rao, C. R., and S. K. Mitra. 1971. Generalized Inverse of Matrices and Its Applications. Wiley.  
Redmon, Joseph, Santosh Kumar Divvala, Ross B. Girshick, and Ali Farhadi. 2015. You Only Look Once: Unified, Real-Time Object Detection. Technical report. arxiv:1506.02640.  
Ren, Shaoqing, Kaiming He, Ross B. Girshick, and Jian Sun. 2015. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. Technical report. arxiv:1506.01497.  
Rezende, Danilo J, Shakir Mohamed, and Daan Wierstra. 2014. "Stochastic backpropagation and approximate inference in deep generative models." In Proceedings of the 31st International Conference on Machine Learning (ICML-14), 1278-1286.  
Ricotti, L. P., S. Ragazzini, and G. Martinelli. 1988. "Learning of word stress in a suboptimal second order backpropagation neural network." In Proceedings of the IEEE International Conference on Neural Networks, 1:355-361. IEEE.  
Robert, C. P., and G. Casella. 1999. Monte Carlo Statistical Methods. Springer.  
Rombach, Robin, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2021. High-Resolution Image Synthesis with Latent Diffusion Models. Technical report. arXiv:2112.10752.  
Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. 2015. "U-Net:

Convolutional Networks for Biomedical Image Segmentation." In Medical Image Computing and Computer-Assisted Intervention - MICCAI, edited by N. Navab, J. Hornegger, W. Wells, and A. Frangi. Springer.  
Rosenblatt, F. 1962. Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms. Spartan.  
Roweis, S. 1998. "EM algorithms for PCA and SPCA." In Advances in Neural Information Processing Systems, edited by M. I. Jordan, M. J. Kearns, and S. A. Solla, 10:626-632. MIT Press.  
Roweis, S., and Z. Ghahramani. 1999. "A unifying review of linear Gaussian models." Neural Computation 11 (2): 305-345.  
Rubin, D. B., and D. T. Thayer. 1982. "EM algorithms for ML factor analysis." Psychometrika 47 (1): 69-76.  
Rumelhart, D. E., G. E. Hinton, and R. J. Williams. 1986. "Learning internal representations by error propagation." In Parallel Distributed Processing: Explorations in the Microstructure of Cognition, edited by D. E. Rumelhart, J. L. McClelland, and the PDP Research Group, vol. 1: Foundations, 318-362. Reprinted in Anderson and Rosenfeld (1988). MIT Press.  
Ruthotto, L., and E. Haber. 2021. An introduction to deep generative modeling. Technical report. arXiv:2103.05180.  
Sagan, H. 1969. Introduction to the Calculus of Variations. Dover.  
Saharia, Chitwan, William Chan, Huiwen Chang, Chris A. Lee, Jonathan Ho, Tim

Salimans, David J. Fleet, and Mohammad Norouzi. 2021. Palette: Image-to-Image Diffusion Models. Technical report. arXiv:2111.05826.  
Saharia, Chitwan, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, et al. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. Technical report. arXiv:2205.11487.  
Saharia, Chitwan, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. 2021. Image SuperResolution via Iterative Refinement. Technical report. arXiv:2104.07636.  
Santurkar, S., D. Tsipras, A. Ilyas, and A. Madry. 2018. How does batch normalization help optimization? Technical report. arXiv:1805.11604.  
Satorras, Victor Garcia, Emiel Hoogeboom, and Max Welling. 2021.  $E(n)$  Equivariant Graph Neural Networks. Technical report. arXiv:2102.09844.  
Scholkopf, B., and A. J. Smola. 2002. Learning with Kernels. MIT Press.  
Schuhmann, Christoph, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. 2021. LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs. Technical report. arXiv:2111.02114.  
Schuster, Mike, and Kaisuke Nakajima. 2012. "Japanese and Korean voice search." In 2012 IEEE International Conference on Acoustics, Speech and Signal Processing

(ICASSP), 5149-5152.  
Selvaraju, Ramprasaath R., Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, and Dhruv Batra. 2016. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. Technical report. arXiv:1610.02391.  
Sennrich, Rico, Barry Haddow, and Alexandra Birch. 2015. Neural Machine Translation of Rare Words with Subword Units. Technical report. arXiv:1508.07909.  
Sermanet, Pierre, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann LeCun. 2013. OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks. Technical report. arXiv:1312.6229.  
Shachter, R. D., and M. Peot. 1990. "Simulation Approaches to General Probabilistic Inference on Belief Networks." In Uncertainty in Artificial Intelligence, edited by P. P. Bonissone, M. Henrion, L. N. Kanal, and J. F. Lemmer, vol. 5. Elsevier.  
Shannon, C. E. 1948. "A mathematical theory of communication." The Bell System Technical Journal 27 (3): 379-423 and 623-656.  
Shen, Sheng, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami, Michael W. Mahoney, and Kurt Keutzer. 2019. Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT. Technical report.arXiv:1909.05840.  
Simard, P., B. Victorri, Y. LeCun, and J. Denker. 1992. "Tangent prop - a formalism for specifying selected invariances in an

adaptive network." In Advances in Neural Information Processing Systems, edited by J. E. Moody, S. J. Hanson, and R. P. Lippmann, 4:895-903. Morgan Kaufmann.  
Simard, P. Y., D. Steinkraus, and J. Platt. 2003. "Best practice for convolutional neural networks applied to visual document analysis." In Proceedings International Conference on Document Analysis and Recognition (ICDAR), 958-962. IEEE Computer Society.  
Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. 2013. "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps." In Computer Vision and Pattern Recognition. ArXiv:1312.6034.  
Simonyan, Karen, and Andrew Zisserman. 2014. Very Deep Convolutional Networks for Large-Scale Image Recognition. Technical report. arXiv:1409.1556.  
Sirovich, L. 1987. "Turbulence and the Dynamics of Coherent Structures." Quarterly Applied Mathematics 45 (3): 561-590.  
Sohl-Dickstein, Jascha, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. Technical report. arXiv:1503.03585.  
Sønderby, C., J. Caballero, L. Theis, W. Shi, and F. Huszar. 2016. Amortised MAP inference for image superresolution. Technical report. arXiv:1610.04490.  
Song, Jiaming, Chenlin Meng, and Stefano Ermon. 2020. Denoising Diffusion Implicit Models. Technical report. arXiv:2010.02502.

Song, Yang, and Stefano Ermon. 2019. "Generative Modeling by Estimating Gradients of the Data Distribution." In Advances in Neural Information Processing Systems, 11895-11907. ArXiv:1907.05600.  
Song, Yang, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. 2019. "Sliced score matching: A scalable approach to density and score estimation." In Uncertainty in Artificial Intelligence, 204. ArXiv:1905.07088.  
Song, Yang, and Diederik P. Kingma. 2021. How to Train Your Energy-Based Models. Technical report. arXiv:2101.03288.  
Song, Yang, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020. Score-Based Generative Modeling through Stochastic Differential Equations. Technical report. arXiv:2011.13456.  
Srivastava, N., G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. 2014. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." Journal of Machine Learning Research 15:1929-1958.  
Stone, J. V. 2004. Independent Component Analysis: A Tutorial Introduction. MIT Press.  
Sutskever, I., J. Martens, G. Dahl, and G. E. Hinton. 2013. "On the importance of initialization and momentum in deep learning." In Proceedings of the International Conference on Machine Learning (ICML).  
Sutton, R. 2019. The Bitter Lesson. URL: incompleteideas.net/IncIdeas/

BitterLesson.html.  
Szegedy, Christian, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. 2013. Intriguing properties of neural networks. Technical report. arXiv:1312.6199.  
Szeliski, R. 2022. Computer Vision: Algorithms and Applications. Second. Springer.  
Tarassenko, L. 1995. "Novelty detection for the identification of masses in mammograms." In Proceedings of the Fourth IEE International Conference on Artificial Neural Networks, 4:442-447. IEE.  
Tay, Yi, Mostafa Dehghani, Dara Bahri, and Donald Metzler. 2020. Efficient Transformers: A Survey. Technical report. arXiv:2009.06732.  
Tibshirani, R. 1996. "Regression shrinkage and selection via the lasso." Journal of the Royal Statistical Society, B 58:267-288.  
Tipping, M. E., and Christopher M. Bishop. 1997. Probabilistic Principal Component Analysis. Technical report NCRG/97/010. Neural Computing Research Group, Aston University.  
Tipping, M. E., and Christopher M. Bishop. 1999. "Probabilistic Principal Component Analysis." Journal of the Royal Statistical Society, Series B 21 (3): 611-622.  
Vapnik, V. N. 1995. The nature of statistical learning theory. Springer.  
Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. Technical report. arXiv:1706.03762.

Velicković, Petar. 2023. Everything is Connected: Graph Neural Networks. Technical report. arXiv:2301.08210.  
Velicković, Petar, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph Attention Networks. Technical report. arXiv:1710.10903.  
Vidakovic, B. 1999. Statistical Modelling by Wavelets. Wiley.  
Vig, Jesse, Ali Madani, Lav R. Varshney, Caiming Xiong, Richard Socher, and Nazneen Fatema Rajani. 2020. BERTology Meets Biology: Interpreting Attention in Protein Language Models. Technical report. arXiv:2006.15222.  
Vincent, P. 2011. "A connection between score matching and denoising autoencoders." Neural Computation 23:1661-1674.  
Vincent, Pascal, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. "Extracting and Composing Robust Features with Denoising Autoencoders." In Proceedings of the International Conference on Machine Learning (ICML).  
Walker, A. M. 1969. "On the asymptotic behaviour of posterior distributions." Journal of the Royal Statistical Society, B 31 (1): 80-88.  
Wang, Chengyi, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, et al. 2023. Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers. Technical report. arXiv:2301.02111.  
Weisstein, E. W. 1999. CRC Concise Encyclopedia of Mathematics. Chapman /

Hall, / CRC.  
Welling, Max., and Yee Whye Teh. 2011. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." In Proceedings of the International Conference on Machine Learning (ICML).  
Williams, P. M. 1996. "Using neural networks to model conditional multivariate densities." Neural Computation 8 (4): 843-854.  
Williams, R J. 1992. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine Learning 8:229-256.  
Winn, J., C. M. Bishop, T. Diethe, J. Guiver, and Y. Zaykov. 2023. Model-Based Machine Learning.  
Wolpert, D. H. 1996. "The lack of apriori distinctions between learning algorithms." Neural Computation 8:1341-1390.  
Wu, Zhirong, Yuanjun Xiong, Stella Yu, and Dahua Lin. 2018. Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination. Technical report. arXiv:1805.01978.  
Wu, Zonghan, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S. Yu. 2019. A Comprehensive Survey on Graph Neural Networks. Technical report. arXiv:1901.00596.  
Yan, Wilson, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. 2021. VideoGPT: Video Generation using VQ-VAE and Transformers. Technical report. arXiv:2104.10157.  
Yang, Ruihan, Prakhar Srivastava, and Stephan Mandt. 2022. Diffusion Probabilistic Modeling for Video Generation.

Technical report. arXiv:2203.09481.  
Yilmaz, Fatih Furkan, and Reinhard Heckel. 2022. Regularization-wise double descent: Why it occurs and how to eliminate it. Technical report. arXiv:2206.01378.  
Yosinski, Jason, Jeff Clune, Anh Mai Nguyen, Thomas J. Fuchs, and Hod Lipson. 2015. Understanding Neural Networks Through Deep Visualization. Technical report. arXiv:1506.06579.  
Yu, Jiahui, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. 2021. Vector-quantized Image Modeling with Improved VQGAN. Technical report. arXiv:2110.04627.  
Yu, Jiahui, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, et al. 2022. Scaling Autoregressive Models for Content-Rich Text-to-Image Generation. Technical report. arXiv:2206.10789.  
Yu, Lili, Bowen Shi, Ramakanth Pasunuru, Ben-jamin Muller, Olga Golovneva, Tianlu Wang, Arun Babu, et al. 2023. Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning. Technical report. arXiv:2309.02591.  
Zaheer, Manzil, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, and Alexander Smola. 2017. Deep Sets. Technical report. arXiv:1703.06114.  
Zarchan, P., and H. Musoff. 2005. Fundamentals of Kalman Filtering: A Practical Approach. Second. AIAA.  
Zeiler, Matthew D., and Rob Fergus. 2013. Visualizing and Understanding

Convolutional Networks. Technical report. arXiv:1311.2901.  
Zhang, Chiyuan, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. 2016. Understanding deep learning requires rethinking generalization. Technical report. arXiv:1611.03530.  
Zhao, Wayne Xin, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, et al. 2023. A Survey of Large Language Models. Technical report. arXiv:2303.18223.  
Zhou, Jie, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan

Liu, Lifeng Wang, Changcheng Li, and Maosong Sun. 2018. Graph Neural Networks: A Review of Methods and Applications. Technical report. arXiv:1812.08434.  
Zhou, Y., and R. Chellappa. 1988. "Computation of optic flow using a neural network." In International Conference on Neural Networks, 71-78. IEEE.  
Zhu, J-Y, T. Park, P. Isola, and A. Efros. 2017. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. Technical report. arXiv:1703.10593.

## 索引

粗体显示的页码，表示相应主题的主要信息来源。

$1\times 1$  convolution 1×1卷积，255

1-of-  $K$  coding 1-of-  $K$  编码方案，58, 119，392

acceptancecriterion 接受准则，375,378，381

activation 激活，15

activation function 激活函数，15，137, 156

active constraint 激活的约束，528

AdaGrad AdaGrad, 193

Adam optimization Adam 算法, 193

adaptive rejection sampling 适应性拒绝采样，370

adjacency matrix 邻接矩阵，349

adjoint sensitivity method 473

adversarialattack 对抗攻击，263

aggregation 聚合，353

aleatoricuncertainty 偶然不确定性，21

AlexNet AlexNet, 257

alpha family  $\alpha$  家族，53

amortized inference 摊销推理，487

ancestral sampling 祖先采样，382

anchor 锚点，166

annealed Langevin dynamics 退火朗之万动力学，508

AR model, see autoregressive model AR模型，参见“自回归模型”

area under the ROC curve ROC曲线下面 积，130

artificial intelligence 人工智能，1

attention 注意力，306

attention head 注意力头，313

audio data 音频数据，339

auto-associative neural network, see autoencoder 自联想神经网络，参见“自编码器”

autoencoder 自编码器，164，479

automatic differentiation 自动微分，19, 201, 211

autoregressive flow 自回归流，470

autoregressive model 自回归模型，5, 299, 323

average pooling 平均池化，255

backpropagation 反向传播，17，201

backpropagation through time 通过时间反向传播，325

bag of words 词袋，322

bagging 装袋，240

base distribution 基础分布，465

basis function 基函数，98，138，150

batch gradient descent 批量梯度下降，185

batch learning 批量学习，102

batch normalization 批量归一化，196

Bayes net 贝叶斯网，280

Bayes’ theorem 贝叶斯定理, 26

Bayesian network 贝叶斯网络，280

Bayesian probability 贝叶斯概率，47

beam search 束搜索，329

Bernoulli distribution 伯努利分布，56,81

Bernoulli mixture model 伯努利混合模型，409

BERT BERT, 330

bi-gram model 二元模型，323

bias 偏差，35，109

bias parameter 偏置参数，98，116，156

bias-variance trade-off 偏差-方差权衡，108

BigGAN BigGAN, 458

bijective function 466

binomial distribution 二项分布，57

bits 比特，41

blind source separation 盲源分离，437

blocked path 阻塞路径，290，294

boosting 提升，241

bootstrap 自助，239

bottleneck 瓶颈问题，325

bounding box 边界框，265

Box-Muller method Box-Muller 方法, 368

byte pair encoding 字节对编码，321

canonical correlation analysis 典型相关分析，426

canonical link function 规范连接函数, 143

Cauchy distribution 柯西分布，368

causal attention 因果注意力，328

causality 因果，297

central differences 中心差分法，206

central limit theorem 中心极限定理, 60

ChatGPT ChatGPT, 335

child node 子节点，216，281

Cholesky decomposition 楚列斯基分解，368

circular normal distribution 圆形正态分布，78

classical probability 经典概率，47

classification 分类，3

CLIP CLIP, 167

co-parents 298

codebook vector 码本向量，338，396

collider node 碰撞节点，292

combining models 模型整合，127

committee 委员会，239

complete data set 完整数据集，404

completing the square 完全平方，66

computervision 计算机视觉，248

concave function 46

concentration parameter 聚焦参数，78

condition number 条件数，190

conditional entropy 条件熵，47

conditional expectation 条件期望，31

conditional independence 条件独立性, 127, 289

conditional mixture model 条件混合模型, 173

conditional probability 条件概率，25

conditional VAE 条件VAE，491

conditioner 条件函数，469

confusion matrix 混淆矩阵，128

continuous bag of words 连续词袋，320

continuous normalizing flow 连续标准化流，475

contrastivedivergence 对比散度，387

contrastive learning 对比学习，165

convex function 凸函数，45

convolution 卷积，249，275

convolutional network 卷积网络，247

correlation matrix 相关性矩阵，428

cost function 成本函数，124

coupling flow 耦合流，467

coupling function 耦合函数，469

covariance 协方差，32

Cox's axioms Cox定理，48

cross attention 交叉注意力，332

cross-correlation 互相关，251，275

cross-entropy error function 交叉熵误差函数，139，141，170

cross-validation 交叉验证，13

cumulative distribution function 累积分布函数，29

curse of dimensionality 维度诅咒，150

curve fitting 曲线拟合，7

CycleGAN CycleGAN，459

d-separation d分离，289，293，407

DAG, see directed acyclic graph DAG, 参见“有向无环图”

data augmentation 数据增强，166，222

data compression 数据压缩，396

DDIM DDIM, 505

DDPM DDPM, 493

decision 决策，105

decision boundary 决策边界，115，122

decision region 决策区域，115，122

decision surface, see decision boundary决策面，参见“决策边界”

decision theory 决策理论，105，121

decoder 解码器，479

deep double descent 双重下降，232

deep learning 深度学习，17

deep neural networks 深度神经网络，17

deep sets 深度集合，356

DeepDream DeepDream, 264

degrees of freedom 自由度，421

denoising 493

denoising autoencoder 去噪自编码器，482

denoising diffusion implicit model 去噪扩散隐式模型，505

denoising diffusion probabilistic model 去噪扩散概率模型，493

denoising score matching 去噪得分匹配，507

density estimation 密度估计，33，55

dequantization 去量化，448

descendant node 后代节点，292

design matrix 设计矩阵，101

development set 开发集，13

diagonal covariance matrix 对角协方差矩阵，64

differential entropy 微分熵，44

diffusion kernel 扩散核，495

diffusion model 扩散模型，493

Dirac delta function 狄拉克  $\delta$  函数，30

directed acyclic graph 有向无环图，282

directed cycle 有向环，282

directed factorization 有向分解，298

directed graph 有向图，280

directed graphical model 有向图模型, 280

discriminant function 125

discriminativemodel 判别模型，126, 137，297

disentangled representations 解纠缠表示，462

distributed representation 分布式表示，163

dot-product attention 点积注意力，310

double descent 双重下降，232

dropout Dropout, 241

Estep E步骤，402，405

early stopping 早停法，230

earth mover's distance 推土机距离，457

ECM, see expectation conditional maximization ECM, 参见“期望条件最大化” edge 边, 280, 349

edge detection 252

ELBO, see evidence lower bound ELBO, 参见“证据下界”

EM, see expectation maximization EM, 参见“最大期望”

embedding space 嵌入空间，163

embedding vector 嵌入向量, 349

encoder 编码器，479

energy function 能量函数，384

energy-based models 基于能量的模型，384

ensemble methods 集成，239

entropy 熵, 41

epistemicuncertainty 认知不确定性，21

epoch 训练周期，186

equality constraint 等式约束，527

equivariance 317, 351

erf function erf 函数, 143

error backpropagation, see backpropagation 误差反向传播，参见“反向传播”

error function 8,48,169, 182

Euler-Lagrange equations 欧拉-拉格朗日方程，525

evaluation trace 213

evidence lower bound 证据下界，412, 439, 485, 499

expectation 期望，31

expectation conditional maximization 期望条件最大化，416

expectation maximization 最大期望，400, 404, 440, 442

expectation step, see E step 期望步，参见“E步骤”

expectations 期望，366

explaining away 相消解释，293

exploding gradient 梯度爆炸，196，325

exponential distribution 指数分布，30, 367

exponential family 指数族分布，80, 136, 282

expression swell 表达式膨胀，212

factor analysis 因子分析，436

factor graph 因子图，280

factor loading 因子载荷，436

false negative 假阴性，23

false positive 假阳性，23

fast gradient sign method 快速梯度符号, 263

fast R-CNN fast R-CNN, 270

feature extraction 特征提取，17，98

feature map 特征图，251

features 特征，155

feed-forward network 前馈网络，149, 168

feed-forward networks 前馈神经网络，16

few-shot learning 小样本学习，165，335

filter 滤波器，250

fine-tuning 微调，3，19，165，334

flow matching 476

forward kinematics 正运动学，172

forward problem 正问题，172

forward propagation 203

foundation model 基础模型，19，306, 334，349

frequentist probability 频率学派概率，47

fuel system 燃油系统，292

fully connected graphical model 全连接的图模型，281

fully convolutional network 全卷积网络，272

functional 泛函，524

Gabor filters Gabor滤波器，259

gamma distribution 伽马分布，370

GAN, see generative adversarial network  
GAN, 参见“生成对抗网络”

gatedrecurrentunit 门控循环单元，326

Gaussian 高斯，32，59

Gaussian mixture 高斯混合，74，173,234，397

GEM, see generalized EM algorithm  
GEM, 参见“广义EM算法”

generalization 泛化，7

generalized EM algorithm 广义EM算法，416

generalized linear model 广义线性模型, 138, 143

generative adversarial network 生成对抗网络，453

generative AI 生成式AI，5

generativemodel 生成式模型，5，126,296，454

generative pre-trained transformer 生成式预训练Transformer，5，326

geometric deep learning 几何深度学习, 362

Gibbs sampling 吉布斯采样，380

global minimum 全局最小值，183

GNN, see graph neural network GNN, 参见“图神经网络”

GPT, see generative pre-trained transformer  
GPT, 参见“生成式预训练Transformer”

GPU, see graphics processing unit GPU, 参见“图形处理单元”

gradient descent 梯度下降，181

graph attention network 图注意力网络，359

graph convolutional network 图卷积网络，353

graph neural network 图神经网络，347

graph representation learning 图表示学习，349

graphical model 图模型，279

graphical model factorization 图模型分解，282

graphics processing unit 图形处理单元, 17, 306

group theory 群理论，222

guidance 引导，510

Hadamard product Hadamard 积, 468

Hamiltonian Monte Carlo 哈密顿蒙特卡洛，384

handwritten digit 手写数字，427

He initialization He初始化，188

head-to-head path 头头相接路径，292

head-to-tail path 头尾相接路径，291

Heaviside step function 单位阶跃函数, 140

Hessian matrix 黑塞矩阵，183，209

Hessian outer product approximation 黑塞外积近似法，210

heteroscedastic 异方差，173

hidden Markov model 隐马尔可夫模型, 324, 407

hidden unit 隐藏单元，16，156

hidden variable, see latent variable 隐变量，参见“潜变量”

hierarchical representation 层次化表示，162

histogram density estimation 直方图密度估计，85

history of machine learning 机器学习历史，14

hold-out set 保留集，13

homogeneous Markov chain 均匀马尔可夫链，377

Hooke'slaw 胡克定律，443

Hutchinson's trace estimator Hutchinson的迹估计器，474

hybrid Monte Carlo 混合蒙特卡洛，384

hyperparameter 超参数，12

IAF, see inverse autoregressive flow  
IAF, 参见“逆自回归流”

ICA, see independent component analysis ICA, 参见“独立成分分析”

identifiability 可识别性，400

IID, see independent and identically distributed IID, 参见“独立同分布”

image segmentation 图像分割，270

ImageNet dataset ImageNet 数据集, 257

importance sampling 重要性采样，371,383

importance weight 重要性权重，372

improper distribution 反常分布，30

improper prior 反常先验，227

inactive constraint 非激活的约束，528

incomplete data set 不完整的数据集, 404

independent and identically distributed 独立同分布，37，294

independent component analysis 独立成分分析，437

independent factor analysis 独立因子分析，438

independentvariables 独立变量，28

inductive bias 归纳偏置，17，220

inductive learning 归纳学习，349，358

inequality constraint 不等式约束，527

inference推理，105，121，125，288

InfoNCE InfoNCE, 166

information theory 信息论，46

instance discrimination 个体判别，166

internal covariate shift 内部协变量偏移，197

internal representation 内部表示，265

intersection-over-union 交并比，266

intrinsic dimensionality 内在维度，421

invariance 不变性，222，255，351

inverse autoregressive flow 逆自回归流，471

inverse kinematics 逆运动学，172

inverse problem 逆问题，107，172，220,296

Iris data 鸢尾花数据，151

IRLS, see iterative reweighted least squares IRLS, 参见“迭代重加权最小二乘法”

isotropiccovariancematrix 各向同性协方差矩阵，64

iterativereweightedleast squares 迭代重加权最小二乘法，140

Jacobian matrix 雅可比矩阵，39，207

Jensen's inequality 詹森不等式，46

Jensen-Shannon divergence Jensen-Shannon散度，463

$K$  nearest neighbours  $K$  近邻, 89

$K$  -means clustering algorithm  $K$  均值聚类算法，392，408

Kalman filter 卡尔曼滤波器，302，439

Karush-Kuhn-Tucker conditions 卡鲁什-库恩-塔克条件，528

kernel density estimator 86, 506

kernel function 核函数，87

kernel image 核图像，250

KKT, see Karush-Kuhn-Tucker conditions KTT, 参见“卡鲁什-库恩-塔克条件”

KL divergence, see Kullback-Leibler divergence KL散度，参见“Kullback-Leibler散度”

Kosambi-Karhunen-Loève transform Kosambi-Karhunen-Loève 变换, 422

Kullback-Leibler divergence Kullback-Leibler散度，45，463

Lagrange multiplier 拉格朗日乘子，526

Lagrangian 拉格朗日函数，527

Langevin dynamics 朗之万动力学，386

Langevin sampling 朗之万采样，387

language model 语言模型，326

Laplace distribution 拉普拉斯分布，30

large language model 大语言模型，5, 326, 333

lasso lasso, 229

latent class analysis 潜在类别分析，409

latent diffusion model 潜扩散模型，512

latent variable 潜变量，64，288，391, 421

layer normalization 层归一化，198，315

LDM, see latent diffusion model LDM, 参见“潜扩散模型”

LDS, see linear dynamical system LDS, 参见“线性动态系统”

leakyReLU leakyReLU,161

learning curve 学习曲线，192，230

learning rate parameter 学习率，192

learning to learn 学会学习，165

least-mean-squares algorithm 最小二乘, 103

least-squares GAN 最小二乘GAN，456

leave-one-out 留一法，13

LeNet convolutional network LeNet, 257

Levenberg-Marquardt approximation 列文伯格-马夸尔特近似法，210

likelihood function 似然函数，34，399

likelihood weighted sampling 似然加权采样，383

linear discriminant 线性判别式，116

linear dynamical system 线性动态系统，439

linear independence 线性无关，518

linear regression 线性回归，6，97

linear-Gaussian model 线性高斯模型, 67, 284

linearly separable 线性可分，115

link, see edge 链接，参见“边”

link function 连接函数，137，144

LLM, see large language model LLM, 参见“大语言模型”

LMS, see least-mean-squares algorithm LMS, 参见“平均最小均方算法”

local minimum 局部极小值，182

log odds 对数几率，132

logic sampling 逻辑采样，383

logistic regression 逻辑斯谛回归，138

logistic sigmoid 逻辑斯谛 sigmoid, 81, 99, 131, 139

logit function logit 函数, 132

long short-term memory 长短时记忆, 326

LoRA, see low-rank adaptation LoRA, 参见“低秩自适应”

loss function 损失函数，106，124

loss matrix 损失矩阵，124

lossless data compression 无损数据压缩，396

lossy data compression 有损数据压缩, 396

low-rank adaptation 低秩自适应，335

LSGAN, see least-squares GAN LSGAN, 参见“最小二乘GAN”

LSTM, see long short-term memory LSTM, 参见“长短时记忆”

Mstep M步骤，402，405

macrostate 宏观状态，42

MAE, see masked autoencoder MAE, 参见“掩蔽自编码器”

MAF, see masked autoregressive flow  
MAF, 参见“掩蔽自回归流”

Mahalanobis distance 马哈拉诺比斯距离，60

manifold 流形，154，445

MAP, see maximum a posteriori MAP, 参见“最大后验”

marginal probability 边缘概率，25

Markov blanket 马尔可夫毯，297，382

Markov boundary, see Markov blanket 马尔可夫边界，参见“马尔可夫毯”

Markov chain 马尔可夫链，300，377

Markov chain Monte Carlo 马尔可夫链蒙特卡洛采样，375

Markov model 马尔可夫模型，300

Markov random field 马尔可夫随机场, 280

masked attention 遮掩注意力，328

masked autoencoder 掩蔽自编码器, 483

masked autoregressive flow 掩蔽自回归流，472

max-pooling 最大池化，255

max-unpooling 最大上采样，272

maximization step, see M step 最大化步骤，参见“M步骤”

maximum a posteriori 最大后验，49, 405

maximum likelihood 34, 72, 101, 134

MCMC, see Markov chain Monte Carlo MCMC, 参见“马尔可夫链蒙特卡洛采样”

MDN, see mixture density network

MDN，参见“混合密度网络”

mean 平均值，32

mean value theorem 中值定理，43

measure theory 测度论, 30

mel spectrogram 梅尔频谱，339

message-passing 消息传递，354

message-passing neural network 消息传递神经网络，354

meta-learning 元学习，165

Metropolis algorithm Metropolis 算法, 375

Metropolis-Hastings algorithm Metropolis-Hastings算法，378

microstate 微观状态，42

mini-batches 小批量，187

minimum risk 最小风险，127

Minkowski loss 闵可夫斯基损失，107

missing at random 随机缺失，406，442

missing data 缺失数据，442

mixing coefficient 混合系数，75

mixture component 混合情况，75

mixture density network 混合密度网络，172

mixture distribution 混合分布，391

mixture model 混合模型, 391

mixture of Gaussians 高斯混合分布，74, 173，234，397

MLP, see multilayer perceptron MLP, 参见“多层感知机”

MNIST data MNIST 数据集, 421

mode collapse 模式崩溃，456

model averaging 模型平均，239

model comparison 模型比较, 9

model selection 模型选择，12

moment 矩，33

momentum 动量，190

MonteCarlodropout 蒙特卡洛dropout, 242

Monte Carlo sampling 蒙特卡洛采样,

365

Moore-Penrose pseudo-inverse, see pseudo-inverse 摩尔-彭若斯伪逆，参见“伪逆”

MRF, see Markov random field MRF, 参见“马尔可夫随机场”

multi-class logistic regression 多类逻辑斯谛回归，140

multi-head attention 313

multilayerperceptron 多层感知机，16, 149

multimodal transformer 自注意力，309

multimodality 多模态，173

multinomial distribution 多项分布，59，82

multiplicity 多重数，42

multitask learning 165

mutual information 互信息，47

$n$  -grammodel  $n$  元模型，323

naive Bayes model 朴素贝叶斯模型, 128, 294, 322

nats 纳特，41

natural language processing 自然语言处理，305

natural parameter 自然参数，81

nearest-neighbours 最近邻，88

neocognitron 新认知机，260

Nesterov momentum Nesterov 动量, 191

neural ordinary differential equation 神经常微分方程，472

neuroscience 神经科学，259

NLP, see natural language processing NLP, 参见“自然语言处理”

no free lunch theorem 无免费午餐定理, 221

node 节点，280，349

noise 噪声，21

noiseless coding theorem 无噪编码定理，

41

noisy-OR 噪声 OR, 303

non-identifiability 不可识别性，437

non-max suppression 非最大抑制，269

nonparametric methods 非参数化方法, 56, 85

normal distribution, see Gaussian 正态分布，参见“高斯分布”

normal equations 正规方程，101

normalized exponential, see softmax function 归一化指数，参见“softmax函数”

novelty detection 奇异值检测，126

object detection 目标检测，265

observed variable 观测变量，287

Old Faithful data 老忠实喷泉数据，74

on-hot encoding, see 1-of-  $K$  encoding 独热编码，参见“1-of-  $K$  编码”

one-shot learning 单样本学习，165

one-versus-one classifier 一对一分类器，118

one-versus-the-rest classifier 一对多分类器，117

online gradient descent 在线梯度下降, 186

online learning 在线学习，102

ordered over-relaxation 有序过度放松，382

outer product approximation 外积近似法，210

outlier 离群值，120，126，143

over-fitting 过拟合，9，108，400

over-relaxation 过度放松，382

over-smoothing 过度平滑，362

padding 填充，252

parameter sharing 参数共享，234，284

parameter shrinkage 参数收缩，103

parameter tying, see parameter sharing 参数捆绑，参见“参数共享”

parent node 父节点，214，281

partition function 配分函数, 384

Parzen estimator, see kernel density estimator Parzen 估计器，参见“核密度估计器”

Parzen window Parzen 窗, 87

PCA，see principal component analysis PCA，参见“主成分分析”

perceptron 感知机，15

periodic variables 周期变量，76

permutation matrix 排列矩阵，350

PixelCNN PixelCNN, 338

PixelRNN PixelRNN, 338

plate 板块，286

polynomial curve fitting 多项式曲线拟合，55

pooling 池化，255

positional encoding 位置编码，317

positive definite covariance 正定协方差矩阵，61

positive definite matrix 正定矩阵，523

posterior collapse 后验崩塌，491

posterior probability 后验概率，28

power method 幂方法，424

pre-activation 预激活, 15

pre-processing 预处理，17

pre-training 预训练，164，334

precision matrix 精度矩阵，65

precision parameter 精度，32

predictivedistribution 预测分布，37，105

prefix prompt 前缀提示，335

principal component analysis 主成分分析，422，430，481

principal subspace 主子空间，422

prior 先验，227

prior knowledge 先验知识，17，220

prior probability, 28, 126

probabilistic graphical model, see graphical model 概率图模型，参见“图模型”

probabilistic PCA 概率PCA，430

probability 概率，23

probability density 概率密度，28

probability theory 概率论，22

probit function probit 函数, 142

probit regression probit 回归, 141

product rule of probability 概率的乘积法则，24，25，280

prompt 提示，335，511

prompt engineering 提示工程，335

proposal distribution 提议分布，369, 371, 375

pseudo-inverse 伪逆，101，120

pseudo-random numbers 伪随机数，366

quadratic discriminant 二次判别式，133

radial basis functions 径向基函数，155

random variable 随机变量，24

raster scan 栅格扫描，338

readout layer 读出层，357

real NVP normalizing flow 实数NVP标准化流，468

receiver operating characteristic, see ROC curve 受试者工作特征曲线，参见“ROC曲线”

receptive field 感受野，250，355

recurrent neural network 递归神经网络，324

regression 回归，3

regression function 回归函数，106

regularization 正则化，11，219

regularized least squares 正则化最小二乘法，103

reject option 拒绝选项，125，127

rejection sampling 拒绝采样，369

relative entropy 相对熵，45

reparameterization trick 重参数化技巧, 488

representation learning 表示学习，19, 163

residual block 残差块，237

residual connection 236

residual network 残差网络，237

resnet, see residual network ResNet, 参见“残差网络”

responsibility 责任，75，398

RLHF RLHF, 335

RMS error, see root-mean-square error RMS 误差，参见“均方根误差”

RMSProp RMSProp, 193

RNN, see recurrent neural network RNN, 参见“递归神经网络”

robotarm 机械臂，172

robustness 鲁棒性，120

ROC curve ROC曲线，129

root-mean-square error 均方根误差，9

saliency map 显著性图，262

same convolution 253

sample mean 样本均值，34

samplevariance 样本方差，34

sampling 采样，365

sampling-importance-resampling 采样-重要性-重采样，373

scaleinvariance尺度不变性，222

scaled self-attention 缩放自注意力，312

scaling hypothesis 缩放假设，306

Schur complement 舒尔补，67

score function 得分函数，386，505

scorematching 得分匹配，505

self-attention 自注意力，309

self-supervised learning 自监督学习，5, 320

semi-supervised learning 半监督学习，358

sequential estimation 序贯估计，73

sequential gradient descent 序贯梯度下降，103

sequential learning 序贯学习，102

SGD, see stochastic gradient descent SGD, 参见“随机梯度下降”

shared parameters, see parameter sharing共享参数，参见“参数共享”

shared weights 251

shattered gradients 破碎梯度，236

shrinkage 收缩，11

sigmoid, see logistic sigmoid sigmoid, 参见“逻辑斯谛 sigmoid”

singular value decomposition 奇异值分解，102

SIR, see sampling-importance-resampling SIR, 参见 “采样-重要性-重采样”

skip-grams 跳字，320

skip-layer connections 跳层连接，237

sliding window 滑动窗口，267

smoothing parameter 平滑参数，95

softReLU softReLU,161

soft weight sharing 软权重共享，234

softmax function softmax 函数，83, 132, 171, 174, 310

softplusactivationfunctionsoftplus激活函数，160

sparse autoencoders 稀疏自编码器，482

sparse connections 稀疏连接，251

sparsity 稀疏，229

sphering 球化，429

standard deviation 标准差，32

standardizing 标准化，394，428

state-space model 状态空间模型，301

statistical bias, see bias 统计偏差，参见

“偏差”

statistical independence, see independent variables 统计独立，参见“独立变量”

steepest descent 185

Stein score, see score function 斯坦因得分，参见“得分函数”

Stirling's approximation 斯特林公式，42

stochastic 随机，7

stochastic differential equation 随机微分方程，508

stochastic gradient descent 随机梯度下降，17，186

stochastic variable 随机变量，26

strided convolution 跨步卷积，253

strides 跨步，267

structured data 结构化数据，247，365

style transfer 风格迁移，274

sufficient statistics 57, 58, 72, 84

sum rule of probability 概率的加和法则，24，25，280

sum-of-squares error 平方和误差函数, 8, 37, 119

supervised learning 监督学习，3，358

support vector machine 支持向量机，156

SVD, see singular value decomposition SVD, 参见“奇异值分解”

SVM, see support vector machine SVM, 参见“支持向量机”

swish activation function swish 激活函数, 179

symmetry 对称性，222

symmetrybreaking 对称性破坏，188

tail-to-tail path 尾尾相接，290

tangent propagation 切线传播，223

temperature 温度, 330

tensor 张量，168，253

test set 测试集，9，13

text-to-speech 文本语音转换，340

tied parameters, see parameter sharing 绑参数，参见“参数共享”

token token, 308

tokenization 分词，321

training set 训练集，3

transductive 直推式，349，358

transductive learning 直推式学习, 358

transfer learning 188,331

transformers Transformer, 305

transition probability 377

translation invariance 平移不变性, 222

transpose convolution  转置卷积，272

tri-gram model 三元模型，323

TTS, see text-to-speech TTS, 参见“文本语音转换”

U-net U-Net, 273

undetermined multiplier, see Lagrange multiplier 未定乘子，参见“拉格朗日乘子”

undirected graphical model 无向图模型, 280

uniquenesses 唯一性，436

universal approximation theorems 通用近似器，159

unobserved variable, see latent variable 未观测变量，参见“隐变量”

unsupervised learning 无监督学习，4, 163

utility function 效用函数，124

VAE, see variational autoencoder AE, 参见“变分自编码器”

valid convolution 有效卷积，253

validation set 验证集，13

vanishing gradient 梯度消失，196，325

variance 方差，31，32，109

variational autoencoder 变分自编码器, 484

variational inference 412

variational lower bound, see evidence lower bound 变分下界，参见“证据下界”

vector quantization 向量量化，338，396

vertex, see node 顶点，参见“点”

vision transformer 视觉 Transformer, 336

von Mises distribution 冯·米塞斯分布，76  
voxel 体素，249

Wasserstein distance Wasserstein 距离, 457

Wasserstein GAN Wasserstein GAN, 458

wavelets 小波，99

weakly supervised 弱监督，167

weight decay 权重衰减，11，225

weight parameter 权重参数，15，156

weight sharing, see parameter sharing 权重共享，参见“参数共享”

weight vector 权重向量，116

weight-space symmetry 权重空间的对称性，161

WGAN, see Wasserstein GAN WGAN, 参见“Wasserstein GAN”

whitening 白化，429

Woodbury identity 伍德伯里恒等式，518

word embedding 词嵌入，320

word2vec word2vec，320

wrapped distribution 环绕分布，80

Yellowstone National Park 黄石国家公园，74

![](img/7a2eeeb437c5bfba2002cc61e313baa6d2de8ca61a3183526ce84d628d5ae95e.jpg)
