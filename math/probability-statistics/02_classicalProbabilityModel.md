# 古典概率模型
## 古典概率：离散均匀概率律
给定一个随机试验，所有的结果构成的集合为**样本空间** (sample space) $Ω$。样本空间 Ω 中的每一个元素为一个**样本** (sample)。不同的随机试验有各自的样本空间。样本空间作为集合，也可以划分成不同**子集** (subset)。
### 概率
给定样本空间 $Ω$ 的一个事件 (event) $A$，$Pr(A)$ 为事件 $A$ 发生的概率 (the probability of event A occurring 或 probability of A)。$Pr(A)$ 满足：
$$Pr(A)\ge0$$
### 等可能
设样本空间 $Ω$ 由 $n$ 个等可能事件 (equally likely events 或 events with equal probability) 构成，事件 $A$ 的概率为：
$$Pr(A)=\frac{n_A}{n}$$
## 事件之间的关系：集合运算

### 积事件
事件 $A$ 与事件 $B$ 为样本空间 $Ω$ 中的两个事件，$A\cap B$ 代表 $A$ 和 $B$ 的积事件 (the intersection of events A and B)，指的是某次试验时，事件 $A$ 和事件 $B$ 同时发生。

$\Pr(A\cap B)$ 代表 $A$ 和 $B$ 积事件概率 (probability of the intersection of events A and B 或 joint probability of A and B)。$\Pr(A\cap B)$ 也叫做 A 和 B **联合概率** (joint probability)。$\Pr(A\cap B)$ 也常记做 $\Pr(A,B)$ ：
$$\Pr(A\cap B)=\Pr(A,B)$$
### 互斥
如果事件 $A$ 与事件 $B$ 为两者交集为空$A\cap B =\varnothing$  ，则称事件 $A$ 和事件 $B$ 互斥 (events A and B are disjoint)，或称 $A$ 和 $B$ 互不相容 (two events are mutually exclusive)。

白话说，事件 $A$ 与事件 $B$ 不可能同时发生，也就是说 $\Pr(A\cap B)$ 为 $0$：
$$A\cap B=\varnothing\quad\Rightarrow\quad\Pr(A\cap B)=\Pr(A,B)=0$$

### 和事件
事件 $A\cup B$ 为 $A$ 和 $B$ 的和事件 (union of events A and B)。具体来说，当事件 $A$ 和事件 $B$ 至少有一个发生时，事件 $A\cup B$ 发生。$\Pr(A\cup B)$ 代表事件 $A$ 和 $B$ 和事件概率 (probability of the union of events A and B 或 probability of A or B)。
$$\underbrace{\Pr(A\cup B)}_{Union}=\Pr(A)+\Pr(B)-\underbrace{\Pr(A\cap B)}_{Joint}$$
如果事件 $A$ 和 $B$ 互斥 (events A and B are mutually exclusive)，即 $A\cap B = \varnothing$ 。对于这种特殊情 况，$\Pr(A\cup B)$ 为：
$$\Pr(A\cup B)=\Pr(A)+\Pr(B)$$
![](/img/Pasted%20image%2020230905162754.png ':size=70%')
![](/img/Pasted%20image%2020230905162805.png ':size=70%')

## 条件概率：给定部分信息作推断
**条件概率** (conditional probability) 是在给定部分信息基础上对试验结果的一种推断。

### 条件概率
$A$ 和 $B$ 为样本空间 $Ω$ 中的两个事件，其中 $Pr(B) > 0$。那么，**事件 $B$ 发生的条件下事件 $A$ 发生的条件概率** (conditional probability of event A occurring given B occurs 或 probability of A given B) 可以通过下式计算得到：
$$\underbrace{\Pr(A|B)}_{Conditional}=\frac{\overbrace{\Pr(A\cap B)}^{Joint}}{\underbrace{\Pr(B)}_{Marginal}}$$
其中，$\Pr(A∩B)$ 为 $A$ 和 $B$ 事件的联合概率，$\Pr(B)$ 也叫 $B$ 事件边缘概率。

注意，我们也可以这么理解 $\Pr(A|B)$，$B$ 实际上是“新的样本空间”——$Ω_B$！$Pr(A|B)$ 是在 $Ω_B$ 中计算得到的概率值。

$\Pr(B)$、$\Pr(A∩B)$ 都是在 $Ω$ 中计算得到的概率值。

$Ω_B$ 是 $Ω$ 的子集，两者的联系正是 $\Pr(B)$，即 $B$ 在 $Ω$ 中对应的概率。$\Pr(B)$ 也可以写成“条件概率”的形式 $\Pr(B | Ω)$。 

### 联合概率
利用上式，联合概率 $\Pr(A∩B)$ 可以整理为：
$$\underbrace{\Pr(A\cap B)}_{Joint}=\underbrace{\Pr(A|B)}_{Conditional}\underbrace{\cdot\Pr(B)}_{Marginal}$$
上式可以继续推广，$A_1, A_2, …, A_n$ 为 $n$ 个事件，它们的联合概率可以展开写成一系列条件概率 的乘积：
$$\Pr(A_1\cap A_2\cap\cdots\cap A_n)=\Pr(A_n|A_1,A_2,\cdots,A_{n-1})\Pr(A_{n-1}|A_1,A_2,\cdots,A_{n-2})\cdots\Pr(A_2|A_1)\Pr(A_1)$$
这也叫做条件概率的**链式法则** (chain rule)。

## 贝叶斯定理：条件概率、边缘概率、联合概率关系
**贝叶斯定理** (Bayes' theorem) 是由**托马斯·贝叶斯** (Thomas Bayes) 提出。毫不夸张地说，贝叶 斯定理撑起机器学习、深度学习算法的半边天。 

贝叶斯定理的基本思想是根据**先验概率** (prior) 和**新的证据** (evidence) 来计算**后验概率** (posterior)。在实际应用中，我们通常根据一些已知的先验知识，来计算事件的先验概率。然后， 当我们获取新的证据时，就可以利用贝叶斯定理来计算事件的后验概率，从而更新我们的信念或 概率。

贝叶斯定理描述的是两个条件概率的关系：
$$\underbrace{\Pr(A|B)}_{Conditional}\underbrace{\Pr(B)}_{Marginal}
=\underbrace{\Pr(B|A)}_{Conditional}\underbrace{\Pr(A)}_{Marginal}
=\underbrace{\Pr(A\cap B)}_{Joint}$$
- $\Pr(A|B)$ 是指在 $B$ 发生条件下 $A$ 发生的条件概率 (conditional probability)；也就是说，$\Pr(A|B)$ 的样本空间为 $Ω_B$； 
- $\Pr(B|A)$ 是指在 $A$ 发生条件下 $B$ 发生的条件概率；也就是说，$\Pr(B|A)$ 的样本空间为 $Ω_A$； 
- $\Pr(A)$ 是 $A$ 的边缘概率 (marginal probability)，不考虑事件 $B$ 的因素，样本空间为 $Ω$； 
- $\Pr(B)$ 是 $B$ 的边缘概率，不考虑事件 $A$ 的因素，样本空间为 $Ω$； 
- $\Pr(A∩B)$ 是事件 $A$ 和 $B$ 的联合概率，样本空间为 $Ω$。

![](/img/Pasted%20image%2020230905164515.png ':size=70%')

### 频率学派 vs 贝叶斯学派
贝叶斯学派和频率学派是统计学中两种主要的哲学观点。它们之间的区别在于它们对概率的解释和使用方式不同。
频率学派将概率视为事件发生的频率或可能性，它强调基于大量数据和随机抽样的推断，通过检验假设来得出结论。频率学派侧重于经验数据和实证研究，常常使用假设检验和置信区间等方法来进行统计推断。 

而贝叶斯学派则将概率视为一种个人信念的度量，它关注的是主观先验知识和经验的结合，以推断参数或未知量的后验分布。贝叶斯学派通常使用贝叶斯定理来计算后验分布，同时将不确定性视为一种核心特征，因此贝叶斯学派在处理小样本或缺乏数据的情况下表现更加优秀。 

虽然贝叶斯学派和频率学派的基本理念和方法不同，但它们在某些情况下是相互补充的。例如，当样本数据较大时，频率学派的假设检验方法可以提供可靠的结果，而在缺乏数据或需要考虑主观经验和先验知识时，贝叶斯学派的方法则更为适用。此外，在一些实际应用中，两种方法可以相互结合，以得出更为准确的推断结论。

## 全概率定理：穷举法
假设 $A_1, A_2, …, A_n$ 互不相容，形成对样本空间 $Ω$ 的**分割** (partition)，也就是说每次试验事件 $A_1, A_2, …, A_n$ 中有且仅有一个发生。 假定 $\Pr(A_i) > 0$，对于空间 $Ω$ 中任意事件 $B$，下式成立：
$$\underbrace{\Pr(B)}_{Marginal}
=\sum_{i=1}^n\underbrace{\Pr(A_i\cap B)}_{Joint}=\Pr(A_1\cap B)+\Pr(A_2\cap B)+\cdots+\Pr(A_n\cap B)$$
上式就叫做**全概率定理** (law of total probability)。

这本质上就是穷举法，也叫枚举法。 举个例子，图 24 给出的例子是三个互不相容事件 $A_1、A_2、A_3$ 对 $Ω$ 形成分割。通过全概率定理，即穷举法，$\Pr(B)$ 可以通过下式计算得到：
$$\underbrace{\Pr(B)}_{Marginal}=\underbrace{\Pr(A_1,B)}_{Joint}
+\underbrace{\Pr(A_2,B)}_{Joint}
+\underbrace{\Pr(A_3,B)}_{Joint}$$
![](/img/Pasted%20image%2020230905165042.png ':size=70%')

### 引入贝叶斯定理
利用贝叶斯定理，以 $A_1, A_2, …, A_n$ 条件，展开全概率定理：
$$\begin{split}
\Pr(B)
&=\sum_{i=1}^n\underbrace{\Pr(A_i,B)}_{Joint}
=\sum_{i=1}^n
\underbrace{\Pr(B|A_i)}_{Conditional}
\underbrace{\Pr(A_i)}_{Marginal}\\
&=\Pr(B|A_1)\Pr(A_1)+\Pr(B|A_2)\Pr(A_2)+\cdots+\Pr(B|A_n)\Pr(A_n)
\end{split}$$
![](/img/Pasted%20image%2020230905165418.png ':size=70%')

反过来，根据贝叶斯定理，在给定事件 $B$ 发生条件下 ($\Pr(B) > 0$)，任意事件 $A_i$ 发生的概率为：
$$\Pr(A_i|B)=\frac{\Pr(A_i,B)}{\Pr(B)}=\frac{\Pr(B|A_i)\Pr(A_i)}{\Pr(B)}$$
利用贝叶斯定理，以 $B$ 条件，进一步展开全概率定理：
$$\begin{split}
\Pr(B)
&=\sum_{i=1}^n\underbrace{\Pr(A_i,B)}_{Joint}
=\sum_{i=1}^n
\underbrace{\Pr(A_i|B)}_{Conditional}
\underbrace{\Pr(B)}_{Marginal}\\
&=\Pr(A_1|B)\Pr(B)+\Pr(A_2|B)\Pr(B)+\cdots+\Pr(A_n|B)\Pr(B)
\end{split}$$
上式左右消去 $\Pr(B)$  ( $Pr(B) > 0$ )，得到：
$$\sum_{i=1}^n\Pr(A_i|B)=\Pr(A_1|B)+\Pr(A_2|B)+\cdots+\Pr(A_n|B)=1$$

## 独立、互斥、条件独立
### 独立
有一种特殊的情况，事件 $B$ 发生与否，不会影响事件 $A$ 发生的概率，也就是如下等式成立：
$$\underbrace{\Pr(A|B)}_{Conditional}=\underbrace{\Pr(A)}_{Marginal}$$
如果上式给出的等式成立，则称事件 $A$ 和事件 $B$ 独立，可以得到
$$\Pr(A\cap B)=\Pr(A)\cdot\Pr(B)$$
如果一组事件 $A_1、A_2… A_n$，它们两两相互独立，则下式成立：
$$\Pr(A_1\cap A_2\cap \cdots\cap A_n)=\Pr(A_1)\cdot\Pr(A_2)\cdots\Pr(A_n)=\prod_{i=1}^n\Pr(A_i)$$

### 条件独立
在给定事件 $C$ 发生条件下，如果如下等式成立，则称**事件 $A$ 和事件 $B$ 在 $C$ 发生条件下条件独 立** (events A and B are conditionally independent given an event C)：
$$\Pr(A\cap B|C)=\Pr(A,B|C)=\Pr(A|C)\cdot\Pr(B|C)$$
格外注意，$A$ 和 $B$ 相互独立，无法推导得到 $A$ 和 $B$ 条件独立。而 $A$ 和 $B$ 条件独立，也无法推导得到 $A$ 和 $B$ 相互独立。













































































