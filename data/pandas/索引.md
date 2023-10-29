## 索引器
### 列索引
```python
# 通过df[列名]实现(返回Series)
df['col1']
# 通过列表取出多列(返回DataFrame)
df[['col1','col2']]
# 通过df.列名(列名不含空格)
df.col1
```

### 行索引
#### 以字符串为索引的Series
```python
# 取出对应索引的值(若索引相同则返回Series)
df['row1']
# 使用索引列表
df[['row1','row2']]
# 使用切片(包含两端点,且要求两索引值唯一出现)
df['start','end',step]
```
#### 以整数为索引的Series
```python
# 默认索引从0开始
df[int]
df[[int1,int2]]
# 使用切片(不包含右端点)
df[strat:end:step]
```

### loc索引器
对DataFrame进行**行选取**
#### 基于元素的loc索引器
```python
df.loc['row','col']
```
其中：
- `row`表示行的选择：**单个元素**、**元素列表**、**元素切片**、**布尔列表**、**函数**
- `col`表示列的选择：**单个元素**、**元素列表**、**元素切片**、**布尔列表**、**函数**
##### 单个元素：
若索引值重复，则取出DataFrame，唯一则取出Series
```python
# 取出行索引值为row1的所有行(若索引值不唯一则取出DataFrame)
df.loc['row1']
# 同时选择行和列(返回单元素或Series)
df.loc['row1','col1']# 选择某一对象(row1)的某一属性(col1)
```
##### 元素列表：
```python
# 取出列表中所有元素值对应的行或列
df.loc[['row1','row2'],['col1','col2']]
```
##### 元素切片：
```python
# 字符串索引切片要求起点终点索引值唯一(包含两端点)
df.loc['row1':'row2','col1','col2']
```
##### 布尔列表
```python
# 使用与DataFrame长度相同的布尔列表来选择元素
# 某一属性大于某一值则选中
df.loc[df.col > num]
# 选中col1为opt1或者opt2的行
df.loc[df.col1.isin(['opt1','opt2'])]
# 复合条件 |& ~
condition1 = df.col1 == 'opt1'
condition2 = df.col2 > num
df.loc[condition1 | condition2]
```
##### 函数
```python
# 定义函数(输入为DataFrame本身，返回值必须为上四种形式之一)
def func(x):
	sum = x.col1 + x.col2
	return sum
df.loc[func]
# 匿名函数
df.loc[lambda x:'opt1',lambda x:'opt2']
# 匿名函数返回切片形式(使用slice对象)
df.loc[lambda x:slice('strat','end')]
```
>注意：使用多次索引后赋值是赋在临时返回的copy副本上，并非真正修改元素
#### 基于位置的iloc索引器
```python
# 整数：获取第i行第i列的值(索引默认从0开始)
df.iloc[i,j]
# 整数列表：获取第i~j行第p~q列的值(包含端点)
df.iloc[[i-1,j-1],[p-1,q-1]]
# 整数切片：不包含结束端点
df.iloc[i-1:j,p-1:q]
# 布尔列表
df.iloc[(df.col1 > n).values]# 必须传入value,而不是Series
# 函数
df.iloc[lambda x:slice(i-1,j),lambda x:slice(p-1,q)]
```
#### query方法
```python
# 查询(c1 & c2)|(c3 & c4)
df.query('((condition1)&'
		  '(condition2))|'
		 '((condition3)&
		  '(condition4))')
df.query(c1 or c2 and c3 in c4 not in c5)
# query表达式自动注册了所有来自DataFrame的列名
df.query('col1 > col1.mean()')
```
> 若列名带空格，则需使用`col name`的方式进行引用

#### 随机抽样
```python
df.sample(
		  n = 100  # 抽样数量
		  axis = 0 # 抽样方向(0行1列)
		  frac = 0.3 # 抽样比例
		  replace = True # 是否放回(T放回)
		  weights = df.value # 每个样本的抽样相对概率(以value相对大小为抽样概率)
)
```

## 多级索引
### 多级索引结构
```python
multi_index = pd.MultiIndex.from_product([['index1','index2'],#一级索引
										  df.col1.unique()],#二级索引
										 name = ('name1','name2'))#各级索引名称
multi_column = pd.MultiIndex.from_product([['index3','index4'],#一级索引
										   df.col2.unique()],#二级索引
										 name = ('name3','name4'))#各级索引名称
df = pd.DataFrame(date = 
				 index = multi_index,
				 columns = multi_columns)
```
#### 属性获取
```python
# 索引名字:['一级索引名','二级索引名']
df.index.names
df.columns.names
# 索引值(元组)
df.index.values
df.columns.values
# 获取某一层索引
df.get_level_values(i)
```

### 多级索引中的loc索引器
```python
# 将索引值替换为索引元组
df.loc[('index1','index2'),('index3','index4')]
df.iloc[('index1','index2'),('index3','index4')]
# 传入元组列表需要先进行索引排序(sort_index方法)
df_sort = df.sort_index()
df_sort.loc[[('',''),('','')]]
# 元组切片必须先经过排序
df.sort_index().loc[('',''):]
#  多层元素交叉组合后索引(一级索引12的二级索引12元素)
df.loc[(['一级索引1','一级索引2'],['二级索引1','二级索引2']),:]#读取所有列
# 对比：一级索引1的二级索引1和一级索引2的二级索引2的元素
df.loc[[('一级索引1',['二级索引1']),('一级索引2','二级索引2')],:]
```


