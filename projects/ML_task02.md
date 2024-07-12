### 学习笔记：机器学习竞赛中的进阶模型应用

#### 1. 任务介绍
- **学习目标**：掌握数据可视化（柱状图和折线图）、构建历史平移特征和窗口统计特征、使用LightGBM进行模型训练和预测。

#### 2. 进阶思路与常见方法
- **常见思路**：对于回归预测问题，常规方法包括使用机器学习模型（如LightGBM、XGBoost）或深度学习模型（如神经网络）。深度学习模型需要自行构建模型结构，并对数值数据进行标准化处理。
- **Task2思路**：采用机器学习模型LightGBM进行问题解决，模型简单且数据预处理要求较少。
- **实现步骤**：
  1. 探索性数据分析（EDA）
  2. 数据预处理
  3. 特征提取
  4. 切分训练集与验证集
  5. 训练模型
  6. 预测结果

#### 3. 代码实现详解
##### 3.1 导入模块
```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```

##### 3.2 探索性数据分析（EDA）
- **数据读取**：
  ```python
  train = pd.read_csv('./data/train.csv')
  test = pd.read_csv('./data/test.csv')
  ```
- **数据可视化**：
  - 不同type类型对应target的柱状图：
    ```python
    type_target_df = train.groupby('type')['target'].mean().reset_index()
    plt.figure(figsize=(8, 4))
    plt.bar(type_target_df['type'], type_target_df['target'], color=['blue', 'green'])
    plt.xlabel('Type')
    plt.ylabel('Average Target Value')
    plt.title('Bar Chart of Target by Type')
    plt.show()
    ```
  - 某个id的target值折线图：
    ```python
    specific_id_df = train[train['id'] == '00037f39cf']
    plt.figure(figsize=(10, 5))
    plt.plot(specific_id_df['dt'], specific_id_df['target'], marker='o', linestyle='-')
    plt.xlabel('DateTime')
    plt.ylabel('Target Value')
    plt.title("Line Chart of Target for ID '00037f39cf'")
    plt.show()
    ```

##### 3.3 特征工程
- **合并数据并排序**：
  ```python
  data = pd.concat([test, train], axis=0, ignore_index=True)
  data = data.sort_values(['id', 'dt'], ascending=False).reset_index(drop=True)
  ```
- **构建历史平移特征**：
  ```python
  for i in range(10, 30):
      data[f'last{i}_target'] = data.groupby(['id'])['target'].shift(i)
  ```
- **构建窗口统计特征**：
  ```python
  data['win3_mean_target'] = (data['last10_target'] + data['last11_target'] + data['last12_target']) / 3
  ```
- **切分训练集和测试集**：
  ```python
  train = data[data.target.notnull()].reset_index(drop=True)
  test = data[data.target.isnull()].reset_index(drop=True)
  train_cols = [f for f in data.columns if f not in ['id', 'target']]
  ```

##### 3.4 模型训练与测试集预测
- **定义模型训练函数**：
  ```python
  def time_model(lgb, train_df, test_df, cols):
      trn_x, trn_y = train_df[train_df.dt>=31][cols], train_df[train_df.dt>=31]['target']
      val_x, val_y = train_df[train_df.dt<=30][cols], train_df[train_df.dt<=30]['target']
      train_matrix = lgb.Dataset(trn_x, label=trn_y)
      valid_matrix = lgb.Dataset(val_x, label=val_y)
      lgb_params = {
          'boosting_type': 'gbdt',
          'objective': 'regression',
          'metric': 'mse',
          'min_child_weight': 5,
          'num_leaves': 2 ** 5,
          'lambda_l2': 10,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
          'bagging_freq': 4,
          'learning_rate': 0.05,
          'seed': 2024,
          'nthread': 16,
          'verbose': -1,
      }
      model = lgb.train(lgb_params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], 
                        categorical_feature=[], verbose_eval=500, early_stopping_rounds=500)
      val_pred = model.predict(val_x, num_iteration=model.best_iteration)
      test_pred = model.predict(test_df[cols], num_iteration=model.best_iteration)
      score = mean_squared_error(val_pred, val_y)
      print(score)
      return val_pred, test_pred
  ```
- **训练模型并预测**：
  ```python
  lgb_oof, lgb_test = time_model(lgb, train, test, train_cols)
  test['target'] = lgb_test
  test[['id', 'dt', 'target']].to_csv('submit.csv', index=None)
  ```

#### 4. 总结
- 本次任务中，使用LightGBM进行了模型训练，并添加了时序问题中常见的特征提取方法。通过特征工程的优化，可以显著提升模型预测效果，这是数据挖掘比赛中的主要优化方向。实践中不断尝试与改进模型是提升数据挖掘能力的关键。