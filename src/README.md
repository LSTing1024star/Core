# 学业预警模型代码说明

本文档介绍三个学业预警模型代码版本，分别适用于不同的数据场景（独立样本/时序序列），核心目标是通过学生历史数据（如GPA、挂科数等）预测是否需要学业预警。


## 项目概述
学业预警模型旨在通过学生的历史表现数据（如GPA、挂科次数、奖学金情况等），提前识别可能需要学业干预的学生。随着数据维度的丰富，模型从处理“独立样本”逐步升级为“时序序列”，以捕捉学生表现的趋势变化（如GPA持续下降、挂科累积等）。


## 代码版本说明

### 1. 基础版本（原始代码）
#### 核心功能
基于传统机器学习模型（Gradient Boosting Classifier），将每个（学生-学期）视为独立样本，不考虑同一学生的时序关联，适用于数据无明显时间依赖关系的场景。

#### 主要流程
1. **数据加载**：读取训练集和测试集，处理缺失值，删除学号（XH）列（视为独立样本）。
2. **数据预处理**：
   - 将字符串特征转换为数值型（如GPA、挂科数DCCY、奖学金JXJ）。
   - 对特征进行归一化（MinMax或Standard），避免量纲影响。
   - 调整目标列（学业预警）至数据末尾，方便特征与标签分离。
3. **数据分割**：随机分割训练集为训练/验证集（默认7:3）。
4. **模型训练**：通过GridSearchCV搜索最优超参数（学习率、树深度等），训练梯度提升树模型。
5. **预测与评估**：对测试集预测，输出混淆矩阵、分类报告等指标，并保存结果。

#### 适用场景
- 数据中“同一学生的多学期数据”无明显趋势关联。
- 快速验证特征有效性，搭建基线模型。

#### 核心代码文件
```python
# 基础版本核心类
class ModelTrainer:
    # 包含数据加载、预处理、分割、训练、预测等方法
    ...
```


### 2. 时序特征工程版本
#### 核心改进
在基础版本上增加**时序特征工程**，将同一学生的多学期数据视为序列，通过特征挖掘趋势信息（如历史成绩变化、挂科累积等），仍使用梯度提升树模型。

#### 关键改动
1. **保留时序标识**：保留学号（XH）和学期（XQ）列，用于按学生分组和时间排序。
2. **时序特征构建**：
   - 滞后特征：前1/2学期的GPA、挂科数（如`GPA_prev1`）。
   - 变化量特征：当前与前一学期的差值（如`GPA_change`）。
   - 累计特征：截至当前学期的挂科总数、奖学金累计次数（如`DCCY_cum`）。
   - 滚动统计：过去2学期的GPA均值（如`GPA_roll2`）。
3. **时间导向分割**：按学期顺序分割训练/验证集（早学期训练，晚学期验证），避免“未来数据泄露”。

#### 适用场景
- 学生多学期数据存在明显趋势（如成绩逐步下滑）。
- 希望利用时序信息，但不想引入复杂的深度学习模型。

#### 核心代码片段
```python
# 时序特征构建方法
def create_temporal_features(self, df):
    df_sorted = df.sort_values(by=[self.id_col, self.term_col]).copy()
    grouped = df_sorted.groupby(self.id_col)
    # 滞后特征
    df_sorted['GPA_prev1'] = grouped['GPA'].shift(1)
    # 变化量特征
    df_sorted['GPA_change'] = df_sorted['GPA'] - df_sorted['GPA_prev1']
    # 累计特征
    df_sorted['DCCY_cum'] = grouped['DCCY'].cumsum()
    ...
    return df_sorted
```


### 3. PyTorch LSTM版本
#### 核心改进
使用**循环神经网络（LSTM）** 直接建模时序序列，适用于需要捕捉强时间依赖关系的场景（即使序列较短，如2-3个学期）。

#### 关键改动
1. **序列数据构建**：
   - 按学号分组，将每个学生的多学期数据转换为时序序列（每个时间步对应一个学期的特征）。
   - 统一序列长度（短序列补0，长序列截断最近学期）。
2. **LSTM模型设计**：
   - 轻量级网络（1层LSTM + 全连接层），适配短序列。
   - 使用`Masking`层忽略填充的0值，避免无效数据干扰。
3. **时序训练策略**：
   - 按学生最后一个学期的时间排序，确保训练/验证集的时间先后关系。
   - 加入早停策略（Early Stopping），防止过拟合。

#### 适用场景
- 学生表现的时序依赖较强（如“连续2学期挂科”是预警关键信号）。
- 序列长度较短（2-5个学期），但趋势信息重要。

#### 核心代码片段
```python
# LSTM模型定义（PyTorch）
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1  # 短序列用1层足够
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        return self.sigmoid(self.fc(self.dropout(last_out)))
```


## 环境依赖
- 基础版本/时序特征版本：
  ```
  pandas>=1.0.0
  numpy>=1.18.0
  scikit-learn>=0.23.0
  ```
- PyTorch LSTM版本：
  ```
  pandas>=1.0.0
  numpy>=1.18.0
  torch>=1.7.0
  matplotlib>=3.3.0  # 用于绘制损失曲线
  ```


## 使用步骤（以LSTM版本为例）
1. 准备数据：确保训练集/测试集包含列：`XH`（学号）、`XQ`（学期）、`GPA`、`DCCY`（挂科数）、`JXJ`（奖学金）、`学业预警`（目标列）。
2. 初始化训练器：
   ```python
   trainer = LSTMTrainer(
       train_path='train_data.csv',
       test_path='test_data.csv',
       max_seq_len=3,  # 序列长度（如最多3个学期）
       encoding='gbk'
   )
   ```
3. 执行流程：
   ```python
   trainer.load_data()       # 加载数据
   trainer.preprocess_data() # 构建序列和归一化
   trainer.split_data()      # 按时间分割训练/验证集
   trainer.build_model()     # 构建LSTM模型
   trainer.train()           # 训练模型
   trainer.predict()         # 预测并保存结果
   ```


## 注意事项
1. 数据格式：确保学期列（`XQ`）为可排序的数值（如1、2、3代表连续学期）。
2. 序列长度：`max_seq_len`建议设为实际数据中最大学期数（短序列无需过长）。
3. 类别平衡：若“学业预警”样本占比低，可在训练时调整损失函数权重（如`class_weight`）。


通过选择不同版本，可根据数据的时序特性灵活建模，从简单基线到复杂时序模型逐步优化预测效果。

