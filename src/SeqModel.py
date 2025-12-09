import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import matplotlib.pyplot as plt


# 自定义数据集类（处理时序序列）
class StudentSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # 形状: (N, seq_len, feature_dim)
        self.labels = labels        # 形状: (N,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 转换为PyTorch张量（float32类型）
        seq = torch.FloatTensor(self.sequences[idx])
        label = torch.FloatTensor([self.labels[idx]])  # 二分类标签
        return seq, label


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.2):
        """
        适用于短序列的轻量LSTM模型
        :param input_dim: 特征维度（如3个特征：GPA、DCCY、JXJ）
        :param hidden_dim: LSTM隐藏层维度（短序列建议设小一些，如16-32）
        :param output_dim: 输出维度（二分类为1）
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,  # 输入格式: (batch, seq_len, input_dim)
            num_layers=1       # 短序列用1层足够，避免过拟合
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # 输出层（二分类）
        self.sigmoid = nn.Sigmoid()  # 输出概率

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_len, hidden_dim)
        # 取最后一个时间步的输出（短序列的关键信息通常在最后）
        last_out = lstm_out[:, -1, :]  # shape: (batch, hidden_dim)
        last_out = self.dropout(last_out)
        out = self.fc(last_out)       # shape: (batch, output_dim)
        return self.sigmoid(out)      # 输出概率


class LSTMTrainer:
    def __init__(self, train_path, test_path, target_col='学业预警',
                 id_col='XH', term_col='XQ', max_seq_len=3, encoding='gbk'):
        """
        :param max_seq_len: 最大序列长度（短序列建议设为实际最大长度，如3）
        """
        self.train_path = train_path
        self.test_path = test_path
        self.target_col = target_col
        self.id_col = id_col
        self.term_col = term_col
        self.max_seq_len = max_seq_len  # 短序列适配
        self.encoding = encoding
        
        # 初始化变量
        self.df_train = None
        self.df_test = None
        self.X_train = None  # (N, seq_len, feature_dim)
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.scaler = MinMaxScaler()
        self.feature_cols = ['GPA', 'DCCY', 'JXJ']  # 特征列
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pred_labels = None
        self.pred_proba = None


    def load_data(self):
        """加载数据并检查必要列"""
        self.df_train = pd.read_csv(self.train_path, encoding=self.encoding).fillna(0)
        self.df_test = pd.read_csv(self.test_path, encoding=self.encoding).fillna(0)
        
        required_cols = [self.id_col, self.term_col, self.target_col] + self.feature_cols
        for col in required_cols:
            if col not in self.df_train.columns:
                raise ValueError(f"训练集缺少列：{col}")
            if col not in self.df_test.columns:
                raise ValueError(f"测试集缺少列：{col}")
        
        print(f"训练集形状: {self.df_train.shape}, 测试集形状: {self.df_test.shape}")
        print(f"使用设备: {self.device}")


    def _build_sequences(self, df):
        """构建短序列（每个学生的学期数据）"""
        df_sorted = df.sort_values(by=[self.id_col, self.term_col]).copy()
        sequences = []
        labels = []
        ids = []
        
        for student_id, group in df_sorted.groupby(self.id_col):
            # 提取特征序列（按学期排序）
            feature_seq = group[self.feature_cols].values  # (term_count, feature_dim)
            # 标签取最后一个学期的预警结果
            label = group[self.target_col].iloc[-1]
            
            # 统一序列长度（短序列补0，长序列截断最近的学期）
            if len(feature_seq) > self.max_seq_len:
                feature_seq = feature_seq[-self.max_seq_len:]  # 保留最近的学期
            else:
                pad_length = self.max_seq_len - len(feature_seq)
                feature_seq = np.pad(feature_seq, ((pad_length, 0), (0, 0)), mode='constant')
            
            sequences.append(feature_seq)
            labels.append(label)
            ids.append(student_id)
        
        return pd.DataFrame({'student_id': ids, 'sequence': sequences, 'label': labels})


    def preprocess_data(self):
        """预处理：构建序列+归一化"""
        # 构建训练集和测试集的序列
        self.df_train = self._build_sequences(self.df_train)
        self.df_test = self._build_sequences(self.df_test)
        
        # 归一化特征（用训练集的统计量）
        all_train_feats = np.concatenate(self.df_train['sequence'].values, axis=0)
        self.scaler.fit(all_train_feats)
        
        # 应用归一化
        self.df_train['sequence'] = self.df_train['sequence'].apply(
            lambda seq: self.scaler.transform(seq)
        )
        self.df_test['sequence'] = self.df_test['sequence'].apply(
            lambda seq: self.scaler.transform(seq)
        )


    def split_data(self, val_ratio=0.2):
        """按时间顺序分割训练/验证集（短序列更需避免数据泄露）"""
        # 获取每个学生最后一个学期的时间（用于排序）
        original_train = pd.read_csv(self.train_path, encoding=self.encoding).fillna(0)
        self.df_train['last_term'] = self.df_train['student_id'].apply(
            lambda x: original_train[original_train[self.id_col] == x][self.term_col].max()
        )
        
        # 按最后学期排序后分割（早学期训练，晚学期验证）
        self.df_train = self.df_train.sort_values(by='last_term')
        split_idx = int(len(self.df_train) * (1 - val_ratio))
        
        # 转换为numpy数组
        self.X_train = np.stack(self.df_train['sequence'].iloc[:split_idx].values)
        self.y_train = np.array(self.df_train['label'].iloc[:split_idx].values)
        self.X_val = np.stack(self.df_train['sequence'].iloc[split_idx:].values)
        self.y_val = np.array(self.df_train['label'].iloc[split_idx:].values)
        
        # 测试集
        self.X_test = np.stack(self.df_test['sequence'].values)
        self.y_test = np.array(self.df_test['label'].values)
        
        print(f"训练集: {len(self.X_train)} 样本, 验证集: {len(self.X_val)} 样本")
        print(f"序列形状: {self.X_train.shape} (样本数, 序列长度, 特征数)")


    def build_model(self, hidden_dim=16, dropout=0.2):
        """构建适用于短序列的轻量模型"""
        input_dim = len(self.feature_cols)
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,  # 短序列用小一点的隐藏层
            dropout=dropout
        ).to(self.device)


    def train(self, epochs=20, batch_size=16, lr=0.001):
        """训练模型（带早停策略）"""
        # 构建数据集和DataLoader
        train_dataset = StudentSequenceDataset(self.X_train, self.y_train)
        val_dataset = StudentSequenceDataset(self.X_val, self.y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 损失函数和优化器
        criterion = nn.BCELoss()  # 二分类交叉熵
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 早停参数（防止过拟合）
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        
        # 训练记录
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            # 训练循环
            for seqs, labels in train_loader:
                seqs, labels = seqs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()  # 清零梯度
                
                outputs = self.model(seqs)
                loss = criterion(outputs, labels)
                
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                
                train_loss += loss.item() * seqs.size(0)
            
            # 验证循环
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seqs, labels in val_loader:
                    seqs, labels = seqs.to(self.device), labels.to(self.device)
                    outputs = self.model(seqs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * seqs.size(0)
            
            # 计算平均损失
            train_loss /= len(train_dataset)
            val_loss /= len(val_dataset)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')  # 保存最优模型
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停于第 {epoch+1} 轮（验证损失未改善）")
                    break
        
        # 加载最优模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        # 绘制损失曲线
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.legend()
        plt.show()


    def predict(self, save_path='test_predictions.csv'):
        """预测并评估"""
        self.model.eval()
        test_dataset = StudentSequenceDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for seqs, _ in test_loader:
                seqs = seqs.to(self.device)
                outputs = self.model(seqs)
                all_preds.extend(outputs.cpu().numpy().flatten())
        
        # 转换为标签（阈值0.5）
        self.pred_proba = np.array(all_preds)
        self.pred_labels = (self.pred_proba >= 0.5).astype(int)
        
        # 保存结果
        result_df = pd.DataFrame({
            'student_id': self.df_test['student_id'],
            '真实标签': self.y_test,
            '预测标签': self.pred_labels,
            '预测概率': self.pred_proba
        })
        result_df.to_csv(save_path, index=False, encoding=self.encoding)
        print(f"预测结果已保存至 {save_path}")
        
        # 评估
        self._evaluate()


    def _evaluate(self):
        """评估指标"""
        print("\n混淆矩阵:")
        print(confusion_matrix(self.y_test, self.pred_labels))
        print("\n分类报告:")
        print(classification_report(self.y_test, self.pred_labels, digits=3))
        print(f"RMSE: {np.sqrt(mean_squared_error(self.y_test, self.pred_labels)):.4f}")


# 使用示例
if __name__ == "__main__":
    # 初始化（序列长度设为3，适配短序列）
    trainer = LSTMTrainer(
        train_path='train_data.csv',
        test_path='test_data.csv',
        max_seq_len=3,  # 根据实际数据调整（如学生最多3个学期）
        encoding='gbk'
    )
    
    # 执行流程
    trainer.load_data()
    trainer.preprocess_data()
    trainer.split_data(val_ratio=0.2)
    trainer.build_model(hidden_dim=16)  # 短序列用小隐藏层
    trainer.train(epochs=20, batch_size=16)
    trainer.predict()