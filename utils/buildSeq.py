import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(
    train_path, test_path,
    id_col='XH', term_col='XNXQ', target_col='XYYJ',
    feature_cols=['GPA', 'DCCY', 'JXJ'], max_seq_len=12, encoding='utf-8-sig'
):
    """加载数据并转换为时序序列（样本数, 序列长度, 特征数），每个时间步对应一个标签"""
    # 加载训练集和测试集（已按时间分割）
    df_train = pd.read_csv(train_path, encoding=encoding).fillna(0)
    df_test = pd.read_csv(test_path, encoding=encoding).fillna(0)
    
    # 按学号+学期排序（确保时序正确）
    df_train = df_train.sort_values(by=[id_col, term_col])
    df_test = df_test.sort_values(by=[id_col, term_col])
    
    # 构建训练集/测试集序列（每个时间步对应一个标签）
    def build_sequences(df):
        sequences = []  # 特征序列：(样本数, max_seq_len, 特征数)
        labels = []     # 标签序列：(样本数, max_seq_len)，每个时间步对应一个标签
        student_ids = []
        
        for sid, group in df.groupby(id_col):
            # 1. 提取特征序列（每个时间步对应一个学期的特征）
            feats = group[feature_cols].values  # 形状：(原始学期数, 特征数)
            # 2. 提取标签序列（每个时间步对应一个学期的预警结果）
            labels_seq = group[target_col].values  # 形状：(原始学期数,)
            
            # 3. 统一序列长度（特征和标签需同步处理）
            original_len = len(feats)
            if original_len > max_seq_len:
                # 长序列：截断为最近的max_seq_len个学期
                feats = feats[-max_seq_len:]  # 保留最后max_seq_len个特征
                labels_seq = labels_seq[-max_seq_len:]  # 保留最后max_seq_len个标签
            else:
                # 短序列：前补0（特征补0，标签补0，可根据业务调整补值）
                pad_length = max_seq_len - original_len
                feats = np.pad(feats, ((pad_length, 0), (0, 0)), mode='constant')  # 特征前补0
                labels_seq = np.pad(labels_seq, (pad_length, 0), mode='constant')  # 标签前补0
            
            # 4. 添加到列表
            sequences.append(feats)
            labels.append(labels_seq)
            student_ids.append(sid)
        
        # 转换为numpy数组
        return np.array(sequences), np.array(labels), student_ids
    
    # 构建训练集和测试集的序列（特征+标签一一对应）
    train_sequences, train_labels, train_ids = build_sequences(df_train)
    test_sequences, test_labels, test_ids = build_sequences(df_test)
    
    # 特征归一化（用训练集的统计量，仅对特征序列处理，标签不归一化）
    scaler = MinMaxScaler()
    train_flat = np.concatenate(train_sequences, axis=0)  # 展平为(样本数*max_seq_len, 特征数)
    scaler.fit(train_flat)
    # 对每个序列的每个时间步做归一化
    train_sequences = np.array([scaler.transform(seq) for seq in train_sequences])
    test_sequences = np.array([scaler.transform(seq) for seq in test_sequences])
    
    # 打印形状（确认特征和标签长度一致）
    print(f"训练特征序列形状：{train_sequences.shape}（样本数, 序列长度, 特征数）")
    print(f"训练标签序列形状：{train_labels.shape}（样本数, 序列长度）")
    print(f"测试特征序列形状：{test_sequences.shape}")
    print(f"测试标签序列形状：{test_labels.shape}")
    
    return (train_sequences, train_labels, train_ids,
            test_sequences, test_labels, test_ids)