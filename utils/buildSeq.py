import pandas as pd
import numpy as np

def load_and_preprocess_data(
    train_path, test_path,
    id_col='XH', term_col='XQXN', target_col='学业预警',
    feature_cols=['GPA', 'DCCY', 'JXJ'], max_seq_len=3, encoding='gbk'
):
    """加载数据并转换为时序序列（样本数, 序列长度, 特征数）"""
    # 加载训练集和测试集（已按时间分割）
    df_train = pd.read_csv(train_path, encoding=encoding).fillna(0)
    df_test = pd.read_csv(test_path, encoding=encoding).fillna(0)
    
    # 按学号+学期排序（确保时序正确）
    df_train = df_train.sort_values(by=[id_col, term_col])
    df_test = df_test.sort_values(by=[id_col, term_col])
    
    # 构建训练集序列
    def build_sequences(df):
        sequences = []
        labels = []
        student_ids = []
        for sid, group in df.groupby(id_col):
            # 提取特征序列（每个时间步对应一个学期）
            feats = group[feature_cols].values
            # 标签取最后一个学期的预警结果
            label = group[target_col].iloc[-1]
            # 统一序列长度（短序列前补0，长序列截断最近的max_seq_len个学期）
            if len(feats) < max_seq_len:
                feats = np.pad(feats, ((max_seq_len - len(feats), 0), (0, 0)), mode='constant')
            else:
                feats = feats[-max_seq_len:]
            sequences.append(feats)
            labels.append(label)
            student_ids.append(sid)
        return np.array(sequences), np.array(labels), student_ids
    
    train_sequences, train_labels, train_ids = build_sequences(df_train)
    test_sequences, test_labels, test_ids = build_sequences(df_test)
    
    # 特征归一化（用训练集的统计量）
    scaler = MinMaxScaler()
    train_flat = np.concatenate(train_sequences, axis=0)  # 展平为(样本数*序列长度, 特征数)
    scaler.fit(train_flat)
    # 对每个序列的每个时间步做归一化
    train_sequences = np.array([scaler.transform(seq) for seq in train_sequences])
    test_sequences = np.array([scaler.transform(seq) for seq in test_sequences])
    
    print(f"训练序列形状：{train_sequences.shape}，测试序列形状：{test_sequences.shape}")
    return (train_sequences, train_labels, train_ids,
            test_sequences, test_labels, test_ids)

