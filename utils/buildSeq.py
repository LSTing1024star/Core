import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 修改utils/buildSeq.py中的load_and_preprocess_data函数
def load_and_preprocess_data(
    train_path, test_path,
    id_col='XH', term_col='XNXQ', target_col='XYYJ',
    feature_cols=['GPA', 'DCCY', 'JXJ'], max_seq_len=12, encoding='utf-8-sig'
):
    # 加载数据（不变）
    df_train = pd.read_csv(train_path, encoding=encoding).fillna(0)
    df_test = pd.read_csv(test_path, encoding=encoding).fillna(0)
    df_train = df_train.sort_values(by=[id_col, term_col])
    df_test = df_test.sort_values(by=[id_col, term_col])
    
    # 构建序列（新增掩码）
    def build_sequences(df):
        sequences = []      # 特征序列：(样本数, max_seq_len, 特征数)
        labels = []         # 标签序列：(样本数, max_seq_len)
        masks = []          # 掩码序列：1=真实时间步，0=填充时间步
        student_ids = []
        
        for sid, group in df.groupby(id_col):
            original_len = len(group)  # 实际学期数（真实时间步长度）
            # 提取特征和标签
            feats = group[feature_cols].values
            labels_seq = group[target_col].values
            
            # 统一序列长度（前补0）
            if original_len < max_seq_len:
                pad_length = max_seq_len - original_len
                # 特征前补0
                feats = np.pad(feats, ((pad_length, 0), (0, 0)), mode='constant')
                # 标签前补0（填充部分的标签不参与计算）
                labels_seq = np.pad(labels_seq, (pad_length, 0), mode='constant')
                # 掩码：前pad_length个为0（填充），后original_len个为1（真实）
                mask = np.concatenate([np.zeros(pad_length), np.ones(original_len)]).astype(int)
            else:
                # 长序列截断（全为真实时间步，掩码全1）
                feats = feats[-max_seq_len:]
                labels_seq = labels_seq[-max_seq_len:]
                mask = np.ones(max_seq_len).astype(int)
            
            sequences.append(feats)
            labels.append(labels_seq)
            masks.append(mask)  # 新增掩码
            student_ids.append(sid)
        
        return np.array(sequences), np.array(labels), np.array(masks), student_ids
    
    # 生成训练集和测试集的特征、标签、掩码
    train_sequences, train_labels, train_masks, train_ids = build_sequences(df_train)
    test_sequences, test_labels, test_masks, test_ids = build_sequences(df_test)
    
    # 特征归一化（不变）
    scaler = MinMaxScaler()
    train_flat = np.concatenate(train_sequences, axis=0)
    scaler.fit(train_flat)
    train_sequences = np.array([scaler.transform(seq) for seq in train_sequences])
    test_sequences = np.array([scaler.transform(seq) for seq in test_sequences])
    
    # 打印形状（新增掩码形状）
    print(f"训练特征形状：{train_sequences.shape}，训练标签形状：{train_labels.shape}")
    print(f"训练掩码形状：{train_masks.shape}（1=真实时间步，0=填充）")
    print(f"测试特征形状：{test_sequences.shape}，测试标签形状：{test_labels.shape}")
    return (train_sequences, train_labels, train_masks, train_ids,
            test_sequences, test_labels, test_masks, test_ids)