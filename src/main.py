from ManualRNN import ManualRNN  # 导入修改后的ManualRNN（支持每个时间步预测）
from sklearn.metrics import confusion_matrix, classification_report 
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os

# 路径设置（保持不变）
current_file_path = os.path.abspath(__file__)
a_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(a_dir)
sys.path.append(parent_dir)
from utils.buildSeq import load_and_preprocess_data  # 导入修改后的序列构建函数（每个时间步有标签）


def train_model(model, train_sequences, train_labels, epochs=20, lr=0.01):
    """
    训练模型（适配每个时间步有标签的情况）
    :param train_sequences: 训练特征序列，shape=(样本数, seq_len, input_dim)
    :param train_labels: 训练标签序列，shape=(样本数, seq_len)（每个时间步一个标签）
    """
    # 计算类别权重（需展平所有时间步的标签）
    flat_labels = train_labels.flatten()  # 展平为(样本数*seq_len,)
    n_pos = np.sum(flat_labels == 1)      # 少数类（1）总数量
    n_neg = len(flat_labels) - n_pos      # 多数类（0）总数量
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0  # 避免除零
    print(f"类别比例（0:1）：{n_neg}:{n_pos}，少数类权重：{pos_weight:.2f}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        # 遍历每个样本的特征序列和标签序列
        for seq, label_seq in zip(train_sequences, train_labels):
            # 前向传播：获取每个时间步的预测（y_preds.shape=(seq_len,)）
            y_preds, h_states = model.forward(seq)
            
            # 计算每个时间步的加权损失并累加
            step_loss = 0.0
            for y_pred, label in zip(y_preds, label_seq):
                if label == 1:
                    # 少数类损失加权
                    step_loss += -pos_weight * label * np.log(y_pred + 1e-8) - (1 - label) * np.log(1 - y_pred + 1e-8)
                else:
                    # 多数类损失
                    step_loss += -label * np.log(y_pred + 1e-8) - (1 - label) * np.log(1 - y_pred + 1e-8)
            total_loss += step_loss
            
            # 反向传播：基于每个时间步的预测和标签更新梯度
            model.backward(seq, h_states, y_preds, label_seq)
            # 更新权重
            model.update_weights(lr=lr)
        
        # 平均损失（按总时间步数量计算）
        total_steps = train_sequences.shape[0] * train_sequences.shape[1]
        avg_loss = total_loss / total_steps
        print(f"Epoch {epoch+1}/{epochs} | 平均损失: {avg_loss:.4f}")
    return model


def test_model(model, test_sequences, test_labels, test_ids, max_seq_len=12, save_path='manual_rnn_predictions.csv'):
    """
    测试模型（适配每个时间步有标签的情况）
    :param test_labels: 测试标签序列，shape=(样本数, seq_len)
    :return: 所有样本的每个时间步预测结果
    """
    all_preds = []    # 存储每个样本的每个时间步预测标签，shape=(样本数, seq_len)
    all_probs = []    # 存储每个样本的每个时间步预测概率，shape=(样本数, seq_len)
    all_step_ids = [] # 存储每个时间步的学号+时间步索引（用于保存结果）
    
    # 逐样本预测
    for idx, (seq, sid) in enumerate(zip(test_sequences, test_ids)):
        # 前向传播：获取每个时间步的预测概率
        y_preds, _ = model.forward(seq)  # y_preds.shape=(seq_len,)
        # 转换为标签（阈值可调整）
        preds = (y_preds >= 0.3).astype(int)  # 预警场景建议降低阈值（如0.3）
        all_preds.append(preds)
        all_probs.append(y_preds)
        # 记录每个时间步的学号+步索引（如1001_0, 1001_1表示学生1001的第0/1个时间步）
        all_step_ids.extend([f"{sid}_{t}" for t in range(len(preds))])
    
    # 展平所有时间步的预测和真实标签（用于评估）
    flat_preds = np.concatenate(all_preds)
    flat_trues = test_labels.flatten()
    
    # 输出评估指标
    print("\n所有时间步的混淆矩阵:")
    print(confusion_matrix(flat_trues, flat_preds))
    print("\n所有时间步的分类报告:")
    print(classification_report(flat_trues, flat_preds, digits=3))
    
    # 保存预测结果（包含每个时间步的信息）
    result_df = pd.DataFrame({
        '学号_时间步': all_step_ids,
        '真实标签': flat_trues,
        '预测标签': flat_preds,
        '预测概率': np.concatenate(all_probs)
    })
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存至: {save_path}")
    return all_preds, all_probs


if __name__ == "__main__":
    # 1. 加载并预处理数据（每个时间步有标签）
    (train_seqs, train_labels, train_ids,
     test_seqs, test_labels, test_ids) = load_and_preprocess_data(
        train_path='D:/LST/Core-main/data/train_data_seq.csv',
        test_path='D:/LST/Core-main/data/test_data_seq.csv',
        max_seq_len=12,  # 每个学生最多12个学期的序列
        encoding='utf-8-sig'
    )
    
    # 检查数据形状（确保标签是二维的）
    print(f"训练标签形状：{train_labels.shape}（样本数, 序列长度）")
    print(f"测试标签形状：{test_labels.shape}（样本数, 序列长度）")
    
    # 2. 初始化手动RNN（输入维度=特征数）
    input_dim = train_seqs.shape[2]  # 特征数（如3：GPA, DCCY, JXJ）
    model = ManualRNN(input_dim=input_dim, hidden_dim=16)  # 隐藏层维度可适当增大（因序列较长）
    
    # 3. 训练模型
    model = train_model(
        model, train_seqs, train_labels,
        epochs=30,  # 序列较长可适当增加轮次
        lr=0.005    # 学习率可适当减小（避免震荡）
    )
    
    # 4. 测试模型
    test_preds, test_probs = test_model(
        model, test_seqs, test_labels, test_ids,
        max_seq_len=12,
        save_path='D:/LST/Core-main/data/manual_rnn_step_predictions.csv'
    )