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


def train_model(model, train_sequences, train_labels, train_masks, epochs=20, lr=0.01):
    """
    训练模型（仅用真实时间步计算损失）
    :param train_masks: 训练掩码序列，shape=(样本数, seq_len)，1=真实时间步
    """
    # 计算真实时间步中类别比例（忽略填充）
    flat_labels = train_labels[train_masks == 1]  # 只取真实时间步的标签
    n_pos = np.sum(flat_labels == 1)
    n_neg = len(flat_labels) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"真实时间步类别比例（0:1）：{n_neg}:{n_pos}，少数类权重：{pos_weight:.2f}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_real_steps = 0  # 统计真实时间步总数（用于平均损失）
        
        for seq, label_seq, mask_seq in zip(train_sequences, train_labels, train_masks):
            # 前向传播：获取每个时间步的预测
            y_preds, h_states = model.forward(seq)
            
            # 只计算真实时间步（mask=1）的损失
            step_loss = 0.0
            real_steps = 0  # 单个样本的真实时间步数量
            for t in range(len(label_seq)):
                if mask_seq[t] == 1:  # 仅处理真实时间步
                    y_pred = y_preds[t]
                    label = label_seq[t]
                    # 加权损失
                    if label == 1:
                        step_loss += -pos_weight * label * np.log(y_pred + 1e-8) - (1 - label) * np.log(1 - y_pred + 1e-8)
                    else:
                        step_loss += -label * np.log(y_pred + 1e-8) - (1 - label) * np.log(1 - y_pred + 1e-8)
                    real_steps += 1
            
            total_loss += step_loss
            total_real_steps += real_steps
            
            # 反向传播（仅基于真实时间步的梯度，模型内部已通过mask控制）
            model.backward(seq, h_states, y_preds, label_seq, mask_seq)  # 传入掩码
            model.update_weights(lr=lr)
        
        # 按真实时间步总数计算平均损失
        avg_loss = total_loss / total_real_steps if total_real_steps > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} | 真实时间步平均损失: {avg_loss:.4f}")
    return model


def test_model(model, test_sequences, test_labels, test_masks, test_ids, max_seq_len=12, save_path='manual_rnn_predictions.csv'):
    """测试模型（仅评估真实时间步）"""
    all_preds = []    # 所有样本的预测标签（含填充，后续过滤）
    all_probs = []    # 所有样本的预测概率
    all_real_trues = []  # 过滤后的真实标签（仅真实时间步）
    all_real_preds = []  # 过滤后的预测标签（仅真实时间步）
    all_real_step_ids = []  # 真实时间步的学号+时间步
    
    for idx, (seq, label_seq, mask_seq, sid) in enumerate(zip(test_sequences, test_labels, test_masks, test_ids)):
        # 前向传播
        y_preds, _ = model.forward(seq)
        preds = (y_preds >= 0.3).astype(int)  # 预测标签
        all_preds.append(preds)
        all_probs.append(y_preds)
        
        # 过滤真实时间步（mask=1）
        for t in range(len(mask_seq)):
            if mask_seq[t] == 1:  # 仅保留真实时间步
                all_real_trues.append(label_seq[t])
                all_real_preds.append(preds[t])
                all_real_step_ids.append(f"{sid}_{t}")  # 标记真实时间步
        
    # 转换为数组
    all_real_trues = np.array(all_real_trues)
    all_real_preds = np.array(all_real_preds)
    
    # 仅基于真实时间步评估
    print(f"\n真实时间步总数：{len(all_real_trues)}（排除填充）")
    print("真实时间步的混淆矩阵:")
    print(confusion_matrix(all_real_trues, all_real_preds))
    print("\n真实时间步的分类报告:")
    print(classification_report(all_real_trues, all_real_preds, digits=3))
    
    # 保存结果（仅包含真实时间步）
    result_df = pd.DataFrame({
        '学号_真实时间步': all_real_step_ids,
        '真实标签': all_real_trues,
        '预测标签': all_real_preds,
        '预测概率': [all_probs[i][t] for i, sid in enumerate(test_ids) for t in range(max_seq_len) if test_masks[i][t]==1]
    })
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"真实时间步预测结果已保存至: {save_path}")
    return all_real_preds, all_real_trues


if __name__ == "__main__":
    # 1. 加载数据（包含掩码）
    (train_seqs, train_labels, train_masks, train_ids,
     test_seqs, test_labels, test_masks, test_ids) = load_and_preprocess_data(
        train_path='D:/LST/Core-main/Core-main/data/train_data_seq.csv',
        test_path='D:/LST/Core-main/Core-main/data/test_data_seq.csv',
        max_seq_len=12,
        encoding='utf-8-sig'
    )
    
    # 2. 初始化模型
    input_dim = train_seqs.shape[2]
    model = ManualRNN(input_dim=input_dim, hidden_dim=16)
    
    # 3. 训练模型（传入掩码）
    model = train_model(
        model, train_seqs, train_labels, train_masks,  # 新增train_masks参数
        epochs=30,
        lr=0.005
    )
    
    # 4. 测试模型（传入掩码）
    test_preds, test_trues = test_model(
        model, test_seqs, test_labels, test_masks, test_ids,  # 新增test_masks参数
        max_seq_len=12,
        save_path='D:/LST/Core-main/Core-main/data/manual_rnn_real_step_predictions2.csv'
    )
    # 1. 加载并预处理数据（每个时间步有标签）
    (train_seqs, train_labels, train_ids,
     test_seqs, test_labels, test_ids) = load_and_preprocess_data(
        train_path='D:/LST/Core-main/Core-main/data/train_data_seq.csv',
        test_path='D:/LST/Core-main/Core-main/data/test_data_seq.csv',
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
        save_path='D:/LST/Core-main/Core-main/data/manual_rnn_step_predictions.csv'
    )