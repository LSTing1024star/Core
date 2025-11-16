from ManualRNN import ManualRNN
from sklearn.metrics import confusion_matrix, classification_report 
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
current_file_path = os.path.abspath(__file__)  # 结果类似：parent_dir/A/c.py
a_dir = os.path.dirname(current_file_path)     # 结果类似：parent_dir/A
parent_dir = os.path.dirname(a_dir)            # 结果类似：parent_dir
sys.path.append(parent_dir)
from utils.buildSeq import load_and_preprocess_data


def train_model(model, train_sequences, train_labels, epochs=20, lr=0.01):
    # 计算类别权重（自动适配数据不平衡比例）
    n_pos = np.sum(train_labels == 1)  # 少数类（1）的数量
    n_neg = len(train_labels) - n_pos  # 多数类（0）的数量
    pos_weight = n_neg / n_pos  # 少数类权重 = 多数类数量 / 少数类数量（确保权重>1）
    print(f"类别比例（0:{1}）：{n_neg}:{n_pos}，少数类权重：{pos_weight:.2f}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        for seq, label in zip(train_sequences, train_labels):
            y_pred, h_states = model.forward(seq)
            # 带权重的交叉熵损失（少数类错误损失放大）
            if label == 1:
                # 对少数类（1）的损失乘以权重
                loss = -pos_weight * label * np.log(y_pred + 1e-8) - (1 - label) * np.log(1 - y_pred + 1e-8)
            else:
                # 多数类（0）损失保持不变
                loss = -label * np.log(y_pred + 1e-8) - (1 - label) * np.log(1 - y_pred + 1e-8)
            total_loss += loss
            model.backward(seq, h_states, y_pred, label)
            model.update_weights(lr=lr)
        
        avg_loss = total_loss / len(train_sequences)
        print(f"Epoch {epoch+1}/{epochs} | 平均损失: {avg_loss:.4f}")
    return model
def test_model(model, test_sequences, test_labels, test_ids, save_path='manual_rnn_predictions.csv'):
    """测试模型并输出评估结果"""
    preds = []
    probs = []
    # 逐样本预测
    for seq in test_sequences:
        y_pred, _ = model.forward(seq)
        probs.append(y_pred)
        preds.append(1 if y_pred >= 0.5 else 0)  # 阈值0.5判断标签
    
    # 评估指标
    print("\n混淆矩阵:")
    print(confusion_matrix(test_labels, preds))
    print("\n分类报告:")
    print(classification_report(test_labels, preds, digits=3))
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'XH': test_ids,
        '真实标签': test_labels,
        '预测标签': preds,
        '预测概率': probs
    })
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存至: {save_path}")
    return preds, probs

if __name__ == "__main__":
    # 1. 加载并预处理数据（使用按时间分割后的train和test）
    (train_seqs, train_labels, train_ids,
     test_seqs, test_labels, test_ids) = load_and_preprocess_data(
        train_path='D:/LST/Core-main/data/train_data_seq.csv',    # 按时间分割的训练集
        test_path='D:/LST/Core-main/data/test_data_seq.csv',      # 按时间分割的测试集
        max_seq_len=12,                  # 每个学生最多3个学期的序列
        encoding='utf-8-sig'
    )
    
    # 2. 初始化手动RNN（输入维度=3，隐藏层维度=8）
    input_dim = train_seqs.shape[2]  # 特征数（如3）
    model = ManualRNN(input_dim=input_dim, hidden_dim=8)
    
    # 3. 训练模型
    model = train_model(
        model, train_seqs, train_labels,
        epochs=20,  # 短序列训练轮次不宜过多
        lr=0.01     # 学习率
    )
    
    # 4. 测试模型
    test_preds, test_probs = test_model(
        model, test_seqs, test_labels, test_ids
    )



# from Model import ModelTrainer
# from ConfidenceAnalyzer import ConfidenceAnalyzer
# from ExplainabilityAnalyzer import ExplainabilityAnalyzer

# if __name__ == "__main__":
#     # 1. 模型训练与预测
#     trainer = ModelTrainer(
#         train_path='data_temp/train_data.csv',
#         test_path='data_temp/test_data.csv',
#         target_col='XYYJ'
#     )
#     trainer.load_data()
#     trainer.preprocess_data()
#     trainer.split_data()
#     trainer.train_model()  # 使用默认超参数网格
#     trainer.predict()  # 预测并保存结果
    
#     # 2. 置信度分析
#     confidence_analyzer = ConfidenceAnalyzer(
#         model=trainer.best_model,
#         X_test=trainer.X_test_true,
#         y_true=trainer.y_test_true,
#         pred_proba=trainer.pred_proba,
#         feature_names=trainer.df_train.columns[:-1].tolist()  # 特征名称（排除目标列）
#     )
#     confidence_analyzer.run_full_analysis()
    
#     # 3. 可解释性分析
#     explain_analyzer = ExplainabilityAnalyzer(
#         model=trainer.best_model,
#         X_test=trainer.X_test_true,
#         y_true=trainer.y_test_true,
#         pred_labels=trainer.pred_labels,
#         feature_names=trainer.df_train.columns[:-1].tolist()  # 特征名称
#     )
#     explain_analyzer.run_full_analysis()