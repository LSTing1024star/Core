import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class ConfidenceAnalyzer:
    def __init__(self, model, X_test, y_true, pred_proba, feature_names=None):
        """
        初始化置信度分析器
        :param model: 训练好的模型
        :param X_test: 测试集特征
        :param y_true: 测试集真实标签
        :param pred_proba: 测试集预测正类概率
        :param feature_names: 特征名称列表（可选）
        """
        self.model = model
        self.X_test = X_test
        self.y_true = y_true
        self.pred_proba = pred_proba  # 正类概率
        self.feature_names = feature_names if feature_names is not None else [f"特征{i}" for i in range(X_test.shape[1])]


    def plot_calibration_curve(self, n_bins=10):
        """绘制校准曲线：评估预测概率与实际正确率的匹配程度"""
        prob_true, prob_pred = calibration_curve(self.y_true, self.pred_proba, n_bins=n_bins)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='模型校准曲线')
        plt.plot([0, 1], [0, 1], 'k--', label='理想校准线')
        plt.xlabel('平均预测概率')
        plt.ylabel('实际正例比例')
        plt.title('预测概率校准曲线')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


    def calculate_brier_score(self):
        """计算Brier分数（衡量概率准确性，值越小越好）"""
        brier = brier_score_loss(self.y_true, self.pred_proba)
        print(f"Brier分数（正类）: {brier:.4f} (范围0-1，越小越准确)")
        return brier


    def calculate_ece(self, n_bins=10):
        """计算预期校准误差（ECE）：量化整体校准偏差"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (self.pred_proba >= bin_lower) & (self.pred_proba < bin_upper)
            if np.sum(in_bin) == 0:
                continue
            frac_pos = np.mean(self.y_true[in_bin])  # 区间内实际正例比例
            mean_prob = np.mean(self.pred_proba[in_bin])  # 区间内平均预测概率
            ece += np.abs(frac_pos - mean_prob) * (np.sum(in_bin) / len(self.y_true))
        print(f"预期校准误差（ECE）: {ece:.4f} (越小说明校准越好)")
        return ece


    def analyze_confidence_vs_accuracy(self, n_bins=10):
        """分析不同置信度区间的实际准确率"""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(self.pred_proba, bins)  # 每个样本所属区间
        
        bin_accuracy = []
        bin_counts = []
        for i in range(1, len(bins)):
            mask = (bin_indices == i)
            if np.sum(mask) == 0:
                bin_accuracy.append(np.nan)
                bin_counts.append(0)
                continue
            # 区间内准确率 = 正确预测数 / 区间样本数
            acc = np.mean(self.y_true[mask] == (self.pred_proba[mask] >= 0.5))  # 0.5为决策阈值
            bin_accuracy.append(acc)
            bin_counts.append(np.sum(mask))
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.bar(bins[:-1], bin_accuracy, width=0.1, align='edge', alpha=0.7, label='实际准确率')
        plt.plot(bins[:-1], bins[:-1], 'r--', label='理想：置信度=准确率')
        plt.xlabel('预测置信度区间')
        plt.ylabel('实际准确率')
        plt.title('不同置信度区间的实际准确率分布')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        
        # 高置信度样本准确率
        high_conf_mask = (self.pred_proba > 0.8) | (self.pred_proba < 0.2)
        if np.sum(high_conf_mask) > 0:
            high_conf_acc = np.mean(self.y_true[high_conf_mask] == (self.pred_proba[high_conf_mask] >= 0.5))
            print(f"高置信度样本（>0.8或<0.2）准确率: {high_conf_acc:.4f}")


    def run_full_analysis(self):
        """运行完整的置信度分析流程"""
        print("\n===== 开始置信度分析 =====")
        self.plot_calibration_curve()
        self.calculate_brier_score()
        self.calculate_ece()
        self.analyze_confidence_vs_accuracy()
        print("===== 置信度分析结束 =====")