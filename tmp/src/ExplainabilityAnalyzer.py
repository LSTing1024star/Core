import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import plot_partial_dependence


class ExplainabilityAnalyzer:
    def __init__(self, model, X_test, y_true, pred_labels, feature_names=None):
        """
        初始化可解释性分析器
        :param model: 训练好的模型（需支持特征重要性或SHAP解释）
        :param X_test: 测试集特征
        :param y_true: 测试集真实标签
        :param pred_labels: 测试集预测标签
        :param feature_names: 特征名称列表（必须，用于解释）
        """
        self.model = model
        self.X_test = X_test
        self.y_true = y_true
        self.pred_labels = pred_labels
        self.feature_names = feature_names if feature_names is not None else [f"特征{i}" for i in range(X_test.shape[1])]
        self.shap_explainer = None  # SHAP解释器（延迟初始化）


    def plot_global_feature_importance(self, top_n=5):
        """绘制全局特征重要性（基于模型自带的特征重要性）"""
        if not hasattr(self.model, 'feature_importances_'):
            print("模型不支持特征重要性计算")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]  # 从高到低排序
        
        # 取Top N特征
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [self.feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_features, top_importances, color='skyblue')
        plt.xlabel('特征重要性得分')
        plt.title(f'全局Top{top_n}特征重要性')
        plt.gca().invert_yaxis()  # 重要性高的在上方
        plt.grid(alpha=0.3, axis='x')
        plt.show()
        
        # 打印详细得分
        print(f"Top{top_n}特征重要性:")
        for feat, imp in zip(top_features, top_importances):
            print(f"{feat}: {imp:.4f}")


    def init_shap_explainer(self):
        """初始化SHAP解释器（针对树模型优化）"""
        if self.shap_explainer is None:
            self.shap_explainer = shap.TreeExplainer(self.model)
            print("SHAP解释器初始化完成")


    def plot_shap_summary(self, sample_size=100):
        """绘制SHAP值总结图（全局特征影响分布）"""
        self.init_shap_explainer()
        # 取部分样本加快计算
        sample_indices = np.random.choice(len(self.X_test), min(sample_size, len(self.X_test)), replace=False)
        shap_values = self.shap_explainer.shap_values(self.X_test[sample_indices])
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            self.X_test[sample_indices], 
            feature_names=self.feature_names,
            plot_type="dot"  # 点图：颜色表示特征值大小，X轴表示对预测的影响
        )
        plt.title('SHAP值总结图（特征对预测的影响分布）')
        plt.show()


    def plot_single_sample_explanation(self, sample_idx=None):
        """解释单个样本的预测结果（优先选择预测为正类的样本）"""
        self.init_shap_explainer()
        
        # 自动选择一个预测为正类的样本（若存在）
        if sample_idx is None:
            alert_samples = np.where(self.pred_labels == 1)[0]
            if len(alert_samples) == 0:
                print("无预测为正类的样本，将随机选择一个样本")
                sample_idx = np.random.choice(len(self.X_test), 1)[0]
            else:
                sample_idx = alert_samples[0]  # 取第一个预警样本
        
        # 计算该样本的SHAP值
        shap_values = self.shap_explainer.shap_values(self.X_test[sample_idx:sample_idx+1])[0]
        
        # 绘制力导向图
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            self.shap_explainer.expected_value,  # 模型整体均值
            shap_values,  # 该样本的SHAP值
            self.X_test[sample_idx],  # 该样本的特征值
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f'样本 {sample_idx} 的预测解释（红色：推动正类，蓝色：抑制正类）')
        plt.show()


    def plot_partial_dependence(self, top_n=2):
        """绘制部分依赖图（PDP）：分析Top N特征与预测概率的关系"""
        # 获取Top N重要特征的索引
        if not hasattr(self.model, 'feature_importances_'):
            print("模型不支持特征重要性，无法绘制部分依赖图")
            return
        
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_n]  # Top N特征索引
        
        # 绘制PDP
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_partial_dependence(
            self.model, 
            self.X_test, 
            features=top_indices,
            feature_names=self.feature_names,
            ax=ax
        )
        plt.suptitle(f'Top{top_n}特征与预测概率的关系（部分依赖图）', y=1.02)
        plt.grid(alpha=0.3)
        plt.show()


    def run_full_analysis(self):
        """运行完整的可解释性分析流程"""
        print("\n===== 开始可解释性分析 =====")
        self.plot_global_feature_importance()
        self.plot_shap_summary()
        self.plot_single_sample_explanation()
        self.plot_partial_dependence()
        print("===== 可解释性分析结束 =====")