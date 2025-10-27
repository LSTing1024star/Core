from Model import ModelTrainer
from ConfidenceAnalyzer import ConfidenceAnalyzer
from ExplainabilityAnalyzer import ExplainabilityAnalyzer

if __name__ == "__main__":
    # 1. 模型训练与预测
    trainer = ModelTrainer(
        train_path='data_temp/train.csv',
        test_path='data_temp/test.csv',
        target_col='学业预警'
    )
    trainer.load_data()
    trainer.preprocess_data()
    trainer.split_data()
    trainer.train_model()  # 使用默认超参数网格
    trainer.predict()  # 预测并保存结果
    
    # 2. 置信度分析
    confidence_analyzer = ConfidenceAnalyzer(
        model=trainer.best_model,
        X_test=trainer.X_test_true,
        y_true=trainer.y_test_true,
        pred_proba=trainer.pred_proba,
        feature_names=trainer.df_train.columns[:-1].tolist()  # 特征名称（排除目标列）
    )
    confidence_analyzer.run_full_analysis()
    
    # 3. 可解释性分析
    explain_analyzer = ExplainabilityAnalyzer(
        model=trainer.best_model,
        X_test=trainer.X_test_true,
        y_true=trainer.y_test_true,
        pred_labels=trainer.pred_labels,
        feature_names=trainer.df_train.columns[:-1].tolist()  # 特征名称
    )
    explain_analyzer.run_full_analysis()