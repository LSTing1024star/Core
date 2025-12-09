# 1. 必要的基础库导入（放在顶部）
import argparse  # 用于命令行参数解析


def main(model_type="base"):
    """主函数：根据模型类型执行流程"""
    # 2. 条件导入模型训练器（根据模型类型选择）
    if model_type == "base":
        from Model import ModelTrainer as Trainer  # 导入基础模型训练器
    elif model_type == "seq":
        from SeqModel import SeqModelTrainer as Trainer  # 导入时序模型训练器
    else:
        raise ValueError(f"不支持的模型类型：{model_type}，可选值为'base'或'seq'")

    # 3. 初始化训练器（基础/时序模型参数兼容）
    trainer = Trainer(
        train_path='data_temp/train_data.csv',
        test_path='data_temp/test_data.csv',
        target_col='XYYJ',
        # 时序模型专属参数（仅当model_type为seq时生效）
        id_col='XH' if model_type == 'seq' else None,
        term_col='XQ' if model_type == 'seq' else None,
        max_seq_len=3 if model_type == 'seq' else None
    )

    # 4. 模型训练与预测
    trainer.load_data()
    trainer.preprocess_data()
    trainer.split_data()
    trainer.train_model()
    trainer.predict()

    # 5. 导入分析器（在需要时导入，避免提前加载）
    from ConfidenceAnalyzer import ConfidenceAnalyzer
    from ExplainabilityAnalyzer import ExplainabilityAnalyzer

    # 6. 置信度分析
    confidence_analyzer = ConfidenceAnalyzer(
        model=trainer.best_model,
        X_test=trainer.X_test_true,
        y_true=trainer.y_test_true,
        pred_proba=trainer.pred_proba,
        feature_names=trainer.df_train.columns[:-1].tolist()
    )
    confidence_analyzer.run_full_analysis()

    # 7. 可解释性分析
    explain_analyzer = ExplainabilityAnalyzer(
        model=trainer.best_model,
        X_test=trainer.X_test_true,
        y_true=trainer.y_test_true,
        pred_labels=trainer.pred_labels,
        feature_names=trainer.df_train.columns[:-1].tolist()
    )
    explain_analyzer.run_full_analysis()


if __name__ == "__main__":
    # 解析命令行参数，指定模型类型
    parser = argparse.ArgumentParser(description="选择基础模型或时序模型")
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "seq"],
                        help="模型类型：'base'（基础模型）或'seq'（时序模型）")
    args = parser.parse_args()
    
    # 执行主流程
    main(model_type=args.model_type)


# from Model import ModelTrainer
# # from ConfidenceAnalyzer import ConfidenceAnalyzer
# # from ExplainabilityAnalyzer import ExplainabilityAnalyzer

# if __name__ == "__main__":
#     # 1. 模型训练与预测
#     trainer = ModelTrainer(
#         train_path='D:/LST/Core-main/Core-main/data/train_data.csv',
#         test_path='D:/LST/Core-main/Core-main/data/test_data.csv',
#         target_col='XYYJ'
#     )

#     # 4. 模型训练与预测
#     trainer.load_data()
#     trainer.preprocess_data()
#     trainer.split_data()
#     trainer.train_model()  # 使用默认超参数网格
#     trainer.predict()  # 预测并保存结果
    
#     # # 2. 置信度分析
#     # confidence_analyzer = ConfidenceAnalyzer(
#     #     model=trainer.best_model,
#     #     X_test=trainer.X_test_true,
#     #     y_true=trainer.y_test_true,
#     #     pred_proba=trainer.pred_proba,
#     #     feature_names=trainer.df_train.columns[:-1].tolist()  # 特征名称（排除目标列）
#     # )
#     # confidence_analyzer.run_full_analysis()
    
#     # # 3. 可解释性分析
#     # explain_analyzer = ExplainabilityAnalyzer(
#     #     model=trainer.best_model,
#     #     X_test=trainer.X_test_true,
#     #     y_true=trainer.y_test_true,
#     #     pred_labels=trainer.pred_labels,
#     #     feature_names=trainer.df_train.columns[:-1].tolist()  # 特征名称
#     # )
#     # explain_analyzer.run_full_analysis()
