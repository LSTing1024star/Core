import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error


class ModelTrainer:
    def __init__(self, train_path, test_path, target_col='学业预警', encoding='gbk'):
        """
        初始化模型训练器
        :param train_path: 训练数据路径
        :param test_path: 测试数据路径
        :param target_col: 目标列名称（二分类标签）
        :param encoding: 文件编码
        """
        self.train_path = train_path
        self.test_path = test_path
        self.target_col = target_col
        self.encoding = encoding
        
        # 初始化变量
        self.df_train = None  # 训练数据
        self.df_test = None  # 测试数据（原始）
        self.X_train = None  # 训练特征（标准化后）
        self.X_test = None  # 验证特征（标准化后）
        self.y_train = None  # 训练标签
        self.y_test = None  # 验证标签
        self.X_test_true = None  # 测试集特征（标准化后）
        self.y_test_true = None  # 测试集真实标签
        self.scaler = StandardScaler()  # 标准化器
        self.best_model = None  # 最优模型
        self.pred_labels = None  # 测试集预测标签
        self.pred_proba = None  # 测试集预测概率（正类）


    def load_data(self):
        """加载并初始化训练集和测试集"""
        self.df_train = pd.read_csv(self.train_path, encoding=self.encoding).fillna(0)
        self.df_test = pd.read_csv(self.test_path, encoding=self.encoding).fillna(0)
        print(f"训练集形状: {self.df_train.shape}, 测试集形状: {self.df_test.shape}")
        print(f"训练集缺失值总数: {self.df_train.isnull().sum().sum()}")
        print(f"测试集缺失值总数: {self.df_test.isnull().sum().sum()}")


    def preprocess_data(self):
        """数据预处理：字符串转数值、特征缩放、调整目标列位置"""
        # 处理训练集
        self._numerical_convert(self.df_train)
        self._reorder_target_col(self.df_train)
        self._feature_scaling(self.df_train)
        
        # 处理测试集
        self._numerical_convert(self.df_test)
        self._reorder_target_col(self.df_test)
        self._feature_scaling(self.df_test)


    def split_data(self, test_size=0.3, random_state=4):
        """分割训练集为训练/验证集，并提取测试集特征和标签"""
        # 训练集拆分
        data_train = self.df_train.to_numpy()
        X = data_train[:, :-1]
        y = data_train[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 标准化特征
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # 测试集特征和标签
        data_test = self.df_test.to_numpy()
        self.X_test_true = self.scaler.transform(data_test[:, :-1])
        self.y_test_true = data_test[:, -1]


    def train_model(self, param_grid=None, cv=3, scoring='accuracy'):
        """训练模型并通过GridSearchCV寻找最优超参数"""
        if param_grid is None:
            # 默认超参数网格
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'criterion': ['friedman_mse', 'squared_error'],
                'max_depth': [3, 5, 8],
                'max_features': ['log2', 'sqrt']
            }
        
        # 初始化模型和网格搜索
        gbc = GradientBoostingClassifier()
        grid_search = GridSearchCV(
            estimator=gbc,
            param_grid=param_grid,
            cv=cv,
            verbose=1,
            n_jobs=-1,
            scoring=scoring
        )
        
        # 训练并获取最优模型
        grid_search.fit(self.X_train, self.y_train.ravel())
        self.best_model = grid_search.best_estimator_
        print(f"最优超参数: {grid_search.best_params_}")


    def predict(self, save_path='data_temp/test_with_predictions.csv'):
        """对测试集预测并保存结果"""
        self.pred_labels = self.best_model.predict(self.X_test_true)  # 预测标签
        self.pred_proba = self.best_model.predict_proba(self.X_test_true)[:, 1]  # 正类概率
        
        # 保存带预测结果的原始测试集
        original_df_test = pd.read_csv(self.test_path, encoding=self.encoding).fillna(0)
        original_df_test['预测标签'] = self.pred_labels
        original_df_test.to_csv(save_path, index=False)
        print(f"预测结果已保存至: {save_path}")
        
        # 输出评估指标
        self._evaluate()


    def _evaluate(self):
        """评估模型在测试集上的性能"""
        print("\n混淆矩阵:")
        print(confusion_matrix(self.y_test_true, self.pred_labels))
        print("\n分类报告:")
        print(classification_report(self.y_test_true, self.pred_labels, digits=3))
        mse = mean_squared_error(self.y_test_true, self.pred_labels)
        print(f"RMSE: {np.sqrt(mse):.4f}")


    # 内部工具方法
    def _numerical_convert(self, df):
        """将字符串特征转换为数值型"""
        numeric_cols = ['GPA', '是否获得奖学金', '是否参加大创', '是否参加暑期征文']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


    def _reorder_target_col(self, df):
        """将目标列移至最后"""
        if self.target_col in df.columns:
            target = df[self.target_col]
            df.drop(columns=[self.target_col], inplace=True)
            df[self.target_col] = target


    def _feature_scaling(self, df):
        """特征缩放：GPA单独归一化，其他特征根据范围选择标准化/归一化"""
        exclude_cols = [self.target_col]
        for col in df.columns:
            if col in exclude_cols:
                continue
            if col == 'GPA':
                # GPA归一化到0-1（假设原始范围0-5）
                df[col] = df[col] / 5.0
                continue
            # 其他特征处理
            max_val = df[col].max()
            min_val = df[col].min()
            if max_val > 6:
                # 大范围特征：标准化（减去均值除以最大值）
                mean_val = df[col].mean()
                if max_val != 0:
                    df[col] = (df[col] - mean_val) / max_val
            else:
                # 小范围特征：归一化到0-1
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)