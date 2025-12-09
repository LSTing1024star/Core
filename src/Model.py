import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

current_path = os.path.abspath(__file__)
parent_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_path)
from utils.split import split_imbalanced_data


class ModelTrainer:
    def __init__(self, train_path, test_path, target_col='XYYJ', encoding='utf-8-sig'):
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
        self.subdf_train_list=[]
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
        if "XH" in self.df_train.columns:
            self.df_train.drop(columns=["XH"],inplace=True)
        if "XH" in self.df_test.columns:
            self.df_test.drop(columns=["XH"],inplace=True)
        print(f"训练集形状: {self.df_train.shape}, 测试集形状: {self.df_test.shape}")
        print(f"训练集缺失值总数: {self.df_train.isnull().sum().sum()}")
        print(f"测试集缺失值总数: {self.df_test.isnull().sum().sum()}")


    def preprocess_data(self):
        """数据预处理：字符串转数值、特征缩放、调整目标列位置"""
        # 处理训练集
        self._numerical_convert(self.df_train)
        self._reorder_target_col(self.df_train)
        # self._feature_scaling(self.df_train)
        
        # 处理测试集
        self._numerical_convert(self.df_test)
        self._reorder_target_col(self.df_test)
        # self._feature_scaling(self.df_test)
        
        feature_cols = ["GPA", "DCCY", "JXJ"]
        self.df_train,self.df_test,_=normalize_data(
            train_df=self.df_train,
            test_df=self.df_test,
            feature_cols=feature_cols,
            method="minmax",
        )


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

    def _project(self,df):
        data=self.df.to_numpy()
        X = data[:,:-1]
        y = data[:,-1]
        X_train,X_test,y_train,y_test=train_test_split(
            X,y,
            test_size=test_size,
            random_state=random_state
        )

        # 每个子数据集单独用自己的scaler（避免子集间数据泄露）
        sub_scaler = StandardScaler()
        X_train_scaled = sub_scaler.fit_transform(X_train)
        X_test_scaled = sub_scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, sub_scaler

    def train_model(self, param_grid=None, cv=3, scoring='accuracy'):
        """训练模型并通过GridSearchCV寻找最优超参数"""
        # self.subdf_train_list=split_imbalanced_data(self.train_path,df_train=self.df_train,encoding="utf-8-sig")
        # print(f"共生成 {len(self.subdf_train_list)} 个子训练集")
        
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


    def predict(self, save_path='D:/LST/Core-main/Core-main/data/test_with_predictions3.csv'):
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
        numeric_cols = ['GPA', 'DCCY', 'JXJ']
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
                    
    def analyze_feature_importance(self):
        """分析并打印特征重要性"""
        if self.best_model is None:
            print("请先训练模型（调用train_model方法）")
            return
        
        # 获取特征名称（排除目标列）
        feature_names = self.df_train.columns[:-1].tolist()
        
        # 获取特征重要性得分
        importances = self.best_model.feature_importances_
        
        # 组合特征名称和重要性，并按重要性排序
        feature_importance = pd.DataFrame({
            '特征名称': feature_names,
            '重要性得分': importances
        }).sort_values(by='重要性得分', ascending=False)
        
        # 打印结果
        print("\n【特征重要性排序】")
        print(feature_importance.round(4))  # 保留4位小数


def normalize_data(
    train_df,
    test_df,
    feature_cols,  # 需要归一化的特征列名列表
    method="minmax",  # 归一化方法："minmax" 或 "standard"
    train_save_path=None,
    test_save_path=None
):
    """
    对训练集和测试集的特征进行归一化（用训练集的统计量）
    
    参数：
    - train_df: 训练集DataFrame
    - test_df: 测试集DataFrame
    - feature_cols: 需要归一化的特征列（如 ["GPA", "DCCY", "JXJ"]）
    - method: 归一化方法，"minmax"（[0,1]区间）或 "standard"（均值0，标准差1）
    - train_save_path: 归一化后的训练集保存路径（None则不保存）
    - test_save_path: 归一化后的测试集保存路径（None则不保存）
    """
    # 1. 检查特征列是否存在
    missing_cols = [col for col in feature_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(f"训练集中缺少特征列：{missing_cols}")
    
    # 2. 初始化归一化器
    if method == "minmax":
        scaler = MinMaxScaler()  # Min-Max归一化：(x - min)/(max - min)
    elif method == "standard":
        scaler = StandardScaler()  # Z-score标准化：(x - mean)/std
    else:
        raise ValueError("method必须是'minmax'或'standard'")
    
    # 3. 用训练集拟合归一化器（只使用训练集的统计量，避免数据泄露）
    scaler.fit(train_df[feature_cols])
    
    # 4. 转换训练集和测试集
    train_norm = train_df.copy()
    test_norm = test_df.copy()
    
    # 对特征列进行归一化
    train_norm[feature_cols] = scaler.transform(train_df[feature_cols])
    test_norm[feature_cols] = scaler.transform(test_df[feature_cols])  # 用训练集的scaler转换测试集
    
    # 5. 打印归一化前后的统计量（验证效果）
    print(f"\n【{method}归一化后 - 训练集特征统计量】")
    print(train_norm[feature_cols].describe().round(4))
    
    # 6. 保存结果（如果指定路径）
    if train_save_path:
        train_norm.to_csv(train_save_path, encoding="utf-8-sig", index=False)
        print(f"归一化后的训练集已保存至 {train_save_path}")
    if test_save_path:
        test_norm.to_csv(test_save_path, encoding="utf-8-sig", index=False)
        print(f"归一化后的测试集已保存至 {test_save_path}")
    
    return train_norm, test_norm, scaler  # 返回归一化后的数据和拟合好的scaler（后续可用于新数据）