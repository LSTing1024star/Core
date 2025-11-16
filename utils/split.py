import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_oversample(
    totaldata_path,
    train_save_path,
    test_save_path,
    test_size=0.2,
    random_state=42,
    target_positive_ratio=None  # 训练集XYYJ=1的目标占比（如0.1表示10%）
):
    """
    从totaldata.csv读取数据，分割为训练集和测试集，并对训练集进行过采样提升正样本占比
    
    参数：
    - totaldata_path: totaldata.csv的路径
    - train_save_path: 训练集保存路径
    - test_save_path: 测试集保存路径
    - test_size: 测试集占比（默认0.2）
    - random_state: 随机种子（确保可复现）
    - target_positive_ratio: 训练集XYYJ=1的目标占比（None则不调整）
    """
    # 1. 读取totaldata.csv
    df = pd.read_csv(totaldata_path, encoding="utf-8-sig")
    # 检查是否存在XYYJ列
    if "XYYJ" not in df.columns:
        raise ValueError("totaldata.csv中必须包含'XYYJ'列")
    print(f"已读取totaldata.csv，共 {len(df)} 条记录")
    print(f"原始数据中XYYJ=1的占比：{df['XYYJ'].mean():.2%}")
    
    # 2. 分层抽样分割训练集和测试集（测试集保持原始分布）
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["XYYJ"]  # 确保测试集与原始数据分布一致
    )
    print(f"分割后：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条")
    print(f"测试集XYYJ=1的占比（原始分布）：{test_df['XYYJ'].mean():.2%}")
    
    # 3. 训练集过采样（提升XYYJ=1的占比）
    if target_positive_ratio is not None:
        # 验证目标比例合理性
        if not (0 < target_positive_ratio < 1):
            raise ValueError("target_positive_ratio必须在(0, 1)之间")
        
        # 分离训练集中的正样本（1）和负样本（0）
        train_pos = train_df[train_df["XYYJ"] == 1].copy()
        train_neg = train_df[train_df["XYYJ"] == 0].copy()
        n_pos = len(train_pos)
        n_neg = len(train_neg)
        
        # 计算当前训练集正样本占比
        current_ratio = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0
        print(f"原始训练集XYYJ=1的占比：{current_ratio:.2%}")
        
        if target_positive_ratio <= current_ratio:
            print(f"目标占比({target_positive_ratio:.2%})不大于当前占比，无需过采样")
        else:
            # 计算需要补充的正样本数量
            needed_add = int((target_positive_ratio * n_neg - (1 - target_positive_ratio) * n_pos) / (1 - target_positive_ratio))
            needed_add = max(0, needed_add)  # 确保非负
            
            # 随机过采样（复制正样本）
            if needed_add > 0:
                oversampled_pos = train_pos.sample(needed_add, replace=True, random_state=random_state)
                # 合并并打乱顺序
                train_df = pd.concat([train_neg, train_pos, oversampled_pos], ignore_index=True)
                train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # 验证过采样后的比例
            new_ratio = train_df["XYYJ"].mean()
            print(f"过采样后训练集XYYJ=1的占比：{new_ratio:.2%}（目标：{target_positive_ratio:.2%}）")
    
    # 4. 保存训练集和测试集
    train_df.to_csv(train_save_path, encoding="utf-8-sig", index=False)
    test_df.to_csv(test_save_path, encoding="utf-8-sig", index=False)
    print(f"训练集已保存至 {train_save_path}")
    print(f"测试集已保存至 {test_save_path}")
    
    return train_df, test_df


import pandas as pd
import numpy as np

def split_train_test_by_time(
    data_path,
    id_col='XH',          # 学号列名
    term_col='XQXN',      # 学期列名（字符串格式，如201601）
    target_col='学业预警',  # 目标列名
    test_ratio=0.2,       # 测试集占比（如0.2表示20%数据作为test）
    save_train_path='train_data.csv',
    save_test_path='test_data.csv',
    encoding='gbk'
):
    """
    按时间顺序分割时序数据集为train和test
    """
    # 1. 加载数据集
    df = pd.read_csv(data_path, encoding=encoding).fillna(0)
    print(f"原始数据集形状：{df.shape}")
    
    # 2. 检查必要列
    required_cols = [id_col, term_col, target_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"数据集缺少必要列：{col}")
    
    # 3. 按学期排序，确定时间顺序
    # 转换学期列为字符串（避免数字格式干扰排序）
    df[term_col] = df[term_col].astype(str)
    # 提取所有不重复的学期，并按时间升序排序
    unique_terms = sorted(df[term_col].unique())
    print(f"所有学期（按时间排序）：{unique_terms}")
    
    # 4. 确定分割点（按比例选择较晚的学期作为test）
    # 例如：80%早学期为train，20%晚学期为test
    split_idx = int(len(unique_terms) * (1 - test_ratio))
    if split_idx <= 0 or split_idx >= len(unique_terms):
        raise ValueError("测试集比例不合理，导致分割点无效（建议test_ratio在0.1-0.3之间）")
    
    train_terms = unique_terms[:split_idx]  # 训练集包含的学期（早）
    test_terms = unique_terms[split_idx:]   # 测试集包含的学期（晚）
    print(f"训练集学期：{train_terms}（共{len(train_terms)}个学期）")
    print(f"测试集学期：{test_terms}（共{len(test_terms)}个学期）")
    
    # 5. 分割数据
    train_df = df[df[term_col].isin(train_terms)].copy()
    test_df = df[df[term_col].isin(test_terms)].copy()
    
    # 6. 验证分割有效性（test的学期均晚于train）
    max_train_term = max(train_terms)
    min_test_term = min(test_terms)
    if max_train_term >= min_test_term:
        raise ValueError(f"分割错误：训练集最晚学期{max_train_term} >= 测试集最早学期{min_test_term}")
    
    # 7. 打印分割后信息
    print(f"训练集形状：{train_df.shape}，测试集形状：{test_df.shape}")
    print(f"训练集目标列分布：\n{train_df[target_col].value_counts(normalize=True).round(3)}")
    print(f"测试集目标列分布：\n{test_df[target_col].value_counts(normalize=True).round(3)}")
    
    # 8. 保存分割结果
    train_df.to_csv(save_train_path, index=False, encoding=encoding)
    test_df.to_csv(save_test_path, index=False, encoding=encoding)
    print(f"训练集已保存至：{save_train_path}")
    print(f"测试集已保存至：{save_test_path}")
    
    return train_df, test_df

# #############使用示例#############
if __name__ == "__main__":
    # 路径设置（根据实际情况修改）
    totaldata_path = "D:/LST/Core-main/Core-main/data/totaldata.csv"  # 已生成的总数据
    train_save_path = "D:/LST/Core-main/Core-main/data/train_data.csv"  # 训练集保存路径
    test_save_path = "D:/LST/Core-main/Core-main/data/test_data.csv"    # 测试集保存路径
    
    # 调用函数：设置训练集XYYJ=1的目标占比为10%（可调整为0.2即20%等）
    # train_df, test_df = split_and_oversample(
    #     totaldata_path=totaldata_path,
    #     train_save_path=train_save_path,
    #     test_save_path=test_save_path,
    #     test_size=0.2,  # 测试集占20%
    #     random_state=42,
    #     target_positive_ratio=0.1  # 目标：训练集中正样本占比10%
    # )

    train_df, test_df = split_train_test_by_time(
        data_path=totaldata_path,
        test_ratio=0.2,
        save_train_path=train_save_path,
        save_test_path=test_save_path,
        encoding="utf-8-sig"
    )

