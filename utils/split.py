import pandas as pd
import re
from sklearn.model_selection import train_test_split

def process_and_split_data(
    xqjd_path,
    xyyj_path,
    dccy_path,
    jxj_path,
    output_total_path,
    test_size=0.2,
    random_state=42,
    target_positive_ratio=None  # 新增：训练集目标XYYJ=1的占比（如0.1表示10%）
):
    """
    数据处理、合并、分割及训练集过采样的完整流程
    
    参数：
    - xqjd_path: XQJD.csv路径
    - xyyj_path: XYYJ.csv路径
    - dccy_path: DCCY.csv路径
    - jxj_path: JXJ.csv路径
    - output_total_path: 合并后总数据保存路径
    - test_size: 测试集占比（默认0.2）
    - random_state: 随机种子（确保可复现）
    - target_positive_ratio: 训练集XYYJ=1的目标占比（如0.1表示10%，None则不调整）
    """
    
    #############1. 处理XQJD#############
    df_xqjd = pd.read_csv(xqjd_path, encoding="utf-8-sig")
    df_xqjd = df_xqjd[["XH", "XNXQ", "GPA"]].copy()
    
    def convert_xnxq(xnxq_str):
        match = re.match(r"(\d+)年(春季|秋季)学期", str(xnxq_str))
        if not match:
            return None
        year = match.group(1)
        semester = match.group(2)
        return f"{year}1" if semester == "春季" else f"{year}2"
    
    df_xqjd["XNXQ"] = df_xqjd["XNXQ"].apply(convert_xnxq)
    df_xqjd = df_xqjd.dropna(subset=["XNXQ"])
    df_xqjd["XNXQ"] = df_xqjd["XNXQ"].astype(str)
    
    #############2. 处理XYYJ#############
    df_xyyj = pd.read_csv(xyyj_path, encoding="utf-8-sig")
    df_xyyj = df_xyyj.rename(columns={"XQN": "XNXQ"})
    df_xyyj = df_xyyj[["XH", "XNXQ", "XQSHJG", "JMSHJG", "JNCSHJG"]].copy()
    
    has_remove = df_xyyj[["XQSHJG", "JMSHJG", "JNCSHJG"]].apply(
        lambda x: x.str.contains("解除警示", na=False)
    ).any(axis=1)
    
    df_xyyj_filtered = df_xyyj[~has_remove].copy()
    # 确保XH和XNXQ为字符串，避免类型不匹配
    warning_pairs = set(zip(
        df_xyyj_filtered["XH"].astype(str), 
        df_xyyj_filtered["XNXQ"].astype(str)
    ))
    
    #############3. 处理DCCY#############
    df_dccy = pd.read_csv(dccy_path, encoding="utf-8-sig")
    df_dccy = df_dccy[["XH", "TSTAMP"]].copy()
    
    def convert_tstamp_to_xnxq(tstamp_str):
        try:
            year_month = int(str(tstamp_str)[:6])
        except:
            return None
        year = year_month // 100
        month = year_month % 100
        if month < 9:
            return f"{year}1"
        else:
            return f"{year}2"
    
    df_dccy["DCCY_XNXQ"] = df_dccy["TSTAMP"].apply(convert_tstamp_to_xnxq)
    df_dccy = df_dccy.dropna(subset=["DCCY_XNXQ"])
    df_dccy["DCCY_XNXQ"] = df_dccy["DCCY_XNXQ"].astype(str)
    
    dccy_new = df_dccy.groupby(["XH", "DCCY_XNXQ"]).size().reset_index()
    dccy_new.columns = ["XH", "XNXQ", "new_count"]
    
    dccy_cumulative = []
    for xh, group in dccy_new.groupby("XH"):
        group_sorted = group.sort_values(by="XNXQ").reset_index(drop=True)
        group_sorted["DCCY"] = group_sorted["new_count"].cumsum()
        dccy_cumulative.append(group_sorted[["XH", "XNXQ", "DCCY"]])
    
    dccy_count = pd.concat(dccy_cumulative, ignore_index=True)
    
    #############4. 处理JXJ#############
    df_jxj = pd.read_csv(jxj_path, encoding="utf-8-sig")
    jxj_counts = df_jxj["XH"].value_counts().reset_index()
    jxj_counts.columns = ["XH", "JXJ"]
    
    #############5. 合并总数据#############
    df = df_xqjd.copy()
    df = df.sort_values(by=["XH", "XNXQ"])
    dccy_count = dccy_count.sort_values(by=["XH", "XNXQ"])
    
    # 合并DCCY
    df = pd.merge(
        df, dccy_count, on=["XH", "XNXQ"], how="left"
    )
    df["DCCY"] = df["DCCY"].fillna(0).astype(int)
    
    # 合并JXJ
    df = df.merge(jxj_counts, on="XH", how="left")
    df["JXJ"] = df["JXJ"].fillna(0).astype(int)
    
    # 生成XYYJ标签
    df["XYYJ"] = df.apply(
        lambda row: 1 if (str(row["XH"]), str(row["XNXQ"])) in warning_pairs else 0,
        axis=1
    )
    
    # 保存总数据
    df.to_csv(output_total_path, encoding="utf-8-sig", index=False)
    print(f"总数据已保存至 {output_total_path}，共 {len(df)} 条记录")
    print(f"原始数据中XYYJ=1的占比：{df['XYYJ'].mean():.2%}")
    
    #############6. 分割训练集和测试集（分层抽样）#############
    # 测试集保持原始分布，仅训练集可能过采样
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["XYYJ"]  # 分层抽样确保测试集分布与原始一致
    )
    print(f"分割后：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条")
    print(f"测试集XYYJ=1的占比（原始分布）：{test_df['XYYJ'].mean():.2%}")
    
    #############7. 训练集过采样（提升XYYJ=1的占比）#############
    if target_positive_ratio is not None:
        # 验证目标比例合理性（0 < 目标占比 < 1）
        if not (0 < target_positive_ratio < 1):
            raise ValueError("target_positive_ratio必须在(0, 1)之间")
        
        # 分离训练集中的正样本（1）和负样本（0）
        train_pos = train_df[train_df["XYYJ"] == 1].copy()
        train_neg = train_df[train_df["XYYJ"] == 0].copy()
        n_pos = len(train_pos)
        n_neg = len(train_neg)
        
        # 计算当前训练集正样本占比
        current_ratio = n_pos / (n_pos + n_neg)
        print(f"原始训练集XYYJ=1的占比：{current_ratio:.2%}")
        
        if target_positive_ratio <= current_ratio:
            print(f"目标占比({target_positive_ratio:.2%})不大于当前占比，无需过采样")
        else:
            # 计算需要补充的正样本数量
            # 公式：(n_pos + add) / (n_pos + add + n_neg) = target_ratio
            # 推导得：add = (target_ratio * n_neg - (1 - target_ratio) * n_pos) / (1 - target_ratio)
            needed_add = int((target_positive_ratio * n_neg - (1 - target_positive_ratio) * n_pos) / (1 - target_positive_ratio))
            needed_add = max(0, needed_add)  # 确保非负
            
            # 随机过采样（复制正样本）
            if needed_add > 0:
                # 若需要补充的数量超过现有正样本，循环复制
                oversampled_pos = train_pos.sample(needed_add, replace=True, random_state=random_state)
                # 合并原始负样本和过采样后的正样本
                train_df = pd.concat([train_neg, train_pos, oversampled_pos], ignore_index=True)
                # 打乱顺序
                train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # 验证过采样后的比例
            new_ratio = train_df["XYYJ"].mean()
            print(f"过采样后训练集XYYJ=1的占比：{new_ratio:.2%}（目标：{target_positive_ratio:.2%}）")
    
    return train_df, test_df


# #############使用示例#############
if __name__ == "__main__":
    # 数据路径（根据实际情况修改）
    xqjd_path = "D:/LST/Core-main/Core-main/data/XQJD.csv"
    xyyj_path = "D:/LST/Core-main/Core-main/data/XYYJ.csv"
    dccy_path = "D:/LST/Core-main/Core-main/data/DCCY.csv"
    jxj_path = "D:/LST/Core-main/Core-main/data/JXJ.csv"
    output_total_path = "D:/LST/Core-main/Core-main/data/totaldata.csv"
    
    # 调用函数：设置训练集XYYJ=1的目标占比为10%（可根据需求调整）
    train_df, test_df = process_and_split_data(
        xqjd_path=xqjd_path,
        xyyj_path=xyyj_path,
        dccy_path=dccy_path,
        jxj_path=jxj_path,
        output_total_path=output_total_path,
        test_size=0.2,  # 测试集占20%
        random_state=42,
        target_positive_ratio=0.1  # 目标：训练集中XYYJ=1的占比为10%
    )
    
    # 可选：保存分割后的训练集和测试集
    train_df.to_csv("D:/LST/Core-main/Core-main/data/train_data.csv", encoding="utf-8-sig", index=False)
    test_df.to_csv("D:/LST/Core-main/Core-main/data/test_data.csv", encoding="utf-8-sig", index=False)
    print("训练集和测试集已保存")