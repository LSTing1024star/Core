import pandas as pd
import re  # 用于处理XNXQ的格式转换


# -------------------------- 1. 处理XQJD.csv --------------------------
# 读取文件并提取所需列
df_xqjd = pd.read_csv("XQJD.csv", encoding="gbk")  # 假设编码为gbk，根据实际情况调整
df_xqjd = df_xqjd[["XH", "XNXQ", "GPA"]].copy()  # 提取需要的列

# 处理XNXQ格式："2021年秋季学期"→20212，"2021年春季学期"→20211
def convert_xnxq(xnxq_str):
    # 用正则提取年份和学期（匹配"YYYY年春季学期"或"YYYY年秋季学期"）
    match = re.match(r"(\d+)年(春季|秋季)学期", str(xnxq_str))
    if not match:
        return None  # 格式不匹配的返回空（可根据需求调整）
    year = match.group(1)  # 提取年份（如"2021"）
    semester = match.group(2)  # 提取学期（"春季"或"秋季"）
    # 春季→1，秋季→2，拼接为数字（如2021+1=20211）
    return f"{year}1" if semester == "春季" else f"{year}2"

# 应用转换函数
df_xqjd["XNXQ"] = df_xqjd["XNXQ"].apply(convert_xnxq)
# 过滤掉转换失败的行（如果需要保留可删除此行）
df_xqjd = df_xqjd.dropna(subset=["XNXQ"])
df_xqjd["XNXQ"] = df_xqjd["XNXQ"]


# -------------------------- 2. 处理XYJJ.csv --------------------------
# 读取文件并提取所需列
df_xyjj = pd.read_csv("XYJJ.csv", encoding="gbk")
df_xyjj = df_xyjj[["XH", "XQBM", "XQSHJG", "JMSHJG", "JWCSHJG"]].copy()

# 过滤掉包含"解除警示"的行（三个字段中任意一个出现则删除）
# 先判断每个字段是否包含"解除警示"（忽略NaN）
has_remove = df_xyjj[["XQSHJG", "JMSHJG", "JWCSHJG"]].apply(
    lambda x: x.str.contains("解除警示", na=False)  # na=False表示NaN视为不包含
).any(axis=1)  # 行方向任意一列满足则为True
df_xyjj_filtered = df_xyjj[~has_remove].copy()  # ~取反：保留不包含"解除警示"的行

# 提取有效警示记录的(XH, XQBM)组合，用于后续判断
# 转换XQBM为整数（确保与XNXQ格式一致）
df_xyjj_filtered["XQBM"] = df_xyjj_filtered["XQBM"].astype(int)
warning_pairs = set(zip(df_xyjj_filtered["XH"], df_xyjj_filtered["XQBM"]))


# -------------------------- 3. 处理DCCY.csv和JXJ.csv（统计次数） --------------------------
# 处理DCCY.csv：按XH统计出现次数（即DCCY次数）
df_dccy = pd.read_csv("DCCY.csv", encoding="gbk")
# 假设文件中"XH"是唯一标识，统计每个XH的出现次数
dccy_counts = df_dccy["XH"].value_counts().reset_index()
dccy_counts.columns = ["XH", "DCCY"]  # 重命名列为XH和DCCY（次数）

# 处理JXJ.csv：按XH统计出现次数（即JXJ次数）
df_jxj = pd.read_csv("JXJ.csv", encoding="gbk")
jxj_counts = df_jxj["XH"].value_counts().reset_index()
jxj_counts.columns = ["XH", "JXJ"]  # 重命名列为XH和JXJ（次数）


# -------------------------- 4. 合并所有数据为新表 --------------------------
# 以XQJD的数据为基础（包含XH、XNXQ、GPA）
new_df = df_xqjd.copy()

# 合并DCCY次数（左连接：保留所有XQJD中的记录，没有DCCY的填0）
new_df = new_df.merge(dccy_counts, on="XH", how="left")
new_df["DCCY"] = new_df["DCCY"].fillna(0).astype(int)  # 缺失值视为0次

# 合并JXJ次数（左连接）
new_df = new_df.merge(jxj_counts, on="XH", how="left")
new_df["JXJ"] = new_df["JXJ"].fillna(0).astype(int)  # 缺失值视为0次

# 处理XYJJ字段：默认1，若(XH, XNXQ)在有效警示记录中则改为0
new_df["XYJJ"] = 1  # 默认值1
# 判断每行的(XH, XNXQ)是否在warning_pairs中
new_df["XYJJ"] = new_df.apply(
    lambda row: 0 if (row["XH"], row["XNXQ"]) in warning_pairs else 1,
    axis=1
)


# -------------------------- 5. 输出结果 --------------------------
print("合并后的新表：")
print(new_df.head())
# 保存为CSV（可选）
new_df.to_csv("/data/total.csv", index=False, encoding="gbk")