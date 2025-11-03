import pandas as pd
import re

# 工具函数：清理字符串（去除空格和特殊字符，确保格式统一）
def clean_str(val):
    if pd.isna(val):
        return ""
    return str(val).strip().replace(r"[^\w\d]", "")  # 去除空格和非数字字母字符


#############1. 处理XQJD（所有列转为字符串）#############
# 读取时强制所有列为字符串
df_xqjd = pd.read_csv(
    "D:/LST/Core-main/Core-main/data/XQJD.csv", 
    encoding="utf-8-sig",
    dtype=str  # 关键：强制读取为字符串
)
# 提取并清理所需列
df_xqjd = df_xqjd[["XH", "XNXQ", "GPA"]].copy()
df_xqjd["XH"] = df_xqjd["XH"].apply(clean_str)  # 清理学号
df_xqjd["GPA"] = df_xqjd["GPA"].apply(clean_str)  # GPA转为字符串

# 转换XNXQ为字符串格式（如"2021年春季学期"→"20211"）
def convert_xnxq(xnxq_str):
    match = re.match(r"(\d+)年(春季|秋季)学期", str(xnxq_str))
    if not match:
        return ""  # 无效值返回空字符串（后续过滤）
    year = match.group(1)
    semester = "1" if match.group(2) == "春季" else "2"
    return f"{year}{semester}"

df_xqjd["XNXQ"] = df_xqjd["XNXQ"].apply(convert_xnxq)
df_xqjd = df_xqjd[df_xqjd["XNXQ"] != ""]  # 过滤无效XNXQ
df_xqjd["XNXQ"] = df_xqjd["XNXQ"].astype(str)  # 确保是字符串


#############2. 处理XYYJ（所有列转为字符串）#############
df_xyyj = pd.read_csv(
    "D:/LST/Core-main/Core-main/data/XYYJ.csv", 
    encoding="utf-8-sig",
    dtype=str  # 强制字符串
)
df_xyyj = df_xyyj.rename(columns={"XQN": "XNXQ"})
df_xyyj = df_xyyj[["XH", "XNXQ", "XQSHJG", "JMSHJG", "JNCSHJG"]].copy()

# 清理关键列
df_xyyj["XH"] = df_xyyj["XH"].apply(clean_str)
df_xyyj["XNXQ"] = df_xyyj["XNXQ"].apply(clean_str)  # XNXQ转为字符串

# 过滤包含"解除警示"的行
has_remove = df_xyyj[["XQSHJG", "JMSHJG", "JNCSHJG"]].apply(
    lambda x: x.str.contains("解除警示", na=False)
).any(axis=1)
df_xyyj_filtered = df_xyyj[~has_remove].copy()

# 构建警示组合（均为字符串）
warning_pairs = set(zip(
    df_xyyj_filtered["XH"], 
    df_xyyj_filtered["XNXQ"]
))


#############3. 处理DCCY（所有列转为字符串）#############
df_dccy = pd.read_csv(
    "D:/LST/Core-main/Core-main/data/DCCY.csv", 
    encoding="utf-8-sig",
    dtype=str  # 强制字符串
)
df_dccy = df_dccy[["XH", "TSTAMP"]].copy()
df_dccy["XH"] = df_dccy["XH"].apply(clean_str)  # 清理学号

# TSTAMP转换为学期字符串（如"20240222 181859"→"20241"）
def convert_tstamp_to_xnxq(tstamp_str):
    try:
        year_month_str = str(tstamp_str)[:6].strip()  # 取前6位（字符串）
        if not year_month_str.isdigit():
            return ""
        year = year_month_str[:4]
        month = int(year_month_str[4:6])  # 月份转为数字判断
    except:
        return ""
    return f"{year}1" if month < 9 else f"{year}2"

df_dccy["DCCY_XNXQ"] = df_dccy["TSTAMP"].apply(convert_tstamp_to_xnxq)
df_dccy = df_dccy[df_dccy["DCCY_XNXQ"] != ""]  # 过滤无效值
df_dccy["DCCY_XNXQ"] = df_dccy["DCCY_XNXQ"].astype(str)

# 统计新增次数（转为字符串）
dccy_new = df_dccy.groupby(["XH", "DCCY_XNXQ"]).size().reset_index()
dccy_new.columns = ["XH", "XNXQ", "new_count"]
dccy_new["new_count"] = dccy_new["new_count"].astype(str)  # 次数转为字符串

# 计算累积次数（转为字符串）
dccy_cumulative = []
for xh, group in dccy_new.groupby("XH"):
    # 按学期字符串排序（字典序有效）
    group_sorted = group.sort_values(by="XNXQ").reset_index(drop=True)
    # 累积求和（先转为整数计算，再转回字符串）
    group_sorted["cumulative_count"] = group_sorted["new_count"].astype(int).cumsum().astype(str)
    dccy_cumulative.append(group_sorted[["XH", "XNXQ", "cumulative_count"]])

dccy_count = pd.concat(dccy_cumulative, ignore_index=True)


#############4. 处理JXJ（所有列转为字符串）#############
df_jxj = pd.read_csv(
    "D:/LST/Core-main/Core-main/data/JXJ.csv", 
    encoding="utf-8-sig",
    dtype=str  # 强制字符串
)
df_jxj["XH"] = df_jxj["XH"].apply(clean_str)  # 清理学号

# 统计次数（转为字符串）
jxj_counts = df_jxj["XH"].value_counts().reset_index()
jxj_counts.columns = ["XH", "JXJ"]
jxj_counts["JXJ"] = jxj_counts["JXJ"].astype(str)  # 次数转为字符串


#############5. 合并（确保所有列均为字符串）#############
df = df_xqjd.copy()

# 处理DCCY累积次数（字符串合并+向前填充）
# 由于merge_asof不支持字符串on列，改用merge+ffill
df = df.merge(
    dccy_count,
    on=["XH", "XNXQ"],  # 精确匹配
    how="left"
)

# 按学生分组，向前填充累积次数（继承上一学期值）
def fill_cumulative(group):
    group_sorted = group.sort_values(by="XNXQ").reset_index(drop=True)
    group_sorted["cumulative_count"] = group_sorted["cumulative_count"].ffill()  # 向前填充
    group_sorted["cumulative_count"] = group_sorted["cumulative_count"].fillna("0")  # 初始值填"0"
    return group_sorted

df = df.groupby("XH", group_keys=False).apply(fill_cumulative)
df = df.rename(columns={"cumulative_count": "DCCY"})  # 重命名为DCCY


# 合并JXJ（字符串匹配）
df = df.merge(jxj_counts, on="XH", how="left")
df["JXJ"] = df["JXJ"].fillna("0")  # 缺失值填"0"（字符串）


# 处理XYYJ（字符串"0"/"1"）
df["XYYJ"] = "1"  # 默认"1"
df["XYYJ"] = df.apply(
    lambda row: "0" if (row["XH"], row["XNXQ"]) in warning_pairs else "1",
    axis=1
)


#############6. 输出结果（所有列均为字符串）#############
# 确认所有列类型
print("各列数据类型：")
print(df.dtypes)  # 应全部为object（字符串）

df.to_csv(
    "D:/LST/Core-main/Core-main/data/totaldata.csv", 
    encoding="utf-8-sig", 
    index=False
)
print("输出完成，所有列均为字符串类型")