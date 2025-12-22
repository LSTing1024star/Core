import pandas as pd
import re

#############1. 处理XQJD#############
df_xqjd = pd.read_csv("D:/LST/Core-main/Core-main/data/XQJD.csv", encoding="utf-8-sig")
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
df_xyyj = pd.read_csv("D:/LST/Core-main/Core-main/data/XYYJ.csv", encoding="utf-8-sig")
df_xyyj = df_xyyj.rename(columns={"XQN": "XNXQ"})
df_xyyj = df_xyyj[["XH", "XNXQ", "XQSHJG", "JMSHJG", "JNCSHJG"]].copy()

has_remove = df_xyyj[["XQSHJG", "JMSHJG", "JNCSHJG"]].apply(
    lambda x: x.str.contains("解除警示", na=False)
).any(axis=1)

df_xyyj_filtered = df_xyyj[~has_remove].copy()
warning_pairs = set(zip(df_xyyj_filtered["XH"].astype(str), df_xyyj_filtered["XNXQ"].astype(str)))

#############3. 处理DCCY#############
df_dccy = pd.read_csv("D:/LST/Core-main/Core-main/data/DCCY.csv", encoding="utf-8-sig")
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
df_jxj = pd.read_csv("D:/LST/Core-main/Core-main/data/JXJ.csv", encoding="utf-8-sig")
jxj_counts = df_jxj["XH"].value_counts().reset_index()
jxj_counts.columns = ["XH", "JXJ"]

#############5. 合并#############
df = df_xqjd.copy()
df = df.sort_values(by=["XH", "XNXQ"])
dccy_count = dccy_count.sort_values(by=["XH", "XNXQ"])

df = pd.merge(
    df,  # 左表：学生学期GPA数据
    dccy_count,  # 右表：学生学期累计违纪次数
    on=["XH", "XNXQ"],  # 基于“学号+学期”两个键进行精确匹配
    how="left"  # 保留左表所有记录，右表没有匹配的用NaN填充
)
df["DCCY"] = df["DCCY"].fillna(0).astype(int)

df = df.merge(jxj_counts, on="XH", how="left")
df["JXJ"] = df["JXJ"].fillna(0).astype(int)

df["XYYJ"] = 0
df["XYYJ"] = df.apply(
    lambda row: 1 if (row["XH"], row["XNXQ"]) in warning_pairs else 0,
    axis=1
)

#############6. 输出结果#############
df.to_csv("D:/LST/Core-main/Core-main/data/totaldata.csv", encoding="utf-8-sig", index=False)