import pandas as pd
import numpy as np
from tqdm import tqdm
from termcolor import colored

import os
import sys
current_file_path=os.path.abspath(__file__)
parent_path=os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_path)
from src.config import (
    FEATURE_COLS, 
    TARGET_COL, 
    MAX_HISTORY_LEN, 
    ENCODING,
    GPA_EMPTY_MARK
)


def load_and_preprocess_data(data_path, encoding=ENCODING,splitnum=None,mode=None):
    """
    轻量级数据加载与预处理（split.py已完成大部分工作）：
    - 仅做类型校验和排序，避免重复处理
    """
    df = pd.read_csv(data_path, encoding=encoding)
    if splitnum is not None:
        if mode is None:
            df=df[:splitnum]
        elif mode == "shuffle":
            df=df.sample(n=splitnum)
    
    # 确保关键列存在
    for col in ["XH", "XNXQ", TARGET_COL]:
        if col not in df.columns:
            raise ValueError(f"数据中缺少关键列：{col}")
    
    # 确保目标列是整数（0/1）
    df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)
    
    # 按学生和学期排序（保证时序顺序）
    df = df.sort_values(by=["XH", "XNXQ"]).reset_index(drop=True)
    
    print(f"加载数据完成：{len(df)}条记录，{df['XH'].nunique()}个学生")
    return df

def add_term_index(df):
    term_index_list=[]
    for sid in tqdm(df['XH'].unique(),desc=colored("add_term_index","blue")):
        student_data=df[df["XH"]==sid].reset_index(drop=True)
        student_data["term_index"]=range(1,len(student_data)+1)
        term_index_list.append(student_data)
    return pd.concat(term_index_list,ignore_index=True)

def build_node_level_samples(df):
    """
    构建节点级样本：
    - 适配split.py处理后的数据结构
    - 跳过无历史的学生（仅1个学期）
    """
    samples = []
    student_ids = df["XH"].unique()
    
    for sid in student_ids:
        student_data = df[df["XH"] == sid].reset_index(drop=True)
        term_count = len(student_data)
        
        # 跳过无历史的学生（仅1个学期）
        if term_count == 1:
            continue
        
        # 为每个学期节点构建样本
        for i in range(1, term_count):
            # 提取历史数据范围
            if MAX_HISTORY_LEN == "all":
                history_start = 0
            else:
                history_start = max(0, i - int(MAX_HISTORY_LEN))
            
            history_data = student_data.iloc[history_start:i]
            current_data = student_data.iloc[i]
            
            # 构造历史特征描述（保留GPA空值标识）
            history_terms = history_data["XNXQ"].tolist()
            history_gpas = history_data["GPA"].tolist()
            history_dccy = history_data["DCCY"].tolist()
            history_jxj = history_data["JXJ"].tolist()
            
            # 标注历史长度
            if MAX_HISTORY_LEN == "all":
                history_desc = f"所有{len(history_terms)}个"
            else:
                history_desc = f"最近{len(history_terms)}个"
            
            feature_desc = f"""
            学生{sid}的历史学业特征（{history_desc}学期：{history_terms}）：
            - GPA序列：{history_gpas}
            - 累计挂科/违纪数序列：{history_dccy}
            - 奖学金次数序列：{history_jxj}
            """
            feature_desc = feature_desc.strip().replace("\n", "").replace("  ", " ")
            
            # 样本信息
            samples.append({
                "XH": sid,
                "current_XNXQ": current_data["XNXQ"],
                "history_feature_desc": feature_desc,
                "true_label": current_data[TARGET_COL],
                "history_term_count": len(history_terms)
            })
    
    sample_df = pd.DataFrame(samples)
    print(f"共构建{len(sample_df)}个节点级样本（涉及{len(student_ids)}个学生）")
    return sample_df


def build_node_level_features(df, core_feats=None):
    """新增：构建节点级（学生-学期）特征（修复：处理字符串值）"""
    if core_feats is None:
        core_feats = ["current_gpa", "current_dccy", "current_jxj", "cum_gpa_mean"]
    
    node_feats = []
    student_ids = df["XH"].unique()
    
    for sid in student_ids:
        student_data = df[df["XH"] == sid].reset_index(drop=True)
        term_count = len(student_data)
        
        for i in range(term_count):
            current_data = student_data.iloc[i]
            xnxq = current_data["XNXQ"]
            term_index = current_data["term_index"]
            
            # ------------- 关键修复：强制转换为数值 -------------
            current_gpa = pd.to_numeric(current_data["GPA"], errors="coerce")
            current_gpa = current_gpa if current_gpa != GPA_EMPTY_MARK else np.nan
            current_dccy = pd.to_numeric(current_data["DCCY"], errors="coerce").fillna(0)
            current_jxj = pd.to_numeric(current_data["JXJ"], errors="coerce").fillna(0)
            current_warning = pd.to_numeric(current_data[TARGET_COL], errors="coerce").fillna(0)
            
            # 提取累计特征（到当前学期）
            cumulative_data = student_data.iloc[:i+1]
            cum_gpa = pd.to_numeric(cumulative_data["GPA"], errors="coerce").replace(GPA_EMPTY_MARK, np.nan)
            cum_gpa_mean = cum_gpa.mean()
            cum_dccy_max = pd.to_numeric(cumulative_data["DCCY"], errors="coerce").max()
            cum_jxj_ratio = (pd.to_numeric(cumulative_data["JXJ"], errors="coerce") > 0).sum() / len(cumulative_data) if len(cumulative_data) > 0 else 0
            
            node_feats.append({
                "XH": sid,
                "XNXQ": xnxq,
                "term_index": int(term_index),  # 确保为整数
                "current_gpa": float(current_gpa) if not np.isnan(current_gpa) else 0.0,
                "current_dccy": int(current_dccy),
                "current_jxj": int(current_jxj),
                "cum_gpa_mean": float(cum_gpa_mean) if not np.isnan(cum_gpa_mean) else 0.0,
                "cum_dccy_max": int(cum_dccy_max) if not np.isnan(cum_dccy_max) else 0,
                "cum_jxj_ratio": float(cum_jxj_ratio),
                "current_warning": int(current_warning)
            })
    
    node_feat_df = pd.DataFrame(node_feats).fillna(0)
    # 强制转换所有数值列的dtype
    for col in node_feat_df.columns:
        if col not in ["XH", "XNXQ"]:
            node_feat_df[col] = pd.to_numeric(node_feat_df[col], errors="coerce").fillna(0)
    
    print(f"节点级特征构建完成：{len(node_feat_df)}个（学生-学期）节点")
    return node_feat_df

def build_student_level_features(df, core_feats=None):
    """
    构建学生级数值特征（修复：强制转换为数值类型，处理字符串）
    """
    if core_feats is None:
        core_feats = ["gpa_mean", "dccy_max", "jxj_ratio", "warning_ratio"]
    
    student_feats = []
    student_ids = df["XH"].unique()
    
    for sid in student_ids:
        student_data = df[df["XH"] == sid].reset_index(drop=True)
        term_count = len(student_data)
        
        # ------------- 关键修复：强制转换为数值类型，处理字符串 -------------
        # 处理GPA列：将非数值转换为NaN
        gpa_series = pd.to_numeric(student_data["GPA"], errors="coerce").replace(GPA_EMPTY_MARK, np.nan)
        # 处理DCCY/JXJ/TARGET_COL：确保为数值
        dccy_series = pd.to_numeric(student_data["DCCY"], errors="coerce").fillna(0)
        jxj_series = pd.to_numeric(student_data["JXJ"], errors="coerce").fillna(0)
        warning_series = pd.to_numeric(student_data[TARGET_COL], errors="coerce").fillna(0)
        
        # 计算统计特征（避免字符串参与运算）
        gpa_mean = np.nanmean(gpa_series) if term_count > 0 else 0.0
        dccy_max = np.max(dccy_series) if term_count > 0 else 0.0
        dccy_mean = np.mean(dccy_series) if term_count > 0 else 0.0
        jxj_max = np.max(jxj_series) if term_count > 0 else 0.0
        jxj_ratio = (jxj_series > 0).sum() / term_count if term_count > 0 else 0.0
        warning_ratio = np.mean(warning_series) if term_count > 0 else 0.0
        gpa_std = np.nanstd(gpa_series) if term_count > 0 else 0.0
        
        # 整合特征（确保所有值为数值）
        student_feats.append({
            "XH": sid,
            "term_count": term_count,
            "gpa_mean": float(gpa_mean),
            "dccy_max": int(dccy_max),
            "dccy_mean": float(dccy_mean),
            "jxj_max": int(jxj_max),
            "jxj_ratio": float(jxj_ratio),
            "warning_ratio": float(warning_ratio),
            "gpa_std": float(gpa_std)
        })
    
    # 转换为DataFrame并确保dtype为数值
    feat_df = pd.DataFrame(student_feats).fillna(0)
    # 强制转换数值列的dtype
    for col in feat_df.columns:
        if col != "XH":
            feat_df[col] = pd.to_numeric(feat_df[col], errors="coerce").fillna(0)
    
    print(f"学生级特征构建完成：{len(feat_df)}个学生，{len(feat_df.columns)-2}个特征")
    return feat_df

def build_node_level_features(df, core_feats=None):
    """新增：构建节点级（学生-学期）特征（推荐用核心）"""
    if core_feats is None:
        core_feats = ["current_gpa", "current_dccy", "current_jxj", "cum_gpa_mean"]
    
    node_feats = []
    student_ids = df["XH"].unique()
    
    for sid in student_ids:
        student_data = df[df["XH"] == sid].reset_index(drop=True)
        term_count = len(student_data)
        
        for i in range(term_count):
            current_data = student_data.iloc[i]
            xnxq = current_data["XNXQ"]
            term_index = current_data["term_index"]
            
            # 提取当前学期特征
            current_gpa = current_data["GPA"] if current_data["GPA"] != GPA_EMPTY_MARK else np.nan
            current_dccy = current_data["DCCY"]
            current_jxj = current_data["JXJ"]
            current_warning = current_data[TARGET_COL]
            
            # 提取累计特征（到当前学期）
            cumulative_data = student_data.iloc[:i+1]
            cum_gpa_mean = cumulative_data["GPA"].replace(GPA_EMPTY_MARK, np.nan).mean()
            cum_dccy_max = cumulative_data["DCCY"].max()
            cum_jxj_ratio = sum(cumulative_data["JXJ"] > 0) / len(cumulative_data) if len(cumulative_data) > 0 else 0
            
            node_feats.append({
                "XH": sid,
                "XNXQ": xnxq,
                "term_index": term_index,
                "current_gpa": current_gpa,
                "current_dccy": current_dccy,
                "current_jxj": current_jxj,
                "cum_gpa_mean": cum_gpa_mean,
                "cum_dccy_max": cum_dccy_max,
                "cum_jxj_ratio": cum_jxj_ratio,
                "current_warning": current_warning
            })
    
    node_feat_df = pd.DataFrame(node_feats).fillna(0)
    # 处理GPA为0的情况（空值）
    node_feat_df["current_gpa"] = node_feat_df["current_gpa"].replace(0, 0.1)
    print(f"节点级特征构建完成：{len(node_feat_df)}个（学生-学期）节点")
    return node_feat_df
