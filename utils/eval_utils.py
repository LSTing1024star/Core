import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

import os
import sys
current_file_path=os.path.abspath(__file__)
parent_path=os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_path)
# 补充缺失的依赖（根据实际项目路径调整）
try:
    from src.config import METRICS_SAVE_PATH,ENCODING, RECOMMEND_METRICS_PATH, TARGET_COL  # 新增TARGET_COL
    from utils.llm_utils import call_local_llm
except ImportError:
    # 本地测试用默认值
    ENCODING = 'utf-8'
    RECOMMEND_METRICS_PATH = 'data/recommendation_metrics.csv'
    TARGET_COL = 'true_label'
    # 模拟LLM调用（测试用）
    def call_local_llm(prompt):
        return "匹配度：5分\n参考价值：4分\n理由合理性：5分"

# ===================== 预测评估（基础分类指标） =====================
def evaluate_prediction(result_df, dataset_name="Train"):
    """评估节点预测的分类指标（准确率、召回率、F1）"""
    # 校验必要列
    for col in ["true_label", "pred_label"]:
        if col not in result_df.columns:
            raise ValueError(f"结果表缺少必要列：{col}")
    
    true_labels = result_df["true_label"]
    pred_labels = result_df["pred_label"]
    
    accuracy = accuracy_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    print(f"\n=== {dataset_name}集节点预测评估指标 ===")
    print(f"准确率（Accuracy）：{accuracy:.4f}")
    print(f"召回率（Recall）：{recall:.4f}")
    print(f"F1-score：{f1:.4f}")
    print("\n分类报告：")
    print(classification_report(true_labels, pred_labels, zero_division=0))
    
    return {
        "dataset": dataset_name,
        "type": "normal",
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1
    }

def evaluate_new_students_prediction(new_student_df):
    """评估新入学学生的预测结果（无真实标签时仅统计数量）"""
    if len(new_student_df) == 0:
        print("警告：新入学学生数据为空")
        return None
    
    print(f"\n=== 纯新入学学生预测评估 ===")
    print(f"新入学学生数：{new_student_df['XH'].nunique()}")
    print(f"预测为预警（1）的学生数：{new_student_df['pred_label'].sum()}")
    print(f"预测为无预警（0）的学生数：{len(new_student_df) - new_student_df['pred_label'].sum()}")
    
    # 有真实标签时计算分类指标
    if TARGET_COL in new_student_df.columns:
        true_labels = new_student_df[TARGET_COL]
        pred_labels = new_student_df["pred_label"]
        accuracy = accuracy_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
        print(f"准确率（Accuracy）：{accuracy:.4f}")
        print(f"召回率（Recall）：{recall:.4f}")
        print(f"F1-score：{f1:.4f}")
        
        return {
            "dataset": "New Student",
            "type": "new_student",
            "accuracy": accuracy,
            "recall": recall,
            "f1": f1
        }
    else:
        return {
            "dataset": "New Student",
            "type": "new_student",
            "pred_1_count": new_student_df["pred_label"].sum(),
            "pred_0_count": len(new_student_df) - new_student_df["pred_label"].sum()
        }

# ===================== 推荐评估（经典指标） =====================
def get_positive_examples(test_node, train_nodes, core_feats, k=5, pos_thresholds=None):
    """
    获取伪正例（自定义正例规则，提升灵活性）
    :param test_node: 单个测试节点（Series）
    :param train_nodes: 训练节点集（DataFrame）
    :param core_feats: 核心特征列
    :param k: 正例数量
    :param pos_thresholds: 正例筛选阈值，格式：{"gpa": 0.1, "dccy": 0}
    :return: 正例学生ID列表
    """
    pos_thresholds = pos_thresholds or {"gpa": 0.1, "dccy": 0}
    
    # 计算余弦相似度
    test_vec = test_node[core_feats].values.reshape(1, -1)
    train_vecs = train_nodes[core_feats].values
    similarities = cosine_similarity(test_vec, train_vecs)[0]
    train_nodes = train_nodes.copy()
    train_nodes["similarity"] = similarities
    
    # 定义伪正例：GPA更高、违纪更少（可扩展其他规则）
    mask = (
        (train_nodes["current_gpa"] >= test_node["current_gpa"] + pos_thresholds["gpa"]) &
        (train_nodes["current_dccy"] <= test_node["current_dccy"] + pos_thresholds["dccy"])
    )
    positive_candidates = train_nodes[mask].copy()
    
    if len(positive_candidates) == 0:
        return []
    
    # 按相似度排序取Top-k
    positive_candidates = positive_candidates.sort_values("similarity", ascending=False)
    return positive_candidates.head(k)["XH"].tolist()

def calculate_recommendation_metrics(result_df, test_nodes, train_nodes, core_feats, k=5):
    """
    计算推荐经典指标：Precision@k、Recall@k、HR@k、MAP、MRR
    适配单候选推荐（k=1）场景
    """
    # 校验必要列
    required_cols = ["test_xh", "test_xnxq", "recommend_train_xh"]
    for col in required_cols:
        if col not in result_df.columns:
            raise ValueError(f"结果表缺少必要列：{col}")
    
    precisions = []
    recalls = []
    hit_rates = []
    avg_precisions = []
    reciprocal_ranks = []
    
    for _, rec_row in result_df.iterrows():
        test_xh = rec_row["test_xh"]
        test_xnxq = rec_row["test_xnxq"]
        rec_xh = rec_row["recommend_train_xh"]
        
        # 获取测试节点（鲁棒处理）
        test_node_mask = (test_nodes["XH"] == test_xh) & (test_nodes["XNXQ"] == test_xnxq)
        test_node = test_nodes[test_node_mask]
        if len(test_node) == 0:
            precisions.append(0)
            recalls.append(0)
            hit_rates.append(0)
            avg_precisions.append(0)
            reciprocal_ranks.append(0)
            continue
        test_node = test_node.iloc[0]
        
        # 获取伪正例
        positive_xh = get_positive_examples(test_node, train_nodes, core_feats, k)
        if len(positive_xh) == 0:
            precisions.append(0)
            recalls.append(0)
            hit_rates.append(0)
            avg_precisions.append(0)
            reciprocal_ranks.append(0)
            continue
        
        # 计算指标（单候选场景）
        is_hit = rec_xh in positive_xh and rec_xh != "无"
        precision = 1 if is_hit else 0
        recall = 1 if is_hit else 0
        hit_rate = 1 if is_hit else 0
        
        # MAP和MRR（单候选时等于precision）
        ap = 1 / (positive_xh.index(rec_xh) + 1) if is_hit else 0
        rr = 1 / (positive_xh.index(rec_xh) + 1) if is_hit else 0
        
        precisions.append(precision)
        recalls.append(recall)
        hit_rates.append(hit_rate)
        avg_precisions.append(ap)
        reciprocal_ranks.append(rr)
    
    metrics = {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"HR@{k}": np.mean(hit_rates),
        "MAP": np.mean(avg_precisions),
        "MRR": np.mean(reciprocal_ranks),
        "k_value": k
    }
    
    print(f"\n=== 推荐算法经典指标（k={k}）===")
    for key, value in metrics.items():
        print(f"{key}：{value:.4f}")
    
    return metrics

# ===================== LLM主观评估 =====================
def llm_evaluate_recommendation(result_df, test_nodes, train_nodes, sample_rate=0.2, random_state=42):
    """
    LLM主观评估推荐结果（匹配度、参考价值、理由合理性）
    :param sample_rate: 抽样比例（0-1）
    :param random_state: 随机种子（保证可复现）
    :return: 平均评分指标
    """
    sample_size = max(int(len(result_df) * sample_rate), 1)  # 至少抽样1条
    sample_df = result_df.sample(n=sample_size, random_state=random_state)
    
    print(f"\n=== LLM主观评估（抽样{sample_size}条/{len(result_df)}条）===")
    
    match_scores = []
    value_scores = []
    reason_scores = []
    
    for idx, (_, rec_row) in enumerate(sample_df.iterrows()):
        test_xh = rec_row["test_xh"]
        test_xnxq = rec_row["test_xnxq"]
        rec_xh = rec_row["recommend_train_xh"]
        
        # 进度打印
        if (idx + 1) % 5 == 0:
            print(f"LLM评估进度：{idx+1}/{sample_size}")
        
        # 获取节点特征（鲁棒处理）
        test_node_mask = (test_nodes["XH"] == test_xh) & (test_nodes["XNXQ"] == test_xnxq)
        test_node = test_nodes[test_node_mask].iloc[0] if len(test_nodes[test_node_mask]) > 0 else None
        
        train_node_mask = train_nodes["XH"] == rec_xh
        train_node = train_nodes[train_node_mask].iloc[0] if (rec_xh != "无" and len(train_nodes[train_node_mask]) > 0) else None
        
        # 构造Prompt
        prompt = _construct_llm_prompt(rec_row, test_node, train_node)
        llm_output = call_local_llm(prompt)
        
        # 解析评分（增加容错）
        match_score = _parse_llm_score(llm_output, "匹配度")
        value_score = _parse_llm_score(llm_output, "参考价值")
        reason_score = _parse_llm_score(llm_output, "理由合理性")
        
        match_scores.append(match_score)
        value_scores.append(value_score)
        reason_scores.append(reason_score)
    
    avg_metrics = {
        "平均匹配度": np.mean(match_scores),
        "平均参考价值": np.mean(value_scores),
        "平均理由合理性": np.mean(reason_scores),
        "抽样数": sample_size
    }
    
    print(f"\n=== LLM主观评估均值 ===")
    for key, value in avg_metrics.items():
        if "平均" in key:
            print(f"{key}：{value:.2f}分（满分5分）")
    
    return avg_metrics

def _parse_llm_score(llm_output, score_name):
    """
    解析LLM输出的评分（增加鲁棒性）
    :param llm_output: LLM输出文本
    :param score_name: 评分项（如“匹配度”）
    :return: 评分（1-5分，默认3分）
    """
    pattern = re.compile(rf"{score_name}：(\d+)分")
    match = pattern.search(llm_output)
    if match:
        score = int(match.group(1))
        return max(1, min(5, score))  # 限制在1-5分
    else:
        print(f"警告：未解析到{score_name}评分，默认赋值3分")
        return 3

def _construct_llm_prompt(rec_row, test_node, train_node):
    """构造LLM评估Prompt（鲁棒处理空值）"""
    # 测试节点描述
    if test_node is not None:
        test_desc = f"""
        测试学生{rec_row['test_xh']}（{rec_row['test_xnxq']}学期）特征：
        - GPA：{round(test_node['current_gpa'], 2)}
        - 违纪次数：{test_node['current_dccy']}
        - 学期索引：{test_node['term_index']}
        """
    else:
        test_desc = f"测试学生{rec_row['test_xh']}（{rec_row['test_xnxq']}学期）：特征未获取"
    
    # 榜样节点描述
    if train_node is not None:
        train_desc = f"""
        推荐榜样{rec_row['recommend_train_xh']}特征：
        - GPA：{round(train_node['current_gpa'], 2)}
        - 违纪次数：{train_node['current_dccy']}
        - 学期索引：{train_node['term_index']}
        """
    else:
        train_desc = "推荐榜样：无"
    
    # 推荐理由（处理空值）
    recommend_reason = rec_row.get("recommend_reason", "无推荐理由")
    
    prompt = f"""
    你是学业指导评估专家，请对以下推荐结果进行评分（评分范围1-5分，1分最差，5分最好）：
    1. 匹配度：评估榜样与测试学生的学期阶段、学业能力的匹配程度
    2. 参考价值：评估榜样对测试学生的学业提升、行为改进的参考价值
    3. 理由合理性：评估推荐理由的逻辑性、充分性和针对性

    【输出格式要求】（严格按以下格式输出，仅包含评分）：
    匹配度：X分
    参考价值：X分
    理由合理性：X分

    【推荐结果信息】
    - 测试学生：{rec_row['test_xh']}（{rec_row['test_xnxq']}学期）
    - 推荐榜样：{rec_row['recommend_train_xh']}
    - 推荐理由：{recommend_reason}

    【学生特征】
    {test_desc}
    {train_desc}
    """
    
    # 清理格式（去除多余换行和空格）
    return re.sub(r"\s+", " ", prompt).strip()

# ===================== 综合评估与结果保存 =====================
def evaluate_recommendation_comprehensive(
    result_df, test_nodes, train_nodes, core_feats,
    k=1, sample_rate=0.2, save_path=None
):
    """
    综合评估推荐结果（整合基础指标、经典指标、LLM指标）
    :param save_path: 指标保存路径（默认使用配置文件路径）
    :return: 综合评估指标字典
    """
    save_path = save_path or RECOMMEND_METRICS_PATH
    
    # 1. 基础指标（覆盖率）
    total_test = len(result_df)
    has_recommend = result_df[result_df["recommend_train_xh"] != "无"]
    recommend_coverage = len(has_recommend) / total_test if total_test > 0 else 0
    candidate_coverage = len(result_df[result_df["candidate_count"] > 0]) / total_test if total_test > 0 else 0
    
    # 2. 经典推荐指标
    classic_metrics = calculate_recommendation_metrics(result_df, test_nodes, train_nodes, core_feats, k=k)
    
    # 3. LLM主观评估
    llm_metrics = llm_evaluate_recommendation(result_df, test_nodes, train_nodes, sample_rate=sample_rate)
    
    # 4. 整合所有指标
    metrics = {
        "总测试节点数": total_test,
        "推荐覆盖率": recommend_coverage,
        "候选覆盖率": candidate_coverage,
        **classic_metrics,
        **llm_metrics
    }
    
    # 保存指标到CSV
    pd.DataFrame([metrics]).to_csv(save_path, index=False, encoding=ENCODING)
    print(f"\n=== 综合评估完成 ===")
    print(f"综合评估指标已保存至：{save_path}")
    print(f"关键指标：推荐覆盖率={recommend_coverage:.4f}，HR@{k}={classic_metrics[f'HR@{k}']:.4f}")
    
    return metrics

def save_recommendation_detail(result_df, train_nodes, save_path="data/recommendation_detail.csv"):
    """保存推荐详细结果（关联榜样特征）"""
    # 去重训练节点特征（按学生ID）
    train_feat = train_nodes[["XH", "XNXQ", "term_index", "current_gpa", "current_dccy"]].drop_duplicates(subset=["XH"])
    
    # 合并结果
    result_detail = result_df.merge(
        train_feat,
        left_on="recommend_train_xh",
        right_on="XH",
        how="left",
        suffixes=("_test", "_role_model")
    )
    
    # 保存
    result_detail.to_csv(save_path, index=False, encoding=ENCODING)
    print(f"\n推荐详细结果已保存至：{save_path}")
    return result_detail

def save_metrics_summary(metrics_list):
    """保存指标汇总"""
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(METRICS_SAVE_PATH, index=False, encoding=ENCODING)
    print("\n===== 整体预测指标汇总 =====")
    print(metrics_df)
    return metrics_df

