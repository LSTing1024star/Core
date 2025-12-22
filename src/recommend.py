import pandas as pd
import numpy as np
import re

import os
import sys
current_file_path=os.path.abspath(__file__)
parent_path=os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_path)
from src.clustering import StudentClustering
from utils.data_utils import load_and_preprocess_data, build_node_level_features
from utils.llm_utils import call_local_qwen3
from src.config import (
    TRAIN_PATH, TEST_PATH, ENCODING, RECOMMEND_SAVE_PATH
)

# ---------------------- 1. 原始节点特征提取（学生-学期节点） ----------------------
def extract_node_features(student_node_df, node_col="node_label"):
    """提取学生-学期节点的特征描述"""
    node_features = {}
    for node in student_node_df[node_col].unique():
        node_df = student_node_df[student_node_df[node_col] == node]
        feat_desc = {
            "node_label": node,
            "student_xh": node_df["XH"].iloc[0],
            "term_index": node_df["term_index"].iloc[0],
            "current_gpa": round(node_df["current_gpa"].iloc[0], 2),
            "current_dccy": node_df["current_dccy"].iloc[0],
            "desc": f"节点{node}：学生{node_df['XH'].iloc[0]}的{node_df['XNXQ'].iloc[0]}学期，学期索引{node_df['term_index'].iloc[0]}，GPA{round(node_df['current_gpa'].iloc[0],2)}，违纪数{node_df['current_dccy'].iloc[0]}"
        }
        node_features[node] = feat_desc
    return node_features

# ---------------------- 3. 核心推荐类（学生级聚类+节点级推荐） ----------------------
class NodeLevelRecommender:
    """
    推荐器：
    1. 学生级聚类（全局特征）
    2. 目标节点（学生-学期）基于学生全局特征匹配簇
    3. 榜样筛选：term_index ≥ 目标学生当前学期
    """
    def __init__(self, train_path="./train_data.csv", test_path="./test_data.csv"):
        # 配置参数
        self.train_path = train_path
        self.test_path = test_path
        # 学生级核心聚类特征（从data_utils的build_student_level_features中选取）
        self.student_core_feats = ["gpa_mean", "dccy_max", "jxj_ratio", "term_count"]
        # 节点级核心特征（用于候选相似度计算）
        self.node_core_feats = ["current_gpa", "current_dccy", "term_index", "cum_gpa_mean"]

        # ---------------------- 步骤1：加载并预处理数据 ----------------------
        # 加载原始数据
        self.train_raw = load_and_preprocess_data(self.train_path)
        self.test_raw = load_and_preprocess_data(self.test_path)

        # 空数据检查
        if self.train_raw.empty or self.test_raw.empty:
            raise ValueError("训练集/测试集加载失败，无法继续推荐")

        # 生成term_index
        self.train_raw = add_term_index(self.train_raw)
        self.test_raw = add_term_index(self.test_raw)

        # 构建节点级特征（目标学生的原始节点数据）
        self.train_node_df = build_node_level_features(self.train_raw)
        self.test_node_df = build_node_level_features(self.test_raw)

        # 定义原始节点标识（XH+XNXQ）
        self.train_node_df["node_label"] = self.train_node_df["XH"].astype(str) + "_" + self.train_node_df["XNXQ"].astype(str)
        self.test_node_df["node_label"] = self.test_node_df["XH"].astype(str) + "_" + self.test_node_df["XNXQ"].astype(str)

        # ---------------------- 步骤2：构建学生级特征 ----------------------
        self.train_student_feat = build_student_level_features(self.train_raw)
        self.test_student_feat = build_student_level_features(self.test_raw)

        # 映射学生的簇标签到节点数据
        self.train_node_df = self._map_student_cluster_to_node(self.train_node_df, self.train_student_feat)

        # ---------------------- 步骤3：提取测试集节点特征 ----------------------
        self.test_node_features = extract_node_features(self.test_node_df)

        # ---------------------- 步骤4：学生级聚类 ----------------------
        self.clustering = StudentLevelClustering()
        self._cluster_students()

        # 初始化信息
        print(f"\n=== 初始化完成 ===")
        print(f"训练集：{len(self.train_student_feat)}个学生（{len(self.train_node_df)}个节点），{len(self.clustering.cluster_features)}个学生级簇")
        print(f"测试集：{len(self.test_student_feat)}个学生（{len(self.test_node_df)}个节点），{len(self.test_node_features)}个原始节点")

    def _cluster_students(self):
        """执行学生级聚类"""
        # 聚类训练集学生
        self.clustering.kmeans_cluster(self.train_student_feat, self.student_core_feats)

    def _map_student_cluster_to_node(self, node_df, student_feat_df):
        """将学生的簇标签映射到节点数据"""
        # 构建学生-簇的映射
        student_cluster_map = student_feat_df[["XH", "cluster_label"]].set_index("XH").to_dict()["cluster_label"]
        # 映射到节点
        node_df["student_cluster"] = node_df["XH"].map(student_cluster_map)
        return node_df

    def _get_student_feat_desc(self, xh, is_test=True):
        """获取学生的全局特征描述（供LLM选簇）"""
        if is_test:
            student_feat = self.test_student_feat[self.test_student_feat["XH"] == xh]
        else:
            student_feat = self.train_student_feat[self.train_student_feat["XH"] == xh]

        if student_feat.empty:
            return "学生特征未知"

        feat = student_feat.iloc[0]
        desc = f"""
        学生{xh}的全局学业特征：
        - 总学期数：{feat['term_count']}
        - 平均GPA：{round(feat['gpa_mean'], 2)}
        - 最大挂科/违纪数：{feat['dccy_max']}
        - 奖学金获得比例：{round(feat['jxj_ratio'], 2)}
        - 学业预警比例：{round(feat['warning_ratio'], 2)}
        """.strip().replace("\n", "").replace("  ", " ")
        return desc

    def _construct_cluster_select_prompt(self, test_node):
        """构造LLM选簇的Prompt（基于学生全局特征）"""
        # 目标学生的全局特征
        xh = test_node["XH"]
        student_feat_desc = self._get_student_feat_desc(xh)
        # 目标节点的学期信息
        node_label = test_node["node_label"]
        node_term = test_node["term_index"]

        # 所有学生级簇的特征描述
        cluster_list = "\n".join([feat["desc"] for feat in self.clustering.cluster_features.values()])

        # Prompt
        prompt = f"""
        你是学业推荐专家，请根据目标学生的全局特征和当前学期索引（{node_term}），选择最匹配的学生级簇，仅输出簇编号。

        目标学生全局特征：
        {student_feat_desc}

        目标学生当前学期：{node_label}（学期索引{node_term}）

        可选学生级簇列表：
        {cluster_list}

        最匹配的簇编号：
        """.strip()
        return prompt

    def _select_suitable_cluster_by_llm(self, test_node):
        """调用LLM选择匹配的学生级簇"""
        prompt = self._construct_cluster_select_prompt(test_node)
        llm_output = call_local_llm(prompt)

        # 解析簇编号
        cluster_id = re.findall(r"\d+", llm_output)
        if not cluster_id:
            # 兜底：学生数最多的簇
            max_count_cluster = max(self.clustering.cluster_features.values(), key=lambda x: x["student_count"])
            return max_count_cluster["cluster_id"]

        cluster_id = int(cluster_id[0])
        # 验证有效性
        if cluster_id not in self.clustering.cluster_features:
            max_count_cluster = max(self.clustering.cluster_features.values(), key=lambda x: x["student_count"])
            return max_count_cluster["cluster_id"]

        return cluster_id

    def _retrieve_top_k_candidates(self, test_node, cluster_id):
        """
        从匹配的学生级簇中检索候选：
        1. 筛选簇内学生的所有节点
        2. 过滤：榜样节点的term_index ≥ 目标节点的term_index
        3. 按节点特征相似度选Top-3
        """
        # 1. 筛选簇内学生的节点
        cluster_students = self.clustering.student_feat_df[self.clustering.student_feat_df["cluster_label"] == cluster_id]["XH"].tolist()
        candidate_pool = self.train_node_df[self.train_node_df["XH"].isin(cluster_students)].reset_index(drop=True)
        if len(candidate_pool) == 0:
            return pd.DataFrame()

        # 2. 时间约束：term_index ≥ 目标节点
        target_term = test_node["term_index"]
        candidate_pool = candidate_pool[candidate_pool["term_index"] >= target_term].reset_index(drop=True)
        if len(candidate_pool) == 0:
            return pd.DataFrame()

        # 3. 计算节点特征的欧氏距离
        feat_vectors = candidate_pool[self.node_core_feats].values
        feat_vectors = (feat_vectors - feat_vectors.mean(axis=0)) / (feat_vectors.std(axis=0) + 1e-8)

        test_feat = test_node[self.node_core_feats].values
        test_feat_scaled = (test_feat - candidate_pool[self.node_core_feats].mean(axis=0)) / (candidate_pool[self.node_core_feats].std(axis=0) + 1e-8)

        # 4. 选Top-3
        distances = np.linalg.norm(feat_vectors - test_feat_scaled, axis=1)
        top_k_indices = np.argsort(distances)[:3]

        return candidate_pool.iloc[top_k_indices].reset_index(drop=True)

    def _construct_recommend_prompt(self, test_node, top_k_candidates):
        """构造推荐Prompt（含学生级簇信息）"""
        # 目标节点信息
        test_info = f"""
        目标学生-学期节点信息：
        - 学号：{test_node['XH']}
        - 学期：{test_node['XNXQ']}
        - 原始节点：{test_node['node_label']}
        - 学期索引：{test_node['term_index']}
        - 当前GPA：{round(test_node['current_gpa'], 2)}
        - 违纪数：{test_node['current_dccy']}
        """

        # 候选节点信息
        candidate_info = []
        for idx, (_, candidate) in enumerate(top_k_candidates.iterrows()):
            candidate_info.append(
                f"候选{idx+1}：\n"
                f"  学号：{candidate['XH']}\n"
                f"  学期：{candidate['XNXQ']}\n"
                f"  原始节点：{candidate['node_label']}\n"
                f"  学期索引：{candidate['term_index']}\n"
                f"  当前GPA：{round(candidate['current_gpa'],2)}\n"
                f"  违纪数：{candidate['current_dccy']}次"
            )
        candidate_info_str = "\n\n".join(candidate_info)

        # Prompt
        prompt = f"""
        你是学业指导专家，请从候选中选择最优榜样，并按以下格式输出：
        榜样学号：XXX
        推荐理由：XXX（不超过100字，说明核心依据）
        参考方法：XXX（不超过150字，具体可操作的学习建议）

        选择规则：
        1. 榜样学期索引≥目标学生，且属于匹配的学生级簇；
        2. GPA比目标高0.1~0.6，违纪数更少或相等。

        {test_info}

        Top-K候选列表：
        {candidate_info_str}
        """.strip()
        return prompt

    def _llm_generate_recommendation(self, test_node, top_k_candidates):
        """调用LLM生成推荐结果"""
        if len(top_k_candidates) == 0:
            return "无", "暂无合适榜样", "无参考方法"

        prompt = self._construct_recommend_prompt(test_node, top_k_candidates)
        llm_output = call_local_llm(prompt)

        # 解析结果
        xh_match = re.search(r"榜样学号：(\w+)", llm_output)
        reason_match = re.search(r"推荐理由：(.+?)(?=参考方法：|$)", llm_output, re.DOTALL)
        method_match = re.search(r"参考方法：(.+)", llm_output, re.DOTALL)

        # 提取并清洗
        best_xh = xh_match.group(1).strip() if xh_match else "无"
        recommend_reason = reason_match.group(1).strip().replace("\n", "").replace("  ", "") if reason_match else "暂无合适榜样"
        reference_method = method_match.group(1).strip().replace("\n", "").replace("  ", "") if method_match else "无具体参考方法"

        # 验证学号
        candidate_xh_list = top_k_candidates["XH"].tolist()
        if best_xh != "无" and best_xh not in candidate_xh_list:
            best_xh = "无"
            recommend_reason = "筛选的榜样不在候选列表中"
            reference_method = "无具体参考方法"

        return best_xh, recommend_reason, reference_method

    def generate_recommendations(self):
        """生成推荐结果"""
        results = []
        total_nodes = len(self.test_node_df)

        for idx, (_, test_node) in enumerate(self.test_node_df.iterrows()):
            if (idx + 1) % 50 == 0:
                print(f"推荐进度：{idx+1}/{total_nodes}")

            # 1. 选簇
            suitable_cluster = self._select_suitable_cluster_by_llm(test_node)

            # 2. 检索候选
            top_k_candidates = self._retrieve_top_k_candidates(test_node, suitable_cluster)

            # 3. 生成推荐
            rec_xh, rec_reason, ref_method = self._llm_generate_recommendation(test_node, top_k_candidates)

            # 收集结果
            results.append({
                "test_xh": test_node["XH"],
                "test_xnxq": test_node["XNXQ"],
                "original_node": test_node["node_label"],
                "suitable_cluster": suitable_cluster,
                "recommend_xh": rec_xh,
                "recommend_reason": rec_reason,
                "reference_method": ref_method,
                "candidate_count": len(top_k_candidates)
            })

        # 保存结果
        result_df = pd.DataFrame(results)
        result_df.to_csv("./recommend_result.csv", index=False, encoding=ENCODING)

        # 统计
        success_count = len(result_df[result_df["recommend_xh"] != "无"])
        print(f"\n=== 推荐完成 ===")
        print(f"成功推荐：{success_count}/{total_nodes}个学生-学期节点")
        print(f"结果文件：./recommend_result.csv")

        return result_df

# ---------------------- 运行入口 ----------------------
if __name__ == "__main__":
    try:
        recommender = NodeLevelRecommender(
            train_path="./train_data.csv",
            test_path="./test_data.csv"
        )
        recommender.generate_recommendations()
    except Exception as e:
        print(f"运行失败：{str(e)}")