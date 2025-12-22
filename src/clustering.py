import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.config import DEFAULT_CLUSTER_RANGE
from termcolor import colored
import matplotlib.pyplot as pyplot

import os
import sys
current_file_path=os.path.abspath(__file__)
parent_path=os.path.dirname(os.path.dirname(current_path))
sys.path.append(parent_path)
from utils.data_utils import(
    load_and_preprocess_data,
    add_term_index,
    build_node_level_features,
    build_student_level_features
)
from src.config import(
    DEFAULT_CLUSTER_RANGE,
    ENCODING,
    GPA_EMPTY_MARK,
    TARGET_COL
)

# ---------------------- 2. 学生级特征处理与聚类类 ----------------------
class StudentClustering:
    """学生级聚类类（基于全局统计特征）"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.optimal_k = None
        self.cluster_features = None  # 簇特征字典
        self.student_feat_df = None   # 学生级特征DataFrame

    def fit_scaler(self, feat_df, core_features):
        """训练标准化器"""
        feature_matrix = feat_df[core_features].values
        self.scaler.fit(feature_matrix)

    def transform_features(self, feat_df, core_features):
        """转换特征"""
        feature_matrix = feat_df[core_features].values
        return self.scaler.transform(feature_matrix)

    def find_optimal_k(self, feature_scaled, k_range=DEFAULT_CLUSTER_RANGE):
        """选择最优簇数K"""
        sse = []
        silhouette_scores = []
        valid_silhouette = False

        max_k = min(k_range[1], len(feature_scaled) // 2)
        k_range = (k_range[0], max_k)

        for k in range(*k_range):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(feature_scaled)

                if len(np.unique(labels)) < 2:
                    silhouette_scores.append(-1)
                    continue

                score = silhouette_score(feature_scaled, labels)
                silhouette_scores.append(score)
                sse.append(kmeans.inertia_)
                valid_silhouette = True

            except Exception as e:
                silhouette_scores.append(-1)
                sse.append(np.inf)
                continue

        self.optimal_k = np.argmax(silhouette_scores) + k_range[0] if valid_silhouette else max(k_range[0], 2)
        return self.optimal_k, sse, silhouette_scores

    def kmeans_cluster(self, student_feat_df, core_features, n_clusters=None):
        """执行学生级聚类并提取簇特征"""
        self.student_feat_df = student_feat_df.copy()

        # 训练标准化器
        self.fit_scaler(self.student_feat_df, core_features)

        # 转换特征
        feat_scaled = self.transform_features(self.student_feat_df, core_features)

        # 确定簇数并聚类
        if n_clusters is None:
            n_clusters, _, _ = self.find_optimal_k(feat_scaled)

        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.student_feat_df["cluster_label"] = self.cluster_model.fit_predict(feat_scaled)

        # 提取簇特征
        self.cluster_features = self._extract_cluster_features(core_features)
        return self.student_feat_df["cluster_label"]

    def _extract_cluster_features(self, core_features):
        """提取学生级簇的特征描述"""
        cluster_features = {}
        for cluster in self.student_feat_df["cluster_label"].unique():
            cluster_df = self.student_feat_df[self.student_feat_df["cluster_label"] == cluster]
            feat_desc = {
                "cluster_id": cluster,
                "student_count": len(cluster_df),
                "avg_gpa_mean": round(cluster_df["gpa_mean"].mean(), 2),
                "avg_dccy_max": round(cluster_df["dccy_max"].mean(), 2),
                "avg_term_count": round(cluster_df["term_count"].mean(), 2),
                "desc": f"簇{cluster}：包含{len(cluster_df)}个学生，平均GPA{round(cluster_df['gpa_mean'].mean(),2)}，平均最大挂科数{round(cluster_df['dccy_max'].mean(),2)}，平均学期数{round(cluster_df['term_count'].mean(),2)}"
            }
            # 补充核心特征的统计
            for feat in core_features:
                feat_desc[f"avg_{feat}"] = round(cluster_df[feat].mean(), 2)
            cluster_features[cluster] = feat_desc
        return cluster_features