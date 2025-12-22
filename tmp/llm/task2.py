import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

# ====================== 1. 数据准备与模拟（模拟你现有代码的数据集结构） ======================
def generate_simulation_data():
    """
    模拟学生数据：包含静态特征和时序特征（贴近实际教育场景）
    返回：
        df: 静态特征数据（学号、GPA、挂科数、奖学金次数）
        seq_df: 时序特征数据（学号、多学期GPA序列、趋势）
    """
    # 静态特征数据
    student_ids = [f"S{i:03d}" for i in range(1, 51)]  # 50个学生
    gpas = np.round(np.random.uniform(2.0, 4.0, 50), 2)
    dccy = np.random.randint(0, 5, 50)  # 挂科数0-4
    jxj = np.random.randint(0, 4, 50)   # 奖学金次数0-3

    df = pd.DataFrame({
        "XH": student_ids,
        "GPA": gpas,
        "DCCY": dccy,
        "JXJ": jxj
    })

    # 时序特征数据（最近3学期GPA）
    seq_data = []
    for sid in student_ids:
        base_gpa = df[df["XH"] == sid]["GPA"].values[0]
        # 生成3学期GPA，带轻微趋势
        semester_gpas = np.round(np.random.uniform(base_gpa-0.3, base_gpa+0.3, 3), 2)
        trend = "上升" if semester_gpas[-1] > semester_gpas[0] else "下降" if semester_gpas[-1] < semester_gpas[0] else "平稳"
        seq_data.append({
            "student_id": sid,
            "sequence": semester_gpas.tolist(),
            "trend": trend
        })
    seq_df = pd.DataFrame(seq_data)

    return df, seq_df

# ====================== 2. 通用工具函数（LLM调用、学生特征描述） ======================
def call_local_qwen(prompt, api_url="http://localhost:8000/generate", max_length=1500, temperature=0.3):
    """
    调用本地部署的Qwen8B模型
    :param prompt: 提示词
    :param api_url: 本地API地址
    :param max_length: 生成文本最大长度
    :param temperature: 生成温度（越低越稳定）
    :return: LLM生成的结果
    """
    try:
        response = requests.post(
            api_url,
            json={
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": 0.9
            },
            timeout=30
        )
        # 适配常见的LLM输出格式（若你的接口返回字段不同，需调整）
        return response.json().get("result", response.json().get("response", "生成失败"))
    except Exception as e:
        return f"LLM调用失败：{str(e)}"

def generate_student_profile(student_id, df, seq_df):
    """
    生成学生的自然语言特征描述（整合静态+时序特征）
    :param student_id: 学生学号
    :param df: 静态特征数据
    :param seq_df: 时序特征数据
    :return: 自然语言描述字符串
    """
    # 静态特征
    static = df[df["XH"] == student_id].iloc[0]
    gpa = static["GPA"]
    dccy = static["DCCY"]
    jxj = static["JXJ"]

    # 时序特征
    seq = seq_df[seq_df["student_id"] == student_id].iloc[0]
    recent_gpas = seq["sequence"]
    trend = seq["trend"]

    return f"""
    学生{student_id}的学业情况：
    - 总GPA：{gpa}，总挂科数：{dccy}，获得奖学金次数：{jxj}
    - 最近3学期GPA：{recent_gpas}，趋势：{trend}
    """

# ====================== 3. KMeans聚类（缩小LLM筛选范围） ======================
def cluster_students(df, seq_df, n_clusters=5):
    """
    对学生进行KMeans聚类（结合静态特征）
    :param df: 静态特征数据
    :param seq_df: 时序特征数据
    :param n_clusters: 聚类数量
    :return: 带聚类标签的df和seq_df
    """
    # 提取特征并标准化
    features = df[["GPA", "DCCY", "JXJ"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(features_scaled)

    # 关联聚类标签到时序数据
    seq_df = seq_df.merge(df[["XH", "cluster"]], left_on="student_id", right_on="XH", how="left").drop("XH", axis=1)

    return df, seq_df

# ====================== 4. LLM筛选榜样学生 ======================
def llm_select_role_models(target_id, df, seq_df, top_k=3):
    """
    从同簇学生中用LLM筛选榜样
    :param target_id: 目标学生ID
    :param df: 带聚类标签的静态数据
    :param seq_df: 带聚类标签的时序数据
    :param top_k: 推荐榜样数量
    :return: 榜样列表+LLM推荐理由
    """
    # 获取目标学生的聚类标签和特征描述
    target_cluster = df[df["XH"] == target_id]["cluster"].iloc[0]
    target_profile = generate_student_profile(target_id, df, seq_df)

    # 获取同簇其他学生
    cluster_students = df[(df["cluster"] == target_cluster) & (df["XH"] != target_id)]["XH"].tolist()
    if not cluster_students:
        return [], "同簇无其他学生，无法推荐"

    # 生成候选学生的特征描述
    candidate_profiles = [generate_student_profile(sid, df, seq_df) for sid in cluster_students]
    candidates_str = "\n".join([f"学生{i+1}：{p}" for i, p in enumerate(candidate_profiles)])

    # 构造筛选Prompt（加入Few-Shot示例提升准确性）
    prompt = f"""
    你是教育领域的学生榜样推荐专家，需要为目标学生从候选中筛选{top_k}名最合适的榜样。

    ### 筛选标准
    1. 榜样需比目标学生稍好（如GPA高0.1-0.5，挂科数少1-2门，或趋势更优），但差距不能过大（GPA不超过0.8）；
    2. 榜样的优势需具有可借鉴性（如从挂科1门提升到0门，而非天生高GPA）；
    3. 优先选择与目标学生有相似进步轨迹的（如同样从下降趋势转为平稳/上升）。

    ### Few-Shot示例
    示例1（合理推荐）：
    目标学生：GPA2.8，挂科2门，趋势下降；候选学生A：GPA3.0，挂科1门，趋势平稳 → 推荐，理由：GPA差距0.2，挂科少1门，趋势更优，优势可学习。
    示例2（不合理推荐）：
    目标学生：GPA2.8，挂科2门；候选学生B：GPA3.8，挂科0门 → 不推荐，理由：GPA差距1.0，过大，无借鉴性。

    ### 评测数据
    目标学生情况：{target_profile}
    候选学生情况（与目标学生学业水平相近）：{candidates_str}

    ### 输出要求
    请严格按以下格式输出：
    推荐榜样：[学生ID1, 学生ID2, 学生ID3]
    推荐理由：
    1. 学生ID1：具体理由（符合哪些筛选标准）
    2. 学生ID2：具体理由
    3. 学生ID3：具体理由

    若无可推荐的榜样，输出：
    推荐榜样：[]
    推荐理由：无合适榜样的原因
    """

    # 调用LLM
    llm_result = call_local_qwen(prompt)

    # 解析LLM输出（提取榜样ID和理由）
    import re
    candidate_ids = re.findall(r"学生(S\d{3})", llm_result)  # 匹配S001格式的学号
    reason = llm_result.split("推荐理由：")[-1].strip()

    return candidate_ids[:top_k], reason

# ====================== 5. 推荐算法指标计算 ======================
class RecommendationMetrics:
    """推荐算法指标计算类（适配学生榜样推荐场景）"""
    def __init__(self, df, threshold_gpa=0.5, threshold_dccy=2):
        self.df = df
        self.threshold_gpa = threshold_gpa  # GPA最大差距
        self.threshold_dccy = threshold_dccy  # 挂科数最大差距
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(df[["GPA", "DCCY"]])
        self.feature_df = pd.DataFrame(self.features, columns=["GPA_scaled", "DCCY_scaled"], index=df["XH"])

    def get_ideal_candidates(self, target_id):
        """为目标学生生成“理想榜样集”（符合阈值的同簇学生）"""
        target = self.df[self.df["XH"] == target_id].iloc[0]
        cluster = target["cluster"]
        target_gpa = target["GPA"]
        target_dccy = target["DCCY"]

        ideal = self.df[
            (self.df["cluster"] == cluster) &
            (self.df["GPA"] > target_gpa) &
            (self.df["GPA"] - target_gpa <= self.threshold_gpa) &
            (self.df["DCCY"] < target_dccy) &
            (target_dccy - self.df["DCCY"] <= self.threshold_dccy) &
            (self.df["XH"] != target_id)
        ]["XH"].tolist()
        return ideal

    def cosine_similarity(self, target_id, candidate_id):
        """计算目标与榜样的特征余弦相似度"""
        target_feat = self.feature_df.loc[target_id].values.reshape(1, -1)
        candidate_feat = self.feature_df.loc[candidate_id].values.reshape(1, -1)
        return round(cosine_similarity(target_feat, candidate_feat)[0][0], 4)

    def mae(self, target_id, candidate_id):
        """计算特征平均绝对误差"""
        target = self.df[self.df["XH"] == target_id].iloc[0]
        candidate = self.df[self.df["XH"] == candidate_id].iloc[0]
        gpa_diff = abs(candidate["GPA"] - target["GPA"])
        dccy_diff = abs(candidate["DCCY"] - target["DCCY"])
        return round((gpa_diff + dccy_diff) / 2, 4)

    def ndcg_k(self, target_id, candidate_ids, k=3):
        """计算NDCG@k（衡量排序质量）"""
        ideal = self.get_ideal_candidates(target_id)
        if not ideal:
            return 0.0

        # 给推荐的榜样打分
        rel = [1 if c in ideal else 0 for c in candidate_ids[:k]]
        # 计算DCG
        dcg = rel[0] + sum([r / np.log2(i+2) for i, r in enumerate(rel[1:])])
        # 计算理想DCG
        ideal_rel = [1] * min(k, len(ideal))
        idcg = ideal_rel[0] + sum([r / np.log2(i+2) for i, r in enumerate(ideal_rel[1:])])

        return round(dcg / idcg if idcg > 0 else 0.0, 4)

    def gpa_gap_rate(self, target_id, candidate_id):
        """计算GPA差距率"""
        target = self.df[self.df["XH"] == target_id].iloc[0]
        candidate = self.df[self.df["XH"] == candidate_id].iloc[0]
        gap_rate = (candidate["GPA"] - target["GPA"]) / target["GPA"] if target["GPA"] > 0 else 0.0
        return round(gap_rate, 4)

    def calculate_all_metrics(self, target_id, candidate_ids):
        """计算单个目标学生的所有指标"""
        metrics = {}
        # 单榜样指标（取第一个榜样）
        candidate_id = candidate_ids[0] if candidate_ids else None
        if candidate_id:
            metrics["cosine_similarity"] = self.cosine_similarity(target_id, candidate_id)
            metrics["mae"] = self.mae(target_id, candidate_id)
            metrics["gpa_gap_rate"] = self.gpa_gap_rate(target_id, candidate_id)
        # 列表指标
        metrics["ndcg@3"] = self.ndcg_k(target_id, candidate_ids, k=3)
        # 合理性判断
        metrics["is_gpa_gap_valid"] = metrics.get("gpa_gap_rate", 0) <= 0.2  # 差距率≤20%为合理
        metrics["is_similarity_valid"] = metrics.get("cosine_similarity", 0) >= 0.7  # 相似度≥0.7为合理

        return metrics

# ====================== 6. LLM评价推荐结果（结合指标） ======================
def llm_evaluate_recommendation(target_id, candidate_ids, metrics, df, seq_df):
    """
    用LLM评价推荐结果（结合定量指标）
    :param target_id: 目标学生ID
    :param candidate_ids: 榜样列表
    :param metrics: 计算好的指标
    :param df: 静态数据
    :param seq_df: 时序数据
    :return: LLM的评价结果
    """
    # 获取特征描述
    target_profile = generate_student_profile(target_id, df, seq_df)
    candidate_profiles = [generate_student_profile(cid, df, seq_df) for cid in candidate_ids]
    candidate_profile_str = "\n".join([f"榜样{cid}：{cp}" for cid, cp in zip(candidate_ids, candidate_profiles)])

    # 构造指标描述
    metrics_str = f"""
    推荐算法指标结果：
    1. 特征余弦相似度：{metrics.get('cosine_similarity', '无')}（≥0.7为合理，值越高相似度越高）
    2. 特征MAE：{metrics.get('mae', '无')}（≤0.3为合理，值越低差距越小）
    3. GPA差距率：{metrics.get('gpa_gap_rate', '无')}（≤20%为合理，反映“好一点”的程度）
    4. NDCG@3：{metrics.get('ndcg@3', '无')}（≥0.8为优秀，反映推荐排序的合理性）
    5. 合理性判断：GPA差距{'' if metrics.get('is_gpa_gap_valid', False) else '不'}合理，相似度{'' if metrics.get('is_similarity_valid', False) else '不'}合理
    """

    # 构造评测Prompt
    prompt = f"""
    你是教育领域的推荐系统评测专家，需要结合定量的推荐算法指标和定性的业务逻辑，评价学生榜样推荐的合理性。

    ### 评测背景
    目标：为学生推荐“稍优且可借鉴”的榜样（比自己好一点，但差距不大，优势可学习）。
    定量指标说明：
    - 余弦相似度≥0.7：榜样与目标学生学业水平相近，具有可比性；
    - GPA差距率≤20%：榜样的GPA优势在可学习范围内，避免“好太多”；
    - NDCG@3≥0.8：推荐的榜样排序符合业务优先级。

    ### 评测数据
    目标学生{target_id}的学业特征：{target_profile}
    推荐的榜样学生特征：{candidate_profile_str}
    {metrics_str}

    ### 评测要求
    1. **指标解读**：分析上述定量指标反映了推荐的哪些优点/问题？
    2. **合理性评分**：从以下维度打1-5分（1分最差，5分最优），并说明理由：
       - 维度1：相似度与差距合理性（结合余弦相似度、GPA差距率）
       - 维度2：推荐排序质量（结合NDCG@3）
       - 维度3：榜样的可借鉴性（业务角度，与指标结果关联）
       - 维度4：整体推荐合理性（综合所有指标和特征）
    3. **改进建议**：基于指标和特征，给出优化推荐的具体建议（若有问题）。

    ### 输出格式
    【指标解读】
    ...
    【合理性评分】
    维度1：X分，理由：...
    维度2：X分，理由：...
    维度3：X分，理由：...
    维度4：X分，理由：...
    【改进建议】
    ...
    """

    # 调用LLM
    return call_local_qwen(prompt)

# ====================== 7. 主流程整合 ======================
def main():
    # 1. 生成模拟数据
    print("步骤1：生成模拟学生数据...")
    df, seq_df = generate_simulation_data()

    # 2. 学生聚类
    print("步骤2：对学生进行KMeans聚类...")
    df, seq_df = cluster_students(df, seq_df, n_clusters=5)

    # 3. 初始化指标计算类
    metrics_calculator = RecommendationMetrics(df, threshold_gpa=0.5, threshold_dccy=2)

    # 4. 为每个学生筛选榜样并计算指标、评价
    print("步骤3：LLM筛选榜样并评价...")
    results = []
    target_ids = df["XH"].tolist()[:10]  # 取前10个学生测试（可改为全部）

    for target_id in target_ids:
        # 筛选榜样
        candidate_ids, select_reason = llm_select_role_models(target_id, df, seq_df, top_k=3)
        # 计算指标
        metrics = metrics_calculator.calculate_all_metrics(target_id, candidate_ids)
        # LLM评价
        evaluation = llm_evaluate_recommendation(target_id, candidate_ids, metrics, df, seq_df)
        # 保存结果
        results.append({
            "target_id": target_id,
            "candidate_ids": candidate_ids,
            "select_reason": select_reason,
            "metrics": metrics,
            "llm_evaluation": evaluation
        })

    # 5. 结果整合与保存
    print("步骤4：保存结果...")
    result_df = pd.DataFrame(results)
    # 转换指标为可存储的格式
    result_df["metrics_str"] = result_df["metrics"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    # 保存CSV
    result_df.to_csv("recommendation_evaluation_result.csv", index=False, encoding="utf-8")

    # 6. 整体总结
    print("步骤5：生成整体评价总结...")
    # 计算整体指标均值
    all_metrics = [r["metrics"] for r in results if r["metrics"]]
    if all_metrics:
        avg_similarity = np.mean([m.get("cosine_similarity", 0) for m in all_metrics])
        avg_ndcg = np.mean([m.get("ndcg@3", 0) for m in all_metrics])
        valid_gpa_rate = np.mean([m.get("is_gpa_gap_valid", False) for m in all_metrics]) * 100

        # 构造总结Prompt
        summary_prompt = f"""
        基于以下整体统计数据，总结本次学生榜样推荐的整体效果：
        1. 平均余弦相似度：{round(avg_similarity, 4)}
        2. 平均NDCG@3：{round(avg_ndcg, 4)}
        3. GPA差距合理的推荐占比：{round(valid_gpa_rate, 2)}%
        请从指标表现、业务合理性、优化方向三个角度总结，给出整体评分（1-10分）。
        """

        # 调用LLM生成总结
        overall_summary = call_local_qwen(summary_prompt)
        print("\n=== 整体评价总结 ===")
        print(overall_summary)

    print("\n所有任务完成！结果已保存至 recommendation_evaluation_result.csv")

if __name__ == "__main__":
    main()
