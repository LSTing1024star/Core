# ====================== 数据路径配置 ======================
# 原始数据路径
HOME="/home/lcteam/Project_Core/"
DATAHOME=f"{HOME}data/"
RESULTHOME=f"{HOME}result/"
RAW_DATA_PATH = f"{DATAHOME}totaldata.csv"
# 拆分后的数据集路径
TRAIN_PATH = f"{DATAHOME}train_data_34.csv"
TEST_PATH = f"{DATAHOME}test_data_34.csv"
NEW_STUDENT_DATA_PATH = f"{DATAHOME}new_student_data.csv"
# 预测结果路径
TRAIN_PRED_SAVE_PATH = f"{RESULTHOME}train_node_prediction.csv"
TEST_PRED_SAVE_PATH = f"{RESULTHOME}test_node_prediction.csv"
NEW_STUDENT_PRED_PATH = f"{RESULTHOME}new_student_prediction.csv"
# 指标汇总路径
METRICS_SAVE_PATH = f"{RESULTHOME}prediction_metrics_summary.csv"

# ====================== 数据处理配置 ======================
TEST_TERMS = ["20232", "20242"]  # 测试集学生的最早学期
NEW_STUDENT_TERM = "20252"       # 新入学学生的学期标识
GPA_EMPTY_MARK = "暂无GPA"       # GPA空值的语义标识

# ====================== 预测任务配置 ======================
FEATURE_COLS = ["GPA", "DCCY", "JXJ"]  # 特征列
TARGET_COL = "XYYJ"                   # 预测目标列
MAX_HISTORY_LEN = "all"               # 历史学期使用模式（all/数字）

# ====================== 本地大模型配置 ======================
LLM_API_URL = "http://0.0.0.0:59368/v1/"
LLM_MAX_LENGTH = 5120
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.0
LLM_MAX_NEW_TOKENS=256

# ====================== 通用配置 ======================
ENCODING = "utf-8-sig"

# ====================== 聚类扩展配置（新增，兼容原有功能） ======================
DEFAULT_CLUSTER_RANGE = (2, 11)  # K-Means最优簇数搜索范围

# ====================== 推荐任务配置（新增） ======================
RECOMMEND_SAVE_PATH=f"{RESULTHOME}recommendation_detail.csv"
RECOMMEND_METRICS_PATH=f"{RESULTHOME}recommendation_metrics_detail.csv"
TERM_DIFF_THRESHOLD=1
TOP_K_CANDIDATES=5
SAMPLE_RATE=0.2
