from openai import OpenAI

# ===================== 客户端配置 ======================
LOCAL_API_URL = "http://localhost:8000/v1"  # 本地服务地址
MODEL_NAME = "qwen-8b-chat"  # 只需和服务端的模型名一致即可（无实际校验）
# ======================================================

# 初始化OpenAI客户端，指向本地服务
client = OpenAI(
    base_url=LOCAL_API_URL,  # 核心：指向本地OpenAI兼容服务
    api_key="dummy-key"      # 本地服务无需鉴权，任意字符串即可
)
