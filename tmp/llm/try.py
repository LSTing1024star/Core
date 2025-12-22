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

# 单轮对话示例
def single_chat():
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "你好，请介绍一下自己"}
        ],
        max_tokens=512,
        temperature=0.7
    )
    print("【单轮对话响应】：", response.choices[0].message.content)

# 多轮对话示例（保持上下文）
def multi_chat():
    messages = [
        {"role": "user", "content": "你好，我叫小明"},
        {"role": "assistant", "content": "你好小明，很高兴认识你！"},
        {"role": "user", "content": "记得我的名字吗？请告诉我"}
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=200
    )
    print("【多轮对话响应】：", response.choices[0].message.content)

# 流式输出示例（实时返回结果）
def stream_chat():
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "请详细解释一下人工智能的发展历程"}
        ],
        max_tokens=512,
        stream=True  # 开启流式输出
    )
    print("【流式响应】：", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # 最后换行

if __name__ == "__main__":
    # 执行不同的对话示例
    single_chat()
    # multi_chat()
    # stream_chat()
