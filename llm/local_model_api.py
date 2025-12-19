from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import time

# ===================== 本地模型配置（根据你的环境修改）=====================
MODEL_PATH = "/home/lcteam/Qwen/qwen8b"  # 本地Qwen-8B模型路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 算力7.0 GPU选cuda
LOAD_IN_4BIT = False  # 若用INT4量化版，改为True（需安装bitsandbytes）
MAX_NEW_TOKENS = 512  # 最大生成Token数
# ==========================================================================

# 初始化FastAPI应用（极简版OpenAI兼容服务）
app = FastAPI(title="Local Qwen-8B OpenAI API")

# 加载模型和Tokenizer（启动时仅加载一次，避免重复加载）
print(f"正在加载本地模型：{MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 算力7.0 GPU支持float16
    device_map="auto",
    load_in_4bit=LOAD_IN_4BIT,  # 量化开关
    # 非量化版显存优化：梯度检查点（若单卡16GB显存不足，开启）
    gradient_checkpointing=True if not LOAD_IN_4BIT else False
)
print("模型加载完成！")

# 定义OpenAI ChatCompletions请求体格式
class ChatRequest(BaseModel):
    model: str  # 模型名（仅需和客户端一致，无实际校验）
    messages: list[dict]  # 对话消息列表，格式：[{"role": "user/assistant", "content": "..."}]
    max_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.7
    top_p: float = 0.8

# 实现OpenAI兼容的/chat/completions接口
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # 1. 构造Qwen-8B专用对话模板（关键：必须遵循Qwen的模板格式）
    prompt = tokenizer.apply_chat_template(
        req.messages,
        tokenize=False,
        add_generation_prompt=True  # 为助理回复添加提示符
    )
    
    # 2. 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 3. 生成响应
    with torch.no_grad():  # 禁用梯度计算，节省显存
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # 防止padding报错
        )
    
    # 4. 解码结果（去掉输入部分，只保留生成的响应）
    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    # 5. 构造OpenAI兼容的响应格式
    return {
        "id": f"chat-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(inputs.input_ids[0]),
            "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
            "total_tokens": len(outputs[0])
        }
    }

# 启动服务（直接运行此文件即可）
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",  # 允许本地/局域网访问
        port=8000,       # 服务端口
        log_level="info"
    )
