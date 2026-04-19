# 配置与模型初始化
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  
BASE_URL = "https://api.deepseek.com"

DIFY_APP_API_KEY = os.getenv("DIFY_APP_API_KEY")
DIFY_WORKFLOW_URL = "http://localhost/v1/workflows/run"


DIFY_DATASET_API_KEY = os.getenv("DIFY_DATASET_API_KEY")
# 【核心】Dify 知识库（数据集）配置
# ⚠️ 2. 填入你从 URL 里提取的知识库 ID
DATASET_ID = "7f5cbb69-1ff7-4f2e-a3bd-baaddf10847c"

# 知识库直接检索的专属 URL 格式
DIFY_RETRIEVAL_URL = f"http://localhost/v1/datasets/{DATASET_ID}/retrieve"

# 路由器模型
llm_router = ChatOpenAI(
    model="deepseek-chat", 
    api_key=DEEPSEEK_API_KEY, 
    base_url=BASE_URL,
    temperature=0.1
)

# 总结器模型
llm_synthesizer = ChatOpenAI(
    model="deepseek-reasoner", 
    api_key=DEEPSEEK_API_KEY, 
    base_url=BASE_URL,
    temperature=0.7,
    max_tokens=4096
)