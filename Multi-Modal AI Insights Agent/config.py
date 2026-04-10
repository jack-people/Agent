# 配置与模型初始化
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  
BASE_URL = "https://api.deepseek.com"

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