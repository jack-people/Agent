import os
import time
import yfinance as yf

from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()  # 从 .env 文件加载环境变量
client = OpenAI()  # 初始化 OpenAI 客户端

my_api_key = os.getenv("DEEPSEEK_API_KEY")
if not my_api_key:
    raise ValueError("🚨 致命错误：找不到 API Key！请检查 .env 文件是否配置了 DEEPSEEK_API_KEY。")

llm = ChatOpenAI(
    api_key=my_api_key,
    base_url="https://api.deepseek.com", 
    model="deepseek-chat",                # 替换模型名称
    temperature=1.0
)


# =====================================
# 工具1：搜索新闻
# =====================================

search_tool = DuckDuckGoSearchRun(
    name="search_stock_news",
    description="搜索公司的最新新闻和市场动态"
)


# =====================================
# 工具2：获取真实股票价格
# =====================================

price_cache = {}

@tool
def get_stock_price(ticker: str) -> str:
    """获取股票最近5天的真实价格"""

    try:

        print(f"\n[系统] 正在查询 {ticker} 股票数据...\n")

        if ticker in price_cache:
            print(f"[系统] 从缓存中获取 {ticker} 的股票数据...\n")
            return price_cache[ticker]

        time.sleep(2)

        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        result = ""

        for date, row in hist.iterrows():
            result += f"{date.date()} : ${round(row['Close'],2)}\n"

        price_cache[ticker] = result  # 将结果缓存起来

        return result
    
    except Exception as e:
        print(f"❌ 查询 {ticker} 股票数据时发生错误: {e}")
        return f"❌ 查询 {ticker} 股票数据时发生错误: {e}"

# 工具列表
tools = [search_tool, get_stock_price]


# =====================================
# Prompt
# =====================================

prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
你是一名华尔街资深股票分析师。

你的任务：

1 搜索公司最新新闻
2 获取股票最近价格
3 写一份专业股票报告

报告必须包含：

【公司最新动态】

【近期股价走势】

【投资建议】

报告使用中文
"""
),
("human","{input}"),
("placeholder","{agent_scratchpad}")
]
)


# =====================================
# 创建 Agent
# =====================================

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


# =====================================
# 运行
# =====================================

if __name__ == "__main__":

    print("🚀 股票分析 Agent 启动...\n")

    user_input = "请分析 AAPL 今天的股票情况"

    response = agent_executor.invoke(
        {"input": user_input}
    )

    print("\n"+"="*50)
    print("📊 AI 股票分析报告")
    print("="*50)

    print(response["output"])
