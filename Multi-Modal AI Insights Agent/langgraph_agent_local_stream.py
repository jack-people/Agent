import os
import requests
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()

# ==========================================
# 0. 配置 API Keys 与服务地址
# ==========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  
DS_BASE_URL = "https://api.deepseek.com"

# ⚠️ 你的本地 5090 服务地址
# 如果你在这个 AutoDL 机器上运行此脚本，用 localhost 即可。
# 如果你在自己电脑上运行，需要替换为 AutoDL 提供的自定义服务公网暴露地址。
LOCAL_VLLM_URL = os.getenv("LOCAL_VLLM_URL", "http://localhost:8000/v1")

# --- 模型 1：路由决策 (DeepSeek-Chat) ---
llm_router = ChatOpenAI(
    model="deepseek-chat", 
    api_key=DEEPSEEK_API_KEY, 
    base_url=DS_BASE_URL,
    temperature=0.1
)

# --- 模型 2：兜底大模型 (DeepSeek-Reasoner R1) ---
llm_synthesizer = ChatOpenAI(
    model="deepseek-reasoner", 
    api_key=DEEPSEEK_API_KEY, 
    base_url=DS_BASE_URL,
    temperature=0.7,
    max_tokens=4096
)

# --- 模型 3：你的私有化技术大牛 (本地 5090) ---
llm_local_guru = ChatOpenAI(
    model="tech-guru",       # 这是你启动 vLLM 时 --served-model-name 指定的名字
    api_key="EMPTY",         # vLLM 不需要 key
    base_url=LOCAL_VLLM_URL, 
    temperature=0.7,
    max_retries=1,           # 不重试，防止卡顿
    timeout=30.0             # 生成超时时间
)

# ==========================================
# 1. 探活函数 (Health Check) - 核心新增！
# ==========================================
def is_local_model_alive(base_url, timeout=1.5):
    """检测本地 5090 模型是否在线（1.5秒内无响应直接判定离线）"""
    try:
        # 请求 OpenAI 标准的 /models 接口测试连通性
        response = requests.get(f"{base_url}/models", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# ==========================================
# 2. 定义 Agent 的工具 (保持不变)
# ==========================================
@tool
def search_local_arxiv_db(query: str) -> str:
    """查询 Arxiv 多模态论文的本地知识库。"""
    return "【本地知识库检索模拟】没有找到相关论文。"

@tool
def search_web_news(query: str) -> str:
    """查询互联网上的最新AI新闻。"""
    search = DuckDuckGoSearchResults(max_results=3)
    try:
        return f"【互联网搜索结果】\n{search.run(query)}"
    except Exception as e:
        return f"网络搜索出错: {str(e)}"

tools =[search_local_arxiv_db, search_web_news]
llm_router_with_tools = llm_router.bind_tools(tools)

# ==========================================
# 3. 构建 LangGraph 状态图
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def router_node(state: AgentState):
    print("🤖 [Router] DeepSeek 正在思考需要调用什么工具...")
    sys_msg = SystemMessage(content="你是多模态AI领域的智能助手。请根据用户的问题，决定是查询本地Arxiv论文库，还是去网上搜索。")
    messages = [sys_msg] + state["messages"]
    response = llm_router_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def synthesizer_node(state: AgentState):
    print("\n" + "="*50)
    
    # ======== 核心逻辑：智能路由与降级兜底 ========
    use_local = is_local_model_alive(LOCAL_VLLM_URL)
    
    if use_local:
        print("🚀 [状态] 检测到本地 5090 满血在线！将使用你的专属【技术大牛】模型解答。")
        active_llm = llm_local_guru
        is_reasoner = False # 只有 R1 才有思维链
    else:
        print("⚠️ [状态] 本地 5090 未响应或未开机，自动 Fallback 切换至【DeepSeek-Reasoner】兜底！")
        active_llm = llm_synthesizer
        is_reasoner = True
    print("="*50)
    
    original_question = state["messages"][0].content
    context_str = "\n".join([msg.content for msg in state["messages"] if msg.type == "tool"])
    if not context_str:
        context_str = "未调用任何外部工具，请凭借内部知识解答。"

    prompt = f"""你是多模态AI领域的技术大牛。
请根据以下参考资料，为用户撰写一份专业、全面、犀利的解答。多用行业黑话。

【用户提问】: {original_question}
【参考资料】: {context_str}
"""
    
    response_chunks =[]
    is_thinking_started = False
    is_answer_started = False
    
    # 流式输出兼容逻辑
    for chunk in active_llm.stream([HumanMessage(content=prompt)]):
        
        # 1. 如果使用的是 DeepSeek-R1，处理它的思考过程
        if is_reasoner:
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                if not is_thinking_started:
                    print("\n🤔 【DeepSeek 内心思考】:\n", end="", flush=True)
                    is_thinking_started = True
                print(reasoning, end="", flush=True)
                
        # 2. 处理最终输出 (无论是 5090 还是 DeepSeek，都会走到这里)
        content = chunk.content
        if content:
            if not is_answer_started:
                print("\n\n✨ 【大牛专业解答】:\n", end="", flush=True)
                is_answer_started = True
            print(content, end="", flush=True)
            
        response_chunks.append(chunk)
        
    final_message = response_chunks[0]
    for chunk in response_chunks[1:]:
        final_message += chunk
        
    print("\n" + "="*50)
    return {"messages": [final_message]}

# ==========================================
# 4. 编排并编译图
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("tools", tool_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", tools_condition, {"tools": "tools", END: "synthesizer"})
workflow.add_edge("tools", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()

# ==========================================
# 5. 运行测试
# ==========================================
if __name__ == "__main__":
    question = "多模态大模型(VLM)最近有什么突破性的最新进展？请去网上找最新的报道。"
    print(f"🧑‍💻 用户提问: {question}\n" + "-"*50)
    
    for output in app.stream({"messages": [HumanMessage(content=question)]}, stream_mode="updates"):
        for node_name, node_state in output.items():
            if node_name == "router":
                msg = node_state["messages"][0]
                if msg.tool_calls:
                    print(f"   -> 决定调用工具: {[t['name'] for t in msg.tool_calls]}")
            elif node_name == "tools":
                print("   -> 工具执行完毕，获取到参考资料。")