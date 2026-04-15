#已废弃，用langgraph_agent_stream.py替代

import os
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

load_dotenv()  # 从 .env 文件加载环境变量

# ==========================================
# 0. 配置你的 DeepSeek API Key
# ==========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  
BASE_URL = "https://api.deepseek.com"

# 实例化两个不同的模型！
# 模型 A：负责调用工具和决策 (快且准)
llm_router = ChatOpenAI(
    model="deepseek-chat", 
    api_key=DEEPSEEK_API_KEY, 
    base_url=BASE_URL,
    temperature=0.1
)

# 模型 B：负责最终的深度推理与总结 (深且强)
llm_synthesizer = ChatOpenAI(
    model="deepseek-reasoner", 
    api_key=DEEPSEEK_API_KEY, 
    base_url=BASE_URL,
    temperature=0.7,
    presence_penalty=0.3,
    frequency_penalty=0.3,
    max_tokens=4096
)

# ==========================================
# 1. 定义 Agent 的工具 (Tools)
# ==========================================

@tool
def search_local_arxiv_db(query: str) -> str:
    """
    当你需要查询 Arxiv 多模态论文的本地知识库时调用此工具。
    输入：搜索关键词。返回：相关论文的标题和摘要。
    """
    # 这里为了代码简洁，我们写一个简化的检索逻辑（实际你可以直接导入 step2 的函数）
    import chromadb
    from chromadb.utils import embedding_functions
    try:
        client = chromadb.PersistentClient(path="./multimodal_papers_db")
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
        collection = client.get_collection(name="arxiv_multimodal", embedding_function=emb_fn)
        results = collection.query(query_texts=[query], n_results=3)
        
        if not results['ids'][0]:
            return "本地数据库中没有找到相关论文。"
            
        res_str = "【本地 Arxiv 论文检索结果】\n"
        for i in range(len(results['ids'][0])):
            title = results['metadatas'][0][i]['title']
            abstract = results['documents'][0][i].split("Abstract: ")[-1][:300]
            res_str += f"- 标题: {title}\n  摘要: {abstract}...\n"
        return res_str
    except Exception as e:
        return f"本地检索出错: {str(e)}"

@tool
def search_web_news(query: str) -> str:
    """
    当本地数据库没有最新信息，或者需要查询互联网上的最新AI新闻、业界动态时调用此工具。
    """
    search = DuckDuckGoSearchResults(max_results=3)
    try:
        results = search.run(query)
        return f"【互联网搜索结果】\n{results}"
    except Exception as e:
        return f"网络搜索出错: {str(e)}"

tools = [search_local_arxiv_db, search_web_news]
# 将工具绑定给路由器模型
llm_router_with_tools = llm_router.bind_tools(tools)

# ==========================================
# 2. 构建 LangGraph 状态图
# ==========================================

# 定义图的状态：一个包含所有对话历史的消息列表
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 节点1：路由决策节点 (Router)
def router_node(state: AgentState):
    print("🤖 [Router 模型 (deepseek-chat)] 正在思考需要调用什么工具...")
    # 系统提示词让它聪明点
    sys_msg = SystemMessage(content="你是多模态AI领域的智能助手。请根据用户的问题，决定是查询本地Arxiv论文库，还是去网上搜索最新资讯。如果需要，你可以同时或先后调用它们。")
    messages = [sys_msg] + state["messages"]
    response = llm_router_with_tools.invoke(messages)
    return {"messages": [response]}

# 节点2：工具执行节点 (Tool Execution)
tool_node = ToolNode(tools)

# 节点3：最终总结节点 (Synthesizer)         
def synthesizer_node(state: AgentState):
    print("🧠[Synthesizer 模型 (deepseek-reasoner)] 正在阅读所有资料，进行深度总结...")
    
    # 1. 提取用户的原始问题
    original_question = state["messages"][0].content
    
    # 2. 从消息历史中提取所有工具搜回来的干货文本
    context_str = ""
    for msg in state["messages"]:
        if msg.type == "tool":
            context_str += f"\n{msg.content}\n"
            
    if not context_str:
        context_str = "未调用任何外部工具，请凭借你的内部知识库进行解答。"

    # 3. 构造一个绝对纯净的 Prompt，不带任何工具调用的历史包袱
    prompt = f"""你是多模态AI领域的技术大牛。
请根据以下收集到的参考资料，为用户撰写一份专业、全面、且带有学术深度的解答。
必须指出最新进展，并对未来的趋势做简要点评。
注意：你现在处于最终总结阶段，绝对不要尝试调用任何工具，不要输出任何 XML 或标签代码！请直接输出 Markdown 格式的回答。

【用户提问】: {original_question}

【参考资料】:
{context_str}
"""
    
    # 4. 只传一个简单的 HumanMessage 给 reasoner，避免它因为复杂的角色历史而精神分裂
    response = llm_synthesizer.invoke([HumanMessage(content=prompt)])
    
    # 注意：DeepSeek-Reasoner (R1) 的思维过程通常藏在额外字段里
    # 我们可以尝试把它深度的“思考过程”也打印出来（可选）
    if hasattr(response, "additional_kwargs") and "reasoning_content" in response.additional_kwargs:
        print("\n🤔[大牛的内心思考 (CoT)]:")
        print(response.additional_kwargs["reasoning_content"])
        
    return {"messages": [response]}



# ==========================================
# 3. 编排并编译图 (Graph)
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("tools", tool_node)
workflow.add_node("synthesizer", synthesizer_node)

# 定义边和流程
workflow.add_edge(START, "router")
# tools_condition 会自动判断：如果 router 决定调用工具，就去 "tools"；如果不调用工具，就说明它可以直接回答，直接去 "synthesizer"
workflow.add_conditional_edges("router", tools_condition, {"tools": "tools", END: "synthesizer"})
workflow.add_edge("tools", "synthesizer")  # 工具执行完后，把结果交给大模型总结
workflow.add_edge("synthesizer", END)      # 总结完后结束

# 编译图
app = workflow.compile()

# ==========================================
# 4. 运行 Agent 测试
# ==========================================
if __name__ == "__main__":
    question = "多模态大模型(VLM)最近有什么突破性的最新进展？如果本地论文不够新，请去网上找最新的报道。"
    print(f"🧑‍💻 用户提问: {question}\n" + "-"*50)
    
    # 运行流式输出，观察 Agent 的每一步
    for output in app.stream({"messages": [HumanMessage(content=question)]}, stream_mode="updates"):
        for node_name, node_state in output.items():
            if node_name == "router":
                msg = node_state["messages"][0]
                if msg.tool_calls:
                    print(f"   -> 决定调用工具: {[t['name'] for t in msg.tool_calls]}")
            elif node_name == "tools":
                print("   -> 工具执行完毕，获取到参考资料。")
            elif node_name == "synthesizer":
                final_answer = node_state["messages"][0].content
                print("\n✨ 【最终解答】:\n")
                print(final_answer)
