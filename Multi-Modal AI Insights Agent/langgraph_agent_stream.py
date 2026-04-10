import os
import uuid
import chromadb
from typing import Annotated
from dotenv import load_dotenv
from datetime import datetime
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
#from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# 注意：MemorySaver 是 LangGraph 内置的一个简单的内存保存器，适合本地测试和小规模使用。
#from langgraph.checkpoint.memory import MemorySaver  
import sqlite3
# from langgraph.checkpoint.sqlite import SqliteSaver
# MemorySaver替换为 SqliteSaver，适合更大规模的持久化存储，并且支持查询和分析历史对话数据。
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()  # 从 .env 文件加载环境变量

# ==========================================
# 0. 配置你的 DeepSeek API Key
# ==========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  
BASE_URL = "https://api.deepseek.com"

# 实例化两个不同的模型！
# 模型 A：负责调用工具、决策、以及最终的 Critic 审查 (快且准)
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

# 加载 Re-ranker 模型（放在文件顶部，避免每次查询都加载）
# FlagReranker和transformers版本冲突改为CrossEncoder
# reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True) 
reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
print("加载完成 Reranker 模型 (BAAI/bge-reranker-base)...")

# ==========================================
# 1. 定义 Agent 的工具 (Tools)
# ==========================================
    
@tool
def search_local_arxiv_db(query: str) -> str:
    """
    当你需要查询 Arxiv 多模态论文的本地知识库时调用此工具。
    输入：搜索关键词。返回：相关论文的标题和摘要。
    """

     # 简化的检索逻辑
    try:
        client = chromadb.PersistentClient(path="./multimodal_papers_db")
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
        collection = client.get_collection(name="arxiv_multimodal", embedding_function=emb_fn)

        # 👉 阶段 1：粗排召回 (Recall Top 20)
        total_docs = collection.count()
        print("【本地数据库】当前文档总数:", total_docs)
        if total_docs == 0:
            return "【本地数据库】当前为空，请先入库数据。"
            
        recall_k = min(20, total_docs) 
        results = collection.query(query_texts=[query], n_results=recall_k)
        if not results['ids'][0]:
            return "【本地数据库】未找到相关论文。"

        # 准备数据对交给重排器
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        sentence_pairs = [[query, doc] for doc in docs]
        
        # 👉 阶段 2：精排打分 (Rerank)
        scores = reranker.compute_score(sentence_pairs)
        
        # 将文档、元数据和得分组合在一起并排序
        scored_docs = list(zip(docs, metadatas, scores))
        # 按照 score 降序排列
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        # 👉 阶段 3：截断 (Top 3)
        top_3_docs = scored_docs[:3]    
        
        res_str = "【本地 Arxiv 论文检索结果】\n"
        for doc, meta, score in top_3_docs:
            # 加上你原来的清洗逻辑，提取真正的摘要内容
            clean_abstract = doc.split("Abstract: ")[-1][:300] 
            
            res_str += f"-[相关度得分: {score:.2f}] 标题: {meta['title']}\n  摘要: {clean_abstract}...\n"
        
        return res_str
    except Exception as e:
        return f"本地检索出错: {str(e)}"

@tool
def search_web_news(query: str) -> str:
    """查询互联网最新资讯的工具"""
    print(f"\n   🌐 [工具调用] 正在去互联网搜索关键词: '{query}' ...")
    search = DuckDuckGoSearchResults(max_results=9)
    try:
        results = search.run(query)
        print(f" 🤖【互联网搜索】结果 : ✅ [工具调用] 搜索成功，获取到参考数据。")
        return f"【互联网搜索结果】\n{results}"
    except Exception as e:
        print(f" 🤖网络搜索出错: ❌ [工具调用] 搜索失败: {e}")
        return f"网络搜索出错: {str(e)}"

tools = [search_local_arxiv_db, search_web_news]
# 将工具绑定给路由器模型
llm_router_with_tools = llm_router.bind_tools(tools)


# ==========================================
# 2. 定义状态与数据结构
# ==========================================

# 定义图的状态：一个包含所有对话历史的消息列表 
# 👈 改进：给 State 增加一个 critic_decision 字段，用于决定图的走向
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    critic_decision: str  

# 👈 改进：定义 Critic 的结构化输出规范，强迫模型输出标准的评分和决策
class CriticOutput(BaseModel):
    score: int = Field(description="对回答的评分，0-10分。8分及以上为合格。")
    feedback: str = Field(description="如果不合格，请给出具体的改进建议或需要进一步搜索的关键词建议。如果合格，可以写'无'。")
    action: str = Field(description="如果不合格填 'REJECT'，如果合格填 'ACCEPT'")


# ==========================================
# 3. 定义节点逻辑 (Nodes)
# ==========================================

# 节点1：路由决策节点 (Router)
def router_node(state: AgentState):
    print("🤖 [Router] 正在思考调用什么工具 (或如何修正)...")
    sys_msg = SystemMessage(content="""你是多模态AI领域的智能助手。
    请根据用户的问题，决定是查询本地Arxiv论文库，还是去网上搜索。
    【致命指令】：
    1. 你的内置知识截止到 2024 年！因此对于任何最新发布、最新动态、或你不确定的知识，
        你【必须】调用 `search_web_news` 去网上搜索！绝对不允许凭空猜测或直接放弃！
    2. 如果在对话末尾看到了『Critic审查未通过』的反馈，说明你刚才查的关键词搜不到东西。
        请你务必【更换一个新的搜索关键词】再次调用工具搜索！""")
    
    messages =[sys_msg] + state["messages"]
    response = llm_router_with_tools.invoke(messages)
    return {"messages": [response]}

# 节点2：工具执行节点 (Tool Execution)
tool_node = ToolNode(tools)

def get_latest_question(messages: list) -> str:
    """辅助函数：从后往前找，提取用户最新的真实提问（跳过 Critic 的系统提示词）"""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human" and "【Critic 审查未通过】" not in msg.content:
            return msg.content
    return messages[0].content

# 节点3：最终总结节点 (Synthesizer)         
def synthesizer_node(state: AgentState):
    print("\n" + "="*50)
    print("🧠 [Synthesizer 模型 (deepseek-reasoner)] 开始深度思考...")
    print("="*50)
    
    #original_question = state["messages"][0].content
    # ✅ 改为动态获取最新一轮提问 （而不是一直第一条消息）
    latest_question = get_latest_question(state["messages"])

    # 从消息历史中提取工具返回的纯文本
    context_str = ""
    for msg in state["messages"]:
        if getattr(msg, "type", "") == "human" and "【Critic 审查未通过】" not in msg.content:
            context_str += f"\n[用户]: {msg.content}"
        elif getattr(msg, "type", "") == "ai" and msg.content:
            context_str += f"\n[AI助手]: {msg.content}"
        elif getattr(msg, "type", "") == "tool":
            context_str += f"\n[工具检索资料]: {msg.content}"

            
    if not context_str:
        context_str = "未调用任何外部工具，请凭借你的内部知识库进行解答。"

    prompt = f"""你是多模态AI领域的技术大牛。
                 请根据以下收集到的参考资料，为用户撰写一份专业、
                 全面、且带有学术深度的解答。必须指出最新进展，
                 并对未来的趋势做简要点评。

【严格警告】：
1. 请保持输出条理清晰，言简意赅。
2. 绝对不允许出现重复的段落或句子！如果你发现自己在重复上一条内容，
   请立即停止并进行下一条。

【最新追问】: {latest_question}
【对话历史与参考资料】: {context_str}
"""
    
    # 流式输出 
    response_chunks =[]
    is_thinking_started = False
    is_answer_started = False
    
    # 1. 使用 .stream() 代替 .invoke()
    for chunk in llm_synthesizer.stream([HumanMessage(content=prompt)]):
        
        # 提取 R1 的“内心独白 (思考过程)”
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        # 提取最终输出的内容
        content = chunk.content
        
        if reasoning:
            print(f"\033[90m{reasoning}\033[0m", end="", flush=True) # 灰色打印思考过程
        if content:
            print(content, end="", flush=True)
            
        # 收集所有的流式块
        response_chunks.append(chunk)
        
    # 2. 将流式的碎片拼接成一个完整的消息对象，返回给 LangGraph 状态机
    # LangChain 的 Chunk 支持直接用 + 号拼接
    final_message = response_chunks[0]
    for chunk in response_chunks[1:]:
        final_message += chunk
        
    print("\n" + "="*50) # 打印结束分割线
    return {"messages": [final_message]}


# 节点4：Critic 审查员
def critic_node(state: AgentState):
    print("\n🕵️ [Critic] 正在严格审查 Synthesizer 生成的答案...")
    
    # ✅ 同样需要动态获取最新提问，以便审查最新回答是否切题
    latest_question = get_latest_question(state["messages"])
    last_answer = state["messages"][-1].content
    
    # ✅ 新增：让 Critic 也能看到工具搜到了什么，防止它凭借老旧的内部知识瞎判！
    context_str = ""
    for msg in state["messages"]:
        if getattr(msg, "type", "") == "tool":
            context_str += f"\n[工具检索资料]: {msg.content}"
            
    if not context_str.strip():
        context_str = "暂无外部参考资料。"

    # 获取当前真实时间，校准大模型的时间钟
    current_date = datetime.now().strftime("%Y年%m月%d日")
    print
    # 1. 初始化 Pydantic 解析器
    parser = PydanticOutputParser(pydantic_object=CriticOutput)
    
    # 2. 将解析器的格式要求注入到提示词中
    prompt = f"""你是一个极其苛刻的AI内容审查专家。
当前系统时间是：{current_date}。
请评估以下生成的答案是否完美解答了用户【最新】的提问。

【核心审查标准】：
1. 🎯 主体精准对齐（最重要）：答案必须与用户询问的具体实体、版本号或事件完全一致！
如果用户问的是“最新版本（如某产品的第4代）”，但答案却用“旧版本（如第2代或第3代）”的信息来敷衍，
说明它没有搜到最新信息，必须打回（0分），让Router重新更换关键词搜索！
并且请【务必】对比下方的[参考资料]来判断是否产生幻觉！只要答案的内容在[参考资料]中有提及，就视为正确。
绝不能因为超出了你的内部陈旧知识，或者出现了未来的年份，就误判为虚构！
2. 💡 实质性解答：答案是否给出了用户想要的细节？如果回答“未查到相关信息”、
“目前没有任何关于XX的报道”、 “作为一个AI模型无法获取最新信息”，视为没有解决问题，必须打回重做（0分）。
3. ⏳ 宽容的时间/来源声明：只要答案中包含了实质性的新知识（确实准确回答了用户询问的主体），
允许模型在句首或句尾说明“根据网络搜索结果”或“截至XXXX年X月的报道”。这是严谨的表现，
绝对【不要】将其误判为AI的推脱之词。

用户最新提问: {latest_question}
工具收集到的参考资料: {context_str}
最新生成的答案: {last_answer}

【重要指令】
请严格按照以下JSON格式输出你的审查结果，不要包含任何多余的解释性文字：
{parser.get_format_instructions()}
请严格打分（0-10分），并给出审查决定。
"""
    # 强制模型输出 JSON 格式的结构化数据
    # structured_llm = llm_router.with_structured_output(CriticOutput)
    # evaluation: CriticOutput = structured_llm.invoke([HumanMessage(content=prompt)])
    
    # print(f"   -> 审查得分: {evaluation.score}/10")
    """上面的代码运行出现API 兼容性问题
    报错信息 This response_format type is unavailable now 是 DeepSeek API 抛出的。
    在较新的 LangChain 版本中，当你调用 .with_structured_output(CriticOutput) 时，LangChain 默认会使用 OpenAI 最新的 Structured Outputs (json_schema) 特性。但 DeepSeek 的 API 目前还不完全支持这种极度严格的 json_schema 格式（它只支持 json_object 或工具调用）。
    解决方案
    我们可以放弃使用 .with_structured_output，改为使用经典的 Prompt 格式指令 + Pydantic 解析器 + 开启 JSON 模式。这种方式具有极高的跨模型兼容性（无论是 DeepSeek 还是开源模型都能完美运行）。"""
    # 3. 使用标准调用，并强制开启 DeepSeek 支持的 json_object 模式
    llm_json = llm_router.bind(response_format={"type": "json_object"})
    response = llm_json.invoke([HumanMessage(content=prompt)])
    
    # 4. 解析结果
    try:
        # 将模型返回的 JSON 字符串解析为 CriticOutput 对象
        evaluation = parser.parse(response.content)
    except Exception as e:
        print(f"   -> ⚠️ 解析 Critic 输出失败，默认放行。错误: {e}")
        return {"critic_decision": "ACCEPT"}
    
    print(f"   -> 审查得分: {evaluation.score}/10")
    
    # 防死循环机制：如果在本次会话中 message 数量超过 15 条，说明一直在反思失败，强制通过
    if len(state["messages"]) > 15:
        print("   -> ⚠️ 警告：反思次数过多，为防止死循环，强制 ACCEPT！")
        return {"critic_decision": "ACCEPT"}

    if evaluation.score < 8 or evaluation.action == "REJECT":
        print(f"   -> ❌ 审查未通过，打回重做！反馈建议: {evaluation.feedback}")
        # 追加一条 HumanMessage 给 Router 看，逼迫 Router 纠正行为
        feedback_msg = HumanMessage(
            content=f"【Critic 审查未通过】由于以下原因，你的上一轮工作不及格：\n{evaluation.feedback}\n请立刻根据建议更换搜索策略重新查询！"
        )
        return {"messages": [feedback_msg], "critic_decision": "REJECT"}
    else:
        print("   -> ✅ 审查通过，准备输出给用户。")
        return {"critic_decision": "ACCEPT"}

# 判断 Critic 的决定
def critic_condition(state: AgentState) -> str:
    """根据 critic_decision 的值决定图的走向"""
    return state.get("critic_decision", "ACCEPT")


# ==========================================
# 4. 编排并编译图 (Graph)
# ==========================================

workflow = StateGraph(AgentState)
workflow.add_node("tools", tool_node)
workflow.add_node("critic", critic_node)
workflow.add_node("router", router_node)
workflow.add_node("synthesizer", synthesizer_node)

# 定义边和流程
workflow.add_edge(START, "router")
# tools_condition 会自动判断：如果 router 决定调用工具，就去 "tools"；如果不调用工具，就说明它可以直接回答，直接去 "synthesizer"
workflow.add_conditional_edges("router", tools_condition, {"tools": "tools", END: "synthesizer"})
workflow.add_edge("tools", "synthesizer")  # 工具执行完后，把结果交给大模型总结
workflow.add_edge("synthesizer", "critic")      # 总结完后交给审查员

# Critic 审查后的条件分发：ACCEPT 则结束，REJECT 则返回 router (形成闭环)
workflow.add_conditional_edges(
    "critic", 
    critic_condition, 
    {"ACCEPT": END, "REJECT": "router"}
)

# 👈 引入 MemorySaver 编译图
#memory = MemorySaver()  # 重新初始化清楚占用，MemorySaver是直接存储在本地存储并且不会自动删除

# 👈 核心改进：引入 SqliteSaver 实现真正的持久化记忆
# 连接到一个本地 SQLite 数据库文件（如果没有，它会自动创建）
db_path = "databank/agent_chat_history.db"
conn = sqlite3.connect(db_path, check_same_thread=False)

# 使用 SqliteSaver
memory = SqliteSaver(conn)


# 编译图，引入checkpointer 参数激活了 LangGraph 的长期记忆功能
app = workflow.compile(checkpointer=memory) 


if __name__ == "__main__":
    print("==================================================")
    print("🤖 欢迎使用多模态智能体 Agent！(输入 'exit' 或 'quit' 退出)")
    print("==================================================")
    
    # 1. 初始化会话 ID。你可以自己定义，或者让用户输入
    session_id = input("请输入您的会话名称 (直接回车则默认使用 'default-session'): ")
    if not session_id.strip():
        session_id = "default-session"
        
    config = {"configurable": {"thread_id": session_id}}
    print(f"✅ 已连接到会话: {session_id} (具备历史记忆)\n")

    # 2. 开启无限循环的聊天窗口
    while True:
        # 获取用户动态输入
        user_input = input("\n🧑‍💻 用户提问: ")
        
        # 如果用户输入 exit 或 quit，则退出程序
        if user_input.lower() in ['exit', 'quit']:
            print("👋 感谢使用，再见！")
            break
            
        # 防止用户直接按回车发送空消息
        if not user_input.strip():
            continue
            
        # 3. 将用户的输入送入 Agent 图中处理
        try:
            for output in app.stream({"messages":[HumanMessage(content=user_input)]}, config=config, stream_mode="updates"):
                # 因为在各个 Node (节点) 函数内部我们已经写了 print 逻辑（包括灰色思考过程等），
                # 所以这里用 pass 即可，控制台会自动打出完整的思考和回复流。
                pass 
                
        except Exception as e:
            print(f"\n❌ 运行时发生错误: {e}")
            print("系统已恢复，您可以继续提问。")


# ==========================================
# 6. 运行 Agent (多轮对话测试)
# ==========================================
# if __name__ == "__main__":
#     # 配置会话信息：同一个 thread_id 代表同一个聊天会话，这样模型就有了"记忆"
#     config = {"configurable": {"thread_id": "session-deepseek-1"}}
    
#     print("================ 第一轮对话 (测试环形纠错) ================")
#     # 我们故意提一个比较生僻或需要深挖的问题，观察 Critic 是否会打回
#     question_1 = "帮我查一下谷歌公司最新发布的关于gemma 4模型的细节，尤其是关于在端侧怎么支持多模态的，如果本地没有请去网上搜。"
#     print(f"🧑‍💻 用户提问: {question_1}\n")
    
#     for output in app.stream({"messages":[HumanMessage(content=question_1)]}, config=config, stream_mode="updates"):
#         pass # stream 中的具体打印已经在 Node 内部处理了

    
#     print("\n\n================ 第二轮对话 (测试长期记忆) ================")
#     # 这一轮我们不提"MM1"，也不提"多模态"，看看 Agent 是否记得第一轮的内容
#     question_2 = "根据你刚才找到的内容，它的参数量最大是多少？有哪些创新点？"
#     print(f"🧑‍💻 用户追问: {question_2}\n")
    
#     for output in app.stream({"messages": [HumanMessage(content=question_2)]}, config=config, stream_mode="updates"):
#         pass
            