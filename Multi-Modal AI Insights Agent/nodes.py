# 核心节点：存放 router_node、synthesizer_node 等
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from datetime import datetime

# 从你自己写的其他文件里导入需要的东西！
from config import llm_router, llm_synthesizer
from state import AgentState, CriticOutput
from tools import tools

# 绑定工具
llm_router_with_tools = llm_router.bind_tools(tools)

def get_latest_question(messages: list) -> str:
    """辅助函数：从后往前找，提取用户最新的真实提问（跳过 Critic 的系统提示词）"""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human" and "【Critic 审查未通过】" not in msg.content:
            return msg.content
    return messages[0].content

# 路由决策节点 (Router)
def router_node(state: AgentState):
    print("🤖 [Router] 正在思考调用什么工具 (或如何修正)...")
    sys_msg = SystemMessage(content="""你是多模态AI领域的智能助手。
请根据用户的问题，决定是查询本地Arxiv论文库，还是去网上搜索。
【搜索决策准则】：
1. **时效性判定**：涉及最新发布、未来计划或你不确定的专有名词，必须调用 `search_web_news`search_web_news 去网上搜索！绝对不允许凭空猜测或直接放弃！。
2. **容错搜索**：如果用户提问中疑似包含拼写错误或不存在的型号，请在搜索时尝试使用你认为正确的术语，或搜索该系列产品的“最新版本”或“发布路线图”。
3. **失败演进**：如果收到审查未通过的反馈，严禁重复之前的搜索词。请尝试：
   - 扩大关键词范围（从具体型号扩大到系列品牌）。
   - 改变搜索维度（从“参数亮点”转向“官方发布新闻”或“技术预测”）。
   - 拆分复杂问题。 """)
    
    messages =[sys_msg] + state["messages"]
    response = llm_router_with_tools.invoke(messages)
    return {"messages": [response]}

# 最终总结节点 (Synthesizer)         
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

【撰写原则】：
1. **事实优先**：如果搜索资料显示用户询问的特定版本、型号或事件尚未发生或不存在，请务必礼貌地指出。不要为了迎合用户而虚构。
2. **关联引导**：若用户询问的实体不存在，请提供最接近的已知现状或官方最新动态作为替代参考。
3. **深度点评**：在回答现状的基础上，结合多模态领域的技术趋势进行简要点评。
4. **拒绝幻觉**：资料中未提及的信息，请明确表示“暂无公开信息”。
5. **拒绝重复**：绝对不允许出现重复的段落或句子！如果你发现自己在重复上一条内容，请立即停止并进行下一条。

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

# Critic 审查节点
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
【核心审查准则】：
1. **纠错合规性**：如果用户询问的实体在现实中不存在，而 AI 正确指出了该错误并提供了最相关的准确信息，这被视为“精准对齐”，应予以 ACCEPT。
2. **拒绝敷衍**：如果回答包含“未查到”、“作为一个AI无法获取最新信息”等推托词，但在参考资料中其实存在相关线索，必须 REJECT。
3. **禁止推演**：如果 AI 在资料不足的情况下，将“猜测”当作“事实”来回答用户询问的具体型号，必须 REJECT。
4. **信息来源**：答案是否体现了最新搜索资料中的实质性内容？

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
