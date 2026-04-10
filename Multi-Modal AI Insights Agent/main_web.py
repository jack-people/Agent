# 这个文件是程序的主入口，负责启动 Streamlit Web UI，并将编译好的 Agent 图连接到前端界面。
#终端输入：streamlit run main.py来启动
import streamlit as st
from langchain_core.messages import HumanMessage

# 👉 这里导入你编译好的 Agent 图。
# 如果你的代码拆分了，就是 from graph import app as agent_app
# 如果你的代码还在一个文件里，就把 langgraph_agent_stream 换成你的文件名
from graph import app as agent_app 

# 1. 设置网页基础配置
st.set_page_config(page_title="多模态 AI 专家", page_icon="🤖", layout="wide")
st.title("🤖 多模态 AI Agent 检索系统")

# 2. 侧边栏：用于控制会话与长期记忆
with st.sidebar:
    st.header("⚙️ 会话设置")
    session_id = st.text_input("当前会话 ID (Thread ID)", value="session-web-001")
    st.caption("💡 提示：更改会话 ID 即可开启全新的无记忆对话；随时换回旧 ID，即可恢复历史长期记忆！")

config = {"configurable": {"thread_id": session_id}}

# 3. 初始化本地 UI 的聊天缓存
if "messages" not in st.session_state:
    st.session_state.messages =[]

# 4. 将缓存中的历史消息渲染到界面上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 如果有思考过程，用折叠面板展示
        if msg.get("reasoning"):
            with st.expander("🤔 查看大牛的内心思考过程"):
                st.markdown(msg["reasoning"])
        st.markdown(msg["content"])

# 5. 处理用户的输入并触发 Agent
if prompt := st.chat_input("请输入您的问题，例如：Gemma 4 在端侧有什么多模态进展？"):
    
    # 将用户的问题马上显示在页面上
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示一个正在运行的占位提示框
    with st.chat_message("assistant"):
        status_container = st.empty()
        status_container.info("🧠 Agent 已接收任务，正在规划...")
        
        final_answer = ""
        reasoning_text = ""
        
        try:
            # 运行 Agent，并实时捕获节点状态
            for output in agent_app.stream({"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode="updates"):
                for node_name, node_state in output.items():
                    # 👉 实时把 Agent 当前运行到哪个节点，反馈给前端用户！
                    if node_name == "router":
                        status_container.info("🔄 [Router]: 正在决定是回答还是去检索...")
                    elif node_name == "tools":
                        status_container.info("🌐 [Tools]: 正在调用工具检索外部资料...")
                    elif node_name == "critic":
                        status_container.warning("🕵️ [Critic]: 苛刻的审查员正在评估刚生成的答案...")
                    elif node_name == "synthesizer":
                        status_container.info("✍️ [Synthesizer]: DeepSeek-R1 正在进行深度推理撰写解答...")
                        
                        # 提取最终回答和思考过程
                        latest_msg = node_state["messages"][-1]
                        final_answer = latest_msg.content
                        reasoning_text = latest_msg.additional_kwargs.get("reasoning_content", "")
            
            # Agent 运行结束，清空提示框
            status_container.empty()
            
            # 渲染思考过程（如果有）
            if reasoning_text:
                with st.expander("🤔 查看大牛的内心思考过程"):
                    st.markdown(reasoning_text)
                    
            # 渲染正式回答
            st.markdown(final_answer)
            
            # 把 AI 的回答存入前端缓存
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer,
                "reasoning": reasoning_text
            })

        except Exception as e:
            st.error(f"Agent 运行出错: {e}")