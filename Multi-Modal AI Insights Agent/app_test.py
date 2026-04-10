#该文件调用langgraph_agent_stream
import streamlit as st
from langgraph_agent_stream import app as agent_app  # 导入你的 Agent
import time

# --- 1. 页面配置 ---
st.set_page_config(page_title="Arxiv 多模态论文助手", page_icon="📚")
st.title("🤖 多模态论文科研助手")
st.markdown("欢迎使用！我可以帮你检索本地 Arxiv 数据库并回答专业问题。")

# --- 2. 初始化聊天历史 ---
# Streamlit 每次操作都会重跑代码，session_state 用于持久化保存对话
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的论文助手，请问你想了解关于多模态的什么研究？"}
    ]

# --- 3. 在界面上渲染历史消息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. 聊天输入框 ---
if prompt := st.chat_input("请输入您的问题..."):
    # 用户输入展示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent 生成回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # 创建一个占位符用于显示动态效果
        full_response = ""
        
        with st.spinner("正在思考并检索论文..."):
            try:
                # 调用你的 LangGraph Agent
                # 这里的输入结构取决于你的 Agent 定义，通常是 {"messages": [...]}
                inputs = {"messages": [("user", prompt)]}
                result = agent_app.invoke(inputs)
                
                # 获取 Agent 的最后一条回答
                # 假设 result['messages'] 列表的最后一项是回答
                full_response = result["messages"][-1].content
                
            except Exception as e:
                full_response = f"抱歉，我遇到了一个错误：{str(e)}"

        # 模拟打字机效果（可选，增加体验感）
        displayed_msg = ""
        for char in full_response:
            displayed_msg += char
            message_placeholder.markdown(displayed_msg + "▌")
            time.sleep(0.01)
        
        message_placeholder.markdown(full_response)
    
    # 将回答保存到历史记录
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    