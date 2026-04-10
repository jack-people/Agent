# 程序入口：负责启动 CLI 终端聊天
# from langchain_core.messages import HumanMessage
# from graph import app # 直接导入编译好的大模型图

# if __name__ == "__main__":
#     print("==================================================")
#     print("🤖 欢迎使用多模态智能体 Agent！(输入 'exit' 退出)")
#     print("==================================================")
    
#     session_id = input("请输入您的会话名称 (默认 default): ") or "default"
#     config = {"configurable": {"thread_id": session_id}}
    
#     while True:
#         user_input = input("\n🧑‍💻 用户提问: \n")
#         if user_input.lower() in ['exit', 'quit']: break
#         if not user_input.strip(): continue
            
#         try:
#             for output in app.stream({"messages":[HumanMessage(content=user_input)]}, config=config, stream_mode="updates"):
#                 pass 
#         except Exception as e:
#             print(f"\n❌ 运行错误: {e}")

import os
import sqlite3
from langchain_core.messages import HumanMessage
from graph import app  # 导入编译好的图

def get_all_sessions(db_path="databank/agent_chat_history.db"):
    """从 SQLite 数据库中提取所有已有的 thread_id"""
    if not os.path.exists(db_path):
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # LangGraph 的 SqliteSaver 默认将数据存在 checkpoints 表中
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        sessions = [row[0] for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception as e:
        print(f"读取历史记录失败: {e}")
        return []

def select_session():
    """交互式选择会话 ID"""
    sessions = get_all_sessions()
    
    print("\n--- 历史会话列表 ---")
    if not sessions:
        print("（暂无历史记录）")
    else:
        for i, session in enumerate(sessions, 1):
            print(f"[{i}] {session}")
    print("[0] 新建会话")
    print("-------------------")
    
    choice = input("\n请选择会话编号 (直接输入名称则新建): ").strip()
    
    # 判断用户输入的是编号还是新名称
    if choice.isdigit():
        idx = int(choice)
        if idx == 0:
            new_id = input("请输入新会话的名称: ").strip() or "default"
            return new_id
        elif 1 <= idx <= len(sessions):
            selected_id = sessions[idx-1]
            print(f"✅ 已加载历史会话: {selected_id}")
            return selected_id
        else:
            print("⚠️ 无效编号，将使用默认名称。")
            return "default"
    else:
        # 如果输入的不是数字，直接作为新 ID
        return choice if choice else "default"

if __name__ == "__main__":
    print("==================================================")
    print("🤖 欢迎使用多模态智能体 Agent！(输入 'exit' 退出)")
    print("==================================================")
    
    # 1. 获取用户选择的会话 ID
    session_id = select_session()
    config = {"configurable": {"thread_id": session_id}}
    
    print(f"\n🚀 当前对话 ID: {session_id} (已进入聊天模式)")

    while True:
        user_input = input("\n🧑‍💻 用户提问: \n")
        if user_input.lower() in ['exit', 'quit']: 
            break
        if not user_input.strip(): 
            continue
            
        try:
            # 使用 stream_mode="updates" 实时观察节点变化
            for output in app.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config, 
                stream_mode="updates"
            ):
                # 打印节点名方便调试，正式使用时可以只打印最终回复
                for node_name, state in output.items():
                    print(f"[{node_name}] 处理完毕...")
                    # 如果你想在控制台直接看到回答，可以取消下面注释：
                    # if node_name == "synthesizer":
                    #    print(f"Assistant: {state['messages'][-1].content}")
                    
        except Exception as e:
            print(f"\n❌ 运行错误: {e}")
