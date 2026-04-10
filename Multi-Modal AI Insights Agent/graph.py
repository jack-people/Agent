# 图结构：专门负责拼装 StateGraph 和记忆组件
import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from state import AgentState
from tools import tools
from nodes import router_node, synthesizer_node, critic_node

def critic_condition(state: AgentState) -> str:
    return state.get("critic_decision", "ACCEPT")

def create_agent_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("tools", ToolNode(tools)) 
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("critic", critic_node)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", tools_condition, {"tools": "tools", END: "synthesizer"})
    workflow.add_edge("tools", "synthesizer")
    workflow.add_edge("synthesizer", "critic")
    workflow.add_conditional_edges("critic", critic_condition, {"ACCEPT": END, "REJECT": "router"})

    # 配置 SQLite 数据库
    db_path = "databank/agent_chat_history.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True) 
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)

app = create_agent_app()