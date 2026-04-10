# 定义状态与数据结构
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    critic_decision: str  

class CriticOutput(BaseModel):
    score: int = Field(description="对回答的评分，0-10分。8分及以上为合格。")
    feedback: str = Field(description="如果不合格，请给出具体的改进建议或需要进一步搜索的关键词建议。如果合格，可以写'无'。")
    action: str = Field(description="如果不合格填 'REJECT'，如果合格填 'ACCEPT'")