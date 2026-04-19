# 工具库：存放本地检索、网络检索等 @tool 函数
import os
import json
import requests
import chromadb
from langchain_core.tools import tool
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from langchain_community.tools import DuckDuckGoSearchResults
from config import DIFY_APP_API_KEY, DIFY_WORKFLOW_URL,DIFY_DATASET_API_KEY,DIFY_RETRIEVAL_URL


# reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
# print("加载完成 Reranker 模型 (BAAI/bge-reranker-base)...")

# @tool
# def search_local_arxiv_db(query: str) -> str:
#     """
#     当你需要查询 Arxiv 多模态论文的本地知识库时调用此工具。
#     输入：搜索关键词。返回：相关论文的标题和摘要。
#     """

#      # 简化的检索逻辑
#     try:
#         client = chromadb.PersistentClient(path="./multimodal_papers_db")
#         emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
#         collection = client.get_collection(name="arxiv_multimodal", embedding_function=emb_fn)

#         # 👉 阶段 1：粗排召回 (Recall Top 20)
#         total_docs = collection.count()
#         print("【本地数据库】当前文档总数:", total_docs)
#         if total_docs == 0:
#             return "【本地数据库】当前为空，请先入库数据。"
            
#         recall_k = min(20, total_docs) 
#         results = collection.query(query_texts=[query], n_results=recall_k)
#         if not results['ids'][0]:
#             return "【本地数据库】未找到相关论文。"

#         # 准备数据对交给重排器
#         docs = results['documents'][0]
#         metadatas = results['metadatas'][0]
#         sentence_pairs = [[query, doc] for doc in docs]
        
#         # 👉 阶段 2：精排打分 (Rerank)
#         scores = reranker.compute_score(sentence_pairs)
        
#         # 将文档、元数据和得分组合在一起并排序
#         scored_docs = list(zip(docs, metadatas, scores))
#         # 按照 score 降序排列
#         scored_docs.sort(key=lambda x: x[2], reverse=True)
        
#         # 👉 阶段 3：截断 (Top 3)
#         top_3_docs = scored_docs[:3]    
        
#         res_str = "【本地 Arxiv 论文检索结果】\n"
#         for doc, meta, score in top_3_docs:
#             # 加上你原来的清洗逻辑，提取真正的摘要内容
#             clean_abstract = doc.split("Abstract: ")[-1][:300] 
            
#             res_str += f"-[相关度得分: {score:.2f}] 标题: {meta['title']}\n  摘要: {clean_abstract}...\n"
        
#         return res_str
#     except Exception as e:
#         return f"本地检索出错: {str(e)}"

@tool
def search_dify_arxiv_db(query: str) -> str:
    """
    用于在本地 Dify 多模态论文知识库中直接检索前沿学术资料的原始片段。
    当你需要回答关于多模态AI的技术细节、最新论文、模型架构等问题时，必须调用此工具。
    输入参数 query 应该是具体的学术关键词或研究方向。
    """
    print(f"\n🔍 [Tool] 正在通过工作流向 Dify 发起检索: '{query}'")
    
    headers = {
        "Authorization": f"Bearer {DIFY_APP_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 工作流的标准请求格式
    data = {
        "inputs": {
            "query": query  # 对应你在开始节点定义的那个变量名
        },
        "response_mode": "blocking", # 必须是 blocking，直接拿到最终结果
        "user": "vscode-agent-tool"
    }
    
    try:
        response = requests.post(
            DIFY_WORKFLOW_URL, 
            headers=headers, 
            json=data)
        
        if response.status_code != 200:
            error_msg = f"工作流 API 报错! 状态码: {response.status_code}, 详情: {response.text}"
            print(f"❌ [Tool] {error_msg}")
            return error_msg
            
        result = response.json()
        
        # 解析工作流的输出
        data_block = result.get("data", {})
        outputs = data_block.get("outputs", {})
        
        # 提取我们在结束节点定义的输出值（假设你定义的名字叫 text 或 result）
        # 遍历 outputs 字典拿到那串纯文本
        answer = ""
        for key, value in outputs.items():
             answer += str(value)
        
        if not answer or answer.strip() == "":
            print("⚠️ [Tool] 知识库检索完成，但未返回内容。")
            return "在知识库中未检索到相关内容。"
            
        print(f"✅ [Tool] Dify 检索成功！总字数: {len(answer)}。")
        return answer
        
    except Exception as e:
        error_msg = f"连接 Dify 发生网络异常: {e}"
        print(f"❌ [Tool] {error_msg}")
        return error_msg

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

# tools = [search_local_arxiv_db, search_web_news,search_dify_arxiv_db]
tools = [search_dify_arxiv_db, search_web_news]