# 工具库：存放本地检索、网络检索等 @tool 函数
import chromadb
from langchain_core.tools import tool
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from langchain_community.tools import DuckDuckGoSearchResults
# 如果加了 reranker，也在这里导入...

reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
print("加载完成 Reranker 模型 (BAAI/bge-reranker-base)...")

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
