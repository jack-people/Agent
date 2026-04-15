# import arxiv
# import chromadb
# from chromadb.utils import embedding_functions

# def fetch_arxiv_papers(query="all:multimodal AND all:LLM", max_results=100):
#     """
#     使用 arxiv API 爬取最新的相关论文
#     """
#     print(f"正在从 Arxiv 搜索关键词: {query}...")
    
#     # 构建搜索客户端，按提交日期倒序排序（获取最新）
#     client = arxiv.Client()
#     search = arxiv.Search(
#         query=query,
#         max_results=max_results,
#         sort_by=arxiv.SortCriterion.SubmittedDate,
#         sort_order=arxiv.SortOrder.Descending
#     )
    
#     results = client.results(search)
#     papers =[]
    
#     for paper in results:
#         papers.append({
#             "id": paper.get_short_id(),
#             "title": paper.title,
#             "abstract": paper.summary.replace("\n", " "), # 清理换行符
#             "authors": ", ".join([author.name for author in paper.authors]),
#             "published": str(paper.published),
#             "url": paper.entry_id
#         })
#         print(f"发现最新论文: {paper.title} ({paper.published})")
        
#     return papers

# def store_in_chromadb(papers, db_path="./multimodal_papers_db"):
#     """
#     将论文数据存入 Chroma 向量数据库
#     """
#     if not papers:
#         print("没有找到论文数据！")
#         return

#     print(f"\n正在初始化 ChromaDB，路径: {db_path}...")
#     # 使用 PersistentClient 将数据持久化到本地文件夹
#     chroma_client = chromadb.PersistentClient(path=db_path)
    
#     # 初始化 Embedding 模型。这里使用智源的 bge-small-zh-v1.5，支持中英文双语检索
#     print("正在加载 Embedding 模型 (首次运行可能需要下载)...")
#     emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="BAAI/bge-small-zh-v1.5"
#     )
    
#     # 创建或获取集合
#     collection = chroma_client.get_or_create_collection(
#         name="arxiv_multimodal",
#         embedding_function=emb_fn,
#         metadata={"hnsw:space": "cosine"} # 使用余弦相似度
#     )
    
#     documents =[]
#     metadatas = []
#     ids =[]
    
#     # 构造向量数据库所需的数据结构
#     for paper in papers:
#         # 将标题和摘要组合作为要向量化的核心文本
#         text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
#         documents.append(text)
        
#         # 将作者、发布时间、URL等作为元数据（Metadata）保存，方便后续过滤
#         metadatas.append({
#             "title": paper["title"],
#             "authors": paper["authors"],
#             "published": paper["published"],
#             "url": paper["url"]
#         })
        
#         ids.append(paper["id"])
    
#     # 存入数据库 (如果已有相同的ID会自动忽略或更新)
#     print("\n正在生成向量并存入数据库，请稍候...")
#     collection.add(
#         documents=documents,
#         metadatas=metadatas,
#         ids=ids
#     )
    
#     # 验证存入了多少条数据
#     print(f"入库成功！当前向量库共有 {collection.count()} 篇论文。")

# if __name__ == "__main__":
#     # 关键词设定：多模态 (multimodal/VLM) 和 大语言模型 (LLM)
#     search_query = 'all:"multimodal" OR all:"vision-language model" OR all:"VLM"'
    
#     # 1. 抓取最新 100 篇论文
#     latest_papers = fetch_arxiv_papers(query=search_query, max_results=100)
    
#     # 2. 存入 Chroma 数据库
#     store_in_chromadb(latest_papers)

import arxiv
import chromadb
from chromadb.utils import embedding_functions

def fetch_arxiv_papers(query="all:multimodal AND all:LLM", max_results=200):
    """
    使用 arxiv API 爬取最新的相关论文
    """
    print(f"正在从 Arxiv 搜索关键词: {query}...")
    print(f"目标获取数量: {max_results} 篇，正在请求中（请耐心等待，包含防反爬延迟）...")
    
    # 【关键修改点】：针对 200 篇的需求配置 Client
    # page_size=100：每次向服务器请求 100 篇。200 篇会自动分成 2 次请求。
    # delay_seconds=5.0：两次请求之间强制等待 5 秒，符合 arXiv 官方要求的 ≥3秒，完美避开 429 报错。
    # num_retries=5：如果遇到网络波动或限制，自动重试 5 次而不是直接报错崩溃。
    client = arxiv.Client(
        page_size=100,
        delay_seconds=5.0,
        num_retries=5
    )
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    # 这一步 generator 会在内部自动处理翻页和延迟
    results = client.results(search)
    papers =[]
    
    for paper in results:
        papers.append({
            "id": paper.get_short_id(),
            "title": paper.title,
            "abstract": paper.summary.replace("\n", " "), # 清理换行符
            "authors": ", ".join([author.name for author in paper.authors]),
            "published": str(paper.published),
            "url": paper.entry_id
        })
        print(f"[{len(papers)}/{max_results}] 发现最新论文: {paper.title} ({paper.published})")
        
    return papers

def store_in_chromadb(papers, db_path="./multimodal_papers_db"):
    """
    将论文数据存入 Chroma 向量数据库
    """
    if not papers:
        print("没有找到论文数据！")
        return

    print(f"\n正在初始化 ChromaDB，路径: {db_path}...")
    # 使用 PersistentClient 将数据持久化到本地文件夹
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # 初始化 Embedding 模型。这里使用智源的 bge-small-zh-v1.5，支持中英文双语检索
    print("正在加载 Embedding 模型 (首次运行可能需要下载)...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    
    # 创建或获取集合
    collection = chroma_client.get_or_create_collection(
        name="arxiv_multimodal",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"} # 使用余弦相似度
    )
    
    documents =[]
    metadatas = []
    ids =[]
    
    # 构造向量数据库所需的数据结构
    for paper in papers:
        # 将标题和摘要组合作为要向量化的核心文本
        text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
        documents.append(text)
        
        # 将作者、发布时间、URL等作为元数据（Metadata）保存，方便后续过滤
        metadatas.append({
            "title": paper["title"],
            "authors": paper["authors"],
            "published": paper["published"],
            "url": paper["url"]
        })
        
        ids.append(paper["id"])
    
    # 存入数据库 (如果已有相同的ID会自动忽略或更新)
    print(f"\n正在将 {len(papers)} 篇论文的向量存入数据库，请稍候...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    # 验证存入了多少条数据
    print(f"入库成功！当前向量库共有 {collection.count()} 篇论文。")

if __name__ == "__main__":
    # 关键词设定：多模态 (multimodal/VLM) 和 大语言模型 (LLM)
    search_query = 'all:"multimodal" OR all:"vision-language model" OR all:"VLM"'
    
    # 1. 抓取最新 200 篇论文
    latest_papers = fetch_arxiv_papers(query=search_query, max_results=200)
    
    # 2. 存入 Chroma 数据库
    store_in_chromadb(latest_papers)

