import chromadb
from chromadb.utils import embedding_functions

# 1. 初始化客户端
chroma_client = chromadb.PersistentClient(path="./multimodal_papers_db")

# 2. 设置 Embedding 模型 (确保和存入时一致)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 3. 获取集合
collection = chroma_client.get_collection(name="arxiv_multimodal", embedding_function=emb_fn)

# 4. 执行中文检索
query_str = "多模态大模型最新进展"
n_results = 3
results = collection.query(
    query_texts=[query_str],
    n_results=n_results
)

# 5. 优化输出格式
print(f"\n" + "="*50)
print(f"🔍 检索词: '{query_str}'")
print(f"结果数量: {n_results}")
print("="*50 + "\n")

# 注意：Chroma 返回的结果是嵌套列表 [[]]，所以需要取索引 [0]
for i in range(len(results['ids'][0])):
    metadata = results['metadatas'][0][i]
    document = results['documents'][0][i]
    distance = results['distances'][0][i]
    
    # 计算一个感性的相似度分数 (距离越小越相似，这里做简单转换)
    score = max(0, 1 - distance) 

    print(f"【Top {i+1}】 相似度评分: {score:.4f}")
    print(f"📌 标题: {metadata['title']}")
    print(f"👨‍💻 作者: {metadata['authors']}")
    print(f"📅 发布时间: {metadata['published']}")
    print(f"🔗 链接: {metadata['url']}")
    
    # 提取摘要（去掉前面的 Title 部分，只保留摘要内容）
    abstract = document.split("Abstract: ")[-1]
    print(f"📝 摘要预览: {abstract[:200]}...") 
    
    print("-" * 50) # 分隔线