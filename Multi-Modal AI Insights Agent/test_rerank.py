import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import jieba

# 🌟 关键改变：不使用 FlagEmbedding，而是使用我们已经装好的 sentence_transformers 里的 CrossEncoder
from sentence_transformers import CrossEncoder

# 1. 初始化环境与模型
db_path = "./multimodal_papers_db"
chroma_client = chromadb.PersistentClient(path=db_path)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
collection = chroma_client.get_collection(name="arxiv_multimodal", embedding_function=emb_fn)

# 初始化 Reranker 模型 (使用 CrossEncoder 完美加载 BGE-reranker)
print("正在加载 Reranker 模型 (BAAI/bge-reranker-base)...")
reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)

# 2. 准备 BM25 索引
print("正在构建 BM25 关键词索引...")
all_data = collection.get()
documents = all_data['documents']
metadatas = all_data['metadatas']
ids = all_data['ids']

if not documents:
    print("错误：数据库是空的！请先运行第一步的代码抓取数据。")
    exit()

# 对文档进行简单的中文分词，提升检索精度
tokenized_corpus =[list(jieba.cut(doc)) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

def hybrid_rerank_search(query, top_k=3, recall_num=10):
    """混合检索 + 重排序 核心逻辑"""
    
    # 为了防止 recall_num 超过数据库总数，做一个限制
    actual_recall = min(recall_num, len(documents))

    # --- 阶段一：向量检索召回 (Dense Retrieval) ---
    vector_results = collection.query(query_texts=[query], n_results=actual_recall)
    vector_ids = vector_results['ids'][0]
    
    # --- 阶段二：BM25 关键词召回 (Sparse Retrieval) ---
    query_tokens = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(query_tokens)
    # 获取 BM25 分数最高的前 actual_recall 个索引
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:actual_recall]
    
    # --- 阶段三：合并去重 ---
    candidate_idx_map = {} # id -> doc_content
    
    # 存入向量检索结果
    for i in range(len(vector_ids)):
        candidate_idx_map[vector_ids[i]] = vector_results['documents'][0][i]
        
    # 存入 BM25 结果
    for idx in top_bm25_indices:
        doc_id = ids[idx]
        candidate_idx_map[doc_id] = documents[idx]
    
    candidates = list(candidate_idx_map.values())
    candidate_ids = list(candidate_idx_map.keys())
    
    # --- 阶段四：Rerank 重排序 ---
    # 构造形如 [[query, doc1], [query, doc2]] 的输入对
    pairs = [[query, doc] for doc in candidates]
    
    # 🌟 关键改变：使用 predict 方法代替原来的 compute_score
    rerank_scores = reranker.predict(pairs)
    
    # 按模型打分从高到低排序
    scored_results = sorted(zip(candidate_ids, candidates, rerank_scores), key=lambda x: x[2], reverse=True)
    
    return scored_results[:top_k]


if __name__ == "__main__":
    # 3. 开始对比实验
    query = "多模态大模型最新进展"

    print(f"\n" + "="*30 + " 实验对比 " + "="*30)
    print(f"🔍 查询词: '{query}'\n")

    # --- 测试 A: 仅向量检索 ---
    print("【模式 A: 纯向量检索 (ChromaDB 原生)】")
    v_res = collection.query(query_texts=[query], n_results=3)
    for i in range(len(v_res['ids'][0])):
        title = v_res['metadatas'][0][i]['title']
        score = v_res['distances'][0][i]
        print(f"Top {i+1} (距离: {score:.4f}) -> {title}")

    print("-" * 70)

    # --- 测试 B: 混合检索 + Rerank ---
    print("【模式 B: 混合召回 (向量+BM25) + Reranker 精排】")
    r_res = hybrid_rerank_search(query, top_k=3)
    for i, (doc_id, content, score) in enumerate(r_res):
        idx = ids.index(doc_id)
        title = metadatas[idx]['title']
        # Reranker 的分数通常没有固定范围，越大越相关，有些甚至会超过 1 或为负数
        print(f"Top {i+1} (Rerank打分: {score:.4f}) -> {title}")
    
    print("="*72)
