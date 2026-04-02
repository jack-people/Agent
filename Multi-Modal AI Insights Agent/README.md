项目名称：多模态哨兵（帮我获取多模态相关的最新信息）

目前第一版：
    架构为LangGraph
    模型用的是deepseek V和R系列加上自己微调的Qwen2.5-7B-Instruct

multimodal_dataget.py：使用 arxiv API 爬取最新的相关论文

test.py：ChromaDB原生的纯向量检索

langgraph_agent.py:主要文件，基于LangGraph架构加上deepseek v和r系列模型完整输出多模态相关结果

langgraph_agent_stream.py:和上面的文件基本相似。主要改进为流式输出，提高用户体验。

langgraph_agent_local_stream.py:在上面的文件的基础上，增加了本地模型选项，这里我在autodl上租了张5090微调了Qwen2.5-7B-Instruct（待改进，运行发现结果输出很慢）

test_rerank.py：对比ChromaDB原生的纯向量检索和混合召回 (向量+BM25) + Reranker的结果对比，如下所示：
【模式 A: 纯向量检索 (ChromaDB 原生)】
Top 1 (距离: 0.5402) -> VideoWeaver: Multimodal Multi-View Video-to-Video Transfer for Embodied Agents
Top 2 (距离: 0.5593) -> From Manipulation to Mistrust: Explaining Diverse Micro-Video Misinformation for Robust Debunking in the Wild
Top 3 (距离: 0.5759) -> Towards Comprehensive Real-Time Scene Understanding in Ophthalmic Surgery through Multimodal Image Fusion
----------------------------------------------------------------------
【模式 B: 混合召回 (向量+BM25) + Reranker 精排】
Top 1 (Rerank打分: 0.9733) -> VideoWeaver: Multimodal Multi-View Video-to-Video Transfer for Embodied Agents
Top 2 (Rerank打分: 0.9012) -> LanteRn: Latent Visual Structured Reasoning
Top 3 (Rerank打分: 0.8284) -> Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification
========================================================================
