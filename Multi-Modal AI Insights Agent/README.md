该项目是一个具备多模态学术论文检索（Arxiv）和相关网络新闻收集、且拥有特定语言（学术风格）的智能 Agent 系统， 架构为DeepSeek V和R系列（云） + Qwen2.5-7B-Instruct（本地） 的混合多模型（第一版）。

开发中遇到的问题及解决方法记录：
1 检索到的数据匹配度不高：在原先的混合检索（向量+关键词）基础上加入Reranker 模型（bge-reranker-base）来做重排序来提高相关性。
2 中文场景实现：在embedding函数里引入"BAAI/bge-small-zh-v1.5"模型
3 Deepseek R输出DSML代码：查询官方文档R系列模型需要加适配层，将XML转到标准格式来支持工具调用。最后用状态扁平化方法，通过对数据处理转为纯文本来解决问题。
4 模型输出结果重复并且非流式输出：在模型实例化里加入frequency_penalty和presence_penalty参数和在 Prompt 中加入硬性要求去解决重复问题；用stream() 代替 invoke()实现流式输出。


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

