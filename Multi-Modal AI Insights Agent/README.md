该项目是一个具备多模态学术论文检索（Arxiv）和相关网络新闻收集、且拥有特定语言（学术风格）的智能 Agent 系统， 架构为DeepSeek V和R系列（云） + Qwen2.5-7B-Instruct（本地） 的混合多模型（第一版）。

开发中遇到的问题及解决方法记录：

1 检索到的数据匹配度不高：在原先的混合检索（向量+关键词）基础上加入Reranker 模型（bge-reranker-base）来做重排序来提高相关性。

2 中文场景实现：在embedding函数里引入"BAAI/bge-small-zh-v1.5"模型

3 Deepseek R输出DSML代码：查询官方文档R系列模型需要加适配层，将XML转到标准格式来支持工具调用。最后用状态扁平化方法，通过对数据处理转为纯文本来解决问题。

4 模型输出结果重复并且非流式输出：在模型实例化里加入frequency_penalty和presence_penalty参数和在 Prompt 中加入硬性要求去解决重复问题；用stream() 代替 invoke()实现流式输出。


multimodal_dataget.py：使用 arxiv API 爬取最新的相关论文

test.py：ChromaDB原生的纯向量检索

test_rerank.py：对比ChromaDB原生的纯向量检索和混合召回 (向量+BM25) + Reranker的结果对比

langgraph_agent.py:主要文件，基于LangGraph架构加上deepseek v和r系列模型完整输出多模态相关结果

langgraph_agent_stream.py:和上面的文件基本相似。主要改进为流式输出，提高用户体验。

langgraph_agent_local_stream.py:在上面的文件的基础上，增加了本地模型选项，这里我在autodl上租了张5090微调了Qwen2.5-7B-Instruct（待改进，运行发现结果输出很慢）


第二版：

废弃langgraph_agent.py，主要代码放在langgraph_agent_stream.py。

并且第二版改进点如下所示：

1 引入“反思与自我纠错”机制 (Self-Reflection & Cyclic Graph) : 加入Critic（审查专家）节点在 Synthesizer 节点后。如果 Critic 发现 Synthesizer 生成的答案质量不高（比如产生幻觉、工具没查出东西），它有权把流程打回给 Router 重新更换关键词搜索。
核心技术点：将有向无环图（DAG）改为带环图（Cyclic Graph）。

2 加入“长期记忆” (Memory Checkpointer) : 引入 LangGraph 的 MemorySaver,显示对话

3 更新本地数据库里查找逻辑


第三版：

1 代码结构调整，模块化方便查看和拓展并且通过Streamlit实现Web UI

├── .env                  # 存放你的 DEEPSEEK_API_KEY 等环境变量

├── config.py             # 配置文件：专门负责初始化 LLM 模型实例

├── state.py              # 数据模型：存放 AgentState、CriticOutput 等类

├── tools.py              # 工具库：存放本地检索、网络检索等 @tool 函数

├── nodes.py              # 核心节点：存放 router_node、synthesizer_node 等

├── graph.py              # 图结构：专门负责拼装 StateGraph 和记忆组件

└── main_cil.py           # 程序入口：负责启动 CLI 终端聊天

└── main_web.py           # 程序入口：负责启动 Web UI 

2 引入 SqliteSaver替代MemorySaver存储对话记录,实现ChatBot

3 更新各节点里prompt，规范输出


第四版

引入Docker 容器本地部署Dify社区版，建立企业版知识库
记录该版本开发遇到的问题和思路：

1 知识库还是通过paper_markdown_get.py从arxiv里爬取最新的多模态论文；

  1.1 由于arxiv网站下载太慢，用vpn又会提示类似的(❌ 请求 arXiv API 失败: <urlopen error [WinError 10054] 远程主机强迫关闭了一个现有的连接。
      或'Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None))
      的错误，这里加入重试机制和把urllib.request.urlopen 彻底删掉，换成requests（这个没用！！！）。4月份网络代理有点崩这部分先这样
  
  1.2 step2_process_pdfs_local函数有问题未解决，先在MinerU官网用在线版本，每天额度5000够用

2 embed_server.py和rerank_server.py提供embedding（bge-m3），rerank（bge-reranker-v2-m3）模型本地服务
  注意：在dify里更新知识库时。需要先启动embed_server.py，rerank_server.py和docker容器（docker compose up -d）来正常服务

3 dify社区版知识库不能直接被本地环境（vscode）里调用（注意：这里测试不行但是没查官方文档不确定），
  需要在dify新建应用（workflow）并且绑定URL和API，具体见config.py。
  还有要改dify_main\docker\docker-compose.yaml文件里api: 下新加ports:  - "5001:5001"  # 将容器的5001端口映射到宿主机的5001，不然会报404
  
4 知识库分段用的是父子文档模式，混合检索加重排序提高准确率
