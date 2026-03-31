import os
import json
import chromadb
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载环境变量

# ==========================================
# 1. 初始化 DeepSeek 客户端
# ==========================================
# 确保你在终端执行过 export DEEPSEEK_API_KEY="sk-..."
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("⚠️ 未找到 DEEPSEEK_API_KEY 环境变量！")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ==========================================
# 2. 从 Chroma 数据库加载我们在第一步爬取的论文
# ==========================================
print("📂 正在连接本地 Chroma 数据库提取论文...")
try:
    chroma_client = chromadb.PersistentClient(path="./multimodal_papers_db")
    collection = chroma_client.get_collection(name="arxiv_multimodal")
    all_data = collection.get()
except Exception as e:
    print(f"❌ 数据库读取失败，请确保路径正确且第一步已成功运行。报错: {e}")
    exit()

documents = all_data['documents']
metadatas = all_data['metadatas']
total_papers = len(documents)
print(f"✅ 成功提取到 {total_papers} 篇多模态论文！")

# ==========================================
# 3. 定义 DeepSeek 的数据生成 Prompt
# ==========================================
system_prompt = """你是一个专门用于构建高质量指令微调（SFT）数据集的 AI 助手。
你的任务是：阅读我提供的多模态AI论文标题和摘要，生成一条符合要求的 JSON 格式数据。

【角色设定】：
你需要让最终的 `output` 读起来像是一个资深的“多模态AI技术大牛”。
大牛的说话风格：
1. 极客、犀利、一针见血，直接指出论文的核心本质（是创新还是水文）。
2. 喜欢夹杂英文专业术语和行业黑话（例如：Scaling Law, Latent Space, End-to-end, Make sense, Solid, 伪需求, 范式转移）。
3. 语气带有一点傲气和前瞻性。

【输出格式要求】：
请严格输出 JSON 对象，不要带其他多余文本，包含以下三个字段：
- "instruction": 模拟一个新人向大牛请教的提问（例如："大牛，帮我看看这篇做视觉推理的论文有意思吗？"、"怎么评价这篇关于VLM的最新工作？"）。
- "input": 将我提供的论文 Title 和 Abstract 原样整理并填入。
- "output": 用大牛的口吻，结合摘要内容，给出一段 200-300 字的深度专业点评。
"""

# ==========================================
# 4. 开始批量生成数据
# ==========================================
print("🚀 开始调用 DeepSeek 批量生成『大牛风格』微调数据...")

dataset =[]
# 如果你想多跑几遍扩充数据量，可以让这里的循环多跑几次（DeepSeek 每次生成的提问和点评都不一样）
# 这里我们遍历所有抓取到的论文：
for i in tqdm(range(total_papers), desc="生成进度"):
    title = metadatas[i]['title']
    # 从 document 中剥离出 Abstract 部分
    abstract = documents[i].split("Abstract: ")[-1]
    
    paper_info = f"Title: {title}\nAbstract: {abstract}"
    
    try:
        # 调用 DeepSeek-Chat，强制要求输出 JSON 格式
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请为以下论文生成微调数据：\n{paper_info}"}
            ],
            response_format={"type": "json_object"}, # 这是 DeepSeek 支持的杀手级功能，保证输出必是 JSON
            temperature=0.8 # 稍微高一点，让大牛的语气更多样化
        )
        
        # 解析返回的 JSON 字符串
        result_str = response.choices[0].message.content
        data_item = json.loads(result_str)
        dataset.append(data_item)
        
    except Exception as e:
        print(f"\n⚠️ 第 {i+1} 篇论文处理出错，跳过。错误信息: {e}")

# ==========================================
# 5. 保存数据集到本地
# ==========================================
output_file = "tech_guru_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    # 格式化写入 JSON，确保中文正常显示
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\n🎉 完美！成功生成 {len(dataset)} 条高质量微调数据，已保存至 {output_file}。")
