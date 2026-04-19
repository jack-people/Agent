import os
import re
import json
import time
import shutil
import requests
import subprocess
import urllib.parse
import urllib.request
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
# 目录配置
RAW_PDF_DIR = Path("./databank/paper_data/01_raw_pdfs")
CLEAN_MD_DIR = Path("./databank/paper_data/02_cleaned_mds")
TEMP_MINERU_DIR = Path("./databank/paper_data/03_temp_mineru")
# arXiv 状态追踪文件（用于保证不重复下载）
TRACKER_FILE = Path("./databank/paper_data/arxiv_tracker.json")

# 每次获取的数量
FETCH_BATCH_SIZE = 50

# 多模态检索关键词 (涵盖 Multimodal, Vision-Language 等)
SEARCH_QUERY = 'all:multimodal OR all:"vision-language model" OR all:"VLM"'

# 初始化目录
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_MD_DIR.mkdir(parents=True, exist_ok=True)
# ==========================================

def load_tracker():
    """加载追踪记录，避免重复下载"""
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"downloaded_ids": [], "last_start_index": 0}

def save_tracker(tracker_data):
    """保存追踪记录"""
    with open(TRACKER_FILE, 'w', encoding='utf-8') as f:
        json.dump(tracker_data, f, indent=4)

def sanitize_filename(name):
    """清洗文件名，防止非法字符"""
    return re.sub(r'[\\/*?:"<>|\n\r]', "", name).strip()

# def step1_fetch_arxiv():
#     """第一步：从 arXiv 抓取 50 篇不重复的多模态论文"""
#     print("\n" + "="*50)
#     print("🚀 开始第一步：从 arXiv 抓取最新的多模态论文")
#     print("="*50)
    
#     tracker = load_tracker()
#     downloaded_ids = set(tracker["downloaded_ids"])
#     start_index = tracker["last_start_index"]
    
#     collected_papers = []
    
#     # 循环请求 API，直到我们收集够 50 篇全新的论文
#     print("正在搜索 arXiv API...")
#     while len(collected_papers) < FETCH_BATCH_SIZE:
#         query = urllib.parse.quote(SEARCH_QUERY)
#         # 每次向 API 请求 100 篇，按提交时间倒序（最新的在前）
#         url = f'http://export.arxiv.org/api/query?search_query={query}&start={start_index}&max_results=100&sortBy=submittedDate&sortOrder=descending'
        
#         try:
#             response = urllib.request.urlopen(url)
#             xml_data = response.read()
#             root = ET.fromstring(xml_data)
            
#             # 定义 XML 命名空间
#             ns = {'atom': 'http://www.w3.org/2005/Atom'}
#             entries = root.findall('atom:entry', ns)
            
#             if not entries:
#                 print("⚠️ 没有更多新论文了！")
#                 break
                
#             for entry in entries:
#                 if len(collected_papers) >= FETCH_BATCH_SIZE:
#                     break
                    
#                 # 提取论文 ID 和 标题
#                 paper_id_url = entry.find('atom:id', ns).text
#                 paper_id = paper_id_url.split('/abs/')[-1].split('v')[0] # 取出纯净ID，如 2401.0001
#                 title = entry.find('atom:title', ns).text.replace('\n', ' ')
#                 title = sanitize_filename(title)
                
#                 # 提取 PDF 下载链接
#                 pdf_url = ""
#                 for link in entry.findall('atom:link', ns):
#                     if link.attrib.get('title') == 'pdf':
#                         pdf_url = link.attrib.get('href')
#                         break
                        
#                 if paper_id not in downloaded_ids and pdf_url:
#                     collected_papers.append({
#                         "id": paper_id,
#                         "title": title,
#                         "pdf_url": pdf_url
#                     })
#                     downloaded_ids.add(paper_id)
            
#             start_index += 100 # 下次 API 请求偏移量
            
#         except Exception as e:
#             print(f"❌ 请求 arXiv API 失败: {e}")
#             break

#     # 开始下载收集到的论文
#     print(f"\n✅ 成功找到 {len(collected_papers)} 篇新论文，开始下载...")
#     for paper in tqdm(collected_papers, desc="下载 PDF"):
#         file_name = f"[{paper['id']}] {paper['title']}.pdf"
#         file_path = RAW_PDF_DIR / file_name
        
#         try:
#             urllib.request.urlretrieve(paper['pdf_url'], str(file_path))
#             # arXiv 限制：为了防止被封 IP，下载每篇后暂停 3 秒
#             time.sleep(3) 
#         except Exception as e:
#             print(f"\n⚠️ 下载失败 {paper['id']}: {e}")
            
#     # 更新追踪器
#     tracker["downloaded_ids"] = list(downloaded_ids)
#     tracker["last_start_index"] = start_index
#     save_tracker(tracker)
    
#     print(f"\n🎉 第一步完成！共下载 {len(collected_papers)} 篇论文。")
#     print(f"👉 请前往 {RAW_PDF_DIR} 文件夹进行人工检查和剔除！")

def step1_fetch_arxiv():
    """第一步：从 arXiv 抓取最新的多模态论文（带自动重试机制）"""
    print("\n" + "="*50)
    print("🚀 开始第一步：从 arXiv 抓取最新的多模态论文")
    print("="*50)
    
    tracker = load_tracker()
    downloaded_ids_set = set(tracker.get("downloaded_ids", []))
    start_index = tracker.get("last_start_index", 0)
    
    collected_papers = []
    
    print("正在搜索 arXiv API...")
    
    while len(collected_papers) < FETCH_BATCH_SIZE:
        query = urllib.parse.quote(SEARCH_QUERY)
        url = f'http://export.arxiv.org/api/query?search_query={query}&start={start_index}&max_results=100&sortBy=submittedDate&sortOrder=descending'
        
        try:
            # 这里用 urllib 请求 API 数据没问题，因为数据量很小1
            response = urllib.request.urlopen(url, timeout=20) 
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            
            if not entries:
                print("⚠️ 没有更多新论文了！")
                break
                
            for entry in entries:
                if len(collected_papers) >= FETCH_BATCH_SIZE:
                    break
                    
                paper_id_url = entry.find('atom:id', ns).text
                paper_id = paper_id_url.split('/abs/')[-1].split('v')[0]
                title = entry.find('atom:title', ns).text.replace('\n', ' ')
                title = sanitize_filename(title)
                
                pdf_url = ""
                for link in entry.findall('atom:link', ns):
                    if link.attrib.get('title') == 'pdf':
                        pdf_url = link.attrib.get('href')
                        break
                        
                if paper_id not in downloaded_ids_set and pdf_url:
                    collected_papers.append({
                        "id": paper_id,
                        "title": title,
                        "pdf_url": pdf_url
                    })
            
            start_index += 100 
            
        except Exception as e:
            print(f"❌ 请求 arXiv API 失败: {e}")
            break

    # ============ 核心升级：强健的下载模块 ============
    print(f"\n✅ 成功找到 {len(collected_papers)} 篇新论文，开始下载...")

    successful_downloads = []

    for paper in tqdm(collected_papers, desc="下载 PDF"):
        file_name = f"[{paper['id']}] {paper['title']}.pdf"
        file_path = RAW_PDF_DIR / file_name
        
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                # 使用 requests 进行流式下载，设置每次读取超时时间为 15 秒
                with requests.get(paper['pdf_url'], stream=True, timeout=15) as r:
                    r.raise_for_status()
                    with open(file_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                success = True
                break  # 下载成功，跳出重试循环
                
            except Exception as e:
                print(f"\n⚠️ 下载中断 {paper['id']} (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)  # 失败后休息 2 秒再重试
                
        if success:
            successful_downloads.append(paper['id'])
            time.sleep(3)  # 成功后暂停 3 秒，遵守 arXiv 防封禁规则
        else:
            print(f"\n❌ 彻底失败 {paper['id']}，已跳过。下次运行将重新尝试下载。")
            if file_path.exists():
                file_path.unlink()  # 删除下载了一半的残缺文件

    # ==================================================

    # 只有真正下载成功的，才写入追踪器！
    downloaded_ids_set.update(successful_downloads)
    tracker["downloaded_ids"] = list(downloaded_ids_set)
    
    # 只有当所有的论文都处理完（成功或彻底跳过），才更新偏移量
    tracker["last_start_index"] = start_index
    save_tracker(tracker)
    
    print(f"\n🎉 第一步完成！本次成功下载 {len(successful_downloads)} 篇论文。")
    print(f"👉 请前往 {RAW_PDF_DIR} 文件夹进行人工检查。")

def clean_markdown(input_md_path, output_md_path):
    """核心清洗逻辑：切除参考文献并去除多余空行"""
    try:
        with open(input_md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 匹配英文 References, Bibliography, 以及中文 参考文献
        pattern = re.compile(r'\n#+\s*(References|Bibliography|Reference|参考文献)\b.*', flags=re.IGNORECASE | re.DOTALL)
        
        match = pattern.search(content)
        if match:
            cleaned_content = content[:match.start()]
            status = "已切除参考文献"
        else:
            cleaned_content = content
            status = "未找到参考文献标识"
            
        cleaned_content = re.sub(r'\n{4,}', '\n\n', cleaned_content)

        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
            
        return True, status
    except Exception as e:
        return False, str(e)

#TODO:目前有问题，先在MinerU官网用在线版本
def step2_process_pdfs_local():
    """第二步：读取剩下的 PDF，调用 MinerU 解析并清洗"""
    print("\n" + "="*50)
    print("⚙️ 开始第二步：批量转换 PDF 并输出纯净 Markdown")
    print("="*50)
    
    pdf_files = list(RAW_PDF_DIR.glob('*.pdf'))
    if not pdf_files:
        print(f"⚠️ 在 {RAW_PDF_DIR} 中没有找到任何 PDF 文件！")
        return
        
    print(f"📁 扫描到 {len(pdf_files)} 篇人工过滤后的 PDF，准备调用 MinerU...\n")
    TEMP_MINERU_DIR.mkdir(exist_ok=True)

    for pdf_path in tqdm(pdf_files, desc="MinerU 解析中"):
        pdf_name = pdf_path.stem
        final_md_file = CLEAN_MD_DIR / f"{pdf_name}_clean.md"
        
        # 如果已经处理过了，跳过（支持断点续传）
        if final_md_file.exists():
            continue
            
        # 调用 MinerU

        try:
            subprocess.run([
                "magic-pdf pdf ", 
                "-p", str(pdf_path), 
                "-o", str(TEMP_MINERU_DIR), 
                "-m", "auto" 
            ], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [MinerU错误] {pdf_name}: {e.stderr}")
            continue

        # 查找生成的 MD 并清洗
        miner_output_folder = TEMP_MINERU_DIR / pdf_name
        md_files = list(miner_output_folder.glob('*.md'))
        
        if md_files:
            success, status = clean_markdown(md_files[0], final_md_file)
            if not success:
                print(f"\n❌ [清洗错误] {pdf_name}: {status}")
        else:
            print(f"\n⚠️ [警告] {pdf_name} MinerU 未能生成 Markdown。")

    # 清理临时文件
    shutil.rmtree(TEMP_MINERU_DIR, ignore_errors=True)
    print(f"\n✅ 全部解析完成！")
    print(f"👉 纯净版 Markdown 已保存在: {CLEAN_MD_DIR}")

def step2_process_pdfs_online():
    print(f"\n✅ 全部解析完成！")
    print(f"👉 纯净版 Markdown 已保存在: {CLEAN_MD_DIR}")

def main():
    while True:
        print("\n" + "*"*40)
        print("   多模态论文处理工作流 (Dify准备阶段)   ")
        print("*"*40)
        print("1. [第一步] 从 arXiv 抓取 50 篇最新论文 (输出至 01_raw_pdfs)")
        print("2. [第二步] 运行 MinerU 进行深度解析并清洗 (输出至 02_cleaned_mds)")
        print("3. 退出程序")
        print("*"*40)
        
        choice = input("请输入你的选择 (1/2/3): ").strip()
        
        if choice == '1':
            step1_fetch_arxiv()
        elif choice == '2':
            step2_process_pdfs()
        elif choice == '3':
            print("👋 退出程序。")
            break
        else:
            print("⚠️ 输入无效，请重新输入。")

if __name__ == "__main__":
    main()