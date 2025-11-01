import os
import time
import json
import re
from dotenv import load_dotenv
from google import genai
from PIL import Image
from typing import List, Dict, Optional, Any

import psycopg2 
from psycopg2 import sql

from sentence_transformers import SentenceTransformer

# --- 1. 環境設定與配置 ---
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", 5432)
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

VLM_MODEL = "gemini-2.5-flash"  
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DOCS_DIR = "木星五式說明書"

if not (GEMINI_API_KEY and DB_HOST and DB_NAME and DB_USER and DB_PASSWORD):
    raise ValueError("請檢查.env，有變數沒有找到。")

def generate_step_description(client: genai.Client, image_path: str, page_number: int) -> Optional[Dict]:
    """
    使用Gemini VLM分析單張說明書圖片，生成結構化的組裝描述。
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"找不到圖片文件: {image_path}")
        return None
        
    prompt = f"""
        請擔任模型組裝專家，分析這張說明書圖片（第 {page_number} 頁）。
        你的首要任務是**絕對完整地**提取圖片中**所有可見的零件編號**，即使它們只是作為背景存在。

        任務要求：
        1. 輸出**僅為** JSON 格式。
        2. 確保 'parts_list' 欄位**包含所有**可見零件編號（例如：A11, E13, C11）。
        3. 描述組裝關係時，請以 'parts_list' 中的零件為準。
        4. 請注意!!說明書中版件編號可能為單個字母，也有可能為單字母加一位數字，
            例如B1，黑色圓底白字才是零件編號，綜合起來就像這樣 B1-18，如遇到該種情況，請用'-'隔開板件編號與零件編號

        JSON 格式要求：
        {{
            "page": {page_number},
            "parts_list": ["A11", "E13", "C11", "B1-18"], 
            "summary": "將零件 C11 插入 A11 的槽中，E13 作為裝甲覆蓋。",
            "detail": "[請根據圖片仔細描述組裝步驟]"
        }}
    """
    
    print(f"-> 正在分析頁面 {page_number}，圖片: {os.path.basename(image_path)}")
    
    try:
        response = client.models.generate_content(
            model=VLM_MODEL,
            contents=[prompt, img] ,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json", 
            )
        )
        
        import json
        data = json.loads(response.text)
        
        part_list_str = ", ".join(data.get("parts_list", [])) 
        summary = str(data.get("summary", ""))
        detail = str(data.get("detail", ""))
        
        rag_data = {
            "page_number": data.get("page", page_number),
            "part_names": part_list_str, 
            "description": summary + " | " + detail, 
            "image_source": image_path
        }
        
        return rag_data
        
    except Exception as e:
        print(f"VLM分析或JSON解析失敗 (頁面 {page_number})。錯誤: {e}")
        print(f"原始VLM輸出: {response.text[:100]}...")
        return None

def natural_keys(text: str) -> List[Any]:
    """將文件名分割為數字和非數字部分，以進行數字排序(page1, page2, page10)。"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def insert_data_to_pgsql(data_list: List[Dict[str, Any]], embed_dim: int):
    """連接 PostgreSQL，並將帶有向量的數據寫入assembly_steps表格。"""

    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST, 
            port=DB_PORT, 
            database=DB_NAME,
            user=DB_USER, 
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        
        print(f"\n--- 成功連接到 PostgreSQL 資料庫: {DB_NAME} ---")

        # SQL 插入模板
        insert_query = sql.SQL("""
            INSERT INTO public.assembly_steps (page_number, part_names, description, embedding, image_source)
            VALUES (%s, %s, %s, %s, %s);
        """)

        total_inserted = 0
        for item in data_list:
            # pgvector 擴展要求向量以字串形式 [x, y, z] 傳遞
            vector_str = "[" + ",".join(map(str, item['embedding'])) + "]"
            
            cur.execute(insert_query, (
                item['page_number'],
                item['part_names'],
                item['description'],
                vector_str,
                item['image_source']
            ))
            total_inserted += 1

        conn.commit()
        print(f"總共{total_inserted}條數據插入assembly_steps表格(維度: {embed_dim})。")

    except Exception as e:
        print(f"PostgreSQL 寫入失敗。請檢查DB權限和連接。{e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()


def build_multimodal_corpus():
    """
    掃描圖片 -> VLM 描述 -> 向量化 -> 寫入 pgvector。
    """
    if not os.path.isdir(DOCS_DIR):
        print(f"找不到說明書資料夾: {DOCS_DIR}。請先創建並放入圖片。")
        return

    try:
        vlm_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        print("無法初始化Gemini客戶端。請檢查GEMINI_API_KEY。")
        return
        
    print(f"-> 載入 Embedding 模型: {EMBEDDING_MODEL_NAME}...")
    try:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embed_dim = embed_model.get_sentence_embedding_dimension()
    except Exception as e:
        print(f"無法載入 Embedding 模型。錯誤: {e}")
        return

    image_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print(f"警告: {DOCS_DIR} 資料夾中沒有找到任何圖片。")
        return

    image_files_sorted = sorted(image_files, key=natural_keys)

    print(f"--- 找到 {len(image_files_sorted)} 張說明書圖片，開始 VLM 分析 (維度: {embed_dim}) ---")
    
    successful_rag_data = []

    for idx, filename in enumerate(image_files_sorted):
        full_path = os.path.join(DOCS_DIR, filename)
        page = idx + 1 
        
        data = generate_step_description(vlm_client, full_path, page)
        
        if data and data.get('description'):
            # 向量化
            embedding = embed_model.encode(data['description']).tolist()
            data['embedding'] = embedding
            successful_rag_data.append(data)
            
        time.sleep(1)

    print(f"\n--- VLM 分析和向量化完成 ---")
    print(f"總共成功處理 {len(successful_rag_data)} 個步驟。")

    if successful_rag_data:
        insert_data_to_pgsql(successful_rag_data, embed_dim)
    else:
        print("有成功生成數據，請檢查 VLM 輸出和 JSON 解析。")

if __name__ == "__main__":
    build_multimodal_corpus()