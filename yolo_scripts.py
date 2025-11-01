import sys
import os
import markdown
from typing import Tuple, Optional, List, Dict, Any

# --- 導入RAG相關庫 ---
import psycopg2
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
# 確保ultralytic資料夾路徑正確導入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_SOURCE_DIR = os.path.join(CURRENT_DIR, 'ultralytics-8.3.221')

GRADIO_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_gradio_output")
INFERENCE_NAME = "current_detection_pic"

sys.path.insert(0, YOLO_SOURCE_DIR)
from ultralytics import YOLO
DEFAULT_WEIGHTS_PATH = r"YOUR_WEIGHT_PATH" 

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

GEMINI_CLIENT = genai.Client(api_key = GEMINI_API_KEY)
LLM_MODEL = "gemini-2.5-flash"

DB_HOST = os.getenv("DB-HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
K_NEAREST = 5

def run_inference_and_rag(weights_path: str, image_path: str) -> Tuple[Optional[str], str]:
    """使用指定的weights_path運行YOLO偵測，並執行RAG查詢。"""

    if not weights_path or not os.path.exists(weights_path):
        weights_path = DEFAULT_WEIGHTS_PATH 
    
    if not os.path.exists(weights_path):
        return None, f"找不到模型權重！請先訓練或設定正確的DEFAULT_WEIGHTS_PATH。"
    
    if not image_path:
        return None, "錯誤：請上傳待偵測的零件圖片！"
    
    yolo = YOLO(model=weights_path, task='detect')
    results_list = yolo(
        source=image_path,
        save=True,                   
        project=GRADIO_OUTPUT_DIR,  
        name=INFERENCE_NAME,          
        exist_ok=True,               
        verbose=False
    )
    result = results_list[0]
    detected_filename = os.path.basename(image_path)
    output_image_path = os.path.join(result.save_dir, detected_filename)
    
    # RAG檢索
    class_ids = result.boxes.cls.tolist()
    part_names = [result.names[int(id)] for id in class_ids]
    
    context_data = retrieve_content(part_names) # return list of dic{"page","description","image"}
    rag_guidance = markdown.markdown(generate_guidence(part_names, context_data))
    rag_guidance_html = f"<div id='rag_output_box' style='max-height: 600px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;'>{rag_guidance}</div>"
    
    # 圖片路徑
    retrieved_image_paths = [item['image'] for item in context_data]
    retrieved_image_paths.insert(0, output_image_path)
    
    return retrieved_image_paths, rag_guidance_html


def retrieve_content(part_names: List[str])-> List[Dict[str,Any]]:
    """
    根據偵測到的零件名稱，查詢pgVector資料庫，獲取相關的組裝步驟。
    Args:
        part_names:YOLO偵測到的零件名稱列表 (e.g., ['A11', 'E13'])
    Returns:
        包含{page_number, description}的上下文列表。
    """
    part_names_str = ", ".join(part_names)
    query = f"請提供組裝說明書，需要使用零件{part_names_str}的步驟和說明。"
    
    # 向量化query
    query_vector = EMBED_MODEL.encode(query).tolist()
    query_vector_str = "[" + ", ".join(map(str, query_vector)) + "]"
    
    conn = None
    retrieved_data = []
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        from psycopg2 import sql
        
        sql_query = sql.SQL("""
                SELECT page_number, description, image_source 
                FROM public.assembly_steps
                ORDER BY embedding <-> %s::public.vector  
                LIMIT %s;
            """)
                
        cur.execute(sql_query, (query_vector_str, K_NEAREST))
        results = cur.fetchall()
        
        for page_num, desc, img_src in results:
            retrieved_data.append({"page":page_num, "description": desc, "image": img_src})
            
    except Exception as e:
        print(f"資料庫檢索失敗。{e}")
        
    finally:
        if conn:
            conn.close()
            
    return retrieved_data

def generate_guidence(part_names: List[str], context: List[Dict[str, Any]]):
    """
    使用Gemini LLM根據檢索到的上下文生成組裝說明。
    """
    if not context:
        return f"偵測到零件{', '.join(part_names)}，但找不到相關的組裝說明，請檢查您的說明書。"

    context = "\n".join([f"[第{item['page']}頁]:{item['description']}" for item in context])
    
    prompt = f"""
    你是一位專業的模型組裝助理。
    任務：根據以下偵測到的零件和提供的組裝說明書片段，為使用者提供**清晰、分步**的組裝指導。
    偵測到的零件: {', '.join(part_names)}
    組裝說明書片段 (Context):
    ---
    {context}
    ---
    請以自然語言回答，總結這些零件的組裝順序、關鍵連接點，並指出它們分別在哪些頁面。
    """
    
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=LLM_MODEL,
            contents=[prompt]
        )
        return response.text
    
    except Exception as e:
        return f"LLM回覆失敗。{e}"