<p align="center">
  <img src="assets/GunplaScan.gif" alt="GunplaScan Demo" style="max-width: 90%; height: auto;">
</p>

# GunplaScan

> 剪了一堆零件下來，卻找不到零件編號多少，忘記在哪一個組裝步驟嗎?GunplaScan 透過 YOLO 物件偵測與 Gemini VLM，查找零件編號與得到組裝說明書和指南。
**⚠️ 免責聲明：本軟體僅供個人學習與研究用途，使用者需自負版權與安全責任。**

## ✨ 核心功能 (Features)
- 實時零件識別： 使用手機鏡頭或上傳圖片，精確鎖定已剪下的零件（例如 A11, E13）。
- 多模態知識庫： 利用 Gemini VLM 分析圖像式說明書，將視覺組裝步驟轉化為可搜索的向量。
- AI組裝指導(RAG)： 根據偵測結果，自動查詢 pgVector 資料庫並生成組裝順序建議和頁碼引用。

## 🛠️ 技術棧 (Tech Stack)
| 類別 | 技術/庫 | 作用 |
|  ----  | ---- | ---- |
| 影像辨識 | Ultralytics YOLOv11 | 實時物件檢測和邊界框定位。 |
| 大型模型 | Google Gemini (VLM/LLM) | 圖像內容分析（多模態 RAG）和檢索生成。 |
| 向量資料庫 | PostgreSQL + pgVector | 儲存和搜索 VLM 輸出的語義向量。 |
| 環境/UI | Gradio, Python (Miniforge) | 創建 Web 應用界面和環境管理。 |

## ⚙️ 安裝與環境準備 (Getting Started)
複製倉庫：
```bash
git clone [https://github.com/Rueicc1144/GunplaScan.git](https://github.com/Rueicc1144/GunplaScan.git)
```
創建環境：
```bash
conda create -n gunpla_env python=3.10
```
啟動環境：
```bash
conda activate gunpla_env
```
安裝依賴：
```bash
pip install -r requirements.txt
```

## 🔑 配置與資料庫設置 (Configuration & Database Setup)
1. API KEY： 創建 `.env` 文件，並填入 `GEMINI_API_KEY` 和所有 `DB_HOST`, `DB_USER` 等資料庫連接資訊。
2. pgVector 啟用： 在 PostgreSQL 中執行 `CREATE EXTENSION vector;` 並確保應用程式使用者擁有 `SELECT`, `INSERT`, `TRUNCATE` 權限。
3. 知識庫建立 (RAG Corpus)：
  - 數據準備： 將說明書圖片放入 `./docs` 資料夾。
  - 填充資料庫： 運行資料構建腳本：
    ```bash
    python rag_data_builder.py
    ```
    
## 🧠 模型權重準備
- YOLO 訓練：該專案以`BANDAI HG JUPITIVE GUNDAM` 中部份零件 ['A11', 'A12', 'A20', 'C11', 'E1', 'E13', 'E3', 'E6', 'E9', 'F1-4'] 進行訓練。您可自行訓練其他各種類型的模型零件，創建屬於自己的模型玩具辨識庫。
- 預設模型： `yolo_scripts.py` 中的 `DEFAULT_WEIGHTS_PATH` 預設為該專案中的 `jupitive_gundam.pt`。若有需要，可變更為您訓練的模型。

## ▶️ 應用程式啟動 (Usage)
- 本地啟動： `python app.py`
- 手機測試： 開啟 `share=True`（或使用區域 IP），並在手機瀏覽器中訪問 HTTPS連結。

## 📝 授權與貢獻 (License & Contribution)
- License： 本專案使用 [MIT License]。詳情請見 LICENSE 文件。
- 貢獻： 歡迎提交 Pull Requests 或 Issue。
