import gradio as gr
import yolo_scripts as ys 

theme = gr.themes.Base(primary_hue="violet")

with gr.Blocks(title="YOLO 零件識別訓練平台", theme=theme) as app:
    
    gr.Markdown("## 組裝模型 YOLO 訓練與零件查找管理平台")
    gr.Markdown("---")
    
    best_weights_state = gr.State(ys.DEFAULT_WEIGHTS_PATH) 

    with gr.Tabs():
        with gr.TabItem("零件查找與RAG應用"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 應用流程：上傳圖片 → 獲取組裝資訊")
                    test_image_input = gr.Image(type="filepath", label="* 上傳或拍照待偵測零件", sources=["upload", "webcam"]) 
                    predict_button = gr.Button("運行零件偵測與查找", variant="secondary")
                    
                                        
                with gr.Column(scale=1):
                    retrieved_gallery = gr.Gallery(label="檢索到的TopK說明書圖片", columns=3,
                                                   height=800
                                                   )
                
                with gr.Column(scale=1):
                    rag_output_text = gr.HTML(label="RAG組裝說明與頁碼", 
                                              value="<div id='rag_output_box' style='max-height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;'>等待模型輸出組裝指導...</div>")
                    
                    
                predict_button.click(
                    fn=ys.run_inference_and_rag,
                    inputs=[best_weights_state, test_image_input],
                    outputs=[retrieved_gallery, rag_output_text]
                )

    app.load(lambda: ys.DEFAULT_WEIGHTS_PATH, outputs=best_weights_state)

# 啟動應用程式
app.queue().launch(
    server_name="0.0.0.0",
    inbrowser=True,
    share=True,
    server_port=7860
)