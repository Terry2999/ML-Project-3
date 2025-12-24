import os
import json
import ollama
import chromadb
from neo4j import GraphDatabase
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import glob # for batch pdf processing

# ================= 設定區域 =================
# 請將這裡換成你的 Neo4j 密碼
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "12345678") 

# PDF 路徑 (請修改為你電腦上實際的 PDF 路徑)
PDF_PATH = "source-pdf/Razer Pro Click Mini - Master Guide -zh-TW.pdf"

PDF_FOLDER_PATH = "./source-pdf"

# ================= 1. 資料庫連線管理 =================
class GraphRAGBuilder:
    def __init__(self):
        # 連線 Neo4j
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        # 連線 ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./graphrag_db")
        self.vector_collection = self.chroma_client.get_or_create_collection("rag_collection")
        print(">>> 資料庫連線成功")

    def close(self):
        self.driver.close()

    # ================= 2. 核心：LLM 實體關係抽取 =================
    def extract_graph_from_text(self, text):
        """
        利用 LLM 從文字中提取 (主體, 關係, 客體)
        """
        prompt = f"""
        你是一個資料科學家。請從以下文本中提取關鍵實體(Entities)和它們之間的關係(Relationships)。
        
        文本：
        {text}

        請嚴格按照以下 JSON 格式輸出，不要包含任何解釋或其他文字：
        [
            {{"head": "實體1", "relation": "關係描述", "tail": "實體2"}},
            {{"head": "實體1", "relation": "關係描述", "tail": "實體3"}}
        ]
        
        重點：
        1. 實體(head/tail)必須是名詞。
        2. 關係(relation)必須簡潔。
        3. 如果沒有明確關係，回傳空陣列 []。
        4. JSON 格式必須合法。
        """

        try:
            response = ollama.chat(model='llama3.1:8b', messages=[
                {'role': 'user', 'content': prompt}
            ])
            content = response['message']['content']
            
            # 清理回應，確保只剩下 JSON 部分 (有時候 LLM 會多話)
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                return json.loads(json_str)
            else:
                return []
        except Exception as e:
            print(f"!!! 提取失敗: {e}")
            return []

    # ================= 3. 資料寫入 =================
    def ingest_document(self, file_path):
        print(f">>> 開始處理文件: {file_path}")
        
        # A. 讀取 PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # B. 切分文本 (太大 LLM 吃不下)
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        print(f">>> 文件已切分為 {len(chunks)} 個區塊，開始分析...")

        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            chunk_id = f"{os.path.basename(file_path)}_part_{i}"
            
            # --- C. 寫入 ChromaDB (向量) ---
            vector = ollama.embeddings(model="nomic-embed-text", prompt=text)['embedding']
            self.vector_collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[text],
                metadatas=[{"source": file_path}]
            )

            # --- D. 寫入 Neo4j (圖譜) ---
            relations = self.extract_graph_from_text(text)
            if relations:
                print(f"    區塊 {i}: 提取到 {len(relations)} 個關係")
                self.save_to_neo4j(relations, chunk_id)
            else:
                print(f"    區塊 {i}: 無法提取關係")

    def save_to_neo4j(self, relations, source_id):
        # --- 增加過濾邏輯 ---
        # 只有當 head, relation, tail 都有值且不為空時才保留
        valid_data = [
            row for row in relations 
            if row.get('head') and row.get('relation') and row.get('tail')
        ]
        
        if not valid_data:
            return

        query = """
        UNWIND $data AS row
        MERGE (h:Entity {name: row.head})
        MERGE (t:Entity {name: row.tail})
        MERGE (h)-[:RELATION {type: row.relation, source: $source}]->(t)
        """
        try:
            with self.driver.session() as session:
                session.run(query, data=valid_data, source=source_id)
        except Exception as e:
            print(f"!!! Neo4j 寫入失敗: {e}")

# # ================= 主程式執行 =================
# if __name__ == "__main__":
#     # 建立一個測試用的文字檔或 PDF，若無 PDF 可用下面代碼產生一個假的 PDF 測試
#     # 這裡假設你有一個 PDF，如果沒有，請將 PDF_PATH 指向你電腦隨便一個 PDF
    
#     if not os.path.exists(PDF_PATH):
#         print(f"錯誤：找不到檔案 {PDF_PATH}，請修改程式碼中的路徑。")
#     else:
#         builder = GraphRAGBuilder()
#         builder.ingest_document(PDF_PATH)
#         builder.close()
#         print("\n>>> 全部完成！請打開 Neo4j Browser 查看圖譜。")

# ================= 主程式執行 =================
if __name__ == "__main__":
    builder = GraphRAGBuilder()
    
    # 1. 取得資料夾內所有的 PDF 檔案路徑
    pdf_files = glob.glob(os.path.join(PDF_FOLDER_PATH, "*.pdf"))
    
    if not pdf_files:
        print(f"!!! 在 {PDF_FOLDER_PATH} 中找不到任何 PDF 檔案。")
    else:
        print(f"發現 {len(pdf_files)} 個檔案，準備開始批次處理...")
        
        for index, file_path in enumerate(pdf_files):
            file_name = os.path.basename(file_path)
            print(f"\n--- [{index + 1}/{len(pdf_files)}] 正在處理: {file_name} ---")
            
            try:
                # 呼叫原本寫好的 ingest_document 函式
                builder.ingest_document(file_path)
                print(f"ok! {file_name} 處理完成。")
            except Exception as e:
                print(f"!!! 處理檔案 {file_name} 時發生錯誤: {e}")
                # 這裡可以選擇繼續下一個檔案，而不是讓整個程式崩潰
                continue
                
        print("\n" + "="*30)
        print(f">>> 所有檔案處理完畢！共計 {len(pdf_files)} 個。")
        print(">>> 現在可以打開 Neo4j Browser 查詢合併後的全局圖譜。")

    builder.close()