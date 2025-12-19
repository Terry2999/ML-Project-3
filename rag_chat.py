import ollama
import chromadb
from neo4j import GraphDatabase

# ================= 設定 =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "12345678")

class HybridRAGChat:
    def __init__(self):
        # 1. 連線 Neo4j
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        
        # 2. 連線 ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./graphrag_db")
        self.collection = self.chroma_client.get_collection("rag_collection")

    def close(self):
        self.driver.close()

    # --- A. 向量搜尋 (Vector Search) ---
    def get_vector_context(self, query, k=3):
        # 將問題轉成向量
        query_embed = ollama.embeddings(model="nomic-embed-text", prompt=query)['embedding']
        # 搜尋
        results = self.collection.query(query_embeddings=[query_embed], n_results=k)
        
        if results['documents']:
            return "\n".join(results['documents'][0])
        return ""

    # --- B. 圖譜搜尋 (Graph Search) ---
    def get_graph_context(self, query):
        """
        簡單策略：用關鍵字去圖資料庫找「鄰居」
        """
        # 這裡簡化處理，直接把 User 的問題當作關鍵字去模糊比對實體名稱
        # 進階做法是先讓 LLM 提取 query 中的關鍵實體 (Entity Extraction)
        
        cypher_query = """
        MATCH (n)-[r]->(m)
        WHERE n.name CONTAINS $keyword OR m.name CONTAINS $keyword
        RETURN n.name, type(r), m.name
        LIMIT 10
        """
        
        # 為了演示，我們簡單拆解問題中的關鍵字 (或是直接傳入整句)
        # 實際使用建議提取名詞，這裡我們先嘗試用整句 query 的部分字串
        # 但為了效果，這裡我們假設 user 會問 "電池" 這種關鍵字
        
        context_lines = []
        with self.driver.session() as session:
            # 這裡做一個簡單的關鍵字提取優化：
            # 在真實場景，我們應該叫 LLM 提取關鍵字，這裡我們先用簡單的 split
            keywords = query.replace("?", "").replace("？", "").split()
            
            for kw in keywords:
                if len(kw) < 2: continue # 跳過太短的字
                
                result = session.run(cypher_query, keyword=kw)
                for record in result:
                    line = f"{record['n.name']} --[{record['type(r)']}]--> {record['m.name']}"
                    context_lines.append(line)
        
        return "\n".join(list(set(context_lines))) # 去重

    # --- C. 混合生成 (Hybrid Generation) ---
    def chat(self, user_query):
        print(f"正在思考: {user_query} ...")
        
        # 1. 獲取上下文
        vector_ctx = self.get_vector_context(user_query)
        graph_ctx = self.get_graph_context(user_query)
        
        print(f"--- 向量檢索到的片段長度: {len(vector_ctx)}")
        print(f"--- 圖譜檢索到的關係數: {len(graph_ctx.splitlines()) if graph_ctx else 0}")

        # 2. 組合 Prompt
        prompt = f"""
        你是一個智慧助手。請根據以下檢索到的資訊回答使用者的問題。
        
        [來自文本的詳細內容 (Vector DB)]:
        {vector_ctx}
        
        [來自知識圖譜的關聯 (Graph DB)]:
        {graph_ctx}
        
        使用者問題: {user_query}
        
        請用繁體中文回答，並綜合上述兩種資訊來源。如果資訊不足，請誠實告知。
        """

        # 3. 生成回答
        response = ollama.chat(model='llama3.1:8b', messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        return response['message']['content']

# ================= 執行對話 =================
if __name__ == "__main__":
    bot = HybridRAGChat()
    
    # 互動迴圈
    print(">>> 系統就緒！請輸入問題 (輸入 'exit' 離開)")
    while True:
        q = input("\n你: ")
        if q.lower() in ['exit', 'quit']:
            break
            
        try:
            ans = bot.chat(q)
            print(f"\nAI: {ans}")
        except Exception as e:
            print(f"發生錯誤: {e}")
            
    bot.close()