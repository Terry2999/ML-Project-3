import ollama
from neo4j import GraphDatabase
import chromadb

# --- 設定連線 ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "12345678")
CHROMA_PATH = "./graphrag_db"

class HybridRetriever:
    def __init__(self):
        # 初始化 Neo4j
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        # 初始化 ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection("rag_docs")

    def close(self):
        self.neo4j_driver.close()

    def add_data(self, text, doc_id, entities):
        """
        同時存入向量與圖譜
        entities: 格式為 [('實體1', '關係', '實體2')]
        """
        # 1. 存入 ChromaDB (向量)
        embed = ollama.embeddings(model="nomic-embed-text", prompt=text)['embedding']
        self.collection.add(ids=[doc_id], embeddings=[embed], documents=[text])

        # 2. 存入 Neo4j (圖譜)
        with self.neo4j_driver.session() as session:
            for s, r, o in entities:
                session.run(
                    "MERGE (a:Entity {name: $s}) "
                    "MERGE (b:Entity {name: $o}) "
                    "MERGE (a)-[:REL {type: $r}]->(b)",
                    s=s, r=r, o=o
                )

    def hybrid_search(self, query):
        # 1. 向量檢索
        query_embed = ollama.embeddings(model="nomic-embed-text", prompt=query)['embedding']
        v_results = self.collection.query(query_embeddings=[query_embed], n_results=1)
        v_context = v_results['documents'][0][0] if v_results['documents'] else ""

        # 2. 圖譜檢索 (簡單搜尋相關節點)
        # 這裡假設 query 包含實體名稱，實際應用會用 LLM 提取實體
        g_context = []
        with self.neo4j_driver.session() as session:
            # 找尋跟問題相關的節點及其關係
            result = session.run(
                "MATCH (n:Entity)-[r]->(m) WHERE n.name CONTAINS $q OR m.name CONTAINS $q "
                "RETURN n.name, type(r), m.name LIMIT 5", q=query
            )
            for record in result:
                g_context.append(f"{record[0]} -{record[1]}-> {record[2]}")

        return v_context, "\n".join(g_context)

# --- 測試執行 ---
hr = HybridRetriever()

# 模擬存入一筆關於 RTX 2080 Ti 的資料
text_content = "RTX 2080 Ti 是一款由 NVIDIA 推出的高端顯卡，擁有 11GB VRAM。"
relations = [("RTX 2080 Ti", "生產商", "NVIDIA"), ("RTX 2080 Ti", "顯存", "11GB")]
hr.add_data(text_content, "doc_001", relations)

# 進行混合查詢
user_query = "RTX 2080 Ti"
vector_info, graph_info = hr.hybrid_search(user_query)

# 結合 Ollama 生成回答
final_prompt = f"""
請根據以下資訊回答問題。
[向量參考內容]: {vector_info}
[圖譜關聯資料]: {graph_info}

問題: {user_query}
"""

response = ollama.generate(model="llama3.1:8b", prompt=final_prompt)
print("\n--- GraphRAG 回答 ---")
print(response['response'])

hr.close()