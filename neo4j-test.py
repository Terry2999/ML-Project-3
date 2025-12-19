from neo4j import GraphDatabase

# 設定連線資訊 (預設帳號是 neo4j)
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "12345678")

class Neo4jTest:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def create_test_graph(self):
        with self.driver.session() as session:
            # 建立一個測試關係：文件包含關鍵字
            query = """
            MERGE (d:Document {name: 'RAG_Homework.pdf'})
            MERGE (k:Keyword {text: 'Ollama'})
            MERGE (d)-[:HAS_KEYWORD]->(k)
            RETURN d.name, k.text
            """
            result = session.run(query)
            for record in result:
                print(f"成功連線！已建立關係: {record['d.name']} -> {record['k.text']}")

    def verify(self):
        try:
            self.driver.verify_connectivity()
            print("連線狀態：正常")
        except Exception as e:
            print(f"連線失敗: {e}")

# 執行測試
if __name__ == "__main__":
    tester = Neo4jTest(URI, AUTH)
    tester.verify()
    tester.create_test_graph()
    tester.close()