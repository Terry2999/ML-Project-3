import ollama

try:
    response = ollama.chat(model='llama3.1:8b', messages=[
        # {'role': 'user', 'content': '嘿！如果我的顯卡是 2080 Ti，你會建議我怎麼使用你？'}
        # {'role': 'user', 'content': '嘿！如果我的顯卡是 2080 Ti，你會建議我怎麼使用你？'}
        # {'role': 'user', 'content': 'Do you understand english?'}
        # {'role': 'user', 'content': '你懂中文嗎？'}
        # {'role': 'user', 'content': '你懂幾種語言？'}
        # {'role': 'user', 'content': '法文怎麼說「你好」？'}
        # {'role': 'user', 'content': '你會記住我剛剛說的話嗎？'}
        {'role': 'user', 'content': 'ollama適合用conda跑，還是用pip跑？'}
    ])
    print("Ollama 連線成功！")
    print("AI 回應：", response['message']['content'])
except Exception as e:
    print(f"連線失敗，請確認 Ollama 是否正在執行。錯誤訊息: {e}")