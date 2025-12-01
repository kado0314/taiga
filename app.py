import os
import gc
import time
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# 1. Renderの環境変数からAPIキーを取得して設定
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not API_KEY:
        return jsonify({"error": "APIキーが設定されていません"}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400

    # 一時ファイルの保存パス
    temp_path = "temp_audio.mp3"

    try:
        # 【メモリ対策1】メモリに展開せず、一旦ディスクに保存
        file.save(temp_path)
        
        # 【モデル指定】ここで明確に gemini-2.0-flash-lite を指定しています
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        
        # 【メモリ対策2】GeminiのFile APIを使用してアップロード
        # (大きなファイルもサーバーのメモリを使わずに処理できます)
        uploaded_file = genai.upload_file(temp_path, mime_type="audio/mp3")

        # 念のためアップロード完了を少し待機（即時処理でエラーになるのを防ぐ）
        time.sleep(1)

        # プロンプト (ご指定の内容に変更)
        prompt = """
        あなたはプロの書記官です。添付された音声ファイルを聞き、以下の形式で詳細な議事録を作成してください。

        【要件】
        1. **話者の識別**: 文脈から可能な限り「Aさん」「Bさん」のように書き分けてください。
        2. **要約と詳細**: 議論の流れがわかるようにまとめてください。
        3. **重要事項**: 決定事項やネクストアクションは明確に抜き出してください。
        4. **重要**: 喋っていることには喋っている人の名前とカギカッコ [〇さん「」] を付けてください。

        【出力フォーマット】
        # 会議議事録
        ## 1. 概要
        * 日時/参加者: (推定)

        ## 2. 決定事項

        ## 3. ネクストアクション (ToDo)

        ## 4. 詳細な議論内容 (会話形式)
        """

        # 解析実行
        response = model.generate_content([prompt, uploaded_file])

        # 【メモリ対策3】Google側のファイルを削除（ゴミを残さない）
        try:
            uploaded_file.delete()
        except:
            pass

        return jsonify({"text": response.text})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"エラーが発生しました: {str(e)}"}), 500
        
    finally:
        # 【メモリ対策4】Render側の一次ファイルを削除
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()

if __name__ == '__main__':
    app.run(debug=True)
