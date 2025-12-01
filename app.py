import os
import gc
import time
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Renderの環境変数からAPIキーを取得
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
        # メモリ対策: 一旦ディスクに保存
        file.save(temp_path)
        
        # モデル指定 (Gemini 2.0 Flash Lite)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        
        # File APIを使用してアップロード
        uploaded_file = genai.upload_file(temp_path, mime_type="audio/mp3")
        time.sleep(1) # 処理待ち

        # ▼▼▼▼▼ プロンプト修正部分 ▼▼▼▼▼
        prompt = """
        あなたはプロの書記官です。添付された音声ファイルを聞き、以下の形式で詳細な議事録を作成してください。

        【重要要件】
        1. **話者の特定（最優先）**: 
           - 音声内の「呼びかけ」や「自己紹介」から、具体的な名前（例：かいち、きみのり）を特定してください。
           - 名前が特定できた場合は、必ずその名前を使用してください。
           - どうしても名前が不明な人物のみ「Aさん」「Bさん」とアルファベットで表記してください。
        
        2. **会話の記述形式**:
           - 以下の形式で記述してください（カギカッコを使用）。
             かいち「こんにちは」
             きみのり「こんばんは」
             Aさん「こんばんは」

        3. **構成**:
           - 概要の「参加者」欄には、特定できた名前をすべて列挙してください。

        【出力フォーマット】
        # 会議議事録
        ## 1. 概要
        * 日時: (推定)
        * 参加者: (特定できた名前、および不明な人数)

        ## 2. 決定事項
        * ## 3. ネクストアクション (ToDo)
        * ## 4. 詳細な議論内容 (会話形式)
        """
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 解析実行
        response = model.generate_content([prompt, uploaded_file])

        # Google側のファイルを削除
        try:
            uploaded_file.delete()
        except:
            pass

        return jsonify({"text": response.text})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"エラーが発生しました: {str(e)}"}), 500
        
    finally:
        # Render側の一次ファイルを削除
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()

if __name__ == '__main__':
    app.run(debug=True)
