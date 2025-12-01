import os
import gc
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Renderの環境変数からAPIキーを取得
API_KEY = os.environ.get("GEMINI_API_KEY")

# APIキー設定
if API_KEY:
    genai.configure(api_key=API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not API_KEY:
        return jsonify({"error": "サーバー側でAPIキーが設定されていません"}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "音声ファイルが見つかりません"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400

    try:
        # ファイルデータを読み込む
        file_data = file.read()
        
        # モデル設定 (Gemini 2.0 Flash Lite)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        
        # プロンプト (ご希望の内容に更新)
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

        # Geminiへ送信
        response = model.generate_content([
            prompt,
            {
                "mime_type": file.content_type,
                "data": file_data
            }
        ])

        # メモリ解放のおまじない
        del file_data
        gc.collect()

        return jsonify({"text": response.text})

    except Exception as e:
        print(f"Error: {e}")
        # メモリ不足のエラーが見えた場合、クライアントに伝える
        if "ResourceExhausted" in str(e) or "429" in str(e):
             return jsonify({"error": "AIの処理制限、またはメモリ不足が発生しました。ファイルを小さくして試してください。"}), 500
        return jsonify({"error": f"処理中にエラーが発生しました: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
