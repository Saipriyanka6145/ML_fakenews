import os
import sys
import re
import string
import pickle
from flask import Flask, request, jsonify, render_template

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

model = None
vectorizer = None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_artifacts():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("[OK] Model and Vectorizer loaded successfully.", flush=True)
    else:
        print("[WARN] Model files not found. Run train.py first.", flush=True)

load_artifacts()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    global model, vectorizer
    try:
        if model is None or vectorizer is None:
            load_artifacts()
            if model is None:
                return jsonify({'error': 'Model not trained yet. Run train.py first.'}), 500

        data = request.get_json(silent=True) or {}
        news_text = data.get('text', '')
        if not news_text and 'news' in data:
            news_text = data['news']

        if not news_text.strip():
            return jsonify({'error': 'No text provided'}), 400

        cleaned = clean_text(news_text)
        vec = vectorizer.transform([cleaned])
        raw_pred = model.predict(vec)
        pred = int(raw_pred[0])
        label_str = "Real News" if pred == 1 else "Fake News"

        return jsonify({
            'label': pred,
            'prediction': label_str,
            'cleaned_text_preview': cleaned[:100]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5055))
    print(f"Starting Fake News Detection server at http://127.0.0.1:{port}...", flush=True)
    app.run(host='127.0.0.1', port=port, debug=False)
