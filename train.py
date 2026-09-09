import os
import sys
import re
import string
import urllib.request
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Ensure utf-8 output safe for Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

FAKE_URL = "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/main/data/Fake.csv"
TRUE_URL = "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/main/data/True.csv"
DATA_DIR = "data"
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")

def download_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(FAKE_PATH):
        print(f"[1/2] Downloading Fake.csv from GitHub...", flush=True)
        urllib.request.urlretrieve(FAKE_URL, FAKE_PATH)
        print("--> Fake.csv downloaded successfully.", flush=True)
    else:
        print("--> Fake.csv already exists.", flush=True)

    if not os.path.exists(TRUE_PATH):
        print(f"[2/2] Downloading True.csv from GitHub...", flush=True)
        urllib.request.urlretrieve(TRUE_URL, TRUE_PATH)
        print("--> True.csv downloaded successfully.", flush=True)
    else:
        print("--> True.csv already exists.", flush=True)

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

def main():
    print("=" * 60, flush=True)
    print("FAKE NEWS DETECTION - END-TO-END TRAINING PIPELINE", flush=True)
    print("=" * 60, flush=True)

    # 1. Download
    download_datasets()

    # 2. Load
    print("\nLoading datasets into memory...", flush=True)
    df_fake = pd.read_csv(FAKE_PATH)
    df_true = pd.read_csv(TRUE_PATH)

    print(f"Fake news samples: {len(df_fake):,}", flush=True)
    print(f"Real news samples: {len(df_true):,}", flush=True)

    df_fake['class'] = 0  # 0 -> Fake
    df_true['class'] = 1  # 1 -> Real

    # Reserve 10 rows for manual testing
    fake_manual = df_fake.tail(10)
    true_manual = df_true.tail(10)
    df_fake = df_fake.iloc[:-10]
    df_true = df_true.iloc[:-10]

    # Combine
    df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)
    df = df[['text', 'class']].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total combined training samples: {len(df):,}", flush=True)

    # 3. Preprocessing
    print("\nCleaning & preprocessing text...", flush=True)
    df['text'] = df['text'].apply(clean_text)

    X = df['text']
    y = df['class']

    # 4. Train/Test Split
    print("\nSplitting dataset (75% train, 25% test)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 5. TF-IDF Vectorization
    print("Extracting TF-IDF Features...", flush=True)
    vectorizer = TfidfVectorizer(max_features=50000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"TF-IDF Matrix Shape (Train): {X_train_vec.shape}", flush=True)

    # 6. Train Models
    models = {
        "Linear SVM (LinearSVC)": LinearSVC(max_iter=2000),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    }

    results = {}
    best_model_name = None
    best_accuracy = 0.0
    best_model = None

    print("\n" + "=" * 60, flush=True)
    print("TRAINING AND EVALUATING ALL MODELS", flush=True)
    print("=" * 60, flush=True)

    for name, model in models.items():
        print(f"\n[+] Training {name}...", flush=True)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"[OK] {name} Accuracy: {acc * 100:.2f}%\n", flush=True)
        print("Classification Report:\n" + classification_report(y_test, y_pred, target_names=["Fake News (0)", "Real News (1)"]), flush=True)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    print("\n" + "=" * 60, flush=True)
    print("FINAL ACCURACY LEADERBOARD:", flush=True)
    for name, acc in results.items():
        print(f"  * {name:<26}: {acc * 100:.2f}%", flush=True)
    print(f"\nBest Performing Model: {best_model_name} ({best_accuracy * 100:.2f}%)", flush=True)
    print("=" * 60, flush=True)

    # 7. Save Best Model and Vectorizer
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model.pkl")
    vectorizer_path = os.path.join("models", "vectorizer.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nSaved best model to '{model_path}'", flush=True)
    print(f"Saved vectorizer to '{vectorizer_path}'", flush=True)

    # 8. Sample Predictions Test
    print("\nRunning Verification on Held-out Samples:", flush=True)
    sample_fake = fake_manual.iloc[0]['text']
    sample_true = true_manual.iloc[0]['text']

    def test_predict(sample_text, expected_label):
        cleaned = clean_text(sample_text)
        vec = vectorizer.transform([cleaned])
        pred = best_model.predict(vec)[0]
        label = "Real News (1)" if pred == 1 else "Fake News (0)"
        print(f"\nSample [{expected_label}]:\n  \"{sample_text[:110]}...\"", flush=True)
        print(f"  -> Predicted by AI: {label}", flush=True)

    test_predict(sample_fake, "Expected: Fake News")
    test_predict(sample_true, "Expected: Real News")

    print("\nAll tasks completed successfully!", flush=True)

if __name__ == "__main__":
    main()
