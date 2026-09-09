# 📰 Fake News Detection using Machine Learning & NLP

An end-to-end Machine Learning and Natural Language Processing (NLP) project that classifies news articles as **Real** or **Fake** based on their textual content.

---

## 📌 Overview
Fake news spreads rapidly across online platforms and social media, influencing public perception and spreading misinformation. This project processes large datasets of news articles, performs natural language preprocessing and TF-IDF vectorization, and trains multiple machine learning classifiers to accurately detect fake news.

---

## 🚀 Key Features
- **Data Preprocessing & Cleaning:** Lowercasing, removal of URLs, brackets, punctuation, digits, and special characters using regular expressions.
- **Feature Extraction:** TF-IDF Vectorizer (`TfidfVectorizer`) with vocabulary mapping.
- **Multiple ML Classifiers:**
  - **Support Vector Machine (LinearSVC)** — *99.37% Accuracy*
  - **Random Forest Classifier** — *98.83% Accuracy*
  - **Logistic Regression** — *98.52% Accuracy*
  - **Multinomial Naive Bayes** — *93.85% Accuracy*
- **Interactive Testing:** Interactive prediction function to test any custom news text in real-time.

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear SVM (LinearSVC)** 🏆 | **99.37%** | 0.99 | 0.99 | 0.99 |
| **Random Forest** | **98.83%** | 0.99 | 0.99 | 0.99 |
| **Logistic Regression** | **98.52%** | 0.99 | 0.98 | 0.99 |
| **Multinomial Naive Bayes** | **93.85%** | 0.93 | 0.95 | 0.94 |

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.x
- **Libraries:**
  - `pandas` — Data manipulation & analysis
  - `numpy` — Numerical computations
  - `scikit-learn` — Machine learning models & TF-IDF vectorization
  - `matplotlib` & `seaborn` — Data visualization
  - `re` & `string` — Text preprocessing & regex cleaning

---

## 📂 Project Structure
```
Fake-News-Detection-ML/
├── Fake_News_Detection.ipynb   # Main Jupyter notebook containing training and evaluation
├── README.md                   # Project documentation
└── baton/                      # Compressed resources & archives
```

---

## ⚙️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Saipriyanka6145/ML_fakenews.git
cd ML_fakenews
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
```

### 3. Run the Notebook
Open and run `Fake_News_Detection.ipynb` in **Jupyter Notebook** or **Google Colab**.

---

## 👤 Author
**Sai Priyanka**
- GitHub: [@Saipriyanka6145](https://github.com/Saipriyanka6145)
- Repository: [ML_fakenews](https://github.com/Saipriyanka6145/ML_fakenews)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).