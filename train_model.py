from pathlib import Path
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset" / "news.csv"
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


df = pd.read_csv(DATASET)
required = {"text", "label"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].apply(clean_text)
df["label"] = df["label"].astype(str).str.upper().str.strip()
df = df[df["label"].isin(["REAL", "FAKE"])]

if len(df) < 10:
    raise ValueError("Add a larger labeled dataset before training.")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_DIR / "fake_news_model.pkl")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
print("Model saved in model/")
