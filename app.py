from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from pathlib import Path
from datetime import datetime
import sqlite3, re, joblib, math

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
DB_PATH = BASE / "news_history.db"

app = Flask(__name__)
app.secret_key = "change-this-secret-key-in-production"

MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
vectorizer = joblib.load(VECTORIZER_PATH) if VECTORIZER_PATH.exists() else None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def predict_news(text):
    if not model or not vectorizer:
        return "MODEL_NOT_TRAINED", 0.0

    vector = vectorizer.transform([clean_text(text)])
    prediction = model.predict(vector)[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(vector)[0])) * 100
    else:
        confidence = 50.0

    return str(prediction).upper(), round(confidence, 2)


@app.context_processor
def inject_globals():
    return {"year": datetime.now().year}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/verify", methods=["GET", "POST"])
def verify():
    result = None
    confidence = None
    news = ""

    if request.method == "POST":
        news = request.form.get("news", "").strip()
        if len(news) < 20:
            flash("Please enter at least 20 characters of news text.", "error")
        else:
            result, confidence = predict_news(news)
            if result == "MODEL_NOT_TRAINED":
                flash("The ML model is not trained yet. Run: python train_model.py", "error")
            else:
                conn = db()
                conn.execute(
                    "INSERT INTO history(news,prediction,confidence,created_at) VALUES(?,?,?,?)",
                    (news, result, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                conn.commit()
                conn.close()

    return render_template("verify.html", result=result, confidence=confidence, news=news)


@app.route("/history")
def history():
    conn = db()
    rows = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/dashboard")
def dashboard():
    conn = db()
    total = conn.execute("SELECT COUNT(*) c FROM history").fetchone()["c"]
    real = conn.execute("SELECT COUNT(*) c FROM history WHERE prediction='REAL'").fetchone()["c"]
    fake = conn.execute("SELECT COUNT(*) c FROM history WHERE prediction='FAKE'").fetchone()["c"]
    avg = conn.execute("SELECT AVG(confidence) a FROM history").fetchone()["a"]
    recent = conn.execute("SELECT * FROM history ORDER BY id DESC LIMIT 8").fetchall()
    conn.close()

    return render_template(
        "dashboard.html",
        total=total, real=real, fake=fake,
        average_confidence=round(avg or 0, 2),
        recent=recent,
        model_ready=bool(model and vectorizer)
    )


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": bool(model and vectorizer)
    })


init_db()

if __name__ == "__main__":
    app.run(debug=True)
