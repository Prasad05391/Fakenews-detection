# NewsGuard AI — Professional Fake News Detection Platform

## Features
- Professional responsive UI
- Home, Verify, How It Works, About, Dashboard and History pages
- TF-IDF + Logistic Regression model
- Confidence score
- SQLite verification history
- Flask backend
- Responsive HTML/CSS/JavaScript frontend

## 1. Create environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 3. Train the model

The included CSV is only a tiny demonstration dataset. Replace it with a substantially larger, properly labeled dataset for a meaningful project.

```powershell
python train_model.py
```

This creates:

```text
model/fake_news_model.pkl
model/tfidf_vectorizer.pkl
```

## 4. Start the website

```powershell
python app.py
```

Open:

http://127.0.0.1:5000

## Dataset format

CSV must contain:

```text
text,label
"news article text...",REAL
"news article text...",FAKE
```

## Important project note

A machine-learning classifier does not prove whether a claim is factually true. It predicts patterns learned from its training data. For a stronger final-year project, combine the classifier with source credibility checks, retrieval from trusted sources, explainability, and careful evaluation on a held-out dataset.

## Suggested production improvements

- Use a real, large, balanced dataset.
- Add authentication and role-based admin access.
- Add rate limiting and input validation.
- Store secrets in environment variables.
- Use PostgreSQL/MongoDB for deployment.
- Add unit tests.
- Add model versioning.
- Add precision/recall/F1 and confusion matrix to the dashboard.
