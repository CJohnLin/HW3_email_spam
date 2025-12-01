import joblib
import pandas as pd
from src.preprocessing import clean_text

def load_vectorizer(path):
    return joblib.load(path)

def load_model(path):
    return joblib.load(path)

def predict_single(model, vectorizer, text):
    t = clean_text(text)
    x = vectorizer.transform([t])
    pred = model.predict(x)[0]
    prob = None
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(x)[0][1]
    return pred, prob

def predict_csv(model, vectorizer, csv_path, out_path='prediction_output.csv'):
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError('CSV must contain text column')
    df['text_clean'] = df['text'].astype(str).apply(clean_text)
    X = vectorizer.transform(df['text_clean'])
    df['pred'] = model.predict(X)
    if hasattr(model, 'predict_proba'):
        df['spam_prob'] = model.predict_proba(X)[:,1]
    df.to_csv(out_path, index=False)
    return out_path
