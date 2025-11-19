# src/predict.py
from typing import Tuple, Any
import joblib
import os
from src.preprocessing import clean_text

def load_model_and_vectorizer(models_dir: str, model_key: str):
    vect = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
    model = joblib.load(os.path.join(models_dir, f"{model_key}_model.pkl"))
    return model, vect

def predict_text(models_dir: str, model_key: str, text: str) -> Tuple[int, float]:
    model, vect = load_model_and_vectorizer(models_dir, model_key)
    text_clean = clean_text(text)
    X = vect.transform([text_clean])
    pred = model.predict(X)[0]
    # score: probability if possible else decision_function
    score = None
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[0,1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
    else:
        score = float(pred)
    return int(pred), float(score)
