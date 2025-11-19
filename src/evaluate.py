# src/evaluate.py
from typing import Tuple
import joblib
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt

def load_vectorizer_and_model(models_dir: str, model_key: str):
    vect = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))
    model = joblib.load(os.path.join(models_dir, f"{model_key}_model.pkl"))
    return vect, model

def evaluate_on_csv(models_dir: str, model_key: str, csv_path: str, text_col: str = "text", label_col: str = "label"):
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str)
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    y_true = df[label_col].map(lambda v: 1 if str(v).lower() in ("spam","1","true","yes") else 0).astype(int).values
    vect, model = load_vectorizer_and_model(models_dir, model_key)
    X = vect.transform(df[text_col].values)
    y_pred = model.predict(X)
    report = classification_report(y_true, y_pred, output_dict=False)
    cm = confusion_matrix(y_true, y_pred)
    # ROC
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:,1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    auc = None
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = None
    return {"report": report, "confusion_matrix": cm, "auc": auc}
