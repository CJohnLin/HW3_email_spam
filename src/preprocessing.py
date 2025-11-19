# src/preprocessing.py
from typing import Tuple
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

RANDOM_SEED = 42

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # basic cleaning: lower, remove urls, emails, non-alphanum (keep spaces)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", dtype=str)
    # ensure columns exist
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset must contain columns: {text_col}, {label_col}")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].apply(clean_text)
    # map common labels to 0/1
    df[label_col] = df[label_col].map(lambda v: 1 if str(v).lower() in ("spam", "1", "true", "yes") else 0)
    return df

def vectorize_text(train_texts, val_texts=None, max_features: int = 10000) -> Tuple[TfidfVectorizer, any, any]:
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X_train = vect.fit_transform(train_texts)
    X_val = None
    if val_texts is not None:
        X_val = vect.transform(val_texts)
    return vect, X_train, X_val

def train_val_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_SEED, stratify: bool = True):
    y = df.iloc[:, 1].astype(int)
    X = df.iloc[:, 0]
    strat = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
    return X_train.tolist(), X_val.tolist(), y_train.values, y_val.values

def save_vectorizer(vect: TfidfVectorizer, path: str):
    joblib.dump(vect, path)
