# src/models.py
from typing import Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import joblib
import os

def build_lr(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=random_state)

def build_nb() -> MultinomialNB:
    return MultinomialNB()

def build_svm(random_state: int = 42) -> SVC:
    # probability=True so we can compute ROC
    return SVC(kernel="linear", C=1.0, probability=True, random_state=random_state)

def save_model(model: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
