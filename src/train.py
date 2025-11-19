# src/train.py
import argparse
from typing import Optional
from src.preprocessing import load_dataset, train_val_split, vectorize_text, save_vectorizer, clean_text
from src.models import build_lr, build_nb, build_svm, save_model
import numpy as np
from sklearn.metrics import classification_report
import os

MODEL_MAP = {
    "lr": ("lr_model.pkl", build_lr),
    "nb": ("nb_model.pkl", build_nb),
    "svm": ("svm_model.pkl", build_svm),
}

def train_single(model_key: str, X_train_texts, y_train, X_val_texts=None, y_val=None, models_dir: str = "models"):
    model_name, builder = MODEL_MAP[model_key]
    vect, X_train, X_val = vectorize_text(X_train_texts, X_val_texts)
    model = builder()
    model.fit(X_train, y_train)
    os.makedirs(models_dir, exist_ok=True)
    save_model(model, os.path.join(models_dir, model_name))
    save_vectorizer(vect, os.path.join(models_dir, "vectorizer.pkl"))
    if X_val_texts is not None and y_val is not None:
        y_pred = model.predict(X_val)
        print(f"=== Validation report for {model_key} ===")
        print(classification_report(y_val, y_pred))
    else:
        print(f"Trained {model_key} and saved to {models_dir}/{model_name}")

def train_all(dataset_path: str, models_dir: str = "models"):
    df = load_dataset(dataset_path)
    X_train_texts, X_val_texts, y_train, y_val = train_val_split(df)
    # train each model using same vectorizer (fit on train only)
    from src.preprocessing import vectorize_text
    vect, X_train, X_val = vectorize_text(X_train_texts, X_val_texts)
    save_vectorizer(vect, os.path.join(models_dir, "vectorizer.pkl"))
    for key, (_, builder) in MODEL_MAP.items():
        model = builder()
        model.fit(X_train, y_train)
        save_model(model, os.path.join(models_dir, MODEL_MAP[key][0]))
        y_pred = model.predict(X_val)
        print(f"--- {key} ---")
        from sklearn.metrics import classification_report
        print(classification_report(y_val, y_pred))

def main():
    parser = argparse.ArgumentParser(description="Train spam models")
    parser.add_argument("--dataset", type=str, required=True, help="path to CSV dataset (columns: text,label)")
    parser.add_argument("--model", type=str, default="all", choices=["all", "lr", "nb", "svm"])
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    if args.model == "all":
        train_all(args.dataset, args.models_dir)
    else:
        df = load_dataset(args.dataset)
        X_train_texts, X_val_texts, y_train, y_val = train_val_split(df)
        train_single(args.model, X_train_texts, y_train, X_val_texts, y_val, args.models_dir)

if __name__ == "__main__":
    main()
