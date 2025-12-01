import argparse
import pandas as pd
import joblib
from src.preprocessing import clean_text

def load_model(model_path: str):
    """Load trained model (.joblib)."""
    return joblib.load(model_path)


def predict_single(model, text: str):
    """Predict a single SMS/email."""
    clean = clean_text(text)
    pred = model.predict([clean])[0]

    # 若模型支援機率 (Logistic, NB)，則一起顯示
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([clean])[0][1]

    return pred, prob


def predict_csv(model, csv_path: str):
    """Batch predict from CSV file (must contain column 'text')."""
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("CSV 必須包含 'text' 欄位。")

    df["text_clean"] = df["text"].astype(str).apply(clean_text)
    df["pred"] = model.predict(df["text_clean"])

    # 若支援機率，則加入機率欄位
    if hasattr(model, "predict_proba"):
        df["spam_prob"] = model.predict_proba(df["text_clean"])[:, 1]

    return df


def main():
    parser = argparse.ArgumentParser(description="Spam Predictor")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model, e.g. models/logreg.joblib"
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Text string to classify"
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="CSV file for batch prediction (must contain 'text' column)"
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Single prediction
    if args.text:
        pred, prob = predict_single(model, args.text)
        label = "SPAM" if pred == 1 else "HAM"
        print(f"[Result] {label}")
        if prob is not None:
            print(f"[Spam Probability] {prob:.4f}")

    # Batch prediction
    if args.csv:
        df_result = predict_csv(model, args.csv)
        output = "prediction_output.csv"
        df_result.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"[Batch Prediction Completed] Results saved to {output}")


if __name__ == "__main__":
    main()
