import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model_name):
    model = joblib.load(f"models/{model_name}.joblib")
    df = pd.read_csv("Chapter03/datasets/sms_spam_no_header.csv", header=None, names=["label", "text"])
    df["label"] = df["label"].map({"ham":0,"spam":1})
    from src.preprocessing import preprocess_df
    df = preprocess_df(df)

    y_true = df["label"]
    y_pred = model.predict(df["text_clean"])

    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate("logreg")
