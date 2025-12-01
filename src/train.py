import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from src.preprocessing import preprocess_df

DATA_PATH = "Chapter03/datasets/sms_spam_no_header.csv"

def load_dataset():
    df = pd.read_csv(DATA_PATH, header=None, names=["label", "text"])
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return preprocess_df(df)

def train_models():
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "nb": MultinomialNB(),
        "svm": LinearSVC()
    }

    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
            ("clf", clf)
        ])

        pipe.fit(X_train, y_train)
        joblib.dump(pipe, f"models/{name}.joblib")
        print(f"Saved: models/{name}.joblib")

if __name__ == "__main__":
    train_models()
