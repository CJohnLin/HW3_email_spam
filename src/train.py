import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import joblib
import os
from src.preprocessing import clean_text

DATA_PATH = os.path.join("Chapter03","datasets","sms_spam_no_header.csv")
MODELS_DIR = "models"

def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path, header=None, names=["label","text"])
    df["label"] = df["label"].map({"ham":0,"spam":1})
    df["text_clean"] = df["text"].astype(str).apply(clean_text)
    return df

def train_and_save():
    df = load_dataset()
    X = df["text_clean"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Logistic Regression
    pipe_lr = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
                        ("clf", LogisticRegression(max_iter=2000))])
    pipe_lr.fit(X_train, y_train)
    joblib.dump(pipe_lr, os.path.join(MODELS_DIR,"logreg.joblib"))

    # Multinomial NB
    pipe_nb = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
                        ("clf", MultinomialNB())])
    pipe_nb.fit(X_train, y_train)
    joblib.dump(pipe_nb, os.path.join(MODELS_DIR,"nb.joblib"))

    # Linear SVM
    pipe_svm = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
                         ("clf", LinearSVC(max_iter=5000))])
    pipe_svm.fit(X_train, y_train)
    joblib.dump(pipe_svm, os.path.join(MODELS_DIR,"svm.joblib"))

    print('Models saved to', MODELS_DIR)

if __name__ == '__main__':
    train_and_save()
