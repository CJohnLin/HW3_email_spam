import joblib
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import clean_text

def evaluate(model_path, data_path=os.path.join('Chapter03','datasets','sms_spam_no_header.csv')):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path, header=None, names=['label','text'])
    df['label'] = df['label'].map({'ham':0,'spam':1})
    df['text_clean'] = df['text'].astype(str).apply(clean_text)
    y_true = df['label']
    y_pred = model.predict(df['text_clean'])
    print(classification_report(y_true, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    evaluate(os.path.join('models','logreg.joblib'))
