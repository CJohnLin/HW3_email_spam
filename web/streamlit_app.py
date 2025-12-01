import streamlit as st
import pandas as pd
import joblib
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ensure repo root in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing import clean_text

st.set_page_config(page_title="Spam Classifier", layout="wide")
st.title("Spam Classification Demo")

model_paths = {
    "Logistic Regression": "models/logreg.joblib",
    "Naive Bayes": "models/nb.joblib",
    "Linear SVM": "models/svm.joblib"
}

model_choice = st.sidebar.selectbox("Select model", list(model_paths.keys()))
if not os.path.exists(model_paths[model_choice]):
    st.sidebar.error(f"Model not found: {model_paths[model_choice]}")
    st.stop()
model = joblib.load(model_paths[model_choice])

st.header('Single prediction')
text = st.text_area('Enter message here')
if st.button('Predict'):
    if text.strip()=='':
        st.warning('Please enter text')
    else:
        clean = clean_text(text)
        pred = model.predict([clean])[0]
        if hasattr(model,'predict_proba'):
            prob = model.predict_proba([clean])[0][1]
            st.write('Spam prob:', prob)
        st.write('Prediction:', 'SPAM' if pred==1 else 'HAM')

st.header('Batch upload CSV (text column)')
uploaded = st.file_uploader('Upload CSV', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    if 'text' not in df.columns:
        st.error('CSV must contain text column')
    else:
        df['text_clean'] = df['text'].astype(str).apply(clean_text)
        df['pred'] = model.predict(df['text_clean'])
        if hasattr(model,'predict_proba'):
            df['spam_prob'] = model.predict_proba(df['text_clean'])[:,1]
        st.dataframe(df)
        st.download_button('Download', df.to_csv(index=False).encode('utf-8-sig'), 'predictions.csv')

dataset_path = os.path.join('Chapter03','datasets','sms_spam_no_header.csv')
if os.path.exists(dataset_path):
    df_raw = pd.read_csv(dataset_path, header=None, names=['label','text'])
    df_raw['label'] = df_raw['label'].map({'ham':0,'spam':1})
    df_raw['text_clean'] = df_raw['text'].astype(str).apply(clean_text)
    y_true = df_raw['label']
    y_pred = model.predict(df_raw['text_clean'])
    st.subheader('Classification report')
    st.text(classification_report(y_true, y_pred, target_names=['HAM','SPAM']))
    cm = confusion_matrix(y_true, y_pred)
    fig,ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
else:
    st.info('Dataset not found; upload in Chapter03/datasets/sms_spam_no_header.csv')
