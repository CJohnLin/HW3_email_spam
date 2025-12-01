import streamlit as st
import pandas as pd
import joblib
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ensure repo root in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing import clean_text

st.set_page_config(page_title='垃圾簡訊分類系統', page_icon='📧', layout='centered')

st.markdown("""<h1 style='text-align:center;color:#22577A;'>📧 垃圾簡訊分類系統（自訂模型）</h1>""", unsafe_allow_html=True)

with st.sidebar:
    st.header('⚙️ 模型設定 (使用使用者模型檔案)')
    st.write('期望檔名：')
    st.code('spam_logreg_model.joblib\nspam_tfidf_vectorizer.joblib\nspam_label_mapping.json')
    st.caption('若你有自己的模型檔，請放到 /models 並使用相同檔名。')

# model files
model_file = os.path.join('models','spam_logreg_model.joblib')
vec_file = os.path.join('models','spam_tfidf_vectorizer.joblib')
map_file = os.path.join('models','spam_label_mapping.json')

if not os.path.exists(model_file) or not os.path.exists(vec_file):
    st.error('找不到模型或向量器。請將 spam_logreg_model.joblib 與 spam_tfidf_vectorizer.joblib 放入 models/ 資料夾。')
    st.stop()

model = joblib.load(model_file)
vectorizer = joblib.load(vec_file)

st.subheader('🔍 單筆預測')
txt = st.text_area('輸入簡訊內容', height=120)
if st.button('預測'):
    if txt.strip()=='':
        st.warning('請輸入文字')
    else:
        t = clean_text(txt)
        x = vectorizer.transform([t])
        pred = model.predict(x)[0]
        if hasattr(model,'predict_proba'):
            prob = model.predict_proba(x)[0][1]
            st.info(f'垃圾訊息機率: {prob:.4f}')
        st.write('結果:', 'SPAM' if pred==1 else 'HAM')

st.markdown('---')
st.subheader('📂 批次預測 (CSV)')
uploaded = st.file_uploader('上傳 CSV（需含 text 欄位）', type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    if 'text' not in df.columns:
        st.error('CSV 必須含 text 欄位')
    else:
        df['text_clean'] = df['text'].astype(str).apply(clean_text)
        X = vectorizer.transform(df['text_clean'])
        df['pred'] = model.predict(X)
        if hasattr(model,'predict_proba'):
            df['spam_prob'] = model.predict_proba(X)[:,1]
        st.dataframe(df)
        st.download_button('下載結果', df.to_csv(index=False).encode('utf-8-sig'), 'predictions.csv')

# evaluation if dataset present
ds = os.path.join('Chapter03','datasets','sms_spam_no_header.csv')
if os.path.exists(ds):
    df_all = pd.read_csv(ds, header=None, names=['label','text'])
    df_all['label'] = df_all['label'].map({'ham':0,'spam':1})
    df_all['text_clean'] = df_all['text'].astype(str).apply(clean_text)
    X_all = vectorizer.transform(df_all['text_clean'])
    preds = model.predict(X_all)
    st.subheader('📊 分類報告')
    st.text(classification_report(df_all['label'], preds, target_names=['HAM','SPAM']))
    cm = confusion_matrix(df_all['label'], preds)
    fig,ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
else:
    st.info('資料集缺失：請放 Chapter03/datasets/sms_spam_no_header.csv')
