import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# -----------------------------
# 修正 Streamlit Cloud import 問題
# -----------------------------
import sys
import os

# 取得 web/ 的上層，也就是整個 repo 的根目錄
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing import clean_text


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Spam Classification System",
    page_icon="📧",
    layout="wide"
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📘 系統設定")

model_paths = {
    "Logistic Regression": "models/logreg.joblib",
    "Naive Bayes": "models/nb.joblib",
    "Linear SVM": "models/svm.joblib"
}

model_choice = st.sidebar.selectbox("選擇模型", list(model_paths.keys()))

# 載入模型
if os.path.exists(model_paths[model_choice]):
    model = joblib.load(model_paths[model_choice])
else:
    st.sidebar.error(f"找不到模型檔案：{model_paths[model_choice]}")
    st.stop()

uploaded_csv = st.sidebar.file_uploader("📂 上傳 CSV（需包含 text 欄位）", type=["csv"])


# -----------------------------
# Title
# -----------------------------
st.title("📧 Spam Classification System")
st.markdown("使用 ML 模型即時判斷簡訊是否為 **垃圾訊息 (SPAM)** 或 **正常訊息 (HAM)**。")


# -----------------------------
# Single Prediction
# -----------------------------
st.markdown("## 🔍 單筆訊息預測")

text_input = st.text_area(
    label="請輸入要分析的簡訊內容",
    placeholder="例如：Congratulations! You won a prize...",
    height=120
)

if st.button("✨ 進行預測", use_container_width=True):
    if text_input.strip() == "":
        st.warning("請輸入訊息內容再預測。")
    else:
        clean = clean_text(text_input)
        pred = model.predict([clean])[0]

        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([clean])[0][1]

        st.markdown("### 🎯 預測結果")
        if pred == 1:
            st.error("🔴 **SPAM — 垃圾訊息**")
        else:
            st.success("🟢 **HAM — 正常訊息**")

        if prob is not None:
            st.info(f"📈 垃圾訊息機率：`{prob:.4f}`")


# -----------------------------
# Batch CSV Prediction
# -----------------------------
st.markdown("---")
st.markdown("## 📂 批次 CSV 預測")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    if "text" not in df.columns:
        st.error("CSV 必須有 'text' 欄位。")
    else:
        df["text_clean"] = df["text"].astype(str).apply(clean_text)
        df["pred"] = model.predict(df["text_clean"])

        if hasattr(model, "predict_proba"):
            df["spam_prob"] = model.predict_proba(df["text_clean"])[:, 1]

        st.dataframe(df)

        csv_output = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ 下載預測結果",
            data=csv_output,
            file_name="prediction_results.csv",
            mime="text/csv"
        )


# -----------------------------
# Model Evaluation
# -----------------------------
st.markdown("---")
st.markdown("## 📊 模型效能分析")

# 載入完整資料集
dataset_path = "Chapter03/datasets/sms_spam_no_header.csv"

if not os.path.exists(dataset_path):
    st.error(f"找不到資料集：{dataset_path}")
    st.stop()

df_raw = pd.read_csv(dataset_path, header=None, names=["label", "text"])
df_raw["label"] = df_raw["label"].map({"ham": 0, "spam": 1})
df_raw["text_clean"] = df_raw["text"].apply(clean_text)

y_true = df_raw["label"]
y_pred = model.predict(df_raw["text_clean"])

# Classification Report
st.markdown("### 📄 分類報告")
report = classification_report(y_true, y_pred, target_names=["HAM", "SPAM"])
st.code(report, language="text")


# Confusion Matrix
st.markdown("### 🔵 混淆矩陣")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=["HAM", "SPAM"],
            yticklabels=["HAM", "SPAM"],
            ax=ax)
plt.xlabel("預測")
plt.ylabel("真實")
st.pyplot(fig)


# ROC Curve（若模型支援）
if hasattr(model, "decision_function") or hasattr(model, "predict_proba"):

    st.markdown("### 📈 ROC Curve")

    if hasattr(model, "decision_function"):
        y_score = model.decision_function(df_raw["text_clean"])
    else:
        y_score = model.predict_proba(df_raw["text_clean"])[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)
