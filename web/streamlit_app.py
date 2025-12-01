import streamlit as st
import pandas as pd
import joblib
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 修正 import 問題：加入專案根目錄
# -----------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing import clean_text

# -----------------------------
# Streamlit 頁面設定
# -----------------------------
st.set_page_config(
    page_title="垃圾簡訊分類系統",
    page_icon="📧",
    layout="centered"
)

# -----------------------------
# 頂部標題
# -----------------------------
st.markdown("""
    <h1 style='text-align: center; margin-bottom: 10px;'>📧 垃圾簡訊分類系統</h1>
    <p style='text-align: center; color: #5a5a5a; margin-top: -10px;'>
        使用機器學習模型偵測 SMS 是否為垃圾訊息
    </p>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar 模型選擇
# -----------------------------
with st.sidebar:
    st.title("⚙️ 模型選擇")

    model_paths = {
        "Logistic Regression（邏輯迴歸）": "models/logreg.joblib",
        "Naive Bayes（朴素貝氏）": "models/nb.joblib",
        "Linear SVM（線性 SVM）": "models/svm.joblib"
    }

    model_choice = st.selectbox("選擇模型", list(model_paths.keys()))

# 載入模型
model_path = model_paths[model_choice]
if not os.path.exists(model_path):
    st.error(f"❌ 找不到模型檔案：{model_path}")
    st.stop()

model = joblib.load(model_path)


# ==================================================
# 區塊 1 — 單筆預測
# ==================================================
st.markdown("---")
st.subheader("🔍 單筆訊息預測")

with st.container():
    st.markdown("輸入一段簡訊內容，模型會預測其是否為 **垃圾訊息（SPAM）** 或 **正常訊息（HAM）**。")

    msg = st.text_area(
        "請輸入簡訊內容：",
        height=120,
        placeholder="例如：Congratulations! You won a prize..."
    )

    if st.button("進行預測", use_container_width=True):
        if msg.strip() == "":
            st.warning("⚠️ 請輸入訊息後再預測。")
        else:
            clean = clean_text(msg)
            pred = model.predict([clean])[0]

            st.markdown("### 預測結果")

            if pred == 1:
                st.error("🔴 **SPAM — 垃圾訊息**")
            else:
                st.success("🟢 **HAM — 正常訊息**")

            # 機率（若模型支援）
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba([clean])[0][1]
                st.info(f"📊 垃圾訊息機率：**{prob:.4f}**")


# ==================================================
# 區塊 2 — 批次 CSV 預測
# ==================================================
st.markdown("---")
st.subheader("📂 批次 CSV 預測")

with st.container():
    st.markdown("""
        上傳一份 **CSV 檔案**，需包含欄位：`text`  
        系統會批次預測每一列是否為垃圾簡訊。
    """)

    file = st.file_uploader("上傳 CSV 檔案", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("❌ CSV 檔案必須包含 `text` 欄位。")
        else:
            df["text_clean"] = df["text"].astype(str).apply(clean_text)
            df["pred"] = model.predict(df["text_clean"])

            if hasattr(model, "predict_proba"):
                df["spam_prob"] = model.predict_proba(df["text_clean"])[:, 1]

            st.success("🎉 預測完成")
            st.dataframe(df)

            st.download_button(
                "⬇️ 下載預測結果 CSV",
                df.to_csv(index=False).encode("utf-8-sig"),
                "batch_predictions.csv",
                mime="text/csv"
            )


# ==================================================
# 區塊 3 — 模型評估 Metrics
# ==================================================
st.markdown("---")
st.subheader("📊 模型完整評估結果")

dataset_path = os.path.join("Chapter03", "datasets", "sms_spam_no_header.csv")

if not os.path.exists(dataset_path):
    st.error("❌ 找不到資料集，請確認路徑 Chapter03/datasets/")
    st.stop()

# 載入資料集
df_eval = pd.read_csv(dataset_path, header=None, names=["label", "text"])
df_eval["label"] = df_eval["label"].map({"ham": 0, "spam": 1})
df_eval["text_clean"] = df_eval["text"].apply(clean_text)

y_true = df_eval["label"]
y_pred = model.predict(df_eval["text_clean"])

# 分類報告
st.markdown("### 📄 分類報告")
st.code(classification_report(y_true, y_pred, target_names=["HAM", "SPAM"]), language="text")


# 混淆矩陣
st.markdown("### 🔵 混淆矩陣（Confusion Matrix）")
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["HAM", "SPAM"],
    yticklabels=["HAM", "SPAM"],
    ax=ax
)
st.pyplot(fig)


# ROC 曲線
st.markdown("### 📈 ROC 曲線（ROC Curve）")

if hasattr(model, "decision_function"):
    y_score = model.decision_function(df_eval["text_clean"])
elif hasattr(model, "predict_proba"):
    y_score = model.predict_proba(df_eval["text_clean"])[:, 1]
else:
    y_score = None

if y_score is not None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("此模型不支援 ROC 曲線計算（無 predict_proba / decision_function）。")
