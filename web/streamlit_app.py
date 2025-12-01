import streamlit as st
import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ====== 修正 import 路徑 ======
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing import clean_text


# ====== Streamlit 頁面設定 ======
st.set_page_config(
    page_title="垃圾簡訊分類系統",
    page_icon="📧",
    layout="centered"
)

# ====== 自訂 CSS（簡潔風格，不抄原版） ======
st.markdown("""
<style>
/* 主標題樣式 */
.main-title {
    font-size: 2.2rem;
    text-align: center;
    font-weight: 600;
    color: #22577A;
    margin-bottom: 0.5rem;
}

/* 副標題 */
.sub-title {
    text-align: center;
    color: #555;
    margin-top: -10px;
    margin-bottom: 30px;
}

/* 卡片容器 */
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f8f9fa;
    border: 1px solid #e0e0e0;
    margin-bottom: 25px;
}

/* 分隔線 */
.section-divider {
    margin: 30px 0;
    border-top: 1px solid #ddd;
}

/* 頁腳 */
.footer {
    text-align: center;
    font-size: 0.9rem;
    color: #777;
    padding: 1rem 0 0 0;
}
</style>
""", unsafe_allow_html=True)


# ====== 頁面標題 ======
st.markdown("<h1 class='main-title'>📧 垃圾簡訊分類系統</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>使用機器學習模型辨識 SMS 是否為垃圾訊息</p>", unsafe_allow_html=True)


# ====== Sidebar（模型切換） ======
with st.sidebar:
    st.header("⚙️ 模型選擇")

    model_paths = {
        "Logistic Regression（邏輯迴歸）": "models/logreg.joblib",
        "Naive Bayes（朴素貝氏）": "models/nb.joblib",
        "Linear SVM（線性 SVM）": "models/svm.joblib"
    }

    model_choice = st.selectbox("選擇模型", list(model_paths.keys()))

model_path = model_paths[model_choice]

if not os.path.exists(model_path):
    st.error(f"❌ 找不到模型檔案：{model_path}")
    st.stop()

model = joblib.load(model_path)


# ==========================================================
# 🟦 區塊 1：單筆預測
# ==========================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🔍 單筆訊息預測")
st.write("輸入簡訊內容，系統會自動判斷是否為垃圾訊息（SPAM）。")

text_input = st.text_area("請輸入訊息內容：", placeholder="例如：Congratulations! You won a free ticket...", height=120)

if st.button("進行預測", use_container_width=True):
    if text_input.strip() == "":
        st.warning("⚠️ 請輸入訊息內容！")
    else:
        clean = clean_text(text_input)
        pred = model.predict([clean])[0]

        st.markdown("### 預測結果：")
        if pred == 1:
            st.error("🔴 **SPAM — 垃圾訊息**")
        else:
            st.success("🟢 **HAM — 正常訊息**")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([clean])[0][1]
            st.info(f"📊 垃圾訊息機率：**{prob:.4f}**")

st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# 🟦 區塊 2：批次 CSV 預測
# ==========================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📂 批次 CSV 預測")
st.write("上傳一份包含 `text` 欄位的 CSV 檔案，系統將較大量訊息一次分類。")

uploaded_file = st.file_uploader("上傳 CSV 檔案：", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("❌ CSV 檔案內必須包含 `text` 欄位！")
    else:
        df["text_clean"] = df["text"].apply(clean_text)
        df["pred"] = model.predict(df["text_clean"])

        if hasattr(model, "predict_proba"):
            df["spam_prob"] = model.predict_proba(df["text_clean"])[:, 1]

        st.success("🎉 預測完成！")
        st.dataframe(df)

        st.download_button(
            "⬇️ 下載預測結果（CSV）",
            df.to_csv(index=False).encode("utf-8-sig"),
            "prediction_results.csv",
            mime="text/csv"
        )

st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# 🟦 區塊 3：模型效能評估
# ==========================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📊 模型效能評估")

dataset_path = os.path.join("Chapter03", "datasets", "sms_spam_no_header.csv")

if not os.path.exists(dataset_path):
    st.error("❌ 找不到資料集，請確認路徑是否正確：`Chapter03/datasets/`")
else:
    df_eval = pd.read_csv(dataset_path, header=None, names=["label", "text"])
    df_eval["label"] = df_eval["label"].map({"ham": 0, "spam": 1})
    df_eval["text_clean"] = df_eval["text"].apply(clean_text)

    y_true = df_eval["label"]
    y_pred = model.predict(df_eval["text_clean"])

    st.markdown("### 📄 分類報告")
    st.code(classification_report(y_true, y_pred, target_names=["HAM", "SPAM"]), language="text")

    st.markdown("### 🔵 混淆矩陣")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.markdown("### 📈 ROC 曲線")
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
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("此模型不支援 ROC 計算。")

st.markdown("</div>", unsafe_allow_html=True)


# ====== 頁腳 ======
st.markdown("<div class='footer'>垃圾簡訊分類系統 © 2025</div>", unsafe_allow_html=True)
