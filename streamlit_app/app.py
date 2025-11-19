# streamlit_app/app.py
import sys
import os

# Add project root to PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)


import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
from src.predict import predict_text
from src.evaluate import load_vectorizer_and_model, evaluate_on_csv
from src.utils import list_models
import joblib

st.set_page_config(page_title="Spam Classifier Demo", layout="centered")

st.title("Email/SMS Spam Classifier — Demo")
st.markdown("Upload dataset or enter a single message to classify. Choose model and view metrics.")

models_dir = st.sidebar.text_input("Models directory", value="models")
available = ["lr", "nb", "svm"]
model_choice = st.sidebar.selectbox("Model", available, index=0)

st.sidebar.markdown("---")
st.sidebar.write("Saved models:")
for m in list_models(models_dir):
    st.sidebar.write(f"- {m}")

st.header("Single message prediction")
text_input = st.text_area("Enter message here", height=120)
if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter a message.")
    else:
        try:
            pred, score = predict_text(models_dir, model_choice, text_input)
            st.success(f"Prediction: {'SPAM' if pred==1 else 'HAM'} (score: {score:.4f})")
        except Exception as e:
            st.error(f"Error predicting: {e}")

st.header("Batch evaluation (CSV)")
uploaded = st.file_uploader("Upload CSV (columns: text,label)", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
        if st.button("Evaluate on uploaded CSV"):
            # save temp to disk for evaluate helper
            tmp_path = "tmp_uploaded.csv"
            df.to_csv(tmp_path, index=False)
            res = evaluate_on_csv(models_dir, model_choice, tmp_path)
            st.text("Classification Report:")
            st.text(res["report"])
            cm = res["confusion_matrix"]
            st.write("Confusion matrix:")
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, int(val), ha='center', va='center')
            st.pyplot(fig)
            if res["auc"] is not None:
                st.write(f"AUC: {res['auc']:.4f}")
            else:
                st.write("AUC: N/A")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
