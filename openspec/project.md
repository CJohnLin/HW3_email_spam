# Homework 3 — Email / SMS Spam Classification (OpenSpec project.md)

## Project summary
This project implements a full spam classification machine learning pipeline using the OpenSpec (Spec-Driven Development) workflow. It reproduces and extends Chapter 3 of *Hands-On Artificial Intelligence for Cybersecurity* with additional preprocessing, evaluation, and visualization features.

## Goals
- Build a reproducible ML pipeline (data load → preprocess → vectorize → train → evaluate).
- Compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
- Provide interactive Streamlit UI for predictions & metrics.
- Publish GitHub repo with OpenSpec workflow docs and a working Streamlit demo.

## Dataset
Dataset from textbook:
`Chapter03/datasets/sms_spam_no_header.csv`

Columns: `label`, `text`.

## Tech stack
- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib / seaborn / plotly
- Streamlit
- joblib

## Project structure
- `src/` code modules
- `notebooks/` for experiments
- `web/` for Streamlit
- `openspec/` OpenSpec documentation
- `models/` exported trained models

## Deliverables
- Functional ML pipeline
- Trained models exported in `/models`
- Streamlit deployed demo
- Notebook
