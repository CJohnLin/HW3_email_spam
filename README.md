# 2025ML-spamEmail — Spam Classification (OpenSpec + Streamlit + ML Pipeline)

## 🌐 Streamlit Demo (Click to Open)
👉 https://hw3emailspam-qwkwfgqzaiqg9ezjxkut42.streamlit.app/


This project implements an end-to-end **spam classification system** featuring:

- Scikit-learn models (Logistic Regression / Naïve Bayes / SVM)
- TF-IDF text vectorization
- CLI interface for training, evaluation, prediction
- Streamlit interactive demo web application
- OpenSpec-driven workflow (Project Spec → Proposal → Agent Workflow → Implementation)

This project is structured based on the course design requirements and follows a clean, modular ML development pipeline.

---

# 📁 1. Project Structure

```
2025ML-spamEmail/
│
├── data/                         # Dataset (CSV with text,label)
│   └── sms_spam_no_header.csv
│
├── models/                       # Saved ML models
│   ├── lr_model.pkl
│   ├── nb_model.pkl
│   ├── svm_model.pkl
│   └── vectorizer.pkl
│
├── src/                          # Core ML pipeline modules
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── cli/                          # CLI tool entrypoint
│   └── spam_cli.py
│
├── streamlit_app/                # Streamlit demonstration application
│   └── app.py
│
└── openspec/                     # OpenSpec workflow files
    ├── project.md
    ├── AGENTS.md
    └── proposals/
        └── 001-model-selection.md
```

---

# ⚙️ 2. Installation

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🗂️ 3. Dataset Requirements

Your dataset must be a CSV with **two mandatory columns**:

| column | description              |
|--------|---------------------------|
| text   | Email/SMS content        |
| label  | spam/ham OR 1/0 values   |

### Example

```
text,label
"Congratulations! You won a prize",spam
"Hello, are we still meeting?",ham
```

---

# 🧠 4. Train Models

### Train all three models (LR, NB, SVM)

```bash
python -m src.train --dataset data/sms_spam_no_header.csv --model all
```

### Train a specific model

```bash
python -m src.train --dataset data/sms_spam_no_header.csv --model lr
```

Models will be saved to:

```
models/
    lr_model.pkl
    nb_model.pkl
    svm_model.pkl
    vectorizer.pkl
```

---

# 🖥️ 5. CLI Usage

### Predict a single message

```bash
python cli/spam_cli.py predict --model lr --text "Free prize now!!!"
```

### Evaluate a model on a CSV dataset

```bash
python cli/spam_cli.py evaluate --model nb --csv data/sms_spam_no_header.csv
```

### List available models

```bash
python cli/spam_cli.py list
```

---

# 🌐 6. Run Streamlit Web Application

Launch the interactive demo:

```bash
streamlit run streamlit_app/app.py
```

Streamlit features:

- Single-message spam prediction  
- CSV batch evaluation  
- Confusion matrix heatmap  
- AUC score display  
- Ability to switch between LR / NB / SVM models  

---

# 📘 7. OpenSpec Workflow (Required by Course)

This project follows a complete OpenSpec development cycle.

### Step 1 — Project Specification  
`openspec/project.md`

### Step 2 — Change Proposal  
`openspec/proposals/001-model-selection.md`

### Step 3 — Agent Workflow  
`openspec/AGENTS.md`

### Step 4 — Implementation  
Source code inside `src/`, `cli/`, and `streamlit_app/`

### Every new feature must include:

- A proposal file  
- Reviewer approval  
- Implementation matching the proposal  
- Merge after validation  

---

# 📌 8. Future Improvements

Potential extensions beyond the scope of this course:

- Transformer-based spam classifier (BERT)
- SHAP explainability
- Real-time spam filtering service
- LSTM sequence model
- Web deployment (Railway, HuggingFace Spaces)

---

# 👤 9. Author

2025 Machine Learning Coursework  
Student: **5114056042林佳宏**
