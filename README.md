# 2025ML-spamEmail — Spam Classification (OpenSpec + Streamlit + ML Pipeline)

This project implements an end-to-end **spam classification system** using:
- Scikit-learn (LR / NB / SVM)
- TF-IDF vectorizer
- CLI interface (training, prediction, evaluation)
- Streamlit demo web app
- OpenSpec specification & proposal workflow (required for course)

---

# 📌 1. Project Structure

2025ML-spamEmail/
│
├── data/ # Dataset (CSV with text,label)
│ └── sms_spam_no_header.csv
│
├── models/ # Saved ML models
│ ├── lr_model.pkl
│ ├── nb_model.pkl
│ ├── svm_model.pkl
│ └── vectorizer.pkl
│
├── src/ # ML pipeline implementation
│ ├── preprocessing.py
│ ├── models.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ └── utils.py
│
├── cli/
│ └── spam_cli.py # Command-line interface
│
├── streamlit_app/
│ └── app.py # Web demo (Streamlit)
│
└── openspec/
├── project.md # Project specification (Step 1)
├── AGENTS.md # Step 3 (Agent workflow)
└── proposals/
└── 001-model-selection.md # Step 2 Proposal

---

# 📌 2. Installation

### Create environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
📌 3. Dataset Requirements

Dataset must be a CSV with the following columns:

column	description
text	email/sms content
label	spam/ham OR 1/0
Example:
text,label
"Congratulations! You won a prize",spam
"Hello, are we still meeting?",ham
📌 4. Train Models
Train all models (LR, NB, SVM)
python -m src.train --dataset data/sms_spam_no_header.csv --model all
Train a specific model
python -m src.train --dataset data/sms_spam_no_header.csv --model lr
Models will be saved to:
models/
    lr_model.pkl
    nb_model.pkl
    svm_model.pkl
    vectorizer.pkl
📌 5. Use CLI
Predict a single text
python cli/spam_cli.py predict --model lr --text "Free prize now!!!"
Evaluate model on CSV
python cli/spam_cli.py evaluate --model nb --csv data/sms_spam_no_header.csv
List saved models
python cli/spam_cli.py list
📌 6. Run Streamlit Web App
streamlit run streamlit_app/app.py
Features include:

Single-message prediction

CSV batch evaluation

Interactive model selection (LR / NB / SVM)

Confusion matrix heatmap

Display of precision, recall, F1, AUC
📌 7. OpenSpec Workflow (Required by Course)

This project uses the Spec-Driven Development method:

Step 1 — Project Spec

openspec/project.md

Step 2 — Change Proposal

openspec/proposals/001-model-selection.md

Step 3 — Agent Workflow

openspec/AGENTS.md

Step 4 — Implementation

Source code in src/, cli/, streamlit_app/

Every new feature must include:

A proposal file

Reviewer approval

Implementation

Merge
