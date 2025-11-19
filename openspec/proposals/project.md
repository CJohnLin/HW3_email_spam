# Project Specification — Email Spam Classification (OpenSpec)

## 1. Project Overview
This project implements an end-to-end **email spam classification system** using a Spec-Driven Development workflow (OpenSpec).  
It is based on Chapter 3 of *Hands-On Artificial Intelligence for Cybersecurity* and enhances it with additional preprocessing, evaluation, visualization, CLI tools, and a Streamlit demonstration site.

The goal is to produce:
- a reproducible machine-learning pipeline,
- multiple classification models (Logistic Regression, Naïve Bayes, SVM),
- interpretable evaluation metrics and visualizations,
- both CLI and Web UI front-ends,
- and a complete GitHub repository using OpenSpec change-proposal workflow.

---

## 2. Scope & Deliverables
### Included
- Data preprocessing  
  - text cleaning  
  - tokenization  
  - vectorization (TF-IDF)
- Model training  
  - Logistic Regression  
  - Naïve Bayes  
  - Support Vector Machine (SVM)
- Model evaluation  
  - accuracy, precision, recall, F1  
  - confusion matrix  
  - ROC curves  
  - summary reports
- Visualizations  
  - metric charts  
  - confusion matrix heatmaps  
  - ROC curves
- CLI tools  
  - train model  
  - evaluate model  
  - predict single email  
  - allow choosing model via flags (`--model lr | nb | svm`)
- Streamlit UI  
  - upload CSV to run predictions  
  - choose model interactively  
  - view metrics and charts  
  - run single-message prediction
- Documentation  
  - README with setup, workflow, and usage instructions  
  - OpenSpec change-proposal workflow files

### Not Included
- Deep-learning based NLP models  
- Real-time email filtering system  
- Database integrations  
- Multi-label classification  
- Network-level spam detection

---

## 3. Tech Stack
### Programming Language
- Python 3.10+

### Core Libraries
- pandas  
- numpy  
- scikit-learn  
- matplotlib / seaborn  
- nltk (optional for tokenization)  
- joblib (save models)

### Front-End
- Streamlit

### Development Workflow
- OpenSpec (Spec-Driven Development)
- GitHub version control
- AI Coding CLI (e.g., Copilot CLI, OpenAI platform, or OpenSpec tools)

---

## 4. Project Structure
2025ML-spamEmail/
│
├── data/
│ └── sms_spam_no_header.csv
│
├── src/
│ ├── preprocessing.py
│ ├── models.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ └── utils.py
│
├── cli/
│ └── spam_cli.py
│
├── streamlit_app/
│ └── app.py
│
├── openspec/
│ ├── project.md ←(this file)
│ └── AGENTS.md ←(OpenSpec agent workflow)
│
├── models/
│ ├── lr_model.pkl
│ ├── nb_model.pkl
│ ├── svm_model.pkl
│ └── vectorizer.pkl
│
└── README.md

---

## 5. Conventions
### Code Style
- PEP 8
- Type hints required for all functions
- All ML steps modularized into functions

### File Naming
- snake_case for Python files
- lower_case_for_models.pkl

### Git Workflow
- Each new feature requires an OpenSpec Change Proposal  
- Branch name: `feature/<feature-name>`  
- Commit messages follow:
[spec] Added preprocessing feature
[impl] Implemented vectorizer and tokenizer

---

## 6. Requirements
### Functional Requirements
- MUST load dataset and preprocess text  
- MUST train at least three models (LR, NB, SVM)  
- MUST save trained models to `/models`  
- MUST provide metrics + confusion matrix  
- MUST render Streamlit UI  
- MUST provide CLI commands:
- `train`
- `evaluate`
- `predict`
- `--model <lr|nb|svm>`

### Non-Functional Requirements
- Reproducible (random seed fixed)  
- Train time < 30 seconds on common CPU  
- Model files < 50 MB  
- Streamlit must run on free tier

---

## 7. Personas (OpenSpec)
### Developer Agent
- Writes modular Python code following spec  
- Ensures ML models train successfully  
- Produces CLI and Streamlit UI  
- Uses change proposals for new features

### Reviewer Agent
- Validates spec accuracy  
- Ensures repo consistency  
- Checks implementation vs specification  
- Approves or requests changes on proposals

### User
- Runs Streamlit site  
- Uploads CSV or enters message for prediction  
- Selects model  
- Views evaluation charts

---

## 8. Risks & Mitigation
| Risk | Mitigation |
|------|------------|
| Data imbalance | use stratified split, show metrics |
| Overfitting | simplify preprocessing, tune parameters |
| Streamlit memory limit | restrict dataset size |
| Model size or import errors | use joblib + simple models |

---

## 9. Success Criteria
A complete score requires:
- GitHub repo with OpenSpec workflow  
- Fully working model pipeline  
- CLI + Streamlit both functional  
- README fully documented  
- Clear charts and evaluation outputs  
- Demo site publicly accessible

---

## 10. Future Improvements (Out-of-Scope)
- Attention-based NLP models (BERT, LSTM)  
- Real-time email gateway integration  
- Explainability with SHAP  
- Multilingual spam detection  
