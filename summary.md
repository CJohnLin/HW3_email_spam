# HW3 — Email Spam Classification  
Student: 5114056042林佳宏  
Repository: https://github.com/CJohnLin/HW3_email_spam  

---

## 1. Project Overview
This project implements a full spam email classification pipeline, including:

- TF-IDF preprocessing
- Logistic Regression, Naïve Bayes, and SVM models
- CLI interface for training / predicting / evaluating
- Streamlit interactive demonstration website
- OpenSpec-driven workflow (Project Spec → Proposal → Agents → Implementation)

All components required in the assignment are implemented and version-controlled on GitHub.

---

## 2. OpenSpec Workflow (Required by Assignment)
The repository includes all OpenSpec files:

- openspec/project.md  
- openspec/AGENTS.md  
- openspec/proposals/001-model-selection.md  

These files document:
- Project context  
- Change proposal  
- Developer/Reviewer workflow  
- Traceable ML development cycle

---

## 3. Program Components
### Source Code
Located under:

- `/src`  
- `/cli`  
- `/streamlit_app`  

Includes:
- Preprocessing  
- Model training  
- Prediction  
- Evaluation  
- Utilities  
- CLI entrypoint  
- Streamlit UI  

All code is modular and reproducible.

---

## 4. Features Implemented
- Train 3 ML models (LR / NB / SVM)
- Save trained models to /models
- Predict single message via CLI
- Evaluate dataset via CLI
- Streamlit web interface for:
  - Model switching
  - Single prediction
  - Batch CSV evaluation
  - Confusion matrix
  - AUC score

---

## 5. How to Run
### Train models
