# Change Proposal 001 — Add Model Selection to CLI & Streamlit

## Metadata
- Proposal ID: CP-001
- Title: Add `--model` selection (lr / nb / svm) to CLI and Streamlit UI
- Author: Developer Agent
- Reviewer: Reviewer Agent
- Status: Proposed
- Spec Version: project.md v1.0
- Created: 2025-11-19

---

## 1. Summary
This proposal adds **model selection functionality** to both the CLI interface and the Streamlit application.  
Users will be able to select which trained model (Logistic Regression, Naïve Bayes, SVM) to use for predictions or evaluations.

This feature aligns with the course requirement to integrate multiple models and provides flexible workflows for both command-line and web-based usage.

---

## 2. Motivation
The current project only trains models but does not provide a user-friendly way to switch between them.  
Adding model selection:

- makes evaluation more interactive,  
- allows end-users to compare metrics,  
- and enables switching models in real-time without retraining.

This improves usability and fulfills the Deliverables:  
> *“Model training (Logistic Regression / Naïve Bayes / SVM) + Visualization and Streamlit UI”*

---

## 3. Requirements

### Functional Requirements
1. CLI must allow a parameter:
   ```
   --model <lr | nb | svm>
   ```
2. Streamlit UI must include a model-selection widget:
   - dropdown / radio button
3. All predictions must load the correct model and vectorizer.
4. Streamlit must dynamically update:
   - metrics
   - confusion matrix
   - ROC curve
5. CLI commands supporting model choice:
   ```
   python cli/spam_cli.py train --model lr
   python cli/spam_cli.py evaluate --model nb
   python cli/spam_cli.py predict --model svm --text "hello"
   ```

### Non-Functional Requirements
- Model loading must be fast (< 1s)
- Streamlit UI must show validation when model files are missing
- Must store models in `/models/`

---

## 4. Implementation Plan

### 4.1 Code Modules to Update
| File | Change |
|------|--------|
| `src/train.py` | Ensure saving all three models |
| `src/predict.py` | Add model loading logic |
| `src/evaluate.py` | Evaluate selected model |
| `cli/spam_cli.py` | Add `--model` parameter to each command |
| `streamlit_app/app.py` | Add model selector + dynamic charts |

---

### 4.2 Model Loading Logic
Create a helper function:

```python
def load_model(model_name: str):
    model_path = f"models/{model_name}_model.pkl"
    vectorizer_path = "models/vectorizer.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
```

---

### 4.3 CLI Commands
Example:

```python
@click.option("--model", default="lr", type=click.Choice(["lr", "nb", "svm"]))
def predict(model, text):
    model, vectorizer = load_model(model)
    # run prediction
```

---

### 4.4 Streamlit UI Additions

- A dropdown:
```python
model_name = st.selectbox("Choose model", ["lr", "nb", "svm"])
```

- When model selection changes:
  - reload model  
  - re-render metrics  
  - update charts  

---

## 5. Acceptance Criteria

### ✔ CLI
- `train`, `predict`, `evaluate` all accept `--model`  
- invalid model names return safe errors

### ✔ Streamlit
- user can switch models  
- metrics update correctly  
- charts refresh with correct predictions  

### ✔ Models
- all models saved correctly into `/models`  

---

## 6. Risks
| Risk | Mitigation |
|------|------------|
| User selects model before training | Disable UI elements or show warning |
| Missing model file | Validate existence before loading |
| Streamlit slow on large CSV | Add size checks |

---

## 7. Alternatives Considered
- Auto-detect best model (not required by assignment)
- Ensemble model (out of scope)
- BERT-based text classifier (too heavy, not required)

---

## 8. Final Recommendation
Approve this change proposal and proceed to implementation.  
This feature directly satisfies class requirements and improves usability for both CLI and Streamlit interfaces.
