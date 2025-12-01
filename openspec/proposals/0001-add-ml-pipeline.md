# Change Proposal 0001 — Add data pre-processing and evaluation extensions

## Title
Add data pre-processing steps, more evaluation metrics/plots, and Streamlit UI.

## Rationale
Current baseline in Chapter 3 uses minimal preprocessing. To improve model robustness and meet assignment evaluation, we will:
- add text cleaning (lowercase, punctuation removal), tokenization, stopword removal, lemmatization (optional),
- use TF-IDF vectorization and optionally n-grams,
- train LogisticRegression, MultinomialNB, and LinearSVC with hyperparameter grid search,
- produce confusion matrix, ROC curves, precision-recall curves, and interactive Streamlit view.

## Implementation plan
1. Add `src/preprocessing.py` with `clean_text()` and `preprocess_df()` functions.
2. Add `notebooks/experiment.ipynb` with training and evaluation.
3. Add `src/predict.py` for CLI predictions.
4. Add `web/streamlit_app.py` for interactive demo.
5. Update `README.md` and `requirements.txt`.
6. Commit OpenSpec trace documenting each step.

## Acceptance criteria
- Reproducible notebook producing saved models.
- Streamlit app supports model selection, threshold tuning, and example prediction.
- Project repository passes submission checklist.

## Trace
(attach commit hashes and AGENTS trace after implementation)
