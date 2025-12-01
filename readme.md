# 作業三 — Email / SMS 垃圾郵件分類系統

本專案依照 OpenSpec（Spec-Driven Development）流程，實作完整的垃圾郵件分類機器學習管線，包含資料前處理、特徵擷取、模型訓練、模型比較、視覺化與 Streamlit 互動式介面。

---

## 🚀 專案特色
- 資料前處理（清理、標準化文字內容）
- TF-IDF 特徵向量化
- 三種機器學習模型：
  - Logistic Regression（邏輯迴歸）
  - Multinomial Naive Bayes（多項式朴素貝氏）
  - Linear SVM（線性支援向量機）
- 評估指標：
  - Precision / Recall / F1-score
  - Confusion Matrix（混淆矩陣）
  - ROC Curve（ROC 曲線）
- Streamlit 互動式 Demo
- 完整 OpenSpec 專案文件（project.md + proposals）

---

## 📁 專案結構
```
/openspec
    project.md
    AGENTS.md
    proposals/
        0001-add-ml-pipeline.md
/src
    preprocessing.py
    train.py
    evaluate.py
    predict.py
/web
    streamlit_app.py
/notebooks
    experiments.ipynb
/models
    (訓練後的模型將儲存在此)
requirements.txt
README.md
```

---

## 📊 資料集
教材 Chapter 3 來源：
```
Chapter03/datasets/sms_spam_no_header.csv
```
欄位：
- `label`：ham（正常）或 spam（垃圾）
- `text`：簡訊內容

---

## ▶ 專案執行方式

### 1. 安裝環境
```
pip install -r requirements.txt
```

### 2. 訓練模型
```
python src/train.py
```
訓練後模型會自動儲存於：
```
/models/logreg.joblib
/models/nb.joblib
/models/svm.joblib
```

### 3. 執行 Streamlit Demo
```
streamlit run web/streamlit_app.py
```

---

## 📓 分析 Notebook
`notebooks/experiments.ipynb` 內包含：
- 資料前處理範例
- 模型訓練流程
- 多模型指標比較（Precision / Recall / F1）
- 混淆矩陣視覺化（heatmap）
- ROC 曲線

---

## 🧪 OpenSpec Workflow 文件
###（作業繳交必要項目）
- `openspec/project.md`
- `openspec/proposals/0001-add-ml-pipeline.md`
- `openspec/AGENTS.md`

---

## 📝 繳交檢查清單
- [x] GitHub 專案公開
- [x] 包含 OpenSpec 文件
- [x] Streamlit Demo 可執行
- [x] Notebook 評估內容完整（混淆矩陣 / ROC）
- [x] requirements.txt & README.md 完整
