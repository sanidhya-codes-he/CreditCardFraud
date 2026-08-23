# Credit Card Fraud Detection

A **Machine Learning project** that explores credit card transaction data and builds a predictive model to detect **fraudulent transactions**.

The project includes:
-  Exploratory Data Analysis (EDA)
-  Machine Learning models for fraud detection
-  Performance evaluation of classification algorithms

---

# Dataset

The dataset is sourced from **OpenML** and originates from a well-known credit card fraud detection dataset.

It contains **transactions made by European cardholders in September 2013**. The dataset covers **two days of transactions**, including both legitimate and fraudulent activities.

⚠️ **Class Imbalance Notice**

Fraud cases are extremely rare:

- Total transactions: **284,807**
- Fraudulent transactions: **492**
- Fraud percentage: **0.172%**

This project uses a **10% stratified sample** of the dataset to maintain the same class distribution.

Keeping this in mind, false frauds detected would not be a harm but if a false legitimate transaction is a hazard.

---

# Dataset Features

The dataset contains **31 features** including the target variable.

| Feature Name | Type | Distinct Values | Missing Attributes |
|---------------|---------|----------------|-------------------|
| class (target) | nominal | 2 | 0 |
| time | numeric | 25763 | 0 |
| v1 | numeric | 28271 | 0 |
| v2 | numeric | 28271 | 0 |
| v3 | numeric | 28271 | 0 |
| v4 | numeric | 28271 | 0 |
| v5 | numeric | 28271 | 0 |
| v6 | numeric | 28271 | 0 |
| v7 | numeric | 28271 | 0 |
| v8 | numeric | 28271 | 0 |
| v9 | numeric | 28271 | 0 |
| v10 | numeric | 28271 | 0 |
| v11 | numeric | 28271 | 0 |
| v12 | numeric | 28271 | 0 |
| v13 | numeric | 28271 | 0 |
| v14 | numeric | 28271 | 0 |
| v15 | numeric | 28271 | 0 |
| v16 | numeric | 28271 | 0 |
| v17 | numeric | 28271 | 0 |
| v18 | numeric | 28271 | 0 |
| v19 | numeric | 28271 | 0 |
| v20 | numeric | 28271 | 0 |
| v21 | numeric | 28271 | 0 |
| v22 | numeric | 28271 | 0 |
| v23 | numeric | 28271 | 0 |
| v24 | numeric | 28271 | 0 |
| v25 | numeric | 28271 | 0 |
| v26 | numeric | 28271 | 0 |
| v27 | numeric | 28271 | 0 |
| v28 | numeric | 28271 | 0 |
| amount | numeric | 8807 | 0 |

---

# Dataset Characteristics

| Quality Name | Value |
|---------------|-------|
| number of instances | 28480 |
| number of features | 31 |
| number of classes | 2 |
| number of missing values | 0 |
| number of instances with missing values | 0 |
| number of numeric features | 30 |
| number of symbolic features | 1 |
| percentage of binary features | 3.2258 |
| percentage of instances with missing values | 0 |
| percentage of missing values | 0 |
| auto correlation | 0.9966 |
| percentage of numeric features | 96.77 |
| dimensionality | 0.00109 |
| percentage of symbolic features | 3.2258 |
| majority class percentage | 99.8279 |
| majority class size | 28431 |
| minority class percentage | 0.17205 |
| minority class size | 49 |
| number of binary features | 1 |

---

# Project Goal

The main goal of this project is to **detect fraudulent credit card transactions** using machine learning.

Fraud detection is essential for financial institutions to ensure that **customers are not charged for transactions they did not make**.

The dataset represents a **binary classification problem**:

| Class | Meaning |
|------|--------|
| 0 | Legitimate Transaction |
| 1 | Fraudulent Transaction |

---

# Machine Learning Approach

Since the problem is a **classification task**, we will evaluate several classification algorithms and select the best-performing model.

### Steps in the project

1️⃣ Data Exploration  
- Understand feature distributions  
- Identify patterns in fraudulent transactions  

2️⃣ Exploratory Data Analysis (EDA)  
- Visualize transaction patterns  
- Analyze class imbalance  

3️⃣ Model Training  
Possible algorithms include:
- Logistic Regression
- Random Forest
- Decision Trees
- Gradient Boosting
- Support Vector Machines

4️⃣ Model Evaluation  
Models will be compared using metrics such as:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

5️⃣ Model Selection  
The model with the **best performance for fraud detection** will be selected.

---

# Challenges

This dataset presents several challenges:

-  **Severe class imbalance**
-  Fraud transactions are extremely rare
-  Standard accuracy metrics can be misleading

Therefore, evaluation will focus on **precision, recall, and ROC-AUC**, rather than accuracy alone.

---

# REST API

The trained model is served over HTTP by a **FastAPI** application, so the fraud
detector can be called from anything — a website, a mobile app, or `curl` —
rather than only from inside a notebook.

```bash
pip install -r requirements.txt
python train.py                          # writes artifacts/
python -m uvicorn app.main:app --reload  # serves on http://127.0.0.1:8000
```

Interactive documentation is generated automatically at **`/docs`**, pre-filled
with a real fraudulent transaction so every endpoint can be tried in the browser.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness and model status |
| `POST` | `/predict` | Score one transaction |
| `POST` | `/predict/batch` | Score up to 1000 transactions |
| `GET` | `/model/info` | Model type, training details, held-out metrics |
| `GET` | `/stats` | Summary of the training dataset |

Full documentation, including the design decisions behind the API, is in
**[API_README.md](API_README.md)**.

---

# ⚠️ Known data issue: the CSV header is shifted by one column

`credit_card_fraud.csv` ships with a header row that does not line up with the
columns beneath it:

| Header says | Column actually holds |
|---|---|
| `V1` | **Time** — seconds elapsed, 0 to 172,768 (a 48-hour window) |
| `V2` … `V29` | **V1 … V28** — the 28 PCA components |
| `time` | **Amount** — 0.00 to 8,360.00, mean 86.78 |
| `fraud` | The class label, as the strings `fraud` / `otherwise` |

Confirmed two ways: the first column's range and cardinality are those of a
timestamp, not a PCA component; and ranking columns by class separation puts
`V15, V13, V5, V12, V11` on top, which shifted down by one gives
`V14, V12, V4, V11, V10` — the known top discriminators of this dataset.

**The CSV is left untouched.** The correction is applied on load in
`app/config.py`, so the API trains on correctly named features. The notebooks
still read the file as before, which means **feature names in the EDA notebook
are off by one** — the chart labelled `Amount (V1)` is in fact plotting *Time*.
The models themselves are unaffected, since a classifier does not care what a
column is called.

---
