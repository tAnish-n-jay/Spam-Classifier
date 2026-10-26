# Spam Mail Classifier — Logistic Regression from Scratch
> Binary SMS classification using hand-coded gradient descent | NumPy only | No sklearn for the core model

---

## Overview

A spam SMS classifier built entirely from scratch using NumPy. Every mathematical step is implemented by hand — sigmoid activation, binary cross-entropy loss, gradient derivation using the chain rule, and the gradient descent weight update loop.

---

## The Math

**Sigmoid Function**
```
sigma(z) = 1 / (1 + e^(-z))
```

**Binary Cross-Entropy Loss**
```
L = -1/n * sum[ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]
```

**Gradient of Loss (Chain Rule)**
```
dL/dw = 1/n * X^T * (y_hat - y)
dL/db = 1/n * sum(y_hat - y)
```

**Gradient Descent Update Rule**
```
w = w - lr * dL/dw
b = b - lr * dL/db
```

---

## Dataset

**SMS Spam Collection**
- Source: [UCI ML Repository via Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,572 SMS messages labeled spam / ham
- Class distribution: 86.6% ham, 13.4% spam
- Handled with class weighting during training

---

## Features Extracted

| Feature | Description |
|---|---|
| `msg_length` | Total character count |
| `word_count` | Number of words |
| `digit_count` | Number of digit characters |
| `special_char_count` | Count of !, $, ? characters |
| `uppercase_ratio` | Ratio of uppercase to total letters |
| `spam_keywords` | Count of words: free, win, prize, claim, urgent, cash, offer... |

---

## Results

| Experiment | Learning Rate | Final Loss | Test Accuracy |
|---|---|---|---|
| Too slow | 0.001 | 0.4183 | 85.29% |
| Balanced | 0.01 | 0.2227 | 94.08% |
| Best | 0.1 | 0.1568 | 96.14% |
| Sklearn baseline | — | — | 96.59% |

The from-scratch implementation at lr=0.1 achieves 96.14% accuracy — only 0.45% below sklearn's optimised implementation, validating the correctness of the hand-coded gradient descent.

---

## Key Findings

- `digit_count` is the strongest predictor (weight: 2.99) — spam messages embed phone numbers and shortcodes
- `spam_keywords` follows (weight: 1.21) — promotional vocabulary is a reliable signal
- lr=0.001 converges too slowly — loss still at 0.42 after 1000 epochs
- lr=0.1 converges fast and stable — logistic regression has a convex loss surface so high learning rates do not cause divergence
- The gap between from-scratch (96.14%) and sklearn (96.59%) is negligible — confirms implementation correctness

---

## Project Structure

```
spam-classifier/
│
├── data/
│   └── spam.csv                         # Raw dataset — download from Kaggle
│
├── notebooks/
│   ├── 01_eda_feature_extraction.ipynb  # EDA + feature engineering
│   ├── 02_logistic_regression.ipynb     # From-scratch model + training
│   └── 03_experiments_visuals.ipynb     # Learning rate experiments + plots
│
├── src/
│   └── logistic_regression.py           # Reusable model class
│
├── app/
│   └── streamlit_app.py                 # Streamlit UI
│
├── models/
│   └── logistic_model.pkl               # Saved weights + scaler
│
├── results/
│   ├── class_balance.png
│   ├── feature_distributions.png
│   ├── feature_correlation.png
│   ├── loss_curve.png
│   ├── loss_curve_lr_comparison.png
│   ├── accuracy_vs_lr.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── decision_boundary.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/tAnish-n-jay/spam-classifier.git
cd spam-classifier
pip install -r requirements.txt
```

Download `spam.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in `data/`.

Run notebooks in order:
```
01 → 02 → 03
```

Start the Streamlit app:
```bash
cd app
streamlit run streamlit_app.py
```

---

## Tech Stack

```
Language   : Python 3.10+
Core model : NumPy (from scratch — no sklearn for training)
Data       : Pandas
Plots      : Matplotlib, Seaborn
Baseline   : Scikit-learn (comparison only)
UI         : Streamlit
Notebook   : Jupyter in VS Code
```

---

## What This Project Teaches

- Gradient descent is the same algorithm powering every ML model — only the loss function changes
- Learning rate is the most critical hyperparameter — too small converges slowly, too large may diverge
- Logistic regression is mathematically a single neuron — the foundation of neural networks
- The gradient of binary cross-entropy with sigmoid simplifies cleanly to (y_hat - y) via the chain rule
- Class imbalance must be handled explicitly — raw accuracy is misleading on skewed datasets

---

## Author

**Tanish N Jay**