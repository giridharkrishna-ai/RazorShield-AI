
# RazorShield AI 🛡️

## Intelligent Payment Fraud Risk Detection & Decision Engine

RazorShield AI is a defensive AI prototype designed to detect potentially fraudulent payment transactions and convert fraud probability into an actionable risk decision.

The system analyzes transaction behavior, calculates a fraud probability, assigns a risk level, recommends an action, and explains the factors contributing to the decision.

---

## 🎯 Problem Statement

Payment fraud can cause financial losses and customer trust issues.

A fraud detection system should not only identify suspicious transactions but also provide:

- Fraud probability
- Risk classification
- Recommended action
- Explainable reasons
- Measurable model performance

---

## 💡 Solution

RazorShield AI uses a Machine Learning based fraud detection pipeline.

### Workflow

Transaction Data
↓
Data Preprocessing
↓
Random Forest Classifier
↓
Fraud Probability
↓
Risk Score
↓
LOW / MEDIUM / HIGH
↓
Recommended Action
↓
Explainable AI Reasons

---

## 🤖 AI Model

### Algorithm
Random Forest Classifier

### Input Features

- Transaction amount
- Transaction hour
- Customer age
- Transactions in last 24 hours
- Average transaction amount
- Distance from previous transaction
- New device indicator
- International transaction indicator
- Failed attempts
- Merchant risk score

The model is trained and evaluated using a held-out test dataset.

---

## 🚦 Risk Decision Engine

| Risk Level | Recommended Action |
|---|---|
| LOW | APPROVE |
| MEDIUM | ADDITIONAL VERIFICATION |
| HIGH | MANUAL REVIEW |

The risk level is determined from the model's predicted fraud probability.

---

## 🔍 Explainable AI

RazorShield AI provides human-readable reasons for suspicious transactions.

Examples include:

- Unusually high transaction amount
- New device detected
- International transaction
- Multiple failed attempts
- Unusual transaction distance
- High transaction frequency
- High merchant risk
- Unusual transaction time

---

## 📊 Model Evaluation

The system evaluates the model using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- False Positive Count
- Estimated False Positive Cost

The dataset used in this prototype is synthetic.

False-positive cost is estimated using an assumed cost of ₹50 per false positive.

---

## 🖥️ Dashboard

The Streamlit dashboard allows users to enter transaction information and receive:

1. Fraud probability
2. Risk score
3. Risk level
4. Recommended action
5. Explainable AI reasons
6. Model performance information

---

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Streamlit
- Plotly
- Joblib

---

## 📁 Project Structure

```text
RazorShield-AI/
│
├── app.py
├── RazorShield_AI.ipynb
├── razorshield_transactions.csv
├── razorshield_fraud_model.pkl
├── requirements.txt
└── README.md
