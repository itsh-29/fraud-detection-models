import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===== LOAD MODELS =====
lr = joblib.load("lr_model.pkl")
rf = joblib.load("rf_model.pkl")

# ===== LOAD DATA =====
df = pd.read_csv("creditcard.csv")

X = df.drop('Class', axis=1)
y = df['Class']

# SCALE
scaler = StandardScaler()
X = scaler.fit_transform(X)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== UI =====
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Comparison of Machine Learning Models")

# ===== ROC =====
st.subheader("ROC Curve Comparison")

lr_probs = lr.predict_proba(X_test)[:, 1]
rf_probs = rf.predict_proba(X_test)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

fig1 = plt.figure()
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1], [0,1], linestyle='--')
plt.legend()
st.pyplot(fig1)

# ===== PR CURVE =====
st.subheader("Precision-Recall Curve")

precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_probs)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_probs)

fig2 = plt.figure()
plt.plot(recall_lr, precision_lr, label="LR")
plt.plot(recall_rf, precision_rf, label="RF")
plt.legend()
st.pyplot(fig2)

# ===== CONFUSION MATRIX =====
st.subheader("Confusion Matrix")

fig3, ax = plt.subplots(1, 2, figsize=(10,4))

ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, ax=ax[0])
ax[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, ax=ax[1])
ax[1].set_title("Random Forest")

st.pyplot(fig3)