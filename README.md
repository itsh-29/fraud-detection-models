# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques. The dataset is highly imbalanced, making it a challenging real-world problem. The goal is to build and compare models, handle class imbalance, and improve fraud detection performance.

---

## 🚀 Features

* Built and compared multiple ML models:

  * Logistic Regression (Baseline)
  * Random Forest (Optimized)
* Handled imbalanced data using **SMOTE**
* Evaluated models using:

  * Precision, Recall, F1-score
  * Confusion Matrix
  * ROC Curve & AUC
  * Precision-Recall Curve
* Developed an interactive **Streamlit dashboard** for visualization

---

## 📂 Dataset

* The dataset contains anonymized transaction features (`V1–V28`) obtained using PCA
* Includes:

  * `Time`
  * `Amount`
  * `Class` (Target: 0 = Normal, 1 = Fraud)

⚠️ Dataset not included due to size constraints
👉 You can download it from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🧠 Machine Learning Workflow

### 1. Data Preprocessing

* Scaled `Amount` feature
* Train-test split with stratification

### 2. Model Training

* Logistic Regression (baseline model)
* Random Forest (better performance on imbalanced data)

### 3. Handling Imbalance

* Applied **SMOTE (Synthetic Minority Oversampling Technique)**

### 4. Evaluation Metrics

* Accuracy (not reliable for imbalance)
* Precision
* Recall (most important for fraud detection)
* F1 Score

---

## 📊 Results

| Model                        | Recall (Fraud) | Precision | F1 Score |
| ---------------------------- | -------------- | --------- | -------- |
| Logistic Regression          | 0.66           | 0.82      | 0.73     |
| Random Forest (Before SMOTE) | 0.82           | 0.94      | 0.87     |
| Random Forest (After SMOTE)  | **0.86**       | 0.42      | 0.56     |

### 🔍 Key Insight

* SMOTE improved fraud detection (Recall ↑)
* Precision decreased due to more false positives
* Trade-off is acceptable in fraud detection systems

---

## 📊 Visualizations

* Confusion Matrix
* ROC Curve (AUC)
* Precision-Recall Curve
* Feature Importance

---

## 🖥️ Streamlit Dashboard

An interactive dashboard was built to visualize model performance.

### ▶️ Run locally:

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib
* Streamlit

---

## 📁 Project Structure

```
fraud-detection/
│
├── app.py                # Streamlit dashboard
├── notebook.ipynb       # Model training & analysis
├── README.md
├── .gitignore
```

---

## 💡 Key Learnings

* Handling imbalanced datasets is critical in real-world ML problems
* Accuracy alone is misleading for fraud detection
* Recall is more important than precision in detecting fraud
* Visualization helps in better model interpretation

---

## 👨‍💻 Author

**Ishan Meduri**

---

## ⭐ If you found this useful

Give this repo a ⭐ and feel free to connect!
