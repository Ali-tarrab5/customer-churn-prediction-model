# Customer Churn Prediction System

## 📌 Business Problem
Telecom companies lose significant revenue due to customer churn. This project 
predicts which customers are likely to churn, enabling proactive retention strategies.

## 🔍 Key Insights from EDA
- Month-to-month contract customers churn significantly more than long-term contracts
- Fiber optic internet users show higher churn rates
- Electronic check payment method correlates with higher churn

## 🛠️ Approach
1. Data cleaning (handled hidden missing values in TotalCharges)
2. EDA & feature analysis
3. Handled class imbalance using SMOTE
4. Trained & compared 3 models: Logistic Regression, Random Forest, XGBoost
5. Selected Logistic Regression based on highest Recall (business priority: catch churners)

## 📊 Model Performance
| Model | Accuracy | Recall | ROC-AUC |
|-------|----------|--------|---------|
| Logistic Regression | 76% | 0.63 | 0.81 |
| Random Forest | 77% | 0.59 | 0.82 |
| XGBoost | 76% | 0.57 | 0.81 |

## 🚀 Live Demo
[Try the app here](your-streamlit-link)

## 🧰 Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Streamlit, SMOTE
