# Telco Customer Churn Prediction

This project aims to predict customer churn using machine learning techniques on the Telco Customer Churn dataset. It includes extensive preprocessing, model training with various classifiers, and performance evaluation.

## 📂 Dataset

The dataset used in this project is `Telco-Customer-Churn.csv`, which contains customer demographics, account information, and service usage data.

## 🛠️ Features

- Data Cleaning and Preprocessing
- Handling Categorical Variables
- Feature Scaling
- Outlier Detection (LOF)
- Model Training with:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Trees
  - Random Forest
  - Support Vector Machines
  - XGBoost
  - LightGBM
  - CatBoost
- Model Evaluation:
  - Accuracy, F1 Score, Recall, ROC-AUC
  - Confusion Matrix, Classification Report
- Ensemble Learning with Voting Classifier

##  Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`, `lightgbm`, `catboost`
- `missingno` (for missing data visualization)

##  Evaluation Function

A custom model evaluation function `evaluate_model_performance()` is used (imported from `model_evulation.py`). Make sure this file is present in the working directory.

##  Running the Notebook

1. Clone the repository.
2. Place the dataset file (`Telco-Customer-Churn.csv`) in the project directory.
3. Ensure the required packages are installed:
   ```bash
   pip install -r requirements.txt
