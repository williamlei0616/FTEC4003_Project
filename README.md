# FTEC4003_Project

- Ho Man Hin Vincent - 1155193238
- Huang Ying Lam - 1155192046
- Lei Wai Lun - 1155194117

## Project Overview

This project contains machine learning solutions for financial transaction anomaly detection, divided into two main tasks using various classification algorithms and ensemble methods.

---

## File Descriptions

### Root Directory

#### `README.md`

Project documentation file containing descriptions of all files and project structure.

#### `requirement.txt`

Lists Python dependencies required for the project:

- pandas
- numpy
- scikit-learn
- imblearn
- lightgbm

#### `Course Project.pdf`

Official course project documentation and requirements.

---

### Task 1: Anomalous Transaction Identification (`task1/`)

#### Documentation

- **`Task-1- Anomalous-Transaction-Identification.pdf`**: Detailed Task 1 specifications and requirements.

#### Data Files

- **`finsecure_train.csv`**: Training dataset containing transaction features with `Status` labels.
- **`finsecure_test.csv`**: Test dataset for predictions (no labels).
- **`sample_submission_1.csv`**: Sample submission format for Task 1.

#### Task 1 Executables

- **`evaluate_mac_1`**: macOS evaluation executable to score Task 1 submissions.
- **`evaluate_windows_1.exe`**: Windows evaluation executable to score Task 1 submissions.

#### Main Script

- **`main.py`**: Automated testing script that runs all models (Decision Tree, KNN, Naive Bayes, Random Forest, SVM) with both default and tuned hyperparameters. It evaluates predictions using the provided executable and saves results to the `results/` directory.

#### Model Implementations (`task1/model/`)

- **`decision_tree.py`**:
  - Implements Decision Tree classifier.
  - Supports hyperparameter tuning via GridSearchCV.
  - Usage: `python decision_tree.py <model_type> <train_csv> <test_csv> <output_csv> [tune]`

- **`knn.py`**:
  - K-Nearest Neighbors classifier implementation.
  - Supports hyperparameter tuning.
  - Usage: `python knn.py <train_csv> <test_csv> <output_csv> [tune]`

- **`naive_bayes.py`**:
  - Gaussian Naive Bayes classifier.
  - Usage: `python naive_bayes.py <train_csv> <test_csv> <output_csv>`

- **`random_forest.py`**:
  - Random Forest classifier implementation.
  - Supports hyperparameter tuning.
  - Usage: `python random_forest.py <train_csv> <test_csv> <output_csv> [tune]`

- **`svm.py`**:
  - Support Vector Machine classifier.
  - Supports hyperparameter tuning.
  - Usage: `python svm.py <train_csv> <test_csv> <output_csv> [tune]`

#### Results Directory (`task1/results/`)

Contains subdirectories for each model's predictions and evaluation results.

---

### Task 2: Risk Modeling Marketing (`task2/`)

#### Data Files (`task2/dataset/`)

- **`globalmart_train_transactions.csv`**
- **`globalmart_train_identity.csv`**
- **`globalmart_test_transactions.csv`**
- **`globalmart_test_identity.csv`**

#### Main Model Implementations

- **`lightGBM.py`**:
  - Primary LightGBM implementation with extensive feature engineering.
  - Merges transaction and identity datasets.
  - Implements SMOTE/SMOTENC for class imbalance handling.
  - Features: Missing value handling, categorical encoding, and feature selection.
  - Outputs: `GID_submission_2.csv`.

- **`xgb.py`**:
  - XGBoost implementation with advanced encoding techniques (Phase Key Shifting).
  - Outputs: `submission_xgb.csv`.

#### Feature Analysis Scripts

- **`lightGBM_feature_test.py`**:
  - Automated feature testing framework to evaluate the impact of adding back dropped features.

- **`lightGBM_V_feature_drop_test.py`**:
  - Systematic testing of retained features to identify redundancy.

#### Task 2 Executables

- **`evaluate_mac_2`**: macOS evaluation executable for Task 2.
- **`evaluate_windows_2.exe`**: Windows evaluation executable for Task 2.
