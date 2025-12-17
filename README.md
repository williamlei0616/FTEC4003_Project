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

---

### Materials Directory (`materials/`)

#### `Course Project.pdf`
Official course project documentation and requirements.

---

### Task 1: Basic Classification Models (`materials/Task1/`)

#### Documentation
- **`Task-1- Anomalous-Transaction-Identification.pdf`**: Detailed Task 1 specifications and requirements

#### Data Files
- **`finsecure_train.csv`**: Training dataset containing transaction features with `Status` labels
- **`finsecure_test.csv`**: Test dataset for predictions (no labels)
- **`sample_submission_1.csv`**: Sample submission format for Task 1

#### Executables
- **`evaluate_mac_1`**: macOS evaluation executable to score Task 1 submissions
- **`evaluate_windows_1.exe`**: Windows evaluation executable to score Task 1 submissions
- **`True`**: Binary/configuration file (potentially for model settings)

#### Main Script
- **`test.py`**: Automated testing script that runs all models with both default and tuned hyperparameters, evaluates predictions, and saves results to respective directories

#### Model Implementations (`materials/Task1/model/`)

- **`__init__.py`**: Python package initialization file

- **`decision_tree.py`**: 
  - Implements Decision Tree (DT) and Random Forest (RF) classifiers
  - Supports hyperparameter tuning via GridSearchCV
  - Configurable parameters: max_depth, min_samples_split, n_estimators
  - Usage: `python decision_tree.py <model> <train_csv> <test_csv> <output_csv> [tune] [params...]`

- **`knn.py`**: 
  - K-Nearest Neighbors classifier implementation
  - Supports hyperparameter tuning for n_neighbors, weights, and distance metrics
  - Default configuration: 5 neighbors
  - Usage: `python knn.py <train_csv> <test_csv> <output_csv> [tune] [n_neighbors]`

- **`naive_bayes.py`**: 
  - Gaussian Naive Bayes classifier for anomaly detection
  - Simple probabilistic approach without hyperparameter tuning
  - Usage: `python naive_bayes.py <train_csv> <test_csv> <output_csv>`

- **`random_forest.py`**: 
  - Standalone Random Forest implementation (ensemble method)
  - Note: Uses `class` column instead of `Status` (possibly for different dataset)
  - Configurable estimators and tree depth
  - Usage: `python random_forest.py <train_csv> <test_csv> <output_csv> [n_estimators] [max_depth]`

- **`svm.py`**: 
  - Support Vector Machine classifier with StandardScaler preprocessing
  - Supports hyperparameter tuning for kernel type, C parameter, and gamma
  - Default: RBF kernel with C=1.0
  - Usage: `python svm.py <train_csv> <test_csv> <output_csv> [tune] [kernel] [C]`

#### Results Directory (`materials/Task1/results/`)
Contains subdirectories for each model's predictions:
- **`decision_tree/`**: `dt_pred.csv` - Default decision tree predictions
- **`decision_tree_tuned/`**: `dt_pred_tuned.csv` - Tuned decision tree predictions
- **`knn/`**: `knn_pred.csv` - Default KNN predictions
- **`knn_tuned/`**: `knn_pred_tuned.csv` - Tuned KNN predictions
- **`naive_bayes/`**: `nb_pred.csv` - Naive Bayes predictions
- **`random_forest/`**: `rf_pred.csv` - Default random forest predictions
- **`random_forest_tuned/`**: `rf_pred_tuned.csv` - Tuned random forest predictions
- **`svm/`**: `svm_pred.csv` - Default SVM predictions
- **`svm_tuned/`**: `svm_pred_tuned.csv` - Tuned SVM predictions

---

### Task 2: Advanced Ensemble Methods (`task2/`)

#### Data Files & Submissions
- **`8_submission_2.csv`**: Final predictions using LightGBM model
- **`submission_xgb.csv`**: Previous approach of XGB


#### Main Model Implementations

- **`lightGBM.py`**: 
  - Primary LightGBM implementation with extensive feature engineering
  - Merges transaction and identity datasets
  - Feature dropping: 239 V-features removed based on feature importance analysis
  - Implements SMOTE/SMOTENC for class imbalance handling
  - Missing value handling with binary flags
  - High missing value column removal (>90% threshold)
  - Categorical encoding and missing value imputation
  - Model parameters: 3000 estimators, max_depth=15, learning_rate=0.03
  - GPU support option available
  - Outputs: `submission_lightgbm.csv`, `remaining_features.txt`

- **`xgb.py`**: 
  - XGBoost implementation with Phase Key Shifting Encoding (hypersphere embedding)
  - Advanced categorical encoding using sin/cos transformations
  - SMOTE resampling for balanced training
  - Model parameters: 1200 estimators, max_depth=12, learning_rate=0.03
  - Outputs: `submission_xgb.csv`

#### Feature Analysis Scripts

- **`lightGBM_feature_test.py`**: 
  - Automated feature testing framework for previously dropped features
  - Tests adding back each of 239 dropped V-features individually
  - Evaluates impact on F1 score using subprocess calls to evaluation executable
  - Generates comprehensive reports with timing and performance metrics
  - Identifies features that improve model performance when re-added
  - Outputs: `feature_test_results.txt`, temporary submission files

- **`lightGBM_V_feature_drop_test.py`**: 
  - Systematic testing of currently retained V-features (~90 features)
  - Evaluates impact of dropping each feature individually
  - Uses CPU-based LightGBM training
  - Provides ETA and progress tracking during execution
  - Identifies redundant features that can be removed to improve performance
  - Outputs: `v_feature_drop_test_results.txt`, temporary submission files per feature

---
