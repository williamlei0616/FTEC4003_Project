"""
SVM Classifier for Anomalous Transaction Identification
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import sys

# Load data
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop('Status', axis=1)
    y_train = train['Status']
    X_test = test
    return X_train, y_train, X_test

# SVM with hyperparameter tuning
def tune_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring='f1')
    grid.fit(X_train, y_train)
    print(f"Best SVM Params: {grid.best_params_}")
    return grid.best_estimator_

def run_svm(train_path, test_path, output_path, kernel='rbf', C=1.0, tune=False):
    X_train, y_train, X_test = load_data(train_path, test_path)
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if tune:
        clf = tune_svm(X_train_scaled, y_train)
    else:
        clf = SVC(kernel=kernel, C=C, random_state=42)
        clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Add Index column to output
    output_df = pd.DataFrame({
        'Index': X_test.index + 1,  # If your test set index starts at 0, add 1
        'Status': y_pred
    })
    output_df.to_csv(output_path, index=False)
    print(f"SVM predictions saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python svm.py <train_csv> <test_csv> <output_csv> [tune] [kernel] [C]")
        sys.exit(1)
    
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    output_csv = sys.argv[3]
    tune = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False
    kernel = sys.argv[5] if len(sys.argv) > 5 else 'rbf'
    C = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    run_svm(train_csv, test_csv, output_csv, kernel, C, tune)
