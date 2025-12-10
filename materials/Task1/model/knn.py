"""
KNN Classifier for Anomalous Transaction Identification
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import sys

# Load data
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop('Status', axis=1)
    y_train = train['Status']
    X_test = test
    return X_train, y_train, X_test

# KNN with hyperparameter tuning
def tune_knn(X_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    print(f"Best KNN Params: {grid.best_params_}")
    return grid.best_estimator_

def run_knn(train_path, test_path, output_path, n_neighbors=5, tune=False):
    X_train, y_train, X_test = load_data(train_path, test_path)
    if tune:
        clf = tune_knn(X_train, y_train)
    else:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Add Index column to output
    output_df = pd.DataFrame({
        'Index': X_test.index + 1,  # If your test set index starts at 0, add 1
        'Status': y_pred
    })
    output_df.to_csv(output_path, index=False)
    print(f"KNN predictions saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python knn.py <train_csv> <test_csv> <output_csv> [tune] [n_neighbors]")
        sys.exit(1)
    
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    output_csv = sys.argv[3]
    tune = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False
    n_neighbors = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    run_knn(train_csv, test_csv, output_csv, n_neighbors, tune)
