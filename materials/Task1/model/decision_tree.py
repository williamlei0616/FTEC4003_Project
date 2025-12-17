"""
Decision Tree and Random Forest Classifiers for Anomalous Transaction Identification
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Decision Tree with hyperparameter tuning
def tune_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=3, scoring='f1')
    grid.fit(X_train, y_train)
    
    # Print best parameters in consistent format
    print("Best parameters:")
    for key, value in grid.best_params_.items():
        print(f"  {key}: {value}")
    print("---")
    
    return grid.best_estimator_

# Random Forest with hyperparameter tuning
def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Print best parameters in consistent format
    print("Best parameters:")
    for key, value in grid.best_params_.items():
        print(f"  {key}: {value}")
    print("---")
    
    return grid.best_estimator_

def run_decision_tree(train_path, test_path, output_path, tune=False):
    X_train, y_train, X_test = load_data(train_path, test_path)
    
    if tune:
        clf = tune_decision_tree(X_train, y_train)
    else:
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    output_df = pd.DataFrame({
        'Index': X_test.index + 1,
        'Status': y_pred
    })
    output_df.to_csv(output_path, index=False)
    print(f"Decision Tree predictions saved to {output_path}")

def run_random_forest(train_path, test_path, output_path, tune=False):
    X_train, y_train, X_test = load_data(train_path, test_path)
    
    if tune:
        clf = tune_random_forest(X_train, y_train)
    else:
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    output_df = pd.DataFrame({
        'Index': X_test.index + 1,
        'Status': y_pred
    })
    output_df.to_csv(output_path, index=False)
    print(f"Random Forest predictions saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python decision_tree.py <dt|rf> <train_csv> <test_csv> <output_csv> [tune]")
        sys.exit(1)
    
    model_type = sys.argv[1]
    train_csv = sys.argv[2]
    test_csv = sys.argv[3]
    output_csv = sys.argv[4]
    tune = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else False
    
    if model_type == 'dt':
        run_decision_tree(train_csv, test_csv, output_csv, tune)
    elif model_type == 'rf':
        run_random_forest(train_csv, test_csv, output_csv, tune)
    else:
        print("Unknown model type. Use 'dt' for Decision Tree or 'rf' for Random Forest.")
        sys.exit(1)
