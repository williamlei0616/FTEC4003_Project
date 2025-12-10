"""
Decision Tree and Random Forest Classifier for Anomalous Transaction Identification
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
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    print(f"Best Decision Tree Params: {grid.best_params_}")
    return grid.best_estimator_

def run_decision_tree(train_path, test_path, output_path, max_depth=None, min_samples_split=2, tune=False):
    X_train, y_train, X_test = load_data(train_path, test_path)
    if tune:
        clf = tune_decision_tree(X_train, y_train)
    else:
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    output_df = pd.DataFrame({
        'Index': X_test.index + 1,
        'Status': y_pred
    })
    output_df.to_csv(output_path, index=False)
    print(f"Decision Tree predictions saved to {output_path}")

# Ensemble Method: Random Forest with hyperparameter tuning
def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    print(f"Best Random Forest Params: {grid.best_params_}")
    return grid.best_estimator_

def run_random_forest(train_path, test_path, output_path, n_estimators=100, max_depth=None, min_samples_split=2, tune=False):
    X_train, y_train, X_test = load_data(train_path, test_path)
    if tune:
        clf = tune_random_forest(X_train, y_train)
    else:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
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
        print("Usage: python decision_tree.py <model> <train_csv> <test_csv> <output_csv> [tune] [params...]")
        print("model: dt for Decision Tree, rf for Random Forest")
        print("tune: True to tune hyperparameters, False otherwise")
        sys.exit(1)

    model = sys.argv[1]
    train_csv = sys.argv[2]
    test_csv = sys.argv[3]
    output_csv = sys.argv[4]

    # If tune is present, it should be sys.argv[5], else False
    tune = False
    max_depth = None
    min_samples_split = 2
    n_estimators = 100
    # Parse tune and params
    if len(sys.argv) > 5:
        if sys.argv[5].lower() == 'true' or sys.argv[5].lower() == 'false':
            tune = sys.argv[5].lower() == 'true'
            # For dt: [max_depth] [min_samples_split]
            if model == 'dt':
                if len(sys.argv) > 6:
                    max_depth = int(sys.argv[6]) if sys.argv[6] != 'None' else None
                if len(sys.argv) > 7:
                    min_samples_split = int(sys.argv[7])
            # For rf: [n_estimators] [max_depth] [min_samples_split]
            elif model == 'rf':
                if len(sys.argv) > 6:
                    n_estimators = int(sys.argv[6])
                if len(sys.argv) > 7:
                    max_depth = int(sys.argv[7]) if sys.argv[7] != 'None' else None
                if len(sys.argv) > 8:
                    min_samples_split = int(sys.argv[8])
        else:
            # If tune is not present, treat as param
            if model == 'dt':
                max_depth = int(sys.argv[5]) if sys.argv[5] != 'None' else None
                if len(sys.argv) > 6:
                    min_samples_split = int(sys.argv[6])
            elif model == 'rf':
                n_estimators = int(sys.argv[5])
                if len(sys.argv) > 6:
                    max_depth = int(sys.argv[6]) if sys.argv[6] != 'None' else None
                if len(sys.argv) > 7:
                    min_samples_split = int(sys.argv[7])

    if model == 'dt':
        run_decision_tree(train_csv, test_csv, output_csv, max_depth, min_samples_split, tune)
    elif model == 'rf':
        run_random_forest(train_csv, test_csv, output_csv, n_estimators, max_depth, min_samples_split, tune)
    else:
        print("Unknown model type. Use 'dt' for Decision Tree or 'rf' for Random Forest.")
