"""
Random Forest Classifier (Ensemble Method) for Anomalous Transaction Identification
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop('class', axis=1)
    y_train = train['class']
    X_test = test
    return X_train, y_train, X_test

def run_random_forest(train_path, test_path, output_path, n_estimators=100, max_depth=None):
    X_train, y_train, X_test = load_data(train_path, test_path)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Add Index column to output
    output_df = pd.DataFrame({
        'Index': X_test.index + 1,  # If your test set index starts at 0, add 1
        'Status': y_pred
    })
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # Example usage: python random_forest.py finsecure_train.csv finsecure_test.csv output.csv [n_estimators] [max_depth]
    if len(sys.argv) < 4:
        print("Usage: python random_forest.py <train_csv> <test_csv> <output_csv> [n_estimators] [max_depth]")
        sys.exit(1)
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    output_csv = sys.argv[3]
    n_estimators = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    max_depth = int(sys.argv[5]) if len(sys.argv) > 5 else None
    run_random_forest(train_csv, test_csv, output_csv, n_estimators, max_depth)
