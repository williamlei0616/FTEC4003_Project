"""
Naive Bayes Classifier for Anomalous Transaction Identification
"""
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import sys

# Load data
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop('Status', axis=1)
    y_train = train['Status']
    X_test = test
    return X_train, y_train, X_test

def run_naive_bayes(train_path, test_path, output_path):
    X_train, y_train, X_test = load_data(train_path, test_path)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Add Index column to output
    output_df = pd.DataFrame({
        'Index': X_test.index + 1,  # If your test set index starts at 0, add 1
        'Status': y_pred
    })
    output_df.to_csv(output_path, index=False)
    print(f"Naive Bayes predictions saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python naive_bayes.py <train_csv> <test_csv> <output_csv>")
        sys.exit(1)
    
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    output_csv = sys.argv[3]
    run_naive_bayes(train_csv, test_csv, output_csv)
