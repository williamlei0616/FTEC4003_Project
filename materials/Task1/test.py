"""
Test script for all models in Task1/model
"""
import os
import sys
import subprocess

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
TRAIN_PATH = os.path.join(BASE_DIR, 'finsecure_train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'finsecure_test.csv')
EVAL_EXE = os.path.join(BASE_DIR, 'evaluate_windows_1.exe')

# Output directories for each model/setting
OUTPUT_DIRS = {
    'decision_tree': os.path.join(BASE_DIR, 'results', 'decision_tree'),
    'decision_tree_tuned': os.path.join(BASE_DIR, 'results', 'decision_tree_tuned'),
    'random_forest': os.path.join(BASE_DIR, 'results', 'random_forest'),
    'random_forest_tuned': os.path.join(BASE_DIR, 'results', 'random_forest_tuned'),
    'knn': os.path.join(BASE_DIR, 'results', 'knn'),
    'knn_tuned': os.path.join(BASE_DIR, 'results', 'knn_tuned'),
    'naive_bayes': os.path.join(BASE_DIR, 'results', 'naive_bayes'),
    'svm': os.path.join(BASE_DIR, 'results', 'svm'),
    'svm_tuned': os.path.join(BASE_DIR, 'results', 'svm_tuned'),
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run_and_evaluate(model_key, script_args, output_name):
    out_dir = OUTPUT_DIRS[model_key]
    ensure_dir(out_dir)
    output_path = os.path.join(out_dir, output_name)
    # Insert output_path as the 5th argument (after model, train, test) - index 5
    full_args = script_args[:5] + [output_path] + script_args[5:]
    print(f"\nRunning: {' '.join(full_args)}")
    result = subprocess.run(full_args, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    # Evaluate
    if os.path.exists(output_path):
        print(f"Evaluating {output_path} ...")
        eval_result = subprocess.run([EVAL_EXE, output_path], capture_output=True, text=True)
        print(eval_result.stdout)
        if eval_result.stderr:
            print(f"Eval Errors: {eval_result.stderr}")
    else:
        print(f"Output file not found: {output_path}")
    print('-' * 40)

# Decision Tree
print('='*50)
print('Testing Decision Tree...')
print('='*50)
run_and_evaluate(
    'decision_tree',
    [sys.executable, os.path.join(MODEL_DIR, 'decision_tree.py'), 'dt', TRAIN_PATH, TEST_PATH],
    'dt_pred.csv'
)
run_and_evaluate(
    'decision_tree_tuned',
    [sys.executable, os.path.join(MODEL_DIR, 'decision_tree.py'), 'dt', TRAIN_PATH, TEST_PATH, 'True'],
    'dt_pred_tuned.csv'
)

# Random Forest (Ensemble Method)
print('='*50)
print('Testing Random Forest (Ensemble Method)...')
print('='*50)
run_and_evaluate(
    'random_forest',
    [sys.executable, os.path.join(MODEL_DIR, 'decision_tree.py'), 'rf', TRAIN_PATH, TEST_PATH],
    'rf_pred.csv'
)
run_and_evaluate(
    'random_forest_tuned',
    [sys.executable, os.path.join(MODEL_DIR, 'decision_tree.py'), 'rf', TRAIN_PATH, TEST_PATH, 'True'],
    'rf_pred_tuned.csv'
)

# KNN
print('='*50)
print('Testing KNN...')
print('='*50)
run_and_evaluate(
    'knn',
    [sys.executable, os.path.join(MODEL_DIR, 'knn.py'), TRAIN_PATH, TEST_PATH],
    'knn_pred.csv'
)
run_and_evaluate(
    'knn_tuned',
    [sys.executable, os.path.join(MODEL_DIR, 'knn.py'), TRAIN_PATH, TEST_PATH, 'True'],
    'knn_pred_tuned.csv'
)

# Naive Bayes
print('='*50)
print('Testing Naive Bayes...')
print('='*50)
run_and_evaluate(
    'naive_bayes',
    [sys.executable, os.path.join(MODEL_DIR, 'naive_bayes.py'), TRAIN_PATH, TEST_PATH],
    'nb_pred.csv'
)

# SVM
print('='*50)
print('Testing SVM...')
print('='*50)
run_and_evaluate(
    'svm',
    [sys.executable, os.path.join(MODEL_DIR, 'svm.py'), TRAIN_PATH, TEST_PATH],
    'svm_pred.csv'
)
run_and_evaluate(
    'svm_tuned',
    [sys.executable, os.path.join(MODEL_DIR, 'svm.py'), TRAIN_PATH, TEST_PATH, 'True'],
    'svm_pred_tuned.csv'
)

print('='*50)
print('All model tests and evaluations completed.')
print('='*50)
