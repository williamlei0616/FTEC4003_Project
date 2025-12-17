"""
Test script for all models in Task1/model
"""
import os
import sys
import subprocess
import pandas as pd
import json

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

# Store all results
all_results = []

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_best_params(stdout_text):
    """Extract best parameters from model output"""
    params = {}
    lines = stdout_text.split('\n')
    capture = False
    
    for line in lines:
        # Check for the start marker
        if 'Best parameters:' in line or 'Best Parameters:' in line:
            capture = True
            continue
        # Check for the end marker
        if capture and line.strip() == '---':
            break
        # Parse parameter lines
        if capture and line.strip():
            # Format: "  key: value"
            line_stripped = line.strip()
            if ':' in line_stripped:
                parts = line_stripped.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    params[key] = value
    
    return params if params else None

def run_and_evaluate(model_key, script_args, output_name, is_tuned=False):
    out_dir = OUTPUT_DIRS[model_key]
    ensure_dir(out_dir)
    output_path = os.path.join(out_dir, output_name)
    
    # Build full arguments - insert output_path at correct position
    # For decision_tree: python script.py dt train test output [tune]
    # For others: python script.py train test output [tune]
    
    if 'decision_tree.py' in script_args[1]:
        # decision_tree.py has model type as first arg
        full_args = script_args[:5] + [output_path] + script_args[5:]
    else:
        # Other models: script, train, test, then output
        full_args = script_args[:4] + [output_path] + script_args[4:]
    
    print(f"\nRunning: {' '.join(full_args)}")
    result = subprocess.run(full_args, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")
    
    # Parse best parameters if tuned
    best_params = None
    if is_tuned:
        best_params = parse_best_params(result.stdout)
        if best_params:
            print(f"\nBest Parameters Found:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            
            # Save parameters to JSON
            params_file = os.path.join(out_dir, 'best_params.json')
            with open(params_file, 'w') as f:
                json.dump(best_params, f, indent=2)
            print(f"Parameters saved to: {params_file}")
        else:
            print("Warning: Could not parse best parameters from output")
    
    # Evaluate
    f1_score = None
    if os.path.exists(output_path):
        print(f"Evaluating {output_path} ...")
        eval_result = subprocess.run([EVAL_EXE, output_path], capture_output=True, text=True)
        print(eval_result.stdout)
        if eval_result.stderr:
            print(f"Eval Errors: {eval_result.stderr}")
        
        # Parse F1 score
        for line in eval_result.stdout.split('\n'):
            if 'F1 Score' in line or 'f1' in line.lower():
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        f1_score = float(parts[1].strip())
                    except:
                        pass
    else:
        print(f"Output file not found: {output_path}")
    
    # Store result
    result_entry = {
        'Model': model_key.replace('_', ' ').title(),
        'Tuned': is_tuned,
        'F1 Score': f1_score,
        'Output File': output_name
    }
    
    if is_tuned and best_params:
        result_entry['Best Parameters'] = json.dumps(best_params)
    
    all_results.append(result_entry)
    
    print('-' * 40)

# Decision Tree
print('='*50)
print('Testing Decision Tree...')
print('='*50)
run_and_evaluate(
    'decision_tree',
    [sys.executable, os.path.join(MODEL_DIR, 'decision_tree.py'), 'dt', TRAIN_PATH, TEST_PATH],
    'dt_pred.csv',
    is_tuned=False
)
run_and_evaluate(
    'decision_tree_tuned',
    [sys.executable, os.path.join(MODEL_DIR, 'decision_tree.py'), 'dt', TRAIN_PATH, TEST_PATH, 'True'],
    'dt_pred_tuned.csv',
    is_tuned=True
)

# Random Forest (Ensemble Method)
print('='*50)
print('Testing Random Forest (Ensemble Method)...')
print('='*50)
run_and_evaluate(
    'random_forest',
    [sys.executable, os.path.join(MODEL_DIR, 'decision_tree.py'), 'rf', TRAIN_PATH, TEST_PATH],
    'rf_pred.csv',
    is_tuned=False
)
run_and_evaluate(
    'random_forest_tuned',
    [sys.executable, os.path.join(MODEL_DIR, 'decision_tree.py'), 'rf', TRAIN_PATH, TEST_PATH, 'True'],
    'rf_pred_tuned.csv',
    is_tuned=True
)

# KNN
print('='*50)
print('Testing KNN...')
print('='*50)
run_and_evaluate(
    'knn',
    [sys.executable, os.path.join(MODEL_DIR, 'knn.py'), TRAIN_PATH, TEST_PATH],
    'knn_pred.csv',
    is_tuned=False
)
run_and_evaluate(
    'knn_tuned',
    [sys.executable, os.path.join(MODEL_DIR, 'knn.py'), TRAIN_PATH, TEST_PATH, 'True'],
    'knn_pred_tuned.csv',
    is_tuned=True
)

# Naive Bayes
print('='*50)
print('Testing Naive Bayes...')
print('='*50)
run_and_evaluate(
    'naive_bayes',
    [sys.executable, os.path.join(MODEL_DIR, 'naive_bayes.py'), TRAIN_PATH, TEST_PATH],
    'nb_pred.csv',
    is_tuned=False
)

# SVM
print('='*50)
print('Testing SVM...')
print('='*50)
run_and_evaluate(
    'svm',
    [sys.executable, os.path.join(MODEL_DIR, 'svm.py'), TRAIN_PATH, TEST_PATH],
    'svm_pred.csv',
    is_tuned=False
)
run_and_evaluate(
    'svm_tuned',
    [sys.executable, os.path.join(MODEL_DIR, 'svm.py'), TRAIN_PATH, TEST_PATH, 'True'],
    'svm_pred_tuned.csv',
    is_tuned=True
)

print('='*50)
print('All model tests and evaluations completed.')
print('='*50)

# Save all results to CSV
results_df = pd.DataFrame(all_results)
results_csv = os.path.join(BASE_DIR, 'results', 'all_model_results.csv')
ensure_dir(os.path.join(BASE_DIR, 'results'))
results_df.to_csv(results_csv, index=False)
print(f"\nAll results saved to: {results_csv}")

# Print summary table
print("\n" + "="*80)
print("SUMMARY OF ALL MODELS")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)
