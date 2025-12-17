import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import subprocess
import re
import time

# Load the datasets
train_transactions = pd.read_csv("dataset/globalmart_train_transactions.csv")
train_identity = pd.read_csv("dataset/globalmart_train_identity.csv")
test_transactions = pd.read_csv("dataset/globalmart_test_transactions.csv")
test_identity = pd.read_csv("dataset/globalmart_test_identity.csv")

# Merge with left join (transactions as left table)
merged_df = train_transactions.merge(train_identity, how="left", on="OrderID")
merged_test_df = test_transactions.merge(test_identity, how="left", on="OrderID")

print(f"Merged dataset shape: {merged_df.shape}")
print(f"Merged test dataset shape: {merged_test_df.shape}")

# --- SMOTE Resampling Logic ---
target_col = "IsRisky"
categorical_columns = [
    "IdentityFeature12", "IdentityFeature13", "IdentityFeature14", "IdentityFeature15",
    "IdentityFeature16", "IdentityFeature17", "IdentityFeature18", "IdentityFeature19",
    "IdentityFeature20", "IdentityFeature21", "IdentityFeature22", "IdentityFeature23",
    "IdentityFeature27", "IdentityFeature28", "IdentityFeature29", "IdentityFeature30",
    "IdentityFeature31", "IdentityFeature33", "IdentityFeature34", "IdentityFeature35",
    "IdentityFeature36", "IdentityFeature37", "IdentityFeature38", "DeviceOS",
    "DeviceModel", "PaymentType", "CardInfo1", "CardInfo2", "CardInfo3", "CardInfo5",
    "CardNetwork", "CardType", "BillingRegion", "BillingCountry", "PayerEmailProvider",
    "RecipientEmailProvider", "MatchStatus1", "MatchStatus2", "MatchStatus3",
    "MatchStatus4", "MatchStatus5", "MatchStatus6", "MatchStatus7", "MatchStatus8",
    "MatchStatus9",
]

# Features that were originally dropped
dropped_features = [
    "V2", "V5", "V7", "V9", "V10", "V12", "V15", "V16", "V18", "V19",
    "V21", "V22", "V24", "V25", "V28", "V29", "V31", "V32", "V33", "V34", "V35",
    "V38", "V39", "V42", "V43", "V45", "V46", "V49", "V50", "V51", "V52", "V53",
    "V55", "V57", "V58", "V60", "V61", "V63", "V64", "V66", "V71", "V72",
    "V73", "V74", "V75", "V77", "V79", "V81", "V83", "V84", "V85", "V87", "V90",
    "V92", "V93", "V94", "V95", "V96", "V97", "V98", "V99", "V100", "V101", "V102",
    "V103", "V104", "V105", "V106", "V109", "V110", "V112", "V113", "V114", "V116",
    "V118", "V119", "V122", "V125", "V126", "V128", "V131", "V132", "V133", "V134",
    "V135", "V137", "V140", "V141", "V143", "V144", "V145", "V146", "V148", "V149",
    "V150", "V151", "V152", "V153", "V154", "V155", "V157", "V158", "V159", "V161",
    "V163", "V164", "V167", "V168", "V170", "V172", "V174", "V177", "V179", "V181",
    "V183", "V184", "V186", "V189", "V190", "V191", "V192", "V193", "V194", "V195",
    "V196", "V197", "V199", "V200", "V202", "V204", "V206", "V208", "V211", "V212",
    "V213", "V214", "V216", "V217", "V219", "V222", "V225", "V227", "V230", "V231",
    "V232", "V233", "V236", "V237", "V239", "V241", "V242", "V243", "V244", "V245",
    "V246", "V247", "V248", "V249", "V251", "V254", "V255", "V256", "V259", "V262",
    "V263", "V265", "V268", "V269", "V270", "V272", "V273", "V275", "V276", "V278",
    "V279", "V280", "V282", "V287", "V288", "V290", "V292", "V293", "V295", "V298",
    "V299", "V300", "V302", "V304", "V306", "V308", "V311", "V312", "V313", "V315",
    "V316", "V317", "V318", "V319", "V321", "V322", "V323", "V324", "V325", "V326",
    "V327", "V328", "V329", "V330", "V331", "V332", "V333", "V334", "V335", "V336",
    "V337", "V338", "V339",
]

# V features to test dropping (currently kept)
v_features_to_test = [
    'V88', 'V89', 'V91', 'V107', 'V108', 'V111', 'V115', 'V117', 'V120', 
    'V121', 'V123', 'V124', 'V127', 'V129', 'V130', 'V136', 'V138', 'V139', 'V142', 
    'V147', 'V156', 'V160', 'V162', 'V165', 'V166', 'V169', 'V171', 'V173', 'V175', 
    'V176', 'V178', 'V180', 'V182', 'V185', 'V187', 'V188', 'V198', 'V201', 'V203', 
    'V205', 'V207', 'V209', 'V210', 'V215', 'V218', 'V220', 'V221', 'V223', 'V224', 
    'V226', 'V228', 'V229', 'V234', 'V235', 'V238', 'V240', 'V250', 'V252', 'V253', 
    'V257', 'V258', 'V260', 'V261', 'V264', 'V266', 'V267', 'V271', 'V274', 'V277', 
    'V281', 'V283', 'V284', 'V285', 'V286', 'V289', 'V291', 'V294', 'V296', 'V297', 
    'V301', 'V303', 'V305', 'V307', 'V309', 'V310', 'V314', 'V320'
]



def _build_lgbm_classifier():
    import lightgbm as lgb
    params = dict(
        n_estimators=3000,
        num_leaves=40,
        max_depth=15,
        min_child_samples=70,
        reg_alpha=0.1,
        reg_lambda=0.1,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
    )
    return lgb.LGBMClassifier(**params)


def train_and_evaluate(feature_to_drop=None):
    """Train model with optional feature dropped and return (feature_name, F1 score)"""
    
    try:
        # Start with the originally dropped features
        current_drops = dropped_features.copy()
        
        # Add the feature we're testing to drop
        if feature_to_drop:
            current_drops.append(feature_to_drop)
        
        X = merged_df.drop(columns=[target_col] + current_drops, errors='ignore')
        y = merged_df[target_col]
        
        # Update categorical columns list
        current_categorical = [col for col in categorical_columns if col in X.columns]
        
        # --- Drop High Missing Value Columns ---
        missing_threshold = 0.90
        missing_series = X.isnull().mean()
        cols_to_drop = missing_series[missing_series > missing_threshold].index.tolist()
        
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            current_categorical = [col for col in current_categorical if col not in cols_to_drop]
        
        # Track columns with missing values
        cols_with_missing_in_train = [col for col in X.columns if X[col].isnull().any()]
        
        # Handle Missing Values
        for col in X.columns:
            if col in cols_with_missing_in_train:
                X[f"{col}_is_missing"] = X[col].isnull().astype(int)
            
            if X[col].dtype == "object" or col in current_categorical:
                X[col] = X[col].fillna("-1").astype(str)
            else:
                X[col] = X[col].fillna(0)
        
        # SMOTE Resampling
        if current_categorical:
            valid_cat_cols = [c for c in current_categorical if c in X.columns]
            categorical_indices = [X.columns.get_loc(col) for col in valid_cat_cols]
            smote = SMOTENC(categorical_features=categorical_indices, random_state=42)
        else:
            smote = SMOTE(random_state=42)
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Prepare Training Data
        feature_cols = [col for col in X_resampled.columns if col != "OrderID"]
        X_train = X_resampled[feature_cols].copy()
        y_train = y_resampled
        
        # Prepare Test Data
        base_features = [c for c in feature_cols if not c.endswith("_is_missing")]
        X_test = merged_test_df[base_features].copy()
        
        # Re-create the missing flags for test
        for col in cols_with_missing_in_train:
            if col in X_test.columns:
                X_test[f"{col}_is_missing"] = X_test[col].isnull().astype(int)
        
        X_test = X_test[feature_cols]
        
        # Apply the same missing value handling to Test
        for col in X_test.columns:
            if col.endswith("_is_missing"):
                continue
            if X_train[col].dtype == "object" or X_train[col].dtype.name == "category":
                X_test[col] = X_test[col].fillna("-1").astype(str)
            else:
                X_test[col] = X_test[col].fillna(-1)
        
        # Convert Categorical Columns to 'category' dtype
        for col in X_train.columns:
            if X_train[col].dtype == "object":
                X_train[col] = X_train[col].astype("category")
                X_test[col] = X_test[col].astype("category")
        
        # Train Model
        clf = _build_lgbm_classifier()
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Create Submission
        submission_file = f"submission_lightgbm_{feature_to_drop if feature_to_drop else 'baseline'}.csv"
        submission = pd.DataFrame({"OrderID": merged_test_df["OrderID"], "IsRisky": y_pred})
        submission.to_csv(submission_file, index=False)
        
        # Run evaluation
        result = subprocess.run(
            [".\\evaluate_windows_2.exe", f".\\{submission_file}"],
            capture_output=True,
            text=True
        )
        
        # Parse F1 score from output
        output = result.stdout
        match = re.search(r"F1 Score.*?:\s*([\d.]+)", output)
        if match:
            f1_score = float(match.group(1))
        else:
            f1_score = None
            print(f"Could not parse F1 score for {feature_to_drop}. Output: {output}")
        
        return (feature_to_drop, f1_score)
    
    except Exception as e:
        print(f"Error processing {feature_to_drop}: {str(e)}")
        return (feature_to_drop, None)


def write_result_to_file(feature, f1_score, baseline_f1, file_path):
    """Append result to file immediately after computation"""
    with open(file_path, "a") as f:
        if baseline_f1 and f1_score:
            diff = f1_score - baseline_f1
            line = f"DROP {feature}: {f1_score:.4f} (diff: {diff:+.4f})\n"
        else:
            line = f"DROP {feature}: {f1_score}\n"
        f.write(line)


if __name__ == '__main__':
    # Store results
    results = {}
    output_file = "v_feature_drop_test_results.txt"

    # Initialize output file
    with open(output_file, "w") as f:
        f.write("V Feature Drop Test Results (CPU)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Testing {len(v_features_to_test)} V features\n")
        f.write("Results will be updated as each feature is tested...\n\n")

    # First, get baseline (current features)
    print("=" * 60)
    print("Testing BASELINE (current features) - CPU")
    print("=" * 60)
    start_time = time.time()
    _, baseline_f1 = train_and_evaluate(feature_to_drop=None)
    baseline_time = time.time() - start_time
    results["BASELINE"] = baseline_f1
    print(f"BASELINE F1 Score: {baseline_f1} (took {baseline_time:.2f}s)")

    # Write baseline to file
    with open(output_file, "a") as f:
        f.write(f"Baseline F1 Score: {baseline_f1}\n")
        f.write(f"Baseline computation time: {baseline_time:.2f}s\n")
        f.write("=" * 60 + "\n\n")
        f.write("Individual V Feature Drop Results:\n")
        f.write("-" * 60 + "\n")

    print()

    # Filter out features not in dataset
    features_to_test = [f for f in v_features_to_test if f in merged_df.columns]
    print(f"Testing {len(features_to_test)} V features")
    print(f"Estimated total time: {(baseline_time * len(features_to_test)) / 60:.1f} minutes\n")
    
    start_total = time.time()
    
    for i, feature in enumerate(features_to_test):
        feature_name, f1_score = train_and_evaluate(feature)
        results[feature_name] = f1_score
        write_result_to_file(feature_name, f1_score, baseline_f1, output_file)
        
        if f1_score is not None:
            diff = f1_score - baseline_f1 if baseline_f1 else 0
            elapsed = time.time() - start_total
            avg_time = elapsed / (i + 1)
            remaining = (len(features_to_test) - i - 1) * avg_time
            print(f"[{i+1}/{len(features_to_test)}] DROP {feature}: {f1_score:.4f} (diff: {diff:+.4f}) "
                  f"| Avg: {avg_time:.1f}s | ETA: {remaining/60:.1f}min")
        else:
            print(f"[{i+1}/{len(features_to_test)}] DROP {feature}: Failed")

    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"All tests completed in {total_time/60:.1f} minutes")
    print(f"Average time per feature: {total_time/len(features_to_test):.1f}s")
    print(f"{'='*60}\n")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    # Sort by F1 score (descending)
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v is not None and k != "BASELINE"],
        key=lambda x: x[1],
        reverse=True
    )

    # Append summary to file
    with open(output_file, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"Average time per feature: {total_time/len(features_to_test):.1f}s\n")
        f.write(f"Baseline F1 Score: {baseline_f1}\n\n")
        f.write("Top 20 V Features by F1 Score when DROPPED:\n")
        f.write("-" * 60 + "\n")
        
        for i, (feature, f1) in enumerate(sorted_results[:20]):
            diff = f1 - baseline_f1 if baseline_f1 else 0
            line = f"{i+1}. DROP {feature}: {f1:.4f} (diff: {diff:+.4f})"
            print(line)
            f.write(line + "\n")

    # Print features that improved the score when dropped
    print("\n" + "=" * 60)
    print("V FEATURES THAT IMPROVED F1 SCORE WHEN DROPPED:")
    print("=" * 60)
    improved_features = []
    for feature, f1 in sorted_results:
        if baseline_f1 and f1 > baseline_f1:
            improvement = f1 - baseline_f1
            improved_features.append((feature, improvement))

    if improved_features:
        # Sort by improvement
        improved_features.sort(key=lambda x: x[1], reverse=True)
        
        with open(output_file, "a") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("V FEATURES THAT IMPROVED F1 SCORE WHEN DROPPED:\n")
            f.write("=" * 60 + "\n")
            
            for feature, improvement in improved_features:
                line = f"DROP {feature}: +{improvement:.4f}"
                print(line)
                f.write(line + "\n")
        
        print(f"\nConsider dropping these {len(improved_features)} features!")
    else:
        print("No V features improved the score when dropped.")
        with open(output_file, "a") as f:
            f.write("\nNo V features improved the score when dropped.\n")

    print(f"\nAll results saved to {output_file}")