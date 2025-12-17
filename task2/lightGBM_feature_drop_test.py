import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import subprocess
import re

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

# Load remaining features from file
with open("remaining_features.txt", "r") as f:
    remaining_features = [line.strip() for line in f.readlines()]

print(f"Loaded {len(remaining_features)} remaining features")

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

# Features that were originally dropped (kept as reference)
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


def _build_lgbm_classifier(use_gpu: bool):
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
        verbose=-1,  # Suppress output during training
    )
    if use_gpu:
        params.update({"device": "gpu"})
    return lgb.LGBMClassifier(**params)


def train_and_evaluate(feature_to_drop=None):
    """Train model with optional feature dropped and return F1 score"""
    
    # Start with dropping the originally dropped features
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
    
    # Re-create the missing flags
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
    clf = _build_lgbm_classifier(use_gpu=False)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Create Submission
    submission = pd.DataFrame({"OrderID": merged_test_df["OrderID"], "IsRisky": y_pred})
    submission.to_csv("submission_lightgbm.csv", index=False)
    
    # Run evaluation
    result = subprocess.run(
        [".\\evaluate_windows_2.exe", ".\\submission_lightgbm.csv"],
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
        print(f"Could not parse F1 score. Output: {output}")
    
    return f1_score


# Store results
results = {}

# First, get baseline (current remaining features)
print("=" * 60)
print("Testing BASELINE (with current remaining features)")
print("=" * 60)
baseline_f1 = train_and_evaluate(feature_to_drop=None)
results["BASELINE"] = baseline_f1
print(f"BASELINE F1 Score: {baseline_f1}")
print()

# Test dropping each remaining feature (except OrderID and IsRisky which are special)
features_to_test = [f for f in remaining_features if f not in ["OrderID", target_col]]

for i, feature in enumerate(features_to_test):
    print("=" * 60)
    print(f"[{i+1}/{len(features_to_test)}] Testing DROP feature: {feature}")
    print("=" * 60)
    
    # Check if feature exists in the dataset
    if feature not in merged_df.columns:
        print(f"Feature {feature} not found in dataset. Skipping.")
        results[feature] = None
        continue
    
    f1_score = train_and_evaluate(feature_to_drop=feature)
    results[feature] = f1_score
    
    if f1_score is not None:
        diff = f1_score - baseline_f1 if baseline_f1 else 0
        print(f"F1 Score: {f1_score:.4f} (diff from baseline: {diff:+.4f})")
    print()

# Save results to file
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

# Sort by F1 score (descending)
sorted_results = sorted(
    [(k, v) for k, v in results.items() if v is not None],
    key=lambda x: x[1],
    reverse=True
)

with open("feature_drop_test_results.txt", "w") as f:
    f.write("Feature Drop Test Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Baseline F1 Score: {baseline_f1}\n\n")
    f.write("Features sorted by F1 Score when DROPPED (descending):\n")
    f.write("-" * 60 + "\n")
    
    for feature, f1 in sorted_results:
        diff = f1 - baseline_f1 if baseline_f1 else 0
        line = f"DROP {feature}: {f1:.4f} (diff: {diff:+.4f})"
        print(line)
        f.write(line + "\n")

print(f"\nResults saved to feature_drop_test_results.txt")

# Print features that improved the score when dropped
print("\n" + "=" * 60)
print("FEATURES THAT IMPROVED F1 SCORE WHEN DROPPED:")
print("=" * 60)
improved_features = []
for feature, f1 in sorted_results:
    if baseline_f1 and f1 > baseline_f1:
        improvement = f1 - baseline_f1
        improved_features.append((feature, improvement))
        print(f"DROP {feature}: {f1:.4f} (+{improvement:.4f})")

if improved_features:
    print("\n" + "=" * 60)
    print("SUMMARY: Features to consider dropping")
    print("=" * 60)
    for feature, improvement in sorted(improved_features, key=lambda x: x[1], reverse=True):
        print(f"{feature}: +{improvement:.4f}")
else:
    print("No features improved the score when dropped.")