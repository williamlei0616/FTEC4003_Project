import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Load the datasets
train_transactions = pd.read_csv("dataset/globalmart_train_transactions.csv")
train_identity = pd.read_csv("dataset/globalmart_train_identity.csv")
test_transactions = pd.read_csv("dataset/globalmart_test_transactions.csv")
test_identity = pd.read_csv("dataset/globalmart_test_identity.csv")

# Merge with left join (transactions as left table)
merged_df = train_transactions.merge(train_identity, how="left", on="OrderID")
merged_test_df = test_transactions.merge(test_identity, how="left", on="OrderID")

print(f"Merged train shape: {merged_df.shape}")
print(f"Merged test shape: {merged_test_df.shape}")


# --- SMOTE Resampling Logic ---
target_col = "IsRisky"  # Replace with your actual target column name
categorical_columns = [
    "IdentityFeature12",
    "IdentityFeature15",
    "IdentityFeature16",
    "IdentityFeature23",
    "IdentityFeature27",
    "IdentityFeature28",
    "IdentityFeature29",
    "IdentityFeature30",
    "IdentityFeature31",
    "IdentityFeature33",
    "IdentityFeature34",
    "IdentityFeature35",
    "IdentityFeature36",
    "IdentityFeature37",
    "IdentityFeature38",
    "DeviceOS",
    "DeviceModel",
    "PaymentType",
    "CardInfo1",
    "CardInfo2",
    "CardInfo3",
    "CardInfo5",
    "CardNetwork",
    "CardType",
    "BillingRegion",
    "BillingCountry",
    "PayerEmailProvider",
    "RecipientEmailProvider",
    "MatchStatus1",
    "MatchStatus2",
    "MatchStatus3",
    "MatchStatus4",
    "MatchStatus5",
    "MatchStatus6",
    "MatchStatus7",
    "MatchStatus8",
    "MatchStatus9",
]  # Put your categorical column names here

# Separate Features and Target
X = merged_df.drop(columns=[target_col])
y = merged_df[target_col]

# --- Drop High Missing Value Columns ---
missing_threshold = 0.90
missing_series = X.isnull().mean()
cols_to_drop = missing_series[missing_series > missing_threshold].index.tolist()

print(
    f"Dropping {len(cols_to_drop)} columns with > {missing_threshold*100}% missing values."
)
if cols_to_drop:
    print(f"Columns dropped: {cols_to_drop}")
    X = X.drop(columns=cols_to_drop)
    # Update categorical_columns list to remove any that were dropped
    categorical_columns = [
        col for col in categorical_columns if col not in cols_to_drop
    ]

# Handle Missing Values (SMOTE requires no NaNs)
# Track columns with missing values to replicate flags in test set
cols_with_missing_in_train = [col for col in X.columns if X[col].isnull().any()]


def process_missing_values(df, is_train=True):
    # Add binary flag for missingness based on training set columns
    for col in cols_with_missing_in_train:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isnull().astype(int)

    for col in df.columns:
        if col.endswith("_is_missing") or col == "OrderID":
            continue

        if df[col].dtype == "object" or col in categorical_columns:
            df[col] = df[col].fillna("-1")
            df[col] = df[col].astype(str)
        else:
            df[col] = df[col].fillna(-1)
    return df


X = process_missing_values(X, is_train=True)
# Prepare Test Data early to include in encoding mapping
X_test = merged_test_df.copy()
# Drop the same columns from test
if cols_to_drop:
    cols_to_drop_test = [c for c in cols_to_drop if c in X_test.columns]
    X_test = X_test.drop(columns=cols_to_drop_test)

X_test = process_missing_values(X_test, is_train=False)

# --- Phase Key Shifting Encoding (Hypersphere Embedding) ---
print("Applying Phase Key Shifting Encoding...")
for col in categorical_columns:
    if col not in X.columns:
        continue

    # Get unique values from both train and test to ensure consistent mapping
    train_vals = X[col].unique()
    test_vals = X_test[col].unique() if col in X_test.columns else []

    unique_vals = sorted(list(set(train_vals) | set(test_vals)))
    val_to_idx = {val: i for i, val in enumerate(unique_vals)}
    n_unique = len(unique_vals)

    # Function to apply encoding
    def encode_col(series):
        indices = series.map(val_to_idx).fillna(0)
        angles = 2 * np.pi * indices / n_unique
        return np.sin(angles), np.cos(angles)

    # Transform Train
    sin_train, cos_train = encode_col(X[col])
    X[f"{col}_sin"] = sin_train
    X[f"{col}_cos"] = cos_train
    X = X.drop(columns=[col])

    # Transform Test
    if col in X_test.columns:
        sin_test, cos_test = encode_col(X_test[col])
        X_test[f"{col}_sin"] = sin_test
        X_test[f"{col}_cos"] = cos_test
        X_test = X_test.drop(columns=[col])

# Drop OrderID before SMOTE
if "OrderID" in X.columns:
    X = X.drop(columns=["OrderID"])

print("Starting SMOTE resampling...")
# Since we encoded categoricals into numericals, we use standard SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Original shape: {X.shape}")
print(f"Resampled shape: {X_resampled.shape}")
print(f"Class distribution after resampling:\n{y_resampled.value_counts()}")


def _build_xgb_classifier(use_gpu: bool):
    params = dict(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=12,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=True,
    )
    if use_gpu:
        params.update({"tree_method": "gpu_hist"})
    return xgb.XGBClassifier(**params)


# --- XGBoost Training and Prediction ---

# 1. Prepare Training Data
X_train = X_resampled
y_train = y_resampled

# 2. Prepare Test Data
# Ensure column order matches training
# X_test might still have OrderID, and we need to select only feature columns
feature_cols = X_train.columns.tolist()
X_test_final = X_test[feature_cols].copy()

print(f"Training with {len(feature_cols)} features.")

# 4. Train Model
clf = _build_xgb_classifier(use_gpu=False)
clf.fit(X_train, y_train)

# 5. Predict
print("Predicting...")
y_pred = clf.predict(X_test_final)

# 6. Create Submission
submission = pd.DataFrame({"OrderID": merged_test_df["OrderID"], "IsRisky": y_pred})

print(submission.head())
submission.to_csv("submission_xgb.csv", index=False)
print("Submission saved to submission_xgb.csv")
