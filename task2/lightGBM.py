import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA

# Load the datasets
train_transactions = pd.read_csv("dataset/globalmart_train_transactions.csv")
train_identity = pd.read_csv("dataset/globalmart_train_identity.csv")

# Merge with left join (transactions as left table)
merged_df = train_transactions.merge(train_identity, how="left", on="OrderID")
# Display the result
print(merged_df.head())
print(f"Merged dataset shape: {merged_df.shape}")


# --- SMOTE Resampling Logic ---
target_col = "IsRisky"  # Replace with your actual target column name
categorical_columns = [
    "IdentityFeature12",
    "IdentityFeature13",
    "IdentityFeature14",
    "IdentityFeature15",
    "IdentityFeature16",
    "IdentityFeature17",
    "IdentityFeature18",
    "IdentityFeature19",
    "IdentityFeature20",
    "IdentityFeature21",
    "IdentityFeature22",
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
X = merged_df.drop(
    columns=[
        target_col,
        "V2",
        "V5",
        "V7",
        "V9",
        "V10",
        "V12",
        "V15",
        "V16",
        "V18",
        "V19",
        "V21",
        "V22",
        "V24",
        "V25",
        "V28",
        "V29",
        "V31",
        "V32",
        "V33",
        "V34",
        "V35",
        "V38",
        "V39",
        "V42",
        "V43",
        "V45",
        "V46",
        "V49",
        "V50",
        "V51",
        "V52",
        "V53",
        "V55",
        "V57",
        "V58",
        "V60",
        "V61",
        "V63",
        "V64",
        "V66",
        "V69",
        "V71",
        "V72",
        "V73",
        "V74",
        "V75",
        "V77",
        "V79",
        "V81",
        "V83",
        "V84",
        "V85",
        "V87",
        "V90",
        "V92",
        "V93",
        "V94",
        "V95",
        "V96",
        "V97",
        "V98",
        "V99",
        "V100",
        "V101",
        "V102",
        "V103",
        "V104",
        "V105",
        "V106",
        "V109",
        "V110",
        "V112",
        "V113",
        "V114",
        "V116",
        "V118",
        "V119",
        "V122",
        "V125",
        "V126",
        "V128",
        "V131",
        "V132",
        "V133",
        "V134",
        "V135",
        "V137",
        "V140",
        "V141",
        "V143",
        "V144",
        "V145",
        "V146",
        "V148",
        "V149",
        "V150",
        "V151",
        "V152",
        "V153",
        "V154",
        "V155",
        "V157",
        "V158",
        "V159",
        "V161",
        "V163",
        "V164",
        "V167",
        "V168",
        "V170",
        "V172",
        "V174",
        "V177",
        "V179",
        "V181",
        "V183",
        "V184",
        "V186",
        "V189",
        "V190",
        "V191",
        "V192",
        "V193",
        "V194",
        "V195",
        "V196",
        "V197",
        "V199",
        "V200",
        "V201",
        "V202",
        "V204",
        "V206",
        "V208",
        "V211",
        "V212",
        "V213",
        "V214",
        "V216",
        "V217",
        "V219",
        "V222",
        "V225",
        "V227",
        "V230",
        "V231",
        "V232",
        "V233",
        "V236",
        "V237",
        "V239",
        "V241",
        "V242",
        "V243",
        "V244",
        "V245",
        "V246",
        "V247",
        "V248",
        "V249",
        "V251",
        "V254",
        "V255",
        "V256",
        "V259",
        "V262",
        "V263",
        "V265",
        "V268",
        "V269",
        "V270",
        "V272",
        "V273",
        "V275",
        "V276",
        "V278",
        "V279",
        "V280",
        "V282",
        "V287",
        "V288",
        "V290",
        "V292",
        "V293",
        "V295",
        "V298",
        "V299",
        "V300",
        "V302",
        "V304",
        "V306",
        "V308",
        "V311",
        "V312",
        "V313",
        "V315",
        "V316",
        "V317",
        "V318",
        "V319",
        "V321",
        "V322",
        "V323",
        "V324",
        "V325",
        "V326",
        "V327",
        "V328",
        "V329",
        "V330",
        "V331",
        "V332",
        "V333",
        "V334",
        "V335",
        "V336",
        "V337",
        "V338",
        "V339",
    ]
)
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

for col in X.columns:
    if col in cols_with_missing_in_train:
        # Add binary flag for missingness
        X[f"{col}_is_missing"] = X[col].isnull().astype(int)

    if X[col].dtype == "object" or col in categorical_columns:
        X[col] = X[col].fillna("-1")
        X[col] = X[col].astype(str)
    else:
        X[col] = X[col].fillna(0)


# --- Normalize Numerical Features ---
# Identify numerical columns (exclude categorical, OrderID, and missing flags)
numerical_cols = [
    col
    for col in X.columns
    if col not in categorical_columns
    and col != "OrderID"
    and not col.endswith("_is_missing")
    and X[col].dtype != "object"
]

# print(f"Normalizing {len(numerical_cols)} numerical features...")
# scaler = StandardScaler()
# X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("Starting SMOTE resampling...")
if categorical_columns:
    categorical_indices = [X.columns.get_loc(col) for col in categorical_columns]
    smote = SMOTENC(categorical_features=categorical_indices, random_state=42)
else:
    smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)
# X_resampled, y_resampled = X, y
print(f"Original shape: {X.shape}")
print(f"Resampled shape: {X_resampled.shape}")
print(f"Class distribution after resampling:\n{y_resampled.value_counts()}")

test_transactions = pd.read_csv("dataset/globalmart_test_transactions.csv")
test_identity = pd.read_csv("dataset/globalmart_test_identity.csv")
# Merge with left join (transactions as left table)
merged_test_df = test_transactions.merge(test_identity, how="left", on="OrderID")
# Display the result
print(merged_test_df.head())
print(f"Merged test dataset shape: {merged_test_df.shape}")


def _build_lgbm_classifier(use_gpu: bool):
    import lightgbm as lgb

    params = dict(
        n_estimators=3000,
        num_leaves=40,
        max_depth=15,
        min_child_samples=70,
        reg_alpha=0.1,  # L1 Regularization
        reg_lambda=0.1,  # L2 Regularization
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
    )
    if use_gpu:
        params.update({"device": "gpu"})
    return lgb.LGBMClassifier(**params)


# --- LightGBM Training and Prediction ---

# 1. Prepare Training Data
# Drop OrderID if it exists, as it's not a feature
feature_cols = [col for col in X_resampled.columns if col != "OrderID"]
X_train = X_resampled[feature_cols].copy()
y_train = y_resampled

# 2. Prepare Test Data
# Identify base features (excluding the generated missing flags)
base_features = [c for c in feature_cols if not c.endswith("_is_missing")]
X_test = merged_test_df[base_features].copy()

# Re-create the missing flags
for col in cols_with_missing_in_train:
    # Only create flag if the base column is in our test set
    if col in X_test.columns:
        X_test[f"{col}_is_missing"] = X_test[col].isnull().astype(int)

# Ensure column order matches training
X_test = X_test[feature_cols]

# Apply the same missing value handling to Test
for col in X_test.columns:
    if col.endswith("_is_missing"):
        continue

    # Check dtype in X_train to ensure consistency
    if X_train[col].dtype == "object" or X_train[col].dtype.name == "category":
        X_test[col] = X_test[col].fillna("-1")
        X_test[col] = X_test[col].astype(str)
    else:
        X_test[col] = X_test[col].fillna(-1)

# Apply the same normalization to Test
# Ensure we only transform columns that exist in X_test (should be all of them if logic is correct)
valid_numerical_cols = [c for c in numerical_cols if c in X_test.columns]
# if valid_numerical_cols:
#     X_test[valid_numerical_cols] = scaler.transform(X_test[valid_numerical_cols])

# 3. Convert Categorical Columns to 'category' dtype for LightGBM
# LightGBM handles 'category' dtype efficiently
for col in X_train.columns:
    if X_train[col].dtype == "object":
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

print(f"Training with {len(feature_cols)} features.")

# 4. Train Model
clf = _build_lgbm_classifier(use_gpu=False)
clf.fit(X_train, y_train)

from sklearn.metrics import f1_score, classification_report

print("Calculating metrics on Resampled Training Data...")
y_train_pred = clf.predict(X_train)
train_f1 = f1_score(y_train, y_train_pred)
print(f"Training F1 Score (Resampled): {train_f1:.4f}")

print("\nClassification Report (Training):")
print(classification_report(y_train, y_train_pred))

# 5. Predict
print("Predicting...")
y_pred = clf.predict(X_test)

# 6. Create Submission
submission = pd.DataFrame({"OrderID": merged_test_df["OrderID"], "IsRisky": y_pred})

print(submission.head())
submission.to_csv("submission_lightgbm.csv", index=False)
print("Submission saved to submission_lightgbm.csv")
