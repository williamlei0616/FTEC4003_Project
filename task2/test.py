import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

# 1. Reload and Merge Data
train_identity = pd.read_csv('./globalmart_train_identity.csv')
train_transaction = pd.read_csv('./globalmart_train_transactions.csv')
train = pd.merge(train_transaction, train_identity, on='OrderID', how='left')

# 2. Feature Selection: Drop columns with > 90% missing rate
missing_rate = train.isnull().mean()
drop_cols = missing_rate[missing_rate > 0.90].index.tolist()
# Also drop ID and Target
drop_cols = drop_cols + ['OrderID', 'IsRisky']
print(f"Dropping {len(drop_cols)} features: {drop_cols}")

X = train.drop(columns=drop_cols)
y = train['IsRisky']

# 3. Feature Engineering (Create New Features)
# Feature A: Frequency of CardInfo1 (Detects repeated usage of same card info)
X['CardInfo1_Count'] = X.groupby('CardInfo1')['CardInfo1'].transform('count')

# Feature B: Order Amount relative to the average for that Billing Region
# (Detects anomalously high orders for a specific region)
X['Amount_Ratio_Region'] = X['OrderAmount'] / X.groupby('BillingRegion')['OrderAmount'].transform('mean')

# 4. Preprocessing
# Encode Categorical Columns
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    # Fill missing categorical with "Unknown" to capture the pattern
    X[col] = X[col].fillna('Unknown')
    le = LabelEncoder()
    # Convert to string to ensure uniformity
    X[col] = le.fit_transform(X[col].astype(str))

# Note: HistGradientBoosting handles NaN in numeric columns automatically, 
# so we don't need to fillna(-999) for numeric columns.

# 5. Model Training with Stratified K-Fold
# We use class_weight='balanced' because fraud is usually rare
clf = HistGradientBoostingClassifier(
    learning_rate=0.05, 
    max_iter=500, 
    max_depth=10, 
    class_weight='balanced', 
    random_state=42
)

# 5-Fold Cross Validation to get a reliable F1 Score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')

print(f"\nCross-Validation F1 Scores: {scores}")
print(f"Average F1 Score: {np.mean(scores):.4f}")

# 6. Train on full data for final submission
clf.fit(X, y)
print("Model trained on full dataset.")