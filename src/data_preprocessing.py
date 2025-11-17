# =========================
# src/data_preprocessing.py
# =========================

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# --- Load Dataset ---
DATA_PATH = os.path.join("dataset", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)

# --- Remove useless / constant columns ---
drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# --- Separate features and targets ---
target_col = "Attrition"
X = df.drop(columns=[target_col])
y = df[target_col]

# --- Encode categorical columns individually ---
cat_cols = X.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
print(f"🔠 Encoded categorical columns: {list(cat_cols)}")

# --- Encode target Yes/No → 1/0 ---
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# --- Scaling numerical features ---
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
print(f"📏 Scaled numerical columns ({len(num_cols)}): {num_cols[:8]}{' ...' if len(num_cols)>8 else ''}")

# --- Split train & test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n✅ Data splitted into train/test successfully!")
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

# --- Saving preprocessing results ---
os.makedirs("models", exist_ok=True)
joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(target_encoder, "models/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
joblib.dump(num_cols, "models/num_columns.pkl")
joblib.dump(label_encoders, "models/column_label_encoders.pkl")

print("\n💾 Preprocessed data & encoders saved successfully in 'models/' folder.")
print(f"Total features after encoding: {len(X.columns)}")