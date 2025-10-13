# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# --- 1. Load Dataset ---
DATA_PATH = os.path.join("dataset", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)

# --- 2. Pisahkan fitur dan target ---
target_col = "Attrition"
X = df.drop(columns=[target_col])
y = df[target_col]

# --- 3. Encode kolom kategorikal ---
cat_cols = X.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])
print(f"🔠 Encoded categorical columns: {list(cat_cols)}")

# Encode target
y = le.fit_transform(y)  # Yes/No → 1/0

# --- 4. Scaling fitur numerik ---
scaler = StandardScaler()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
X[num_cols] = scaler.fit_transform(X[num_cols])
print(f"📏 Scaled numerical columns: {list(num_cols)}")

# --- 5. Split train & test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Data splitted into train/test successfully!")
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

# --- 6. Simpan hasil preprocessing ---
os.makedirs("models", exist_ok=True)
joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("💾 Preprocessed data saved successfully in 'models/' folder.")