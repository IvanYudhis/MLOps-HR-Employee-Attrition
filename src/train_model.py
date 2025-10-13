# src/train_model.py

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- 1. Load preprocessed data ---
X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("✅ Preprocessed data loaded successfully!")
print(f"🧮 Total features used: {len(feature_columns)}")

# --- 2. Setup MLflow Tracking ---
mlflow.set_tracking_uri("file://" + os.path.abspath("mlflow_tracking"))
mlflow.set_experiment("HR_Attrition_Prediction")

# --- 3. Train Model (Random Forest Optimized) ---
with mlflow.start_run():
    # Hyperparameters
    params = {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "class_weight": "balanced_subsample"
    }

    model = RandomForestClassifier(**params)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Log parameters & metrics
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)

    # Save model locally
    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_model.pkl"
    joblib.dump(model, model_path)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    print(f"💾 Model saved successfully to {model_path}")
    print("✅ MLflow run completed!")

print("✅ Training and optimization finished successfully!")