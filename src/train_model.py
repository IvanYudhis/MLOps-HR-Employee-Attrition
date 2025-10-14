# src/train_model.py

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import os
import platform

# --- 1. Load preprocessed data ---
X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("✅ Preprocessed data loaded successfully!")
print(f"🧮 Total features used: {len(feature_columns)}")

# --- 2a. Setup MLflow Tracking (Windows Only) ---
tracking_dir = Path("mlflow_tracking").resolve()
tracking_dir.mkdir(parents=True, exist_ok=True)
tracking_uri = tracking_dir.as_uri()

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("HR_Attrition_Prediction")

# --- 2b. Setup MLflow Tracking (Windows + Cloud/Linux)
def get_tracking_uri():
    path = Path("mlflow_tracking").resolve()
    path.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        return path.as_uri()  # file:///C:/...
    else:
        return f"file://{path}"

mlflow.set_tracking_uri(get_tracking_uri())
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
    mlflow.log_metric("accuracy", acc)
    mlflow.log_params(model.get_params())

    # Save model locally
    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_model.pkl"
    joblib.dump(model, model_path)
    
    # Prepare signature and example
    sample_input = X_test.iloc[:5]
    sample_output = model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)
    
    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_model",
        input_example=sample_input,
        signature=signature
    )
    
    print(f"💾 Model saved successfully to {model_path}")
    print("✅ MLflow run completed!")

print("✅ Training and optimization finished successfully!")