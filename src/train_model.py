# ==================
# src/train_model.py
# ==================

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import os
import platform
import socket

# --- 📂 Load preprocessed data ---
X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("✅ Preprocessed data loaded successfully!")
print(f"🧮 Total features used: {len(feature_columns)}")

# --- 🌐 MLflow URI detection (Local & Docker) ---
def detect_mlflow_uri():
    """Deteksi secara otomatis apakah dijalankan di lokal atau Docker."""
    # -- Check variable environment from Docker --
    env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip().lower()

    # -- Check if the service 'mlflow' can be resolved (Docker Compose) --
    try:
        socket.gethostbyname("mlflow")
        in_docker = True
    except socket.gaierror:
        in_docker = False

    if "http://mlflow" in env_uri or in_docker:
        uri = "http://mlflow:5000"
        mode = "docker"
    else:
        # -- Local mode Windows/Linux --
        local_path = Path("mlflow_tracking").resolve()
        local_path.mkdir(parents=True, exist_ok=True)
        uri = local_path.as_uri() if platform.system() == "Windows" else f"file://{local_path}"
        mode = "local"
    return uri, mode

tracking_uri, mode = detect_mlflow_uri()
tracking_dir = Path("mlflow_tracking").resolve()
tracking_dir.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("HR_Attrition_Prediction")

if mode == "docker":
    print(f"🌐 Running in Docker mode → MLflow Tracking: {tracking_uri}")
else:
    print(f"💾 Running locally → MLflow Tracking: {tracking_uri}")

# --- Train Model (Random Forest Optimized) ---
with mlflow.start_run(run_name="RandomForest_Training"):
    # -- Hyperparameters --
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

    # -- Train model --
    model.fit(X_train, y_train)

    # -- Predict --
    y_pred = model.predict(X_test)

    # -- Evaluate performance --
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # -- Log model to MLflow --
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", acc)

    # -- Save model & scaler --
    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_model.pkl"
    joblib.dump(model, model_path)
    scaler_path = "models/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # -- Prepare signature and example --
    sample_input = X_test.iloc[:5]
    sample_output = model.predict(sample_input)
    signature = infer_signature(sample_input, sample_output)
    
    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_model",
        input_example=sample_input,
        signature=signature
    )
    
    print(f"💾 Model saved locally at: {model_path}")
    print(f"📏 Scaler saved at: {scaler_path}")
    print("✅ MLflow run completed successfully!")

print("\n🎉 Training and optimization finished successfully!")
print("---------------------------------------------------")
print(f"Tracking URI  : {tracking_uri}")
print(f"Experiment     : HR_Attrition_Prediction")
print(f"Model File     : {model_path}")
print(f"Scaler File    : {scaler_path}")
print("---------------------------------------------------")