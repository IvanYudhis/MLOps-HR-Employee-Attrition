import os
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

print("⏳ Checking MLflow configuration...\n")

# --- Tracking default location ---
tracking_dir = Path("mlflow_tracking").resolve()
tracking_dir.mkdir(parents=True, exist_ok=True)
tracking_uri = tracking_dir.as_uri().replace("\\", "/")

print(f"✅ Found tracking folder: {tracking_dir}")
print(f"📡 MLflow tracking URI set to: {tracking_uri}")

# -- Set tracking URI ---
mlflow.set_tracking_uri(tracking_uri)

# --- Create / Find existing experiment ---
experiment_name = "HR_Attrition_MLflow_Test"
try:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
        print(f"➕ Created new experiment: {experiment_name} (ID: {exp_id})")
    else:
        print(f"✅ Found existing experiment: {experiment_name} (ID: {exp.experiment_id})")
        exp_id = exp.experiment_id

    # -- Test simple log --
    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.log_param("check_param", "ok")
        mlflow.log_metric("check_metric", 0.99)
        print("🎉 Successfully logged test run to MLflow!")

except Exception as e:
    print(f"⚠️ Error while connecting to MLflow:\n   {e}")

print("\n🔎 Steps to view MLflow UI :")
print("1️⃣ Run: mlflow ui --backend-store-uri mlflow_tracking")
print("2️⃣ Then open → http://localhost:5000")