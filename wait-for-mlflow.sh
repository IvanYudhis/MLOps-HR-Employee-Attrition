# ======================
# 🕒 WAIT-FOR-MLFLOW.SH
# ======================
# Utility script to wait until MLflow Tracking Server is ready
# --- !/bin/sh ---

MLFLOW_URL=${MLFLOW_TRACKING_URI:-http://mlflow:5000}
MAX_RETRIES=30
RETRY_INTERVAL=5

echo "⏳ Waiting for MLflow Server at: $MLFLOW_URL ..."

for i in $(seq 1 $MAX_RETRIES); do
    if curl -s "$MLFLOW_URL/api/2.0/mlflow/experiments/list" > /dev/null; then
        echo "✅ MLflow is Up! Continuing..."
        exit 0
    fi
    echo "⏰ Attempt $i/$MAX_RETRIES: MLflow is not ready yet! Retrying in ${RETRY_INTERVAL}s..."
    sleep $RETRY_INTERVAL
done

echo "❌ MLflow did not start within $(($MAX_RETRIES * $RETRY_INTERVAL)) seconds."
exit 1