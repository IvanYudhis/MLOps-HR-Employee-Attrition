# ======================================================
# 📦 MLOps HR Employee Attrition – All-in-One Container
# ======================================================
# ---- Includes: Streamlit + MLflow Tracking Server ----

# --- Base image Python 3.12 (slim) ---
FROM python:3.12-slim

# --- Set working directory in the container ---
WORKDIR /app

# --- Copy all project files into the container ---
COPY . .

# --- 🔧 Install system dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Upgrade pip & tools build ---
RUN pip install --upgrade pip setuptools wheel

# --- Install main dependencies ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Create required directories ---
RUN mkdir -p /app/dataset /app/models /app/mlflow_tracking /app/src

# --- 🌍 Environment variables for MLflow Tracking ---
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000
ENV MLFLOW_ARTIFACT_ROOT=/app/mlflow_tracking
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# --- Expose ports : ---
# -> 8501 = Streamlit
# -> 5000 = MLflow tracking server
EXPOSE 8501 5000

# --- Metadata ---
LABEL maintainer="Ivan Yudhistira" \
      project="HR Employee Attrition Predictor" \
      version="1.0" \
      description="Streamlit + MLflow All-in-One Container for HR Analytics"

# --- Copy entrypoint script & set permission ---
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# --- Copy wait-for-mlflow script ---
COPY wait-for-mlflow.sh /app/wait-for-mlflow.sh
RUN chmod +x /app/wait-for-mlflow.sh

# --- JSON-form CMD ---
CMD ["bash", "/app/entrypoint.sh"]