# !/bin/bash
set -e

# ============================
# 🌟 ENTRYPOINT FOR CONTAINER
# ============================

# --- Create folder log ---
mkdir -p /app/logs

echo "$(date +"%Y-%m-%d %H:%M:%S") | 🎨 Launching Streamlit app..."
exec streamlit run streamlit_app.py \
  --server.port="${STREAMLIT_PORT:-8501}" \
  --server.address="${STREAMLIT_HOST:-0.0.0.0}"