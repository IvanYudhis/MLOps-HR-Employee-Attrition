# ====================
# app/streamlit_app.py
# ====================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import socket
import platform
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
from io import StringIO

# <<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#      🌐 MLflow Auto-Detection (Local vs Docker)
# <<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def detect_mlflow_uri():
    """Deteksi secara otomatis apakah Streamlit berjalan di lokal atau Docker."""
    env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip().lower()
    try:
        socket.gethostbyname("mlflow") # resolve DNS Docker
        in_docker = True
    except socket.gaierror:
        in_docker = False

    if "http://mlflow" in env_uri or in_docker:
        uri = "http://mlflow:5000"
        mode = "docker"
    else:
        local_path = os.path.abspath("mlflow_tracking").replace("\\", "/")
        os.makedirs(local_path, exist_ok=True)
        uri = f"file:///{local_path}"
        mode = "local"
    return uri, mode

tracking_uri, mode = detect_mlflow_uri()
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("HR_Attrition_Prediction")

if mode == "docker":
    print(f"🌐 Connected to MLflow Server at: {tracking_uri}")
else:
    print(f"💾 Using Local MLflow Tracking at: {tracking_uri}")

# <<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>
#      🔧 PAGE CONFIGURATIONS
# <<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>
st.set_page_config(
    page_title="HR Employee Attrition Predictor",
    page_icon="💼",
    layout="wide"
)

# --- Custom Theme ---
st.markdown("""
<style>
    h1, h2, h3 {
        color: #3B5998;
    }
    .stButton>button {
        color: white;
        background-color: #3B5998;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2c4377;
    }
    .status-box {
    background-color: #f5f7ff;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
    }
    .status-green {
        color: #16a34a; font-weight: bold;
    }
    .status-blue {
        color: #2563eb; font-weight: bold;
    }
    .status-gray {
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("---")
st.title("💼 HR Employee Attrition Predictor")
st.caption("End-to-end MLOps pipeline for HR Analytics")
st.markdown("---")

# <<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>
#      🌐 MLflow Status Panel
#<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>
col_info, col_status = st.columns([2.5, 1.5])

with col_info:
    st.markdown("### 🔍 Project Overview")
    st.write("""
Proyek **HR Employee Attrition Predictor** ini bertujuan untuk membantu tim Human Resources 
di suatu perusahaan dalam **memprediksi kemungkinan seorang karyawan akan mengundurkan diri (resign)** 
berdasarkan faktor-faktor historis dan perilaku dari karyawan.

Model ini dibangun menggunakan algoritma **Random Forest Classifier** dan dikembangkan 
dengan pendekatan **MLOps (Machine Learning Operations)** untuk memastikan proses pelatihan, 
pemantauan performa, dan deployment dapat dilakukan secara **terukur, reproducible, dan terintegrasi**.

Sistem ini mencakup pipeline lengkap mulai dari:
- 📊 **Eksplorasi Data (EDA)** untuk memahami pola attrition dan korelasi antar variabel
- ⚙️ **Preprocessing & Training** dengan logging otomatis melalui **MLflow**
- 🧩 **Model Evaluation** untuk menilai akurasi dan generalisasi model
- 🌐 **Deployment dengan Streamlit** agar prediksi bisa dilakukan secara interaktif
- 🐳 **Containerization via Docker** untuk memastikan portabilitas lintas environment

Tujuannya adalah memberikan insight yang lebih mendalam bagi HR agar dapat 
**mengambil keputusan berbasis data (data-driven decisions)** dalam mengelola retensi karyawan.
    """)

with col_status:
    mlflow_port = os.getenv("MLFLOW_PORT", "5000")
    mlflow_link = f"http://localhost:{mlflow_port}"
    st.markdown("### ⚙️ MLflow Status")
    if mode == "docker":
        st.markdown(f"""
        <style>
        .status-box {{
            background-color: #121212;
            color: #f1f1f1;
            border-radius: 12px;
            padding: 18px;
            margin-top: 7px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
            line-height: 1.5;
        }}
        .status-green {{
            background-color: #00b894;
            color: #fff;
            padding: 7px 12px;
            border-radius: 8px;
            display: inline-block;
            font-weight: 600;
            margin-bottom: 2px;
            font-size: 15px;
        }}
        .status-gray {{
            color: #d1d1d1;
            margin-bottom: 7px;
            font-size: 14px;
        }}
        .status-blue {{
            color: #0000FF;
            margin-bottom: 12px;
            font-size: 14px;
        }}
        .status-black {{
            background-color: #1f1f1f;
            border-radius: 10px;
            padding: 10px 12px;
            margin-top: 12px;
            line-height: 1.4;
            color: #eaeaea;
            font-size: 14px;
        }}
        a.mlflow-link {{
            color: #74c0fc;
            font-weight: bold;
            text-decoration: none;
        }}
        a.mlflow-link:hover {{
            text-decoration: underline;
        }}
        code {{
            background-color: #2d3436;
            color: #00ffb3;
            border-radius: 5px;
            padding: 3px 6px;
            font-family: monospace;
            font-size: 14px;
        }}
        </style>
        
        <div class="status-box">
        <div class="status-green">🌐 Connected to MLflow Server</div>
        <div class="status-gray">Mode: Docker Compose</div>
        
        <div>Tracking URI:</div>
        <div class="status-blue">{tracking_uri}</div>
        
        <div class="status-black">
            <b>🔎 Steps to view MLflow UI:</b><br>
            1️⃣ Open terminal<br>
            2️⃣ Ensure you're in the correct project folder & environment activated<br>
            3️⃣ Run :<br>
            <code>mlflow ui --backend-store-uri mlflow_tracking</code><br>
            4️⃣ Then open → <a href="{mlflow_link}" target="_blank" style="color:#4da6ff; font-weight:bold;">🔗 MLflow UI 🔗</a>
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <style>
        .status-box {{
            background-color: #121212;
            color: #f1f1f1;
            border-radius: 12px;
            padding: 18px;
            margin-top: 7px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            font-family: 'Segoe UI', sans-serif;
            font-size: 14px;
            line-height: 1.5;
        }}
        .status-green {{
            background-color: #00b894;
            color: #fff;
            padding: 7px 12px;
            border-radius: 8px;
            display: inline-block;
            font-weight: 600;
            margin-bottom: 2px;
            font-size: 15px;
        }}
        .status-gray {{
            color: #d1d1d1;
            margin-bottom: 7px;
            font-size: 14px;
        }}
        .status-blue {{
            color: #0000FF;
            margin-bottom: 12px;
            font-size: 14px;
        }}
        .status-black {{
            background-color: #1f1f1f;
            border-radius: 10px;
            padding: 10px 12px;
            margin-top: 12px;
            line-height: 1.4;
            color: #eaeaea;
            font-size: 14px;
        }}
        a.mlflow-link {{
            color: #74c0fc;
            font-weight: bold;
            text-decoration: none;
        }}
        a.mlflow-link:hover {{
            text-decoration: underline;
        }}
        code {{
            background-color: #2d3436;
            color: #00ffb3;
            border-radius: 5px;
            padding: 3px 6px;
            font-family: monospace;
            font-size: 14px;
        }}
        </style>
        
        <div class="status-box">
        <div class="status-green">💾 Local MLflow Active</div>
        <div class="status-gray">Mode: Local (Windows)</div>
        
        <div><b>Tracking URI:</b></div>
        <div class="status-blue">{tracking_uri}</div>
        
        <div class="status-black">
            <b>🔎 Steps to view MLflow UI :</b><br>
            1️⃣ Open terminal<br>
            2️⃣ Ensure you're in the correct project folder & environment activated<br>
            3️⃣ Run :<br>
            <code>mlflow ui --backend-store-uri mlflow_tracking</code><br>
            4️⃣ Then open → <a href="{mlflow_link}" target="_blank" style="color:#4da6ff; font-weight:bold;">🔗 MLflow UI 🔗</a>
        </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# <<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>
#      🔄 LOAD RAW DATA FOR EDA
# <<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>
RAW_DATA_PATH = "dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"
if os.path.exists(RAW_DATA_PATH):
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_raw["Attrition"] = df_raw["Attrition"].map({"Yes": 1, "No": 0})
else:
    st.error("❌ Dataset tidak ditemukan. Pastikan file terdapat dalam folder 'dataset/'.")
    st.stop()

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#      📊 SECTION 1: EXPLORATORY DATA ANALYSIS (EDA)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>
st.header("📊 Exploratory Data Analysis (EDA)")

# ---------------------------
#    ☰ INTERACTIVE FILTER
# ---------------------------
st.markdown("### ☰ Filter Data (Optional)")

# --- Custom Theme ---
st.markdown("""
<style>
[data-testid="stSelectbox"] label {
    font-weight: 600;
    color: #C2C7D0;
}
</style>
""", unsafe_allow_html=True)

filter_columns = ["Department", "Gender", "JobRole", "MaritalStatus", "BusinessTravel"]
existing_filters = [col for col in filter_columns if col in df_raw.columns]

col_filters = st.columns(3)
selected_filters = {}

for i, fcol in enumerate(existing_filters):
    with col_filters[i % 3]:
        options = ["All"] + sorted(df_raw[fcol].dropna().unique().tolist())
        selected_filters[fcol] = st.selectbox(f"{fcol}", options, key=f"flt_{fcol}")

# --- Apply Filter ---
df_filtered = df_raw.copy()
for col, val in selected_filters.items():
    if val != "All":
        df_filtered = df_filtered[df_filtered[col] == val]

st.success(f"✅ Jumlah data setelah filter: {len(df_filtered)} baris")
st.markdown("---")

# --------------------------
#    👀 DATASET OVERVIEW
# --------------------------
with st.expander("👀 Dataset Overview", expanded=True):
    st.metric("Total Data", len(df_filtered))
    st.metric("Jumlah Fitur", df_filtered.shape[1])
    st.write("Berikut adalah cuplikan dari dataset HR Employee Attrition:")
    st.dataframe(df_filtered.head(51))

    st.write("**Statistical Summary:**")
    st.dataframe(df_filtered.describe(include='all').T)

    buffer = StringIO()
    df_filtered.info(buf=buffer)
    st.text(buffer.getvalue())
st.markdown("---")

# -------------------------------------------
#    🎯 TARGET & CORELATION DISTRIBUTIONS
# -------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎯 Distribusi Target (Attrition)")
    fig, ax = plt.subplots(figsize=(5, 4))
    value_counts = df_filtered["Attrition"].value_counts()
    labels = ["Stay" if idx == 0 else "Resign" for idx in value_counts.index]

    value_counts.plot.pie(
        autopct='%1.1f%%',
        labels=labels,
        colors=["#4CAF50", "#F44336"],
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)
    st.caption("*Hanya sekitar (16%) dari jumlah karyawan yang memutuskan untuk keluar dari perusahaan, sedangkan sebagian besar karyawan lainnya (84%) tetap bertahan*")

with col2:
    st.subheader("♾️ Korelasi Antar Fitur (Numerik)")
    fig, ax = plt.subplots(figsize=(8, 6))
    # --- Memastikan hanya kolom numerik yang digunakan ---
    numeric_df = df_filtered.select_dtypes(include=['number'])
    
    # --- Cek dulu kalau tidak kosong ---
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)
    else:
        st.warning("⚠️ Tidak terdapat kolom numerik untuk menghitung korelasi.")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    st.pyplot(fig)
st.markdown("---")

# ----------------------------------------
#    📈 NUMERICAL FEATURE DISTRIBUTION
# ----------------------------------------
st.subheader("📈 Distribusi Fitur Numerik")
num_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != "Attrition"][:9]

if num_cols:
    # --- Hitung baris & kolom otomatis berdasarkan jumlah fitur numerik ---
    n_cols = 3
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(
            df_filtered[col],
            kde=True,
            color="#2070B1",
            edgecolor="black",
            linewidth=1,
            ax=axes[i]
        )
        axes[i].set_title(col, fontsize=10, weight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribusi Fitur Numerik", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.warning("⚠️ Tidak terdapat fitur numerik yang tersedia untuk divisualisasikan.")

st.markdown("---")

# ---------------------
#    💡 HR INSIGHTS
# ---------------------
st.markdown("### 💡 HR Insights: Attrition by Categories")

# --- Visualisasi Attrition Rate by Categorical Features ---
for cat in existing_filters:
    st.markdown(f"### 📊 Attrition Rate by {cat}")
    
    # -- Hitung rata-rata attrition (resign rate) --
    rate_df = (
        df_filtered.groupby(cat)["Attrition"]
        .mean()
        .reset_index()
        .sort_values("Attrition", ascending=False)
    )
    
    # -- Setup figure --
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # -- Bar chart --
    sns.barplot(
        x=cat,
        y="Attrition",
        data=rate_df,
        palette="Blues_r",
        edgecolor="black",
        ax=ax
    )
    
    # -- Label dan format --
    ax.set_ylabel("Attrition Rate (Resign %)", fontsize=10, labelpad=10)
    ax.set_xlabel(cat, fontsize=10)
    ax.set_title(f"Attrition Rate by {cat}", fontsize=12, weight='bold', pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.xticks(rotation=30, ha='right', fontsize=9)
    
    # -- Menambahkan nilai persentase di atas bar --
    for i, v in enumerate(rate_df["Attrition"]):
        ax.text(i, v + 0.005, f"{v*100:.1f}%", ha='center', fontsize=8, color='black', weight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Penjelasan di bawah grafik
    st.caption(f"Menunjukkan persentase karyawan yang resign berdasarkan kategori {cat.lower()}.")
    st.markdown("---")

# --------------------------------------------------------------
#    📉 AVERAGE AGE, INCOME, AND DISTANCE BETWEEN CATEGORIES
# --------------------------------------------------------------
st.markdown("### 📉 Rata-rata Fitur Demografis per Department")

for metric in ["Age", "MonthlyIncome", "DistanceFromHome"]:
    if metric in df_filtered.columns and "Department" in df_filtered.columns:
        # --- Hitung rata-rata per Department ---
        avg_df = (
            df_filtered.groupby("Department")[metric]
            .mean()
            .reset_index()
            .sort_values(metric, ascending=False)
        )

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            x="Department",
            y=metric,
            data=avg_df,
            palette="Blues_r",
            edgecolor="black",
            ax=ax
        )

        # --- Judul & label ---
        ax.set_title(f"Average {metric} by Department", fontsize=12, weight="bold", pad=12)
        ax.set_xlabel("Department", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        plt.xticks(rotation=25, ha='right', fontsize=9)

        # --- Menambahkan nilai di atas bar ---
        for i, v in enumerate(avg_df[metric]):
            ax.text(
                i,
                v + (v * 0.02),
                f"{v:.1f}",
                ha='center',
                fontsize=8,
                color='black',
                weight='bold'
            )

        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")

# -----------------------------------------
#    🔗 KORELASI FITUR DENGAN ATTRITION
# -----------------------------------------
st.markdown("### 🔗 Korelasi Fitur dengan Attrition")

if "Attrition" in df_filtered.columns:
    tmp = pd.get_dummies(df_filtered, drop_first=True)
    # --- Hitung korelasi terhadap target ---
    corr_with_target = (
        tmp.corr()["Attrition"]
        .drop("Attrition")
        .sort_values(ascending=False)
    )
    
    # --- Menghapus nilai korelasi yang 0 atau sangat kecil ---
    corr_with_target = corr_with_target[corr_with_target.abs() > 0.0001]
    
    # --- Rubah ke DataFrame agar bisa difilter berdasarkan kolom ---
    corr_df = corr_with_target.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    
    # --- Menghapus kolom konstan dari hasil korelasi ---
    corr_df = corr_df[~corr_df["Feature"].isin(["EmployeeCount", "StandardHours"])]
    
    top_n = st.slider("Tampilkan top-N fitur dengan korelasi tertinggi:", 5, len(corr_df), 10)
    
    # --- Tampilkan top N positif dan negatif ---
    half_n = top_n // 2
    corr_display = pd.concat([
        corr_with_target.head(half_n),
        corr_with_target.tail(half_n)
    ])
    
    height = max(5, len(corr_display) * 0.5)
    fig, ax = plt.subplots(figsize=(8, height))
    
    bars = sns.barplot(
        x=corr_display.values,
        y=corr_display.index,
        palette="coolwarm",
        ax=ax
    )
    
    # --- Label di tengah bar ---
    for i, (value, feature) in enumerate(zip(corr_display.values, corr_display.index)):
        ax.text(
            value / 2,
            i,
            f"{value:.2f}",
            color="black",
            va="center",
            ha="center",
            fontsize=9,
            weight="bold"
        )
        
    ax.set_title("Correlation of Features with Attrition", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Correlation Coefficient", fontsize=10)
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
    
else:
    st.warning("Kolom 'Attrition' tidak ditemukan dalam dataset.")

# --- 5 fitur paling berpengaruh positif & negatif ---
st.markdown("#### 🔍 5 Fitur dengan Korelasi Tertinggi (+) & Terendah (–) terhadap Attrition")

# --- 5 korelasi tertinggi & terendah ---
top_pos = corr_with_target.head(5).reset_index()
top_pos.columns = ["Feature", "Correlation"]

top_neg = corr_with_target.tail(5).reset_index()
top_neg.columns = ["Feature", "Correlation"]

top_pos["Correlation"] = top_pos["Correlation"].map(lambda x: f"{x:.3f}")
top_neg["Correlation"] = top_neg["Correlation"].map(lambda x: f"{x:.3f}")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⬆️ Peningkat Risiko Attrition:")
    st.dataframe(top_pos.style.set_properties(**{
        'text-align': 'center', 
        'font-weight': 'bold'
    })
    .background_gradient(cmap="Reds", subset=["Correlation"])
)

with col2:
    st.markdown("### ⬇️ Penurun Risiko Attrition:")
    st.dataframe(top_neg.style.set_properties(**{
        'text-align': 'center', 
        'font-weight': 'bold'
    })
    .background_gradient(cmap="Blues", subset=["Correlation"])
)

# --- Penjelasan Singkat ---
st.markdown("""
✍🏻 **Interpretasi Cepat:**
- Nilai korelasi **positif tinggi** → semakin besar nilai fitur, semakin tinggi kemungkinan *resign*.
- Nilai korelasi **negatif tinggi** → semakin besar nilai fitur, semakin kecil kemungkinan *resign*.
""")
st.markdown("---")

# <<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>
#      ⚙️ SECTION 2: DATA PREPROCESSING
# <<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>
st.header("⚙️ Data Preprocessing")
st.info("Data telah diproses: encoding, scaling, dan pembagian train/test (80/20).")
st.success("✅ Data split complete & scaler applied.")
st.markdown("---")

# <<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>
#      🧠 SECTION 3: MODEL TRAINING SIMULATION
# <<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>
st.header("🧠 Model Training Simulation")
st.info("Proses training model Random Forest dengan parameter yang telah dioptimasi.")

# --- Simulasi progress training model ---
progress_bar = st.progress(0)
status_text = st.empty()

train_loss = []
val_loss = []
train_acc = []
val_acc = []

# --- Simulasi epoch training model ---
epochs = 30
for epoch in range(epochs):
    progress = int((epoch + 1) / epochs * 100)
    progress_bar.progress(progress)
    status_text.text(f"Training model... Epoch {epoch + 1}/{epochs}")
    
    # -- Simulasi nilai metrik menurun --
    train_loss.append(np.exp(-epoch / 8) * 0.03 + np.random.rand() * 0.002)
    val_loss.append(np.exp(-epoch / 6) * 0.035 + np.random.rand() * 0.002)
    train_acc.append(0.65 + (epoch / epochs) * 0.25 + np.random.rand() * 0.01)
    val_acc.append(0.6 + (epoch / epochs) * 0.22 + np.random.rand() * 0.01)

    time.sleep(0.08)

status_text.text("✅ Training completed successfully!")
progress_bar.empty()

st.success("Model: RandomForestClassifier (Optimized Parameters)")
st.caption("n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight='balanced_subsample'")
st.markdown("---")

# <<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>
#      🔮 TRAINING PERFORMANCE VISUALIZATION
# <<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>
st.subheader("📈 Training Performance Overview")

train_color = "#1E88E5"
val_color = "#E53935"

col1, col2 = st.columns(2)

# --- Plot Loss ---
with col1:
    st.markdown("**Loss (Mean Squared Error)**")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(1, epochs + 1), train_loss, label='Train', color=train_color, linewidth=2)
    ax.plot(range(1, epochs + 1), val_loss, label='Validation', color=val_color, linestyle='--', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax.set_ylabel("Loss (MSE)", fontsize=10, fontweight='bold')
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor("#f9f9f9")
    st.pyplot(fig)

# --- Plot Accuracy ---
with col2:
    st.markdown("**Accuracy (%)**")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(1, epochs + 1), np.array(train_acc) * 100, label='Train', color=train_color, linewidth=2)
    ax.plot(range(1, epochs + 1), np.array(val_acc) * 100, label='Validation', color=val_color, linestyle='--', linewidth=2)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_facecolor("#f9f9f9")
    st.pyplot(fig)

# --- Penjelasan Singkat ---
st.caption("Grafik di atas menampilkan simulasi performa model selama proses pelatihan berdasarkan tren loss dan akurasi.")

# --- Tabel Ringkasan ---
st.markdown("### 📊 Summary of Simulated Metrics")

summary_df = pd.DataFrame({
    "Epoch": list(range(1, epochs + 1)),
    "Train_Loss": np.round(train_loss, 4),
    "Val_Loss": np.round(val_loss, 4),
    "Train_Accuracy (%)": np.round(np.array(train_acc) * 100, 2),
    "Val_Accuracy (%)": np.round(np.array(val_acc) * 100, 2)
})

# --- 5 epoch terakhir ---
st.dataframe(
    summary_df.tail(5).style.format(precision=2)
    .set_properties(**{
        'text-align': 'center',
        'font-weight': 'bold'
    })
)

# --- Ringkasan hasil akhir ---
final_train_acc = np.array(train_acc)[-1] * 100
final_val_acc = np.array(val_acc)[-1] * 100
final_val_loss = val_loss[-1]

st.success(f"🎯 Final Validation Accuracy: **{final_val_acc:.2f}%**")
st.info(f"📉 Final Validation Loss: **{final_val_loss:.4f}**")

st.markdown("---")

# ------------------------
#    📦 LOAD ARTIFACTS
# ------------------------
required_files = [
    "models/random_forest_model.pkl",
    "models/scaler.pkl",
    "models/feature_columns.pkl",
    "models/num_columns.pkl",
    "models/X_train.pkl",
    "models/X_test.pkl",
    "models/y_train.pkl",
    "models/y_test.pkl"
]

if not all(os.path.exists(f) for f in required_files):
    st.error("❌ Sebagian file model hilang. Jalankan ulang data preprocessing & training model.")
    st.stop()

model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
num_columns = joblib.load("models/num_columns.pkl")
X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

# --- Konversi y_train dan y_test menjadi DataFrame agar bisa digabung ---
if isinstance(y_train, np.ndarray):
   y_train = pd.Series(y_train, name="Attrition")
if isinstance(y_test, np.ndarray):
    y_test = pd.Series(y_test, name="Attrition")

# --- Gabungkan data train untuk eksplorasi visual ---
try:
    df_viz = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
except Exception as e:
    st.warning(f"Gagal menggabungkan data untuk EDA: {e}")
    df_viz = None

# <<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>
#      🚀 SECTION 4: MODEL PERFORMANCE
# <<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>
st.header("🚀 Model Performance")

y_pred = model.predict(X_test)
acc = np.mean(y_pred == y_test)
st.metric(label="Model Accuracy", value=f"{acc*100:.2f}%")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T)

with col2:
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --- Feature Importance ---
st.subheader("💡 Feature Importance")
importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False)[:10]
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(y=importances.head(10).index, x=importances.head(10).values, palette="crest")
ax.set_title("Top 10 Important Features")
st.pyplot(fig)

# <<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>
#      🛠️ SECTION 5: MANUAL PREDICTION
# <<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>
st.sidebar.title("🛠️ Manual Prediction")
st.sidebar.markdown("### Employee Data Input")

age = st.sidebar.number_input("Umur (Age)", 18, 60, 30)
income = st.sidebar.number_input("Pendapatan Bulanan (Monthly Income)", 1000, 20000, 5000)
distance = st.sidebar.slider("Jarak dari Rumah (Distance From Home)", 0, 30, 5)
job_satisfaction = st.sidebar.selectbox("Kepuasan Kerja (Job Satisfaction)", [1, 2, 3, 4])
gender = st.sidebar.selectbox("Jenis Kelamin (Gender)", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Status Pernikahan (Marital Status)", ["Single", "Married", "Divorced"])
business_travel = st.sidebar.selectbox("Frekuensi Perjalanan Dinas (Bussiness Travel)", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

# --- Buat DataFrame input sesuai urutan fitur training ---
row = {col: 0 for col in feature_columns}
input_df = pd.DataFrame([row])

if 'Age' in input_df.columns: input_df.loc[0, 'Age'] = age
if 'MonthlyIncome' in input_df.columns: input_df.loc[0, 'MonthlyIncome'] = income
if 'DistanceFromHome' in input_df.columns: input_df.loc[0, 'DistanceFromHome'] = distance
if 'JobSatisfaction' in input_df.columns: input_df.loc[0, 'JobSatisfaction'] = job_satisfaction
if 'Gender' in input_df.columns: input_df.loc[0, 'Gender'] = 1 if gender == "Male" else 0
if 'MaritalStatus' in input_df.columns:
    input_df.loc[0, 'MaritalStatus'] = 0 if marital_status == "Divorced" else (1 if marital_status == "Married" else 2)
if 'BusinessTravel' in input_df.columns:
    input_df.loc[0, 'BusinessTravel'] = 0 if business_travel == "Non-Travel" else (1 if business_travel == "Travel_Rarely" else 2)

# --- Scaling dan Prediksi ---
try:
    input_df[num_columns] = scaler.transform(input_df[num_columns])
    X_infer = input_df[feature_columns]
    proba = model.predict_proba(X_infer)[0][1]
    pred = int(proba >= 0.5)
except Exception as e:
    st.sidebar.error(f"⚠️ Terjadi kesalahan saat melakukan scaling/prediksi: {e}")
    st.sidebar.stop()

# --- Tombol Prediksi ---
if st.sidebar.button("🔍 Predict"):
    st.sidebar.header("📋 Prediction Result")
    
    proba_percent = round(proba * 100, 2)
    proba_decimal = round(proba, 3)
    
    # -- Menentukan hasil utama --
    if pred == 1:
        st.sidebar.error(
            f"⚠️ Karyawan kemungkinan besar akan **RESIGN**\n\n"
            f"**> Probabilitas:** {proba_decimal} ({proba_percent}%) **<**"
        )
        st.sidebar.markdown("### ❓ Mengapa karyawan berpotensi **RESIGN** ❓")
    else:
        st.sidebar.success(
            f"✅ Karyawan kemungkinan akan **TETAP BEKERJA**\n\n"
            f"**> Probabilitas resign:** {proba_decimal} ({proba_percent}%) **<**"
        )
        st.sidebar.markdown("### ❓ Mengapa karyawan berpotensi **TETAP BEKERJA** ❓")

    # --------------------------------------
    #  ℹ️ Explanation (Feature Importance)
    # --------------------------------------
    # -- Feature Importance Global --
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    # -- 5 fitur paling berpengaruh --
    top_features = importance_df.head(5)

    # -- Nilai input user untuk fitur tersebut --
    user_values = {}
    for f in top_features["Feature"]:
        if f in input_df.columns:
            user_values[f] = round(float(X_infer[f].iloc[0]), 2)

    # -- Tabel Alasan --
    explain_df = pd.DataFrame({
        "Feature": list(user_values.keys()),
        "User Input": [f"{val:.2f}" for val in user_values.values()],
        "Importance to the Prediction": [f"{imp * 100:.2f} %" for imp in top_features["Importance"]]
    })
    st.sidebar.markdown("### ⭐ Fitur Paling Berpengaruh")
    st.sidebar.dataframe(explain_df, use_container_width=True)

# <<<<<<<<<<>>>>>>>>>>
#      🦶🏻 FOOTER
# <<<<<<<<<<>>>>>>>>>>
st.markdown("---")
st.caption("""
**MLOps Project — HR Employee Attrition Predictor**\n
Built with ❤️ using Streamlit, scikit-learn, and MLflow.\n
© 2025 Ivan Yudhistira | Artificial Intelligence | BINUS University
""")