# ===================
# src/eda_analysis.py
# ===================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load Dataset ---
DATA_PATH = os.path.join("dataset", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.read_csv(DATA_PATH)

print("✅ Data loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# --- Dataset Structure Informations ---
print("\n=== Dataset Informations ===")
print(df.info())

# --- Check Missing Values ---
print("\n=== Missing Values ===")
print(df.isnull().sum())

# --- Descriptive Statistics ---
print("\n=== Descriptive Statistics ===")
print(df.describe())

# --- Target Distributions (Attrition) ---
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='Attrition', hue='Attrition', palette='coolwarm', legend=False)
plt.title('Distribusi Karyawan Berdasarkan Status Attrition')
plt.xlabel('Attrition (Resign/Stay)')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

# --- Correlation between numerical features ---
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap (Numerical Features)')
plt.show()

# --- Age vs Attrition Distributions ---
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='Attrition', y='Age', hue='Attrition', palette='viridis', legend=False)
plt.title('Distribusi Umur vs Status Attrition')
plt.show()

# --- Initial Insight ---
print("\n=== Insight Awal ===")
print("1. Data memiliki 1470 baris dan 35 kolom (beragam fitur HR).")
print("2. Tidak ditemukan missing values pada dataset asli Kaggle.")
print("3. Distribusi target tidak seimbang (Attrition: lebih banyak 'No' dibanding 'Yes').")
print("4. Umur dan pendapatan memiliki korelasi dengan Attrition (resign).")
print("5. Beberapa fitur penting yang tampak: Age, MonthlyIncome, JobSatisfaction, YearsAtCompany.")