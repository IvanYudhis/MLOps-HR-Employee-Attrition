# 🚀 HR Employee Attrition – End-to-End MLOps Project

This repository contains a **production-ready MLOps pipeline** built using the [IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

The goal of this project is to **predict whether an employee will leave (attrition)** based on HR data such as job role, salary, satisfaction, and demographics.  
The project demonstrates a complete MLOps workflow — from data preprocessing and model training, to tracking experiments, containerization, and deployment.

---

## 🧠 Project Features

✅ Data exploration and visualization (EDA)  
✅ Data preprocessing (encoding, scaling, splitting)  
✅ Model training using `RandomForestClassifier`  
✅ Experiment tracking using **MLflow**  
✅ Model serving via **Streamlit** web app  
✅ **Docker** containerization for reproducibility  
✅ (Optional) CI/CD automation using GitHub Actions  

---

## 🗂️ Project Structure

```
MLOps-HR-Employee-Attrition/
│
├── data/                      # Dataset (CSV)
├── src/                       # ML pipeline scripts
│   ├── eda_analysis.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
│
├── models/                    # Trained model (.pkl)
├── app/                       # Streamlit web app for deployment
│   └── app.py
│
├── mlflow_tracking/            # MLflow experiment logs
├── requirements.txt            # Required dependencies
├── Dockerfile                  # Docker configuration
├── .gitignore                  # Ignore unnecessary files
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/IvanYudhis/MLOps-HR-Employee-Attrition.git
cd MLOps-HR-Employee-Attrition
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # (Windows)
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧩 Running the Project

### ▶️ Run EDA
```bash
python src/eda_analysis.py
```

### ▶️ Run Model Training (with MLflow tracking)
```bash
python src/train_model.py
```

### ▶️ Run Streamlit App
```bash
streamlit run app/app.py
```

Then open your browser at **http://localhost:8501**

---

## 🐳 Running with Docker
```bash
docker build -t hr-mlops .
docker run -p 8501:8501 hr-mlops
```

---

## 🧾 Report Outline

The final project report will include:
1. Dataset Description  
2. Data Exploration & Insights  
3. Preprocessing Workflow  
4. Model Implementation & Evaluation  
5. CI/CD Integration (optional)  
6. Deployment Link (Streamlit / Hugging Face / Docker)

---

## 👤 Author
**Name:** Ivan Yudhistira  
**University:** BINUS University  
**Course:** Machine Learning / MLOps  
**GitHub:** [@IvanYudhis](https://github.com/IvanYudhis)

---

⭐ If you found this project helpful, don’t forget to give it a star!
