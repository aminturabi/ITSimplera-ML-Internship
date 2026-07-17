# Week 3 Internship Task: Dimensionality Reduction & FastAPI Dashboard

## 📝 Student Information
- **Student Name:** Amin Khan
- **Registration No:** AIMLB01-8657
- **Course:** Machine Learning
- **Assignment:** Week 3 Internship Task

---

## 📖 Project Overview

This repository contains the complete implementation for **Week 3** of the Machine Learning Internship. The project is focused on optimizing and deploying an active energy consumption forecasting system for a steel industry plant using **Principal Component Analysis (PCA)** for dimensionality reduction and **FastAPI** for real-time model serving.

The system scales and compresses 20 high-dimensional plant inputs down to **10 principal components** (capturing **95% explained variance**), cutting computational complexity in half while maintaining a high accuracy ($R^2$ score of **98.99%**).

---

## 📁 Repository Structure

```text
Week_3/
│
├── notebooks/
│   ├── week3 pca.ipynb          # Jupyter Notebook containing the full PCA & model retraining pipeline
│   ├── scree_plot.png           # [LOCAL ONLY] Scree plot showing variance per component
│   ├── cumulative_variance_plot.png # [LOCAL ONLY] Cumulative variance curve (95% threshold)
│   └── pca_loadings_heatmap.png # [LOCAL ONLY] Loading heatmap for the first 3 PCs
│
├── static/
│   ├── css/
│   │   └── style.css            # Custom CSS providing a modern, dark glassmorphic UI
│   └── images/
│       ├── energy_by_hour.png   # [LOCAL ONLY] Pre-rendered hourly energy consumption line chart
│       ├── energy_by_load_type.png # [LOCAL ONLY] Pre-rendered energy consumption by load level bar chart
│       └── correlation_heatmap.png # [LOCAL ONLY] Pre-rendered correlation matrix heatmap
│
├── templates/
│   ├── base.html                # Common base layout (navigation header and footer)
│   ├── index.html               # Welcome page introducing the project & model architecture
│   ├── dashboard.html           # EDA dashboard rendering the exploratory plots
│   └── predict.html             # Real-time predictor form & prediction result card
│
├── generate_eda_plots.py        # [LOCAL ONLY] Helper script to generate the static images under static/images
├── generate_notebook.py         # [LOCAL ONLY] Helper script to generate the Jupyter Notebook template
├── main.py                      # FastAPI web server and route controllers
├── model.joblib                 # [LOCAL ONLY] Serialized ML Pipeline (StandardScaler + PCA + Random Forest)
├── pca_report.md                # Written report analyzing PCA results & recommendations
├── requirements.txt             # Python dependencies for the project
└── README.md                    # This project documentation (instructions & details)
```
*(Note: Files marked `[LOCAL ONLY]` are ignored in Git to adhere to source-only submission rules.)*

---

## 🤖 Part 1 — Dimensionality Reduction with PCA

In real-world manufacturing plants, collecting and processing high-frequency raw features is expensive and resource-intensive. Principal Component Analysis (PCA) was used to transform and project the 20-dimensional encoded feature space onto a lower-dimensional subspace.

### ⚙️ Inference Pipeline Workflow

```text
  Raw Input (11 parameters)
             │
             ▼
  One-Hot Categorical Encoding ──► 20-Dimensional Vector
             │
             ▼
  StandardScaler Scaling (prevents magnitude bias)
             │
             ▼
  PCA Dimensionality Reduction ──► 10 Principal Components (95% Variance)
             │
             ▼
  Random Forest Regressor ──────► Active Energy Prediction (Usage_kWh)
```

### 📊 Model Performance Comparison

A Random Forest Regressor (`n_estimators=50`, `max_depth=15`, `random_state=42`) was trained on three configurations to analyze the accuracy vs. complexity trade-offs:

| Model Configuration | Input Dimensions | Test MAE (kWh) | Test RMSE (kWh) | Test $R^2$ Score | Dimension Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original Model (Week 2)** | 20 features | 0.4344 | 1.1470 | 0.9988 | - (Baseline) |
| **3-Component PCA Model** | 3 components | 3.5846 | 7.3615 | 0.9523 | 85.0% Reduction |
| **95% Variance PCA Model** | 10 components | 1.7685 | 3.3800 | 0.9899 | 50.0% Reduction |

**Key Takeaway**: The **95% Variance Model** (10 components) retains **98.99% accuracy** ($R^2$ score) while cutting feature dimensions in half. This offers substantial memory and bandwidth optimization for edge hardware deployments. Read the full evaluation in [pca_report.md](file:///d:/ITSimplera/Week_3/pca_report.md).

---

## 🌐 Part 2 — FastAPI Web Application

We deployed the 10-component pipeline into an interactive web interface using FastAPI, Jinja2, and a premium **Eco-Industrial** Vanilla CSS theme.

### 🔗 Route Endpoints

1.  **Home Page (`GET /`)**: Displays information about the machine learning model, training metrics, pipeline stages, and project background.
2.  **EDA Analytics Dashboard (`GET /dashboard`)**: Serves pre-rendered charts explaining energy consumption over a 24-hour diurnal cycle, across different load types (Light, Medium, Maximum), and correlations of active/reactive parameters.
3.  **Real-Time Predictor Form (`GET /predict`)**: Loads a responsive, user-friendly HTML form prompting for 11 raw inputs.
4.  **Prediction API (`POST /predict`)**: Accepts form inputs, processes them through the standard-scaler, projects them into the 10 PCA components, runs the Random Forest Regressor, and returns the predicted active energy consumption (Usage_kWh).

---

## 🛠️ Local Setup & Run Guide

### 1. Installation & Environment Configuration
Set up a Python 3.8+ environment and install the required dependencies:
```bash
# Clone the repository and navigate into the folder
cd Week_3

# Install required Python dependencies
pip install -r requirements.txt
```

### 2. (Optional) Run Generators locally
To regenerate the Jupyter Notebook and the static dashboard plots locally:
```bash
# Generate the Jupyter notebook template
python generate_notebook.py

# Run the notebook to retrain the model pipeline and output notebooks/ plots
jupyter nbconvert --to notebook --execute --inplace notebooks/"week3 pca.ipynb"

# Generate the static dashboard charts
python generate_eda_plots.py
```

### 3. Start the FastAPI Development Server
Boot the web dashboard locally using Uvicorn:
```bash
uvicorn main:app --reload
```

Once started, open your browser and navigate to:
*   **Web Dashboard URL**: `http://127.0.0.1:8000/`
*   **Predictor Interface**: `http://127.0.0.1:8000/predict`
*   **Interactive API Docs**: `http://127.0.0.1:8000/docs`



