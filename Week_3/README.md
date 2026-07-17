# Week 3 Internship Task: Dimensionality Reduction & FastAPI Dashboard

## 📝 Student Information
- **Student Name:** Amin Khan
- **Registration No:** AIMLB01-8657
- **Course:** Machine Learning
- **Assignment:** Week 3 Internship Task

---

This repository contains the implementation of the Week 3 internship tasks for the Machine Learning course. The project is split into two parts: Dimensionality Reduction with PCA and Deployment via a FastAPI Web Dashboard.

---

## 📁 Repository Structure

```text
Week_3/
│
├── notebooks/
│   ├── week3 pca.ipynb          # Jupyter Notebook containing full PCA analysis & model retraining
│   ├── scree_plot.png           # Scree plot showing variance per component (saved by notebook)
│   ├── cumulative_variance_plot.png # Cumulative variance curve showing 95% threshold (saved by notebook)
│   └── pca_loadings_heatmap.png # Heatmap showing loadings of first 3 PCs (saved by notebook)
│
├── static/
│   ├── css/
│   │   └── style.css            # Custom CSS providing a modern, dark glassmorphic UI
│   └── images/
│       ├── energy_by_hour.png   # Pre-rendered hourly energy consumption chart
│       ├── energy_by_load_type.png # Pre-rendered energy consumption by load level chart
│       └── correlation_heatmap.png # Pre-rendered correlation matrix heatmap
│
├── templates/
│   ├── base.html                # Common base template (nav, layout, and footer)
│   ├── index.html               # Welcome page introducing the system
│   ├── dashboard.html           # EDA dashboard displaying pre-rendered charts
│   └── predict.html             # Predictor page containing input form and prediction card
│
├── generate_eda_plots.py        # Helper script that generates the EDA charts under static/images
├── generate_notebook.py         # Helper script that creates the Jupyter Notebook structure
├── main.py                      # FastAPI web application code
├── model.joblib                 # Trained scikit-learn Pipeline (Scaler + PCA + Random Forest)
├── pca_report.md                # Markdown report analyzing the PCA results
├── requirements.txt             # Python dependencies for this project
└── README.md                    # This project documentation (instructions & details)
```

---

## 🤖 Part 1 — Dimensionality Reduction with PCA

We performed Principal Component Analysis (PCA) on the preprocessed steel industry dataset. We scaled the features to prevent magnitude bias and applied PCA to identify the minimum number of features required to retain high model performance.

### PCA Comparison Results

A Random Forest Regressor (`n_estimators=50`, `max_depth=15`, `random_state=42`) was trained on three configurations:

| Model Configuration | Input Dimensions | Test MAE (kWh) | Test RMSE (kWh) | Test $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Original Model (Week 2)** | 20 features | 0.4344 | 1.1470 | 0.9988 |
| **3-Component PCA Model** | 3 components | 3.5846 | 7.3615 | 0.9523 |
| **95% Variance PCA Model** | 10 components | 1.7685 | 3.3800 | 0.9899 |

*A detailed explanation of these findings and recommendations for memory-constrained hardware can be found in the [pca_report.md](file:///d:/ITSimplera/Week_3/pca_report.md) file.*

---

## 🌐 Part 2 — FastAPI Dashboard

We deployed the 95% variance PCA model pipeline (comprising `StandardScaler`, `PCA(n_components=10)`, and `RandomForestRegressor`) in a web dashboard built with **FastAPI**, **Jinja2 Templates**, and **Vanilla CSS**.

### Key Pages:
1. **Home (`/`)**: A sleek landing page introducing the platform, the underlying model architecture, and links to other pages.
2. **Dashboard (`/dashboard`)**: Displays pre-rendered charts of energy consumption trends (energy by hour, energy by load type, and the correlation heatmap) saved under `static/images/`.
3. **Real-time Predictor (`/predict`)**: A user-friendly web form that collects 11 raw operational inputs, constructs the 20-column encoded feature vector in the backend, projects it into PCA space, and displays the prediction on the same page.

---

## 🛠️ Getting Started & Installation

### 1. Prerequisites
Make sure Python 3.8+ is installed. Clone the repository and navigate to the `Week_3` directory. Install the dependencies using:

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
The Jupyter Notebook contains all run cells and outputs. If you want to open and run it manually:

```bash
jupyter notebook notebooks/"week3 pca.ipynb"
```

### 3. Start the FastAPI Web Dashboard
Start the local development server using Uvicorn:

```bash
uvicorn main:app --reload
```

Once running, access the dashboard in your web browser:
- Home Page: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Dashboard: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
- Predictor: [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)


