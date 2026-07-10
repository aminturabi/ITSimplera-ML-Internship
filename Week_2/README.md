# Steel Industry Energy Consumption - Deep EDA & Baseline Regression Modeling

This repository contains a deep exploratory analysis and baseline machine learning modeling project analyzing energy consumption patterns in a steel manufacturing plant. 

This project was developed for the **Machine Learning** course assignment: **Week 2 Internship Task**.

---

## 📁 Repository Structure

```text
Week_2/
│
├── Steel_industry_data/
│   └── Steel_industry_data.csv        # Raw energy consumption dataset
│
├── screenshots/                       # High-resolution screenshots of notebook sections & plots
│   ├── screenshot_01.png
│   ├── screenshot_02.png
│   ├── screenshot_03.png
│   ├── screenshot_04.png
│   ├── screenshot_05.png
│   ├── screenshot_06.png
│   ├── screenshot_07.png
│   ├── screenshot_08.png
│   ├── screenshot_09.png
│   ├── screenshot_10.png
│   ├── screenshot_11.png
│   └── screenshot_12.png
│
├── .gitignore                         # Git exclusion rules
├── week2_eda.ipynb                    # Unified Deep EDA, Feature Engineering & Baseline Modeling Notebook (25 Sections, 51 Cells)
│
├── README.md                          # Project documentation (this file)
└── requirements.txt                   # Project dependencies
```

---

## 📊 Exploratory Data Analysis & Feature Engineering

The primary objective of **`week2_eda.ipynb`** is to investigate the data structure, inspect correlations, diagnose outliers, and extract predictive signals from raw features.

### Key Highlights:
1. **Dynamic File Loader**: Automatically scans and loads the CSV from the `Steel_industry_data/` folder, ensuring adaptability.
2. **Temporal Feature Extraction**: Parses dates to construct numerical columns for machine learning:
   - `Hour`: Diurnal cycles (day vs. night shifts).
   - `DayOfWeek`: Weekday operational shifts.
   - `Month`: Seasonal patterns.
   - `IsWeekend`: Non-production weekend regimes.
3. **Engineered Features**:
   - **`Power_Factor_Ratio`**: Defined as $\frac{\text{Leading Current Power Factor}}{\text{Lagging Current Power Factor}}$, indicating inductive vs. capacitive balances.
   - **`High_Load`**: A binary classification indicator (1 if `Usage_kWh` > 75th percentile, else 0).
4. **Outlier Detection**: Utilizes the IQR method to locate peak demand surges. The upper limit was found to be **131.60 kWh**, resulting in **1,114 outlier records (3.18%)** representing authentic high-demand machinery boots.
5. **Load Type and Hourly Visualizations**: Grouped bar charts show that average power usage jumps from **2.92 kWh** (Light Load) to **74.19 kWh** (Maximum Load). Line plots map hourly peaks between **09:00 and 18:00**.

---

## 🤖 Baseline Regression Modeling

The baseline regression modeling aims to build, evaluate, and compare multiple regression models to predict active energy usage (`Usage_kWh`) using our processed features.

### Data Preprocessing:
- The target-leaking column `High_Load` and non-interpretable `date` column were dropped.
- Nominal categorical columns (`Load_Type`, `Day_of_week`, `WeekStatus`) were encoded using **One-Hot Encoding** to prevent linear models from assuming ordinal hierarchies.
- The dataset was split into **80% training** and **20% testing** sets (`random_state=42`).

### Model Evaluation Results:

| Model | Test MAE (kWh) | Test RMSE (kWh) | Test $R^2$ Score | 5-Fold CV Mean RMSE (kWh) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 8.04 | 11.23 | 0.8653 | 11.23 |
| **Ridge Regression** | 8.04 | 11.23 | 0.8653 | 11.23 |
| **Decision Tree** | 1.15 | 2.59 | 0.9928 | 2.85 |
| **Random Forest** | **1.01** | **1.81** | **0.9965** | **2.01** |

### Key Insights:
- **Linear & Ridge Models** underfit the data because they cannot capture non-linear, multi-variable physical interactions (such as the combined impact of lagging power factors and hourly shifts).
- **Random Forest Regressor** achieved outstanding results, capturing **99.65%** of variance with a low test RMSE of **1.81 kWh**. The ensemble averaging stabilizes performance and mitigates overfitting, making it the recommended model for production.

---

## 🛠️ Getting Started & Installation

### 1. Prerequisites
Make sure you have Python 3.8+ installed. You can install all project dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2. Running the Notebook
Start Jupyter Notebook or JupyterLab in this directory:

```bash
jupyter notebook
```
Open **`week2_eda.ipynb`** to inspect the data exploration plots and baseline model evaluations. All outputs, plots, and figures are pre-rendered and saved in the notebook cells.

---

## 📝 Student Information

- **Student Name**: Amin Khan
- **Registration No**: AIMLB01-8657
- **Course**: Machine Learning
- **Assignment**: Week 2 Internship Task
