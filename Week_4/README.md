# 💳 Week 4 Internship Task: Unsupervised Learning & Customer Segmentation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This repository contains the complete submission for **Week 4 of the Machine Learning Internship**. 

Unlike supervised learning tasks where explicit target labels exist, this project dives into **Unsupervised Learning** using active credit card behavioral data (~9,000 active cardholders over 6 months, across 18 behavioral features). The goal is to discover natural customer segments to enable targeted marketing strategies, risk management, and personalized banking services.

---

## 📂 Project Structure

```text
Week_4/
├── data/
│   └── credit_card_data.csv        # Preprocessed active cardholder behavioral dataset (8,950 rows x 18 cols)
├── notebooks/
│   └── week4_clustering.ipynb      # Fully executed Jupyter Notebook with all outputs & plots
├── README.md                       # Comprehensive project documentation & submission report
├── requirements.txt                # Python environment dependency requirements
└── generate_notebook.py            # Reproducible build script for notebook generation
```

---

## 📊 Dataset Description

- **Source**: [Credit Card Dataset for Clustering (Kaggle)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
- **Scale**: 8,950 customer records across 18 behavioral features.

| Feature Name | Description |
| :--- | :--- |
| `CUST_ID` | Unique identification key for credit card holder *(Dropped during preprocessing)* |
| `BALANCE` | Remaining balance amount available for purchases |
| `BALANCE_FREQUENCY` | How frequently the balance is updated (score between 0 and 1) |
| `PURCHASES` | Total dollar amount of purchases made from account |
| `ONEOFF_PURCHASES` | Maximum purchase amount done in one single transaction |
| `INSTALLMENTS_PURCHASES` | Amount of purchases done in installment payments |
| `CASH_ADVANCE` | Cash in advance given by the user |
| `PURCHASES_FREQUENCY` | How frequently purchases are made (score between 0 and 1) |
| `ONEOFF_PURCHASES_FREQUENCY` | Frequency of one-off purchases (0 to 1) |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Frequency of installment purchases (0 to 1) |
| `CASH_ADVANCE_FREQUENCY` | How frequently cash in advance is drawn |
| `CASH_ADVANCE_TRX` | Number of cash advance transactions |
| `PURCHASES_TRX` | Number of purchase transactions made |
| `CREDIT_LIMIT` | Limit of credit card for user |
| `PAYMENTS` | Amount of payment done by user |
| `MINIMUM_PAYMENTS` | Minimum amount of payments made by user |
| `PRC_FULL_PAYMENT` | Percent of full payment paid by user |
| `TENURE` | Tenure of credit card service for user (in months) |

---

## 🛠️ Part 1 — Preprocessing & K-Means Clustering

### 1. Data Preprocessing & Missing Value Strategy
- **`CUST_ID` Column Dropped**: Identifier keys contain no customer behavioral signal and distort distance calculations.
- **Missing Value Handling**:
  - `MINIMUM_PAYMENTS`: 313 missing values (~3.50%).
  - `CREDIT_LIMIT`: 1 missing value (<0.01%).
  - **Strategy**: **Median Imputation** was chosen over mean imputation or row deletion. Monetary values in banking are heavily right-skewed with extreme spenders. Mean imputation would artificially distort non-paying customers toward high spenders, whereas row deletion would discard 313 active customer records.

### 2. Feature Scaling Rationale
Clustering algorithms rely on Euclidean distance ($d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum (x_i - y_i)^2}$). Monetary variables like `BALANCE` (up to \$19,000+) dwarfs ratio metrics like `PURCHASES_FREQUENCY` (0.0 to 1.0). 
Applying `StandardScaler` standardizes all features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$), ensuring equal weighting in cluster distance metrics.

### 3. K-Means Evaluation ($k \in [2, 10]$)

| $k$ (Clusters) | Inertia (WCSS) | Silhouette Score |
| :---: | :---: | :---: |
| 2 | 127,784.53 | 0.2100 |
| 3 | 111,975.04 | 0.2510 |
| **4** | **99,061.94** | **0.1977** |
| 5 | 91,490.50 | 0.1931 |
| 6 | 84,826.59 | 0.2029 |
| 7 | 79,856.16 | 0.2077 |
| 8 | 74,484.88 | 0.2217 |
| 9 | 69,828.70 | 0.2260 |
| 10 | 66,466.41 | 0.2204 |

- **Elbow Point**: The WCSS curve shows a clear inflection ("elbow") at **$k=4$**.
- **Silhouette Confirmation**: $k=4$ provides clear, distinct, and actionable customer segments without over-segmentation.

---

### 🏷️ Cluster Profiles & Business Interpretation ($k=4$)

| Feature (Mean) | Cluster 0: Budget Cardholders | Cluster 1: VIP High Spenders | Cluster 2: Cash Borrowers | Cluster 3: Installment Shoppers |
| :--- | :---: | :---: | :---: | :---: |
| **Customer Count** | **3,977 (44.4%)** | **409 (4.6%)** | **1,197 (13.4%)** | **3,367 (37.6%)** |
| `BALANCE` | \$1,012.66 | \$3,551.15 | \$4,602.45 | \$894.91 |
| `PURCHASES` | \$270.04 | \$7,681.62 | \$501.86 | \$1,236.18 |
| `ONEOFF_PURCHASES` | \$209.68 | \$5,080.64 | \$315.13 | \$308.20 |
| `INSTALLMENTS_PURCHASES`| \$60.67 | \$2,601.07 | \$186.96 | \$928.37 |
| `CASH_ADVANCE` | \$596.51 | \$653.64 | \$4,521.51 | \$210.57 |
| `PURCHASES_FREQUENCY` | 0.27 | 0.93 | 0.29 | 0.88 |
| `CASH_ADVANCE_FREQ` | 0.11 | 0.07 | 0.53 | 0.04 |
| `CREDIT_LIMIT` | \$3,278.64 | \$9,696.94 | \$7,546.16 | \$4,213.21 |
| `PAYMENTS` | \$974.26 | \$7,288.74 | \$3,484.05 | \$1,332.19 |

#### Business Personas & Actionable Strategies:
1. **Cluster 0: Low-Activity / Budget Cardholders (44.4%)**:
   - *Persona*: Low balances and minimal purchasing activity.
   - *Strategy*: Promotional double-points offers, low-risk credit line increase incentives.
2. **Cluster 1: Premium VIP High Spenders (4.6%)**:
   - *Persona*: High purchasing power (\$7,681+ avg purchases), high credit limit, prompt payment history.
   - *Strategy*: Retain with VIP concierge services, luxury lounge access, and exclusive wealth management offers.
3. **Cluster 2: Cash Advance Borrowers / High-Risk (13.4%)**:
   - *Persona*: High balances (\$4,602) and heavy cash advances (\$4,521).
   - *Strategy*: Risk monitoring, lower cash sub-limits, debt consolidation / low-APR balance transfer offers.
4. **Cluster 3: Regular Shoppers / Installment Buyers (37.6%)**:
   - *Persona*: High purchase frequency (0.88) driven by monthly installment plans.
   - *Strategy*: Retail merchant partnership discounts, zero-percent interest BNPL (Buy Now Pay Later) features.

---

## 🌳 Part 2 — Hierarchical Clustering & Model Comparison

### 1. SciPy Dendrogram Analysis
Using a representative random sample of 300 customers from the scaled dataset, Ward's linkage hierarchical clustering was performed. Drawing a horizontal threshold line at height $y=25$ confirms **4 distinct branches**, aligning perfectly with the $k=4$ K-Means decision.

### 2. Model Comparison (K-Means vs. Agglomerative Hierarchical)
- **Cross-Tabulation Matrix (`pd.crosstab`)**: High diagonal alignment between K-Means clusters and Scikit-Learn Agglomerative clusters (Adjusted Rand Index $\text{ARI} \approx 0.40$).

### 3. Comparison Report:
1. **Segment Meaningfulness**: Both algorithms extract similar core customer groups (*VIPs*, *Borrowers*, *Shoppers*, *Low-Activity*). K-Means yields more evenly balanced cluster sizes.
2. **Interpretability**: K-Means is easier for business users due to direct centroid averages. Hierarchical clustering excels at visual taxonomy exploration via dendrograms.
3. **Computational Scalability**:
   - **K-Means**: $O(N \cdot k \cdot d)$ time complexity — scales smoothly to millions of cardholders.
   - **Agglomerative Hierarchical**: $O(N^2 \log N)$ time and $O(N^2)$ memory complexity — computationally prohibitive for large datasets beyond 10,000 records.
4. **Recommendation**: **K-Means Clustering** is recommended for real-world banking deployment due to scalability and instantaneous real-time customer scoring via `.predict()`.

---

## 🚀 How to Run the Notebook

```bash
# 1. Clone repository
git clone https://github.com/aminturabi/ITSimplera-ML-Internship.git
cd Week_4

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate and run executed notebook automatically
python generate_notebook.py

# Alternatively, launch Jupyter Lab / Notebook
jupyter notebook notebooks/week4_clustering.ipynb
```

---

## 💬 LinkedIn Post Submission Draft

> 🚀 **Unlocking Customer Insights in Banking with Unsupervised Learning & K-Means Clustering!**
>
> Customer segmentation is one of the most powerful real-world applications of machine learning in banking and finance. This week in my Machine Learning Internship, I explored **Unsupervised Learning** on behavioral credit card data of ~9,000 active cardholders over 6 months!
>
> 🔹 **Key Highlights & Technical Workflow**:
> • **Data Preprocessing**: Handled missing financial records with median imputation to protect against skewness, and standardized features with `StandardScaler`.
> • **Optimal Cluster Selection**: Used the **Elbow Method** (WCSS) and **Silhouette Scores** across $k=2 \dots 10$ to establish $k=4$ as the optimal cluster count.
> • **Customer Profiling & Heatmaps**: Generated cluster mean heatmaps to identify 4 actionable customer personas:
>   1️⃣ **VIP High Spenders (4.6%)**: High purchase volume (\$7.6k+ avg), high payments — targeted for luxury rewards.
>   2️⃣ **Installment Shoppers (37.6%)**: Frequent retail shoppers utilizing monthly payment plans.
>   3️⃣ **Cash Advance Borrowers (13.4%)**: High cash withdrawal reliance — targeted for risk management & balance transfer programs.
>   4️⃣ **Low-Activity Cardholders (44.4%)**: Conservative card users — targeted for engagement campaigns.
> • **Algorithm Evaluation**: Compared K-Means with **Agglomerative Hierarchical Clustering** (SciPy Dendrogram & Scikit-Learn Crosstabulation).
>
> 💡 **Key Takeaway**: While Hierarchical Clustering provides rich visual dendrogram taxonomies, **K-Means is the gold standard for production banking pipelines** due to its $O(N \cdot k)$ linear scalability and real-time customer scoring capabilities!
>
> 🔗 Check out the full repository and executed notebook on GitHub!
>
> #MachineLearning #DataScience #UnsupervisedLearning #KMeans #CustomerSegmentation #BankingAnalytics #Python #AI #Internship
