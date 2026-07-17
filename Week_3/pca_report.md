# Dimensionality Reduction Report (PCA Analysis)

**Student Name:** Amin Khan  
**Registration No:** AIMLB01-8657  
**Course:** Machine Learning  
**Assignment:** Week 3 Internship Task (Part 1)

---

## 📊 Overview of Results

Principal Component Analysis (PCA) was applied to the preprocessed dataset consisting of **20 encoded features** (after temporal feature extraction and one-hot encoding). 

The table below summarizes the performance comparison of the **Random Forest Regressor** (`n_estimators=50`, `max_depth=15`, `random_state=42`) across three feature configurations:

| Model Configuration | Number of Input Features | Test MAE (kWh) | Test RMSE (kWh) | Test $R^2$ Score | Performance Retained |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original Model (Full Features)** | 20 | 0.4344 | 1.1470 | 0.9988 | **Baseline (100.0%)** |
| **3-Component PCA Model** | 3 | 3.5846 | 7.3615 | 0.9523 | **High compression (95.3%)** |
| **95% Variance PCA Model (10 PCs)** | 10 | 1.7685 | 3.3800 | 0.9899 | **Optimal balance (99.1%)** |

---

## 🔍 Key Findings

### 1. Did accuracy drop significantly?
- **3-Component PCA**: With only 3 input features (an **85% reduction** in dimensions), the model still captures **95.23%** of the target variance. However, the Test RMSE increased from **1.1470 kWh** to **7.3615 kWh**, which is a noticeable accuracy drop in absolute terms.
- **10-Component PCA (95% Variance)**: With 10 principal components (a **50% reduction** in dimensions), the model captures **98.99%** of the target variance, with a Test RMSE of **3.3800 kWh**. The accuracy drop is extremely minimal (only a **0.89%** reduction in $R^2$), representing a highly successful dimensionality reduction.

### 2. How many features can safely be removed?
- Based on the cumulative explained variance curve, **10 features can safely be removed** (by projecting the 20 original features onto the first 10 principal components).
- Retaining 10 principal components preserves **95.02%** of the dataset's total variance, ensuring that the model retains almost all its predictive capability while working with half the number of features.

### 3. Cumulative Variance Breakdown
- **PC01** captures **24.73%** of the variance.
- **PC02** captures **19.95%** of the variance.
- **PC03** captures **11.02%** of the variance.
- The first three components combined capture **55.70%** of the total variance.
- The 95% threshold is crossed at exactly **10 components** (reaching **95.02%** cumulative explained variance).

---

## 🛠️ Feature Contribution & Loadings Analysis
The loading heatmap (y-axis: original features, x-axis: first three principal components) reveals the following contributions:
- **PC1 (24.73% variance)** is highly contributed to by energy-related and power factors:
  - Positive loading with `Lagging_Current_Reactive.Power_kVarh` and `CO2(tCO2)`.
  - Negative loading with power factor columns (`Lagging_Current_Power_Factor` and `Leading_Current_Power_Factor`).
- **PC2 (19.95% variance)** captures the temporal and load schedule shifts:
  - Strong loadings with `NSM` (Number of Seconds from Midnight), `Hour`, and `Load_Type_Maximum_Load` / `Load_Type_Medium_Load`.
- **PC3 (11.02% variance)** is heavily influenced by calendar days:
  - Specifically `DayOfWeek` and dummy variables for specific days (like Monday, Saturday, Sunday).

---

## 💡 Recommendation for Memory-Constrained Devices

### Would you recommend PCA for a memory-constrained device?
**Yes, but with caveats.**

#### Why PCA is Recommended:
1. **Model Storage and Size Reduction**: Tree-based ensembles like Random Forest scale in size and complexity with the number of features. Training on 10 principal components instead of 20 reduces the split search space, resulting in shallower or less complex trees and a smaller model memory footprint.
2. **Prediction Speed (Inference Latency)**: Evaluating splits on 10 features instead of 20 is computationally faster. For a micro-controller or edge device, this leads to lower CPU cycles and power consumption during real-time predictions.
3. **Data Storage Savings**: Instead of storing 20 dense features for processing, only the 10 components need to be cached or processed by the regressor.

#### The Downside (Caveat):
- **Transformation Overhead**: PCA is a linear transformation matrix of size $(N_{\text{components}} \times N_{\text{features}})$, which in our case is $(10 \times 20)$. To make a prediction, the device must first scale the incoming raw features and then perform a matrix multiplication to project them into PCA space before passing them to the Random Forest model. 
- **Conclusion**: Since a $(10 \times 20)$ matrix multiplication and scaling are computationally trivial (takes microseconds and minimal RAM), **the memory and speed savings in the Random Forest model far outweigh the minor overhead of the PCA projection**. Thus, **PCA is highly recommended** for memory-constrained and edge devices in this scenario.
