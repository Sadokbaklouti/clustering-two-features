
# 🏦 Household Segmentation: Clustering with K-Means

> **"An original project by Guillaume Saint-Cirgue, extracted from the Cahier de Vacances Data & IA 2026."**

This project applies an unsupervised machine learning algorithm to segment American households (focusing on those who expressed credit-related fear — `TURNFEAR`) into distinct groups. The objective is to analyze the financial profiles of these households based on two key characteristics: **total household debt** (`DEBT`) and **the value of their primary real estate/properties** (`HOUSES`).

---

## 📊 Exploration & Problem Statement

The analysis relies on data from the Federal Reserve's *Survey of Consumer Finances (SCF)*. 

To understand the behavior of credit-fearful households (`TURNFEAR == 1`), we examine the relationship between:
* **Household Debt (`DEBT`)**: Mortgages, consumer loans, etc.
* **Home Value (`HOUSES`)**: Value of the primary residence and other real estate assets.

In the absence of pre-existing classification labels, leveraging clustering techniques allows unique socio-economic profiles to emerge naturally from the data.

---

## 🛠️ Modeling Pipeline

The project follows standard Data Science lifecycle stages:

1. **Data Preparation (`Wrangle`)**: Filtering the dataset to isolate the specific target demographic (`TURNFEAR == 1`).
2. **Feature Selection**: Extracting the feature matrix $X = [\text{DEBT}, \text{HOUSES}]$.
3. **Model Training**: Configuring and fitting the **K-Means** algorithm with $K = 3$ clusters (reproducibility is ensured using `random_state=42`).
4. **Centroid Extraction & Analysis**: Computing the financial "center of gravity" for each group to interpret their characteristics.

---

## 📈 Results and Visualization

The model separates households into 3 distinct clusters, visually mapped using `seaborn` and `matplotlib` scatter plots. The centers of each group (the centroids) are highlighted with red stars on the final plot.

### Interpretation of the 3 Identified Profiles:
* **Cluster 0 (Standard Households)**: Low debt levels and low real estate value. This represents the vast majority of the studied population.
* **Cluster 2 (Wealthy / High-Debt Households)**: Moderate to high debt levels but backed by solid real estate equity.
* **Cluster 1 (Ultra-Wealthy / Outlier Profiles)**: Exceptionally high debt levels (extreme values) relative to their declared real estate holdings.

---

## 🚀 Technologies Used

* **Python 3**
* **Pandas & NumPy**: For data cleaning, filtering, and array manipulation.
* **Scikit-Learn**: For implementing the `KMeans` model and computing evaluation metrics like `silhouette_score`.
* **Seaborn & Matplotlib**: For creating comprehensive data visualizations.
