# Anomaly Detection in Maritime Engineering: Ship Engine Predictive Maintenance

![Project Type](https://img.shields.io/badge/Project-Unsupervised_Learning-blue)
![Course](https://img.shields.io/badge/University-Cambridge_C101-gold)
![Tools](https://img.shields.io/badge/Stack-Python_|_PCA_|_Isolation_Forest-green)

## 📌 Project Overview
As part of my Data Science curriculum at the **University of Cambridge**, this project addresses a critical challenge in the global maritime industry: the early detection of mechanical anomalies in ship main engines. Leveraging real-world sensor data, I developed an unsupervised machine learning pipeline designed to identify the "vital sign" deviations—representing the bottom 1% to 5% of data points—that signal impending mechanical failure.

## 💡 Business Context
In the supply chain industry, a poorly maintained ship engine leads to exponential costs through fuel inefficiency, unplanned downtime, and significant safety hazards. This project demonstrates how AI-driven predictive maintenance can transition fleet management from **reactive repairs** to **proactive intervention**.

---

## 🛠️ Technical Implementation: Key Logic
The project showcases a rigorous approach to high-dimensional sensor data, focusing on dimensionality reduction and the triangulation of multiple anomaly detection methodologies.

### 1. Dimensionality Reduction (PCA)
To handle the complexity of multi-sensor data and eliminate noise, I utilised **Principal Component Analysis (PCA)** to capture 95% of the operational variance.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardising the feature space
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# PCA to retain 95% variance for a cleaner signal
pca = PCA(n_components=0.95)
pca_transformed = pca.fit_transform(data_scaled)

print(f"Reduced feature space to {pca.n_components_} principal components.")
```

### 2. Isolation Forest (Unsupervised Detection)

I implemented Isolation Forest to isolate anomalies based on the principle that outliers are few and different in the high-dimensional feature space.

```python
from sklearn.ensemble import IsolationForest

# Iterative testing across contamination levels (1% to 5%)
model = IsolationForest(contamination=0.05, random_state=42)
data['anomaly_label'] = model.fit_predict(pca_transformed)

# -1 represents an identified mechanical anomaly
anomalies = data[data['anomaly_label'] == -1]
```

---

## 📊 Result Analysis & Visualisation
### I. Feature Correlation & "Non-Linear" Failure

The Correlation Heatmap revealed weak linear relationships between individual sensors. This confirms that engine failure is rarely a simple linear cascade; rather, it is a complex, non-linear deviation from homeostasis that requires sophisticated models like Isolation Forest rather than simple threshold alerts.

![Correlation Heatmap](/images/image1.png)

### II. PCA Scree Analysis

The Explained Variance Ratio plot indicates that the majority of engine behaviour can be explained by a subset of principal components. By focusing on these, the model effectively ignores operational "noise" and focuses on genuine systemic shifts.

![Correlation Heatmap](/images/image2.png)

### III. Triangulating the "Outlier"

The study visualised anomalies across 1% to 5% contamination levels.

**Findings**: While statistical methods like IQR identified rigid outliers, Isolation Forest and ocSVM provided more nuanced results by considering the interaction of multiple sensors simultaneously.

**Logic Audit**: The divergence between models proves that anomaly detection is a decision-support tool; different models "see" failure differently depending on whether they prioritise distance-based or tree-partitioning logic.

![Correlation Heatmap](/images/image3.png)

## 🔑 Key Takeaways & Applications
* **Model Triangulation**: Relying on a single algorithm is insufficient; a combination of statistical and ML approaches is required to provide a comprehensive "safety net" for maritime assets.

* **The "Say-Do" Gap in Machinery**: Sensors may "say" they are within normal parameters individually, but their collective behaviour (detected via PCA/Isolation Forest) may "do" something highly anomalous.

* **Potential Applications:**

  * Real-time Fleet Monitoring: Integrating this pipeline into IoT dashboards for shore-based technical teams.

  * Cost Reduction: Minimising unplanned downtime through early "yellow-flag" warnings.

  * Environmental Compliance: Detecting fuel-burning anomalies that lead to excessive emissions.

## 📂 Project Structure
`Gao_Yuliang_CAM_C101_W5_Mini_project.ipynb`: Full Jupyter Notebook including EDA, PCA, and Model Evaluation.

`Project_Visualisations.pdf`: Comprehensive PDF including all result plots and cluster analysis.

**Data Source**: Devabrat, M. (2022). Predictive Maintenance on Ship's Main Engine using AI.

---
**Location**: Toronto, ON, Canada 🇨🇦

**Academic Context**: University of Cambridge | Certificate in Data Science & Machine Learning
