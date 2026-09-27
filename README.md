<div align="center">

# 📊 E-Commerce & Retail Customer Analytics Platform

**End-to-end data analytics pipeline transforming raw transactional records into actionable customer cohorts, RFM segmentations, and revenue growth strategies.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Interactive Dashboard](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-dashboard-link.streamlit.app)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-View_Analysis-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://github.com/yourusername/retail-analytics/blob/main/notebooks/customer_segmentation.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 🎯 Executive Summary & Business Impact

Online retailers lose significant revenue through broad, non-targeted marketing and unmonitored customer churn. This project analyzes **500,000+ transactional records** to extract behavioral patterns, evaluate retention lifecycles, and build programmatic customer segmentation.

* **Revenue Concentration:** Discovered that the top **15% of customers drive 68% of total revenue**, enabling high-ROI VIP retention programs.
* **Retention Drop-off:** Identified a critical drop-off point at **Month 3** across new customer cohorts, directly informing targeted re-engagement campaigns.
* **Marketing Efficiency:** Built dynamic **RFM (Recency, Frequency, Monetary)** clusters to replace uniform email blasts with tiered, persona-driven lifecycle messaging.

---

## 📸 Interactive Dashboard Preview

<!-- Replace with an actual screenshot or GIF of your Streamlit / Tableau / PowerBI dashboard -->
![Retail Analytics Dashboard](https://raw.githubusercontent.com/yourusername/retail-analytics/main/assets/dashboard_preview.png)

> **Live Demo:** Explore the interactive dashboard at **[retail-analytics.streamlit.app](https://your-dashboard-link.streamlit.app)** (Filter by cohort, drill into RFM segments, and inspect customer lifetime value).

---

## 🔬 Key Analytical Findings

### 1. Customer Retention (Cohort Analysis)
Tracking monthly acquisition cohorts revealed:
- Average Month-1 retention sits at **~24%**, stabilizing around **14%** by Month 6.
- Holiday-season cohorts (Q4) demonstrate higher initial baskets but churn **18% faster** than organic spring cohorts.

### 2. Behavioral Segmentation (RFM Scoring & K-Means)
Customers were evaluated across three normalized vectors:
- **Recency ($R$):** Days since last order.
- **Frequency ($F$):** Total lifetime transactions.
- **Monetary ($M$):** Cumulative net spend.

| Customer Tier | Population (%) | Revenue Share (%) | Actionable Strategy |
| :--- | :---: | :---: | :--- |
| **Champions / VIPs** | 8.2% | 46.1% | Early product access, loyalty rewards, direct outreach. |
| **Loyal Customers** | 18.5% | 27.3% | Up-sell premium lines, referral incentives. |
| **At Risk / Churning** | 14.1% | 12.8% | Automated win-back discounts before Day 90 threshold. |
| **Hibernating / Lost** | 59.2% | 13.8% | Low-cost retargeting via paid socials; suppress active email lists. |

---

## 🛠 Tech Stack & Methodology

| Component | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Data Processing** | Python, `pandas`, `NumPy` | Cleaning, missing-value imputation, outlier removal |
| **Data Storage / Querying** | PostgreSQL / SQLite | Relational schema modeling and aggregations |
| **Statistical Analysis** | `scipy.stats`, `scikit-learn` | Feature scaling, Log-transforms, K-Means clustering |
| **Visualization** | `matplotlib`, `seaborn`, `Plotly` | Heatmaps, cohort retention grids, distribution curves |
| **Delivery / UI** | Streamlit (or PowerBI / Tableau) | Interactive business intelligence dashboard |

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                 # Original, immutable transactional dataset
│   └── processed/           # Cleaned, RFM-scored data files
├── notebooks/
│   ├── 01_data_cleaning.ipynb       # Anomaly detection & cancellation handling
│   ├── 02_cohort_analysis.ipynb     # Retention matrix calculation
│   └── 03_rfm_segmentation.ipynb    # K-Means clustering & segment profiling
├── src/
│   ├── pipeline.py          # Automated ETL script
│   └── utils.py             # Reusable calculation & plotting helpers
├── app.py                   # Streamlit dashboard application
├── requirements.txt         # Pinned project dependencies
└── README.md
