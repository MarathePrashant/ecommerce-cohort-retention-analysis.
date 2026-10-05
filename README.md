# 📈 E-Commerce Retail Sales & Customer Retention Analysis

[![SQL](https://img.shields.io/badge/SQL-Window_Functions-CC292B?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-Cohort_Analysis-blue?style=flat-square)](#)
[![Excel](https://img.shields.io/badge/Excel-Reporting-217346?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](#)

## 📌 Executive Summary & Business Problem

E-commerce businesses need to understand customer purchasing patterns, retention behavior, revenue concentration, and regional performance to improve customer lifetime value and sustainable growth.

This project analyzes multi-region online retail transactions to evaluate sales performance, customer purchasing behavior, monthly retention patterns, and product-level revenue concentration using SQL, Python, and Excel.

## 🛠️ Data Pipeline & Technical Approach

* **Data Cleaning & Preparation:** Removed cancelled transactions, filtered invalid quantities and prices, handled missing customer identifiers, and prepared the dataset for downstream analysis.
* **Advanced SQL Analysis:** Used CTEs, `DENSE_RANK()`, window functions, aggregations, and date-based calculations to analyze customer order frequency, lifecycle metrics, and revenue trends.
* **Customer Cohort Analysis:** Grouped customers by acquisition month and tracked their purchasing activity across subsequent months to evaluate retention behavior.
* **Pareto Analysis:** Evaluated SKU-level revenue contribution to identify the products responsible for the largest share of overall revenue.
* **Reporting & Visualization:** Used Python and Excel to present sales trends, customer retention patterns, cohort analysis, and product concentration insights.

## 📂 Project Structure

```text id="w8n1cz"
├── data/               # Raw and processed online retail transaction data
├── notebooks/          # Python notebooks for data analysis and cohort visualization
├── sql/                # SQL queries for customer and sales analysis
└── README.md           # Business case study, methodology, and insights
```

## 📊 Key Business Insights

* **Customer Retention:** The analysis identified a significant decline in customer activity after the initial purchase period, highlighting the importance of early-stage customer engagement.
* **Regional AOV Differences:** While the domestic market contributed the majority of order volume, selected international markets demonstrated higher Average Order Value (AOV), indicating potential opportunities for targeted market expansion.
* **Revenue Concentration:** A relatively small group of high-performing SKUs contributed a substantial share of overall revenue, demonstrating the importance of effective inventory planning for core products.
* **Customer Purchasing Patterns:** Order frequency and customer lifecycle analysis revealed distinct purchasing behaviors that can be used to design more targeted retention strategies.

## 💡 Strategic Business Recommendations

* **Post-Purchase Engagement:** Introduce automated re-order reminders and personalized product recommendations based on previous purchases and expected replenishment cycles.
* **International Market Expansion:** Prioritize high-AOV international markets for targeted campaigns after evaluating demand, margins, and customer acquisition economics.
* **Core SKU Inventory Protection:** Maintain appropriate safety-stock levels for high-revenue SKUs to reduce potential revenue loss caused by stockouts.
* **Customer Retention Programs:** Develop early-stage engagement campaigns to encourage customers to make a second purchase and improve long-term retention.
* **Customer Segmentation:** Combine purchase frequency, order value, geography, and lifecycle stage to create more targeted customer segments.

## 🚀 How to Explore This Project

1. **Review the SQL Scripts:** Open `/sql` to explore CTEs, window functions, customer lifecycle calculations, cohort grouping, and sales KPIs.
2. **Review the Python Notebook:** Open `/notebooks` to examine data preparation, cohort analysis, retention calculations, and visualization workflows.
3. **Review the Data:** Explore `/data` to understand the transaction-level attributes used for the analysis.

## 👤 Author

**Prashant Marathe**

* **LinkedIn:** https://www.linkedin.com/in/prashantmarathe17
* **Portfolio:** https://prashant-marathe.framer.website/
* **Email:** [p04747391@gmail.com](mailto:p04747391@gmail.com)
* **Location:** Pune, Maharashtra, India
