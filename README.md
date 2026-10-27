# 📈 E-Commerce Cohort Retention & Customer Analytics

[![Python](https://img.shields.io/badge/Python-Analytics-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical_Analysis-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-Business_Intelligence-F2C811?style=flat-square&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![Status](https://img.shields.io/badge/Status-Analytics_Pipeline_Implemented-success?style=flat-square)](#)

## 📌 Executive Summary & Business Problem

E-commerce businesses need to understand more than total sales. Customer retention, repeat purchasing behavior, product performance, and market-level revenue contribution are essential for evaluating business performance and identifying opportunities for sustainable growth.

This project transforms transaction-level online retail data into structured business KPIs, customer-level metrics, monthly cohort retention tables, product and country performance reports, and return/cancellation diagnostics using **Python, Pandas, NumPy, Matplotlib, and Seaborn**.

The generated analytical outputs can also be used in **Power BI** to build an interactive business intelligence dashboard for sales monitoring, customer retention analysis, and performance reporting.

### Business Questions

- How do monthly revenue, order volume, and average order value change over time?
- How does customer retention evolve across acquisition cohorts?
- Which customers are repeat purchasers or high-value customers?
- Which products contribute the most to gross sales revenue?
- How is revenue distributed across countries?
- What patterns appear in cancellation invoices and negative-quantity transactions?
- How can cleaned, structured outputs support business reporting and decision-making?

## 🛠️ Data Pipeline & Technical Approach

### 1. Data Cleaning & Preparation

- Standardized common variations in transaction column names.
- Converted dates, quantities, and prices into appropriate data types.
- Handled common CSV encoding variations.
- Removed rows missing essential transaction fields.
- Created derived time attributes, including year, month, year-month, weekday, and transaction hour.
- Classified transaction records into sales, returns, and cancellation categories.
- Separated eligible positive-quantity sales from reversal and cancellation records.
- Excluded transactions without identified customer IDs from retention analysis.

### 2. Business KPI Analysis

Calculated key business performance indicators:

- Gross sales revenue
- Units sold
- Sales orders
- Identified customers
- Distinct products
- Average order value (AOV)
- Average revenue per identified customer
- Average units per order
- Source row count and sales-line count

Gross sales metrics are calculated using eligible positive-quantity, non-cancellation transactions.

### 3. Monthly Sales Performance

Aggregated transaction data by month to calculate:

- Revenue
- Number of orders
- Units sold
- Active customers
- Average order value

These metrics support monthly performance monitoring and time-series analysis.

### 4. Customer Cohort & Retention Analysis

Assigned customers to cohorts based on their first observed positive-sale month and tracked customer activity across subsequent calendar months.

The pipeline generates:

- Customer-level cohort assignments
- Customer counts by acquisition cohort and cohort age
- A monthly cohort retention matrix
- A retention heatmap for visual analysis

**Retention Rate = Active Customers in a Cohort Month / Original Customers in the Cohort × 100**

This approach helps evaluate how customer activity changes after the first observed purchase and supports the investigation of retention patterns over time.

### 5. Customer Metrics & Segmentation

Created a customer-level analytical dataset containing:

- First and last purchase dates
- Number of distinct orders
- Total purchase revenue
- Units purchased
- Recency in days
- Customer tenure in days
- Average order value
- Repeat-customer indicator
- First-purchase cohort month
- Rule-based customer segment

The implemented segments are:

| Customer Segment | Classification Logic |
|---|---|
| Loyal High-Value | At least two orders and monetary value at or above the 75th percentile |
| Repeat Customer | At least two orders and not classified as Loyal High-Value |
| At-Risk / Inactive | Recency above the 75th percentile among customers not assigned to the preceding segments |
| New / Occasional Customer | Remaining customers |

These are rule-based segments, not machine-learning predictions. The at-risk classification is a preliminary heuristic and should be validated against business-specific definitions.

### 6. Product Performance Analysis

Aggregated eligible sales transactions by product to calculate:

- Gross sales revenue
- Units sold
- Number of distinct orders
- Average unit price

The pipeline ranks products by revenue and generates a visualization of the top 10 products.

### 7. Country Performance Analysis

Evaluated country-level performance using:

- Gross sales revenue
- Number of orders
- Identified customers
- Units sold
- Average order value
- Percentage contribution to total gross sales revenue

This output supports geographic performance comparisons and market-level analysis.

### 8. Returns & Cancellation Diagnostics

Created a separate diagnostic report containing:

- Distinct cancellation invoice count
- Invoices containing negative-quantity lines
- Absolute negative-quantity units
- Positive sales units
- Negative-quantity units as a percentage of positive sales units

Cancellation identification uses the common `C`-prefixed invoice convention. These diagnostic measures require validation against the source dataset's transaction definitions before being interpreted as official business cancellation or refund rates.

## 📂 Project Structure

```text
ecommerce-cohort-retention-analysis/
│
├── ecommerce_cohort_retention_analysis.py
├── Ecommerce_Cohort_Retention_Analytics.pbix
├── online_retail.csv
├── requirements.txt
├── README.md
│
└── outputs/
    ├── business_kpis.csv
    ├── monthly_sales_metrics.csv
    ├── customer_cohort_assignments.csv
    ├── cohort_customer_counts.csv
    ├── cohort_retention_matrix.csv
    ├── customer_metrics_and_segments.csv
    ├── product_performance.csv
    ├── country_performance.csv
    ├── returns_cancellations_diagnostics.csv
    ├── monthly_revenue.png
    ├── cohort_retention_heatmap.png
    ├── top_products_revenue.png
    └── customer_segments.png
```

*Note: The Power BI file, source dataset, and generated outputs should appear in the repository only if they have actually been added. The Python script generates the CSV reports and four charts in the `outputs/` directory.*

## 📊 Generated Reports & Visualizations

The Python pipeline produces nine CSV reports and four charts.

| Output File | Purpose |
|---|---|
| `business_kpis.csv` | Core sales and business KPIs |
| `monthly_sales_metrics.csv` | Monthly revenue, orders, units, customers, and AOV |
| `customer_cohort_assignments.csv` | Customer acquisition cohort assignments |
| `cohort_customer_counts.csv` | Cohort sizes and monthly active customer counts |
| `cohort_retention_matrix.csv` | Customer retention percentages by cohort age |
| `customer_metrics_and_segments.csv` | Customer purchase metrics and segment classifications |
| `product_performance.csv` | Product-level revenue, units, orders, and average unit price |
| `country_performance.csv` | Country-level sales performance and revenue share |
| `returns_cancellations_diagnostics.csv` | Cancellation and negative-quantity diagnostics |

### Generated Charts

| Chart | Analytical Purpose |
|---|---|
| `monthly_revenue.png` | Tracks monthly gross sales revenue |
| `cohort_retention_heatmap.png` | Visualizes retention across acquisition cohorts |
| `top_products_revenue.png` | Compares the top 10 products by revenue |
| `customer_segments.png` | Shows customer distribution by rule-based segment |

## 📈 Power BI Dashboard

The exported CSV reports provide structured inputs for an interactive Power BI dashboard.

### Recommended Dashboard Pages

**Page 1 — Executive Overview**
- Gross sales revenue
- Sales orders
- Units sold
- Identified customers
- Average order value
- Monthly revenue trend

**Page 2 — Customer Retention**
- Cohort retention heatmap
- Acquisition cohort sizes
- Customer retention by cohort age
- Retention trend comparisons

**Page 3 — Customer Segmentation**
- Customer distribution by segment
- Customer revenue by segment
- Repeat-customer analysis
- Recency and tenure analysis

**Page 4 — Product & Country Performance**
- Top products by gross sales revenue
- Country-level revenue contribution
- Orders and units by country
- Average order value comparisons

**Page 5 — Returns & Data Quality**
- Cancellation invoice counts
- Negative-quantity transaction diagnostics
- Data-quality observations
- Transaction classification limitations

These pages are recommended dashboard designs. The final report should reflect the visuals, measures, and relationships actually implemented in the Power BI file.

## 🚀 How to Run the Project

### Prerequisites

- Python 3.10 or later recommended
- A compatible online retail CSV dataset
- Power BI Desktop for optional dashboard development

### 1. Clone the Repository

```bash
git clone https://github.com/MarathePrashant/ecommerce-cohort-retention-analysis.git
cd ecommerce-cohort-retention-analysis
```

### 2. Install Dependencies

Create a `requirements.txt` file with:

```text
pandas
numpy
matplotlib
seaborn
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

### 3. Add the Dataset

Place your source CSV in the project root and name it:

```text
online_retail.csv
```

The dataset must include invoice, quantity, invoice date, price, and customer ID fields. Common alternative column names are supported by the script.

### 4. Execute the Analytics Pipeline

```bash
python ecommerce_cohort_retention_analysis.py --input online_retail.csv --output-dir outputs
```

You can also supply an absolute dataset path:

```bash
python ecommerce_cohort_retention_analysis.py --input "C:/path/to/online_retail.csv" --output-dir outputs
```

After a successful run, the script saves the CSV reports and charts to `outputs/` and prints a summary of the core KPIs, top products, and customer segments.

### 5. Explore the Results in Power BI

1. Open Power BI Desktop.
2. Import the relevant CSV reports from `outputs/`.
3. Configure relationships where needed, using appropriate customer, cohort, date, product, or country keys.
4. Build the report pages around the business questions described above.
5. Validate totals and retention calculations against the Python outputs.
6. Save the dashboard as `Ecommerce_Cohort_Retention_Analytics.pbix`.

## 💡 Business Applications

The analytical outputs can help business teams:

- Monitor sales performance and monthly KPIs.
- Evaluate customer retention patterns over time.
- Identify repeat and high-value customers.
- Investigate customer-engagement opportunities.
- Compare product-level and country-level performance.
- Examine return and cancellation indicators.
- Build more structured reporting workflows using Python and Power BI.

The pipeline provides analytical evidence for investigation and planning; specific business recommendations should be based on the actual results produced from the dataset.

## ⚠️ Limitations & Assumptions

- Cohorts use the first observed purchase month within the available data, which may not represent the customer's true first-ever purchase.
- Retention calculations include only customers with identifiable IDs and eligible positive-sale transactions.
- Cancellation identification relies on the `C`-prefixed invoice convention.
- Negative-quantity transactions are diagnostic indicators and may not capture every return or refund.
- Customer segments use transparent business rules rather than a predictive model.
- Gross sales revenue is calculated as quantity multiplied by unit price for eligible sales rows; it is not equivalent to net revenue or profit.
- Results depend on source-data quality, date coverage, currency conventions, and transaction definitions.

## 🔮 Future Enhancements

- Add automated data-quality tests and validation summaries.
- Introduce customer lifetime value analysis.
- Develop more detailed customer churn and repeat-purchase metrics.
- Extend retention comparisons across products and customer groups.
- Enhance Power BI interactivity with drill-through pages, tooltips, and validated DAX measures.
- Automate report refresh and document verified business insights.

## 🧠 Skills Demonstrated

`Python` · `Pandas` · `NumPy` · `Data Cleaning` · `Exploratory Data Analysis` · `Business KPI Analysis` · `Cohort Analysis` · `Customer Retention` · `Customer Segmentation` · `Data Visualization` · `Power BI` · `Business Intelligence` · `Analytical Reporting`

## 👤 Author

**Prashant Marathe**  
Data Analyst | Business Intelligence

- **GitHub:** [MarathePrashant](https://github.com/MarathePrashant)
- **LinkedIn:** [Prashant Marathe](https://www.linkedin.com/in/prashantmarathe17/)

---

⭐ If you find this project useful, consider starring the repository. Feedback and suggestions are welcome.
