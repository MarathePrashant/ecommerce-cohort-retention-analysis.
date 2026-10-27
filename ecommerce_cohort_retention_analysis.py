
"""
ONLINE RETAIL ANALYTICS
Author: Prashant Marathe

Skills demonstrated:
- Python, Pandas and NumPy
- Data cleaning and data-quality auditing
- Retail sales and transaction analysis
- Return and cancellation analysis
- Product and country performance
- Customer analytics and RFM segmentation
- K-Means clustering
- Business reporting and data visualization

Expected input:
    online_retail_II.csv

Outputs:
    outputs/*.csv
    outputs/business_findings.txt
    outputs/charts/*.png
"""

from pathlib import Path
import os
import re
import warnings

# Reduce avoidable CPU-core detection warnings from joblib on Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 2))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "online_retail_II.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TOP_N = 20
CURRENCY_SYMBOL = ""  # Dataset currency is not assumed here.

# Known non-merchandise/service stock codes.
NON_MERCHANDISE_CODES = {
    "POST", "DOT", "M", "D", "CRUK",
    "BANK CHARGES", "AMAZONFEE", "PADS",
    "C2", "S", "B", "ADJUST",
}

# Description-based rules are a starting classification, not
# a definitive substitute for validating the source data.
NON_MERCHANDISE_PATTERNS = [
    r"\bPOSTAGE\b",
    r"\bCARRIAGE\b",
    r"\bBANK CHARGES?\b",
    r"\bMANUAL\b",
    r"\bADJUSTMENT\b",
    r"\bDISCOUNT\b",
    r"\bAMAZON FEE\b",
    r"\bDOTCOM POSTAGE\b",
    r"\bSAMPLES?\b",
    r"\bDAMAGES?\b",
    r"\bCRUK\b",
    r"\bPACKING CHARGE\b",
    r"\bNEXT DAY CARRIAGE\b",
    r"\bPOSTAL CHARGE\b",
]

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "figure.dpi": 120,
    "savefig.dpi": 160,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def save_csv(dataframe, filename):
    """Save a DataFrame to the output directory."""
    path = OUTPUT_DIR / filename
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path.name}")


def save_chart(filename):
    """Save and close the current Matplotlib figure."""
    path = CHART_DIR / filename
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {filename}")


def money(value):
    """Format a number for console reporting."""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def clean_customer_id(series):
    """
    Convert IDs such as 13085.0 into '13085'.
    Missing IDs remain missing instead of becoming a fake customer.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="string")

    valid = numeric.notna()
    integer_like = valid & np.isclose(
        numeric.fillna(0) % 1, 0, atol=1e-9
    )

    result.loc[integer_like] = (
        numeric.loc[integer_like].round().astype("int64").astype(str)
    )

    non_integer = valid & ~integer_like
    result.loc[non_integer] = numeric.loc[non_integer].astype(str)

    return result


def score_quantile(series, ascending=True):
    """
    Assign robust 1–5 scores without relying on qcut.
    Ties are handled consistently using percentile ranks.
    """
    ranks = series.rank(method="average", pct=True, ascending=ascending)
    scores = np.ceil(ranks * 5).clip(1, 5)
    return scores.fillna(1).astype(int)


def safe_divide(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def is_non_merchandise(stock_code, description):
    code = str(stock_code).strip().upper()
    desc = str(description).strip().upper()

    if code in NON_MERCHANDISE_CODES:
        return True

    return any(
        re.search(pattern, desc, flags=re.IGNORECASE)
        for pattern in NON_MERCHANDISE_PATTERNS
    )


def save_bar_chart(series, title, xlabel, ylabel, filename,
                   horizontal=False, top_n=None):
    """Reusable chart function with empty-data protection."""
    data = series.dropna()

    if top_n is not None:
        data = data.head(top_n)

    if data.empty:
        print(f"Skipped chart (no data): {filename}")
        return

    plt.figure()

    if horizontal:
        data = data.sort_values()
        data.plot(kind="barh")
    else:
        data.plot(kind="bar")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_chart(filename)


# ============================================================
# 3. LOAD AND VALIDATE THE DATASET
# ============================================================

print_section("ONLINE RETAIL ANALYTICS")

print(f"Dataset: {INPUT_FILE}")
print(f"Output folder: {OUTPUT_DIR}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "\nDataset file not found.\n"
        f"Expected location:\n{INPUT_FILE}\n\n"
        "Place online_retail_II.csv in the same folder as online_retail.py."
    )

try:
    df_raw = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8",
        low_memory=False
    )
except UnicodeDecodeError:
    df_raw = pd.read_csv(
        INPUT_FILE,
        encoding="latin1",
        low_memory=False
    )

print_section("INITIAL DATA PROFILE")
print(f"Rows: {len(df_raw):,}")
print(f"Columns: {len(df_raw.columns)}")
print(f"Original columns: {df_raw.columns.tolist()}")
print("\nFirst five records:")
print(df_raw.head().to_string(index=False))
print("\nOriginal data types:")
print(df_raw.dtypes.to_string())
print("\nMissing values in source data:")
print(df_raw.isna().sum().sort_values(ascending=False).to_string())


# ============================================================
# 4. STANDARDIZE COLUMN NAMES
# ============================================================

df = df_raw.copy()

df.columns = [
    str(column).strip().replace("\ufeff", "")
    for column in df.columns
]

column_aliases = {
    "Invoice": "InvoiceNo",
    "Invoice No": "InvoiceNo",
    "Invoice Number": "InvoiceNo",
    "Stock Code": "StockCode",
    "Invoice Date": "InvoiceDate",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
    "Customer Id": "CustomerID",
}

df = df.rename(columns=column_aliases)

required_columns = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

missing_required = [
    column for column in required_columns
    if column not in df.columns
]

if missing_required:
    raise ValueError(
        "Required dataset columns are missing: "
        + ", ".join(missing_required)
        + "\nAvailable columns: "
        + ", ".join(map(str, df.columns))
    )

df = df[required_columns].copy()


# ============================================================
# 5. DATA TYPE CONVERSION AND DATA-QUALITY AUDIT
# ============================================================

df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
df["StockCode"] = df["StockCode"].astype("string").str.strip()
df["Description"] = df["Description"].astype("string").str.strip()
df["Country"] = df["Country"].astype("string").str.strip()
df["CustomerID"] = clean_customer_id(df["CustomerID"])

df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

# The source dataset uses month-first dates in the reported sample.
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce",
    dayfirst=False
)

print("\nDate validation")
print(f"Earliest valid date: {df['InvoiceDate'].min()}")
print(f"Latest valid date:   {df['InvoiceDate'].max()}")

invalid_dates = int(df["InvoiceDate"].isna().sum())
invalid_date_pct = safe_divide(invalid_dates, len(df)) * 100

print(f"Invalid dates: {invalid_dates:,}")
print(
    "Invalid date percentage: "
    f"{invalid_date_pct:.4f}%"
    if pd.notna(invalid_date_pct)
    else "Invalid date percentage: N/A"
)

quality_summary = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": [int(df[column].isna().sum()) for column in df.columns],
    "Missing_Percentage": [
        round(df[column].isna().mean() * 100, 4)
        for column in df.columns
    ],
    "Data_Type": [str(df[column].dtype) for column in df.columns],
})

print_section("DATA QUALITY AUDIT")
print(quality_summary.to_string(index=False))
save_csv(quality_summary, "data_quality_summary.csv")

source_rows = len(df)
exact_duplicates = int(df.duplicated().sum())
negative_quantity_rows = int((df["Quantity"] < 0).sum())
negative_price_rows = int((df["UnitPrice"] < 0).sum())

print(f"\nSource rows: {source_rows:,}")
print(f"Exact duplicate rows: {exact_duplicates:,}")
print(f"Negative quantity rows: {negative_quantity_rows:,}")
print(f"Negative unit-price rows: {negative_price_rows:,}")


# ============================================================
# 6. CLEANING AND TRANSACTION VALIDATION
# ============================================================

print_section("DATA CLEANING")

rows_before = len(df)

# Remove exact duplicate records.
df = df.drop_duplicates().copy()

# Retain missing CustomerID records for sales reporting, but exclude
# them from customer-level analysis later.
# Rows without essential transaction fields are removed.
required_transaction_fields = [
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "Country",
]

invalid_required_mask = df[required_transaction_fields].isna().any(axis=1)

invalid_required_rows = int(invalid_required_mask.sum())

df = df.loc[~invalid_required_mask].copy()

# Remove empty invoice/stock-code identifiers.
empty_identifier_mask = (
    df["InvoiceNo"].fillna("").str.strip().eq("")
    | df["StockCode"].fillna("").str.strip().eq("")
)

empty_identifier_rows = int(empty_identifier_mask.sum())
df = df.loc[~empty_identifier_mask].copy()

# Fill descriptions only for readability; preserve the fact that the
# original description was missing.
df["Description"] = df["Description"].fillna("Unknown description")

# Flag invalid prices; do not silently rewrite the source UnitPrice.
df["Has_Negative_UnitPrice"] = df["UnitPrice"] < 0

# Remove rows that have no parseable date or essential numeric fields.
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

df = df.dropna(
    subset=["InvoiceDate", "Quantity", "UnitPrice"]
).copy()

rows_after = len(df)

print(f"Rows before cleaning: {rows_before:,}")
print(f"Exact duplicates removed: {exact_duplicates:,}")
print(f"Rows missing required transaction fields removed: {invalid_required_rows:,}")
print(f"Rows with empty invoice/stock code removed: {empty_identifier_rows:,}")
print(f"Rows after cleaning: {rows_after:,}")
print(f"Rows with negative unit prices flagged: {int(df['Has_Negative_UnitPrice'].sum()):,}")
print(f"Rows with missing customer IDs retained: {int(df['CustomerID'].isna().sum()):,}")


# ============================================================
# 7. TRANSACTION CLASSIFICATION AND VALUE CALCULATIONS
# ============================================================

print_section("TRANSACTION CLASSIFICATION")

df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
df["InvoiceNo_Upper"] = df["InvoiceNo"].str.upper()

# Cancellation invoices are identified by the conventional C prefix.
df["Is_Cancellation"] = df["InvoiceNo_Upper"].str.startswith(
    "C", na=False
)

# Non-cancellation negative-quantity lines are return-like records.
# The dataset convention should still be validated against documentation.
df["Is_Return_Line"] = (
    (df["Quantity"] < 0)
    & ~df["Is_Cancellation"]
)

df["Is_Sales_Line"] = (
    (df["Quantity"] > 0)
    & ~df["Is_Cancellation"]
)

df["Is_Zero_Quantity_Line"] = df["Quantity"] == 0

df["Line_Value"] = df["Quantity"] * df["UnitPrice"]

# Exclude negative prices from financial measures and flag them for review.
valid_price = df["UnitPrice"] >= 0

df["Sales_Revenue"] = np.where(
    df["Is_Sales_Line"] & valid_price,
    df["Line_Value"].clip(lower=0),
    0.0
)

# Return values remain negative. Invalid negative prices are not assigned
# a financial value, but the associated rows remain available for review.
df["Return_Revenue"] = np.where(
    df["Is_Return_Line"] & valid_price,
    -np.abs(df["Line_Value"]),
    0.0
)

# Cancellation values are shown as negative adjustments for reporting.
# This is a transaction-analysis convention, not audited accounting revenue.
df["Cancellation_Revenue"] = np.where(
    df["Is_Cancellation"] & valid_price,
    -np.abs(df["Line_Value"]),
    0.0
)

df["Recorded_Net_Transaction_Value"] = (
    df["Sales_Revenue"]
    + df["Return_Revenue"]
    + df["Cancellation_Revenue"]
)

# Merchandise classification used only for product ranking.
df["Is_Non_Merchandise"] = [
    is_non_merchandise(code, description)
    for code, description in zip(
        df["StockCode"], df["Description"]
    )
]

print(df[
    ["Is_Sales_Line", "Is_Return_Line", "Is_Cancellation"]
].sum().to_string())

sales_line_count = int(df["Is_Sales_Line"].sum())
return_line_count = int(df["Is_Return_Line"].sum())
cancellation_line_count = int(df["Is_Cancellation"].sum())

print("\nTransaction validation:")
print(f"Sales lines: {sales_line_count:,}")
print(f"Return lines: {return_line_count:,}")
print(f"Cancellation lines: {cancellation_line_count:,}")

save_csv(df.drop(columns=["InvoiceNo_Upper"]), "cleaned_online_retail.csv")


# ============================================================
# 8. CALENDAR FEATURES
# ============================================================

df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Year_Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
df["Day_Name"] = df["InvoiceDate"].dt.day_name()
df["Hour"] = df["InvoiceDate"].dt.hour
df["Date"] = df["InvoiceDate"].dt.date

earliest_date = df["InvoiceDate"].min()
latest_date = df["InvoiceDate"].max()

first_month_start = earliest_date.to_period("M").start_time
last_month_end = latest_date.to_period("M").end_time

partial_months = set()

if earliest_date > first_month_start:
    partial_months.add(str(earliest_date.to_period("M")))

if latest_date < last_month_end:
    partial_months.add(str(latest_date.to_period("M")))


# ============================================================
# 9. BUSINESS KPI SUMMARY
# ============================================================

print_section("BUSINESS KPI DASHBOARD")

gross_sales = float(df["Sales_Revenue"].sum())
return_value = float(df["Return_Revenue"].sum())
cancellation_value = float(df["Cancellation_Revenue"].sum())
recorded_net_value = float(df["Recorded_Net_Transaction_Value"].sum())

sales_df = df.loc[df["Is_Sales_Line"]].copy()
return_df = df.loc[df["Is_Return_Line"]].copy()
cancellation_df = df.loc[df["Is_Cancellation"]].copy()

sales_invoices = int(sales_df["InvoiceNo"].nunique())
return_invoices = int(return_df["InvoiceNo"].nunique())
cancellation_invoices = int(cancellation_df["InvoiceNo"].nunique())

units_sold = float(
    df.loc[df["Is_Sales_Line"] & valid_price, "Quantity"].sum()
)

identified_customers = int(
    sales_df.loc[sales_df["CustomerID"].notna(), "CustomerID"].nunique()
)

unique_products = int(
    sales_df.loc[sales_df["Is_Non_Merchandise"].eq(False), "StockCode"].nunique()
)

average_order_value = safe_divide(gross_sales, sales_invoices)

average_selling_price = safe_divide(
    gross_sales,
    units_sold
)

return_invoice_pct_of_sales = (
    safe_divide(return_invoices, sales_invoices) * 100
)

kpi_summary = pd.DataFrame([
    ("Gross Sales", gross_sales),
    ("Return Value (negative)", return_value),
    ("Cancellation Value (negative adjustment)", cancellation_value),
    ("Recorded Net Transaction Value", recorded_net_value),
    ("Sales Invoices", sales_invoices),
    ("Distinct Return Invoices", return_invoices),
    ("Distinct Cancellation Invoices", cancellation_invoices),
    ("Units Sold", units_sold),
    ("Identified Customers", identified_customers),
    ("Unique Merchandise Stock Codes", unique_products),
    ("Average Order Value", average_order_value),
    ("Average Selling Price per Unit", average_selling_price),
    ("Return Invoices as % of Sales Invoices", return_invoice_pct_of_sales),
], columns=["KPI", "Value"])

print(kpi_summary.to_string(index=False))
save_csv(kpi_summary, "business_kpi_summary.csv")

print("\nFinancial reconciliation check")
print(
    "Gross sales + returns + cancellations: "
    f"{money(gross_sales + return_value + cancellation_value)}"
)
print(f"Recorded net transaction value: {money(recorded_net_value)}")
print(f"Dataset date range: {earliest_date} to {latest_date}")


# ============================================================
# 10. MONTHLY SALES PERFORMANCE
# ============================================================

print_section("MONTHLY SALES PERFORMANCE")

monthly_sales = (
    df.loc[df["Is_Sales_Line"]]
    .groupby("Year_Month", as_index=False)
    .agg(
        Gross_Sales=("Sales_Revenue", "sum"),
        Units_Sold=("Quantity", "sum"),
        Sales_Invoices=("InvoiceNo", "nunique"),
        Unique_Customers=("CustomerID", "nunique"),
    )
    .sort_values("Year_Month")
    .reset_index(drop=True)
)

monthly_sales["Average_Order_Value"] = (
    monthly_sales["Gross_Sales"]
    / monthly_sales["Sales_Invoices"].replace(0, np.nan)
)

monthly_sales["Month_Over_Month_Growth_Pct"] = (
    monthly_sales["Gross_Sales"].pct_change() * 100
)

monthly_sales["Is_Partial_Month"] = (
    monthly_sales["Year_Month"].isin(partial_months)
)

# Growth comparisons involving a partial current or previous month are
# suppressed so incomplete months are not compared as full periods.
for index in range(len(monthly_sales)):
    current_partial = monthly_sales.loc[index, "Is_Partial_Month"]
    previous_partial = (
        monthly_sales.loc[index - 1, "Is_Partial_Month"]
        if index > 0 else False
    )

    if current_partial or previous_partial:
        monthly_sales.loc[index, "Month_Over_Month_Growth_Pct"] = np.nan

print(monthly_sales.to_string(index=False))
save_csv(monthly_sales, "monthly_sales_trend.csv")

if not monthly_sales.empty:
    peak_month_row = monthly_sales.loc[
        monthly_sales["Gross_Sales"].idxmax()
    ]

    print(
        f"\nHighest-sales month: {peak_month_row['Year_Month']} "
        f"with gross sales of {money(peak_month_row['Gross_Sales'])}"
    )

    if peak_month_row["Is_Partial_Month"]:
        print("Note: this peak month is a partial month.")


# ============================================================
# 11. COUNTRY PERFORMANCE
# ============================================================

print_section("COUNTRY PERFORMANCE")

country_sales = (
    df.loc[df["Is_Sales_Line"]]
    .groupby("Country", as_index=False)
    .agg(
        Gross_Sales=("Sales_Revenue", "sum"),
        Units_Sold=("Quantity", "sum"),
        Sales_Invoices=("InvoiceNo", "nunique"),
        Unique_Customers=("CustomerID", "nunique"),
    )
)

country_sales["Sales_Contribution_Pct"] = (
    country_sales["Gross_Sales"]
    / gross_sales * 100
    if gross_sales else 0
)

country_sales["Average_Order_Value"] = (
    country_sales["Gross_Sales"]
    / country_sales["Sales_Invoices"].replace(0, np.nan)
)

country_sales = country_sales.sort_values(
    "Gross_Sales", ascending=False
).reset_index(drop=True)

print(country_sales.head(15).to_string(index=False))
save_csv(country_sales, "country_sales_performance.csv")

if not country_sales.empty:
    leading_country = country_sales.iloc[0]
    print(
        f"\nLeading country: {leading_country['Country']} "
        f"({leading_country['Sales_Contribution_Pct']:.2f}% of gross sales)"
    )


# ============================================================
# 12. PRODUCT PERFORMANCE AND ANOMALY FLAGS
# ============================================================

print_section("PRODUCT PERFORMANCE")

merchandise_sales = df.loc[
    df["Is_Sales_Line"]
    & ~df["Is_Non_Merchandise"]
    & valid_price
].copy()

product_performance = (
    merchandise_sales
    .groupby("StockCode", as_index=False, dropna=False)
    .agg(
        Description=("Description", "first"),
        Gross_Sales=("Sales_Revenue", "sum"),
        Units_Sold=("Quantity", "sum"),
        Sales_Invoices=("InvoiceNo", "nunique"),
        Unique_Customers=("CustomerID", "nunique"),
    )
)

# Flag unusually concentrated products for review instead of allowing a
# single exceptional invoice to dominate the standard top-product list.
product_performance["Flag_Single_Invoice_High_Volume"] = (
    (product_performance["Sales_Invoices"] <= 1)
    & (product_performance["Units_Sold"] >= 10000)
)

product_performance["Flag_Single_Invoice_High_Value"] = (
    (product_performance["Sales_Invoices"] <= 1)
    & (product_performance["Gross_Sales"] >= 100000)
)

product_performance["Requires_Review"] = (
    product_performance["Flag_Single_Invoice_High_Volume"]
    | product_performance["Flag_Single_Invoice_High_Value"]
)

eligible_products = product_performance.loc[
    ~product_performance["Requires_Review"]
].copy()

eligible_product_sales_total = float(
    eligible_products["Gross_Sales"].sum()
)

eligible_products = eligible_products.sort_values(
    "Gross_Sales", ascending=False
).reset_index(drop=True)

if eligible_product_sales_total > 0:
    eligible_products["Sales_Contribution_Pct"] = (
        eligible_products["Gross_Sales"]
        / eligible_product_sales_total * 100
    )
    eligible_products["Cumulative_Sales_Pct"] = (
        eligible_products["Sales_Contribution_Pct"].cumsum()
    )
else:
    eligible_products["Sales_Contribution_Pct"] = 0.0
    eligible_products["Cumulative_Sales_Pct"] = 0.0

# Keep all product rows in the comprehensive export and make the
# ranking eligibility explicit.
product_performance = product_performance.merge(
    eligible_products[
        ["StockCode", "Sales_Contribution_Pct", "Cumulative_Sales_Pct"]
    ],
    on="StockCode",
    how="left",
)

product_performance["Ranking_Eligible"] = (
    ~product_performance["Requires_Review"]
)

product_performance = product_performance.sort_values(
    "Gross_Sales", ascending=False
).reset_index(drop=True)

print("Top products eligible for standard merchandise ranking:")
print(
    eligible_products.head(TOP_N).to_string(index=False)
)

save_csv(product_performance, "product_sales_performance.csv")

product_exceptions = product_performance.loc[
    product_performance["Requires_Review"]
].copy()

save_csv(product_exceptions, "product_anomalies_for_review.csv")

if not eligible_products.empty:
    top_product = eligible_products.iloc[0]
    print(
        "\nTop ranking-eligible merchandise entry: "
        f"{top_product['Description']} "
        f"({money(top_product['Gross_Sales'])})"
    )

    products_to_80 = int(
        (eligible_products["Cumulative_Sales_Pct"] < 80).sum() + 1
    )

    print(
        "Products needed to reach approximately 80% of "
        f"ranking-eligible merchandise sales: {products_to_80:,}"
    )

if not product_exceptions.empty:
    print(
        f"\nReview required for {len(product_exceptions):,} "
        "product entries with unusual invoice concentration."
    )


# ============================================================
# 13. CUSTOMER SALES ANALYSIS
# ============================================================

print_section("CUSTOMER SALES ANALYSIS")

identified_sales = df.loc[
    df["Is_Sales_Line"]
    & df["CustomerID"].notna()
    & valid_price
].copy()

customer_sales = (
    identified_sales
    .groupby("CustomerID", as_index=False)
    .agg(
        Gross_Sales=("Sales_Revenue", "sum"),
        Units_Purchased=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"),
        First_Purchase=("InvoiceDate", "min"),
        Last_Purchase=("InvoiceDate", "max"),
        Products_Purchased=("StockCode", "nunique"),
    )
)

customer_sales["Average_Order_Value"] = (
    customer_sales["Gross_Sales"]
    / customer_sales["Orders"].replace(0, np.nan)
)

customer_sales["Customer_Tenure_Days"] = (
    customer_sales["Last_Purchase"]
    - customer_sales["First_Purchase"]
).dt.days

customer_sales = customer_sales.sort_values(
    "Gross_Sales", ascending=False
).reset_index(drop=True)

print(customer_sales.head(20).to_string(index=False))
save_csv(customer_sales, "customer_sales_summary.csv")

identified_customer_revenue = float(
    customer_sales["Gross_Sales"].sum()
)

if not customer_sales.empty:
    median_customer_sales = customer_sales["Gross_Sales"].median()
    median_customer_orders = customer_sales["Orders"].median()

    top_10_customer_revenue = float(
        customer_sales.head(10)["Gross_Sales"].sum()
    )

    top_10_customer_share = (
        safe_divide(
            top_10_customer_revenue,
            identified_customer_revenue
        ) * 100
    )

    print(f"\nIdentified customers: {len(customer_sales):,}")
    print(
        "Median gross sales per customer: "
        f"{money(median_customer_sales)}"
    )
    print(
        "Median sales invoices per customer: "
        f"{median_customer_orders:.0f}"
    )
    print(
        "Top 10 customers' share of identified-customer gross sales: "
        f"{top_10_customer_share:.2f}%"
    )
else:
    median_customer_sales = np.nan
    median_customer_orders = np.nan
    top_10_customer_share = np.nan


# ============================================================
# 14. RFM CUSTOMER SEGMENTATION
# ============================================================

print_section("RFM CUSTOMER SEGMENTATION")

if not customer_sales.empty:
    reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    rfm = customer_sales[
        ["CustomerID", "Last_Purchase", "Orders", "Gross_Sales"]
    ].copy()

    rfm = rfm.rename(columns={
        "Orders": "Frequency",
        "Gross_Sales": "Monetary",
    })

    rfm["Recency"] = (
        reference_date - rfm["Last_Purchase"].dt.normalize()
    ).dt.days

    # A sales-based RFM model: Monetary reflects gross sales, not a
    # customer-level net value after linking returns to original orders.
    rfm["R_Score"] = score_quantile(rfm["Recency"], ascending=False)
    rfm["F_Score"] = score_quantile(rfm["Frequency"], ascending=True)
    rfm["M_Score"] = score_quantile(rfm["Monetary"], ascending=True)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    def assign_segment(row):
        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 4:
            return "Loyal Customers"
        if r >= 4 and f <= 2:
            return "Recent Customers"
        if r <= 2 and f >= 3 and m >= 3:
            return "At-Risk High-Value"
        if r <= 2 and f <= 2:
            return "Hibernating"
        if m >= 4:
            return "High-Spending Customers"
        return "Needs Attention"

    rfm["Customer_Segment"] = rfm.apply(assign_segment, axis=1)

    print(f"RFM reference date: {reference_date.date()}")
    print(f"Customers segmented: {len(rfm):,}")
    print(rfm.sort_values(
        "Monetary", ascending=False
    ).head(20).to_string(index=False))

    save_csv(rfm, "customer_rfm_segmentation.csv")

    segment_summary = (
        rfm.groupby("Customer_Segment", as_index=False)
        .agg(
            Customers=("CustomerID", "nunique"),
            Average_Recency_Days=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Monetary=("Monetary", "mean"),
            Total_Monetary=("Monetary", "sum"),
        )
    )

    segment_summary["Customer_Share_Pct"] = (
        segment_summary["Customers"] / len(rfm) * 100
    )

    segment_summary["Revenue_Share_Pct"] = (
        segment_summary["Total_Monetary"]
        / rfm["Monetary"].sum() * 100
        if rfm["Monetary"].sum() else 0
    )

    segment_summary = segment_summary.sort_values(
        "Customers", ascending=False
    ).reset_index(drop=True)

    print("\nCustomer segment summary:")
    print(segment_summary.to_string(index=False))

    save_csv(segment_summary, "customer_segment_summary.csv")

    largest_segment = segment_summary.iloc[0]["Customer_Segment"]

else:
    rfm = pd.DataFrame()
    segment_summary = pd.DataFrame()
    largest_segment = "Unavailable"
    reference_date = df["InvoiceDate"].max()


# ============================================================
# 15. OPTIONAL K-MEANS CUSTOMER CLUSTERING
# ============================================================

print_section("K-MEANS CUSTOMER CLUSTERING")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    if len(customer_sales) >= 3:
        clustering = customer_sales[
            ["CustomerID", "Gross_Sales", "Orders", "Last_Purchase"]
        ].copy()

        clustering["Recency"] = (
            reference_date - clustering["Last_Purchase"].dt.normalize()
        ).dt.days

        clustering["Frequency"] = clustering["Orders"]
        clustering["Monetary"] = clustering["Gross_Sales"]

        features = clustering[
            ["Recency", "Frequency", "Monetary"]
        ].clip(lower=0)

        # Log transformation reduces skew in spending and frequency.
        features_log = np.log1p(features)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_log)

        evaluation_rows = []
        max_k = min(6, len(clustering) - 1)

        for k in range(2, max_k + 1):
            model = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=10,
            )

            labels = model.fit_predict(features_scaled)

            # Use a deterministic sample for practical runtime on large data.
            if len(clustering) > 5000:
                silhouette = silhouette_score(
                    features_scaled,
                    labels,
                    sample_size=5000,
                    random_state=RANDOM_STATE,
                )
            else:
                silhouette = silhouette_score(features_scaled, labels)

            evaluation_rows.append({
                "K": k,
                "Inertia": float(model.inertia_),
                "Silhouette_Score": float(silhouette),
            })

        model_evaluation = pd.DataFrame(evaluation_rows)

        print("Model evaluation:")
        print(model_evaluation.to_string(index=False))
        save_csv(model_evaluation, "kmeans_model_evaluation.csv")

        # Select K with the highest silhouette score.
        # This is a useful diagnostic, not proof that the clusters
        # are the uniquely correct business segmentation.
        best_k = int(
            model_evaluation.loc[
                model_evaluation["Silhouette_Score"].idxmax(), "K"
            ]
        )

        final_model = KMeans(
            n_clusters=best_k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        clustering["KMeans_Cluster"] = final_model.fit_predict(
            features_scaled
        )

        cluster_summary = (
            clustering.groupby("KMeans_Cluster", as_index=False)
            .agg(
                Customers=("CustomerID", "nunique"),
                Average_Recency=("Recency", "mean"),
                Average_Frequency=("Frequency", "mean"),
                Average_Monetary=("Monetary", "mean"),
                Total_Monetary=("Monetary", "sum"),
            )
            .sort_values("Total_Monetary", ascending=False)
        )

        print("\nCluster summary:")
        print(cluster_summary.to_string(index=False))
        print(f"Selected K: {best_k}")

        save_csv(
            clustering,
            "customer_rfm_with_kmeans.csv"
        )

        save_csv(
            cluster_summary,
            "kmeans_customer_cluster_summary.csv"
        )

    else:
        print(
            "K-Means skipped: at least three identified customers "
            "are required."
        )

except ImportError:
    print(
        "K-Means skipped because scikit-learn is not installed. "
        "Install it using: python -m pip install scikit-learn"
    )

except Exception as error:
    print(
        "K-Means could not be completed. The rest of the analytics "
        f"pipeline will continue. Reason: {error}"
    )


# ============================================================
# 16. RETURNS AND CANCELLATIONS ANALYSIS
# ============================================================

print_section("RETURNS AND CANCELLATIONS ANALYSIS")

return_units = float(return_df["Quantity"].sum())
cancellation_units = float(cancellation_df["Quantity"].sum())

returns_summary = pd.DataFrame([
    ("Return transaction lines", return_line_count),
    ("Distinct return invoices", return_invoices),
    ("Return units (signed)", return_units),
    ("Return value (negative)", return_value),
    ("Cancellation transaction lines", cancellation_line_count),
    ("Distinct cancellation invoices", cancellation_invoices),
    ("Cancellation units (signed)", cancellation_units),
    ("Cancellation value (negative adjustment)", cancellation_value),
], columns=["Metric", "Value"])

print(returns_summary.to_string(index=False))
save_csv(returns_summary, "returns_cancellations_summary.csv")

# Monthly aggregation uses signed return units and negative return value.
return_monthly = (
    df.loc[df["Is_Return_Line"]]
    .groupby("Year_Month", as_index=False)
    .agg(
        Return_Lines=("InvoiceNo", "size"),
        Distinct_Return_Invoices=("InvoiceNo", "nunique"),
        Return_Units=("Quantity", "sum"),
        Return_Value=("Return_Revenue", "sum"),
    )
)

cancellation_monthly = (
    df.loc[df["Is_Cancellation"]]
    .groupby("Year_Month", as_index=False)
    .agg(
        Cancellation_Lines=("InvoiceNo", "size"),
        Distinct_Cancellation_Invoices=("InvoiceNo", "nunique"),
        Cancellation_Units=("Quantity", "sum"),
        Cancellation_Value=("Cancellation_Revenue", "sum"),
    )
)

monthly_returns = return_monthly.merge(
    cancellation_monthly,
    on="Year_Month",
    how="outer",
)

numeric_monthly_columns = [
    "Return_Lines",
    "Distinct_Return_Invoices",
    "Return_Units",
    "Return_Value",
    "Cancellation_Lines",
    "Distinct_Cancellation_Invoices",
    "Cancellation_Units",
    "Cancellation_Value",
]

for column in numeric_monthly_columns:
    if column not in monthly_returns.columns:
        monthly_returns[column] = 0
    monthly_returns[column] = monthly_returns[column].fillna(0)

monthly_returns = monthly_returns.sort_values(
    "Year_Month"
).reset_index(drop=True)

print("\nMonthly returns and cancellations:")
print(monthly_returns.to_string(index=False))
save_csv(monthly_returns, "monthly_returns_analysis.csv")


# ============================================================
# 17. SALES BY DAY AND HOUR
# ============================================================

print_section("SALES BY DAY AND HOUR")

sales_only = df.loc[df["Is_Sales_Line"]].copy()

day_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

sales_by_day = (
    sales_only.groupby("Day_Name", as_index=False)
    .agg(
        Gross_Sales=("Sales_Revenue", "sum"),
        Sales_Invoices=("InvoiceNo", "nunique"),
        Units_Sold=("Quantity", "sum"),
    )
)

sales_by_day["Day_Name"] = pd.Categorical(
    sales_by_day["Day_Name"],
    categories=day_order,
    ordered=True,
)

sales_by_day = sales_by_day.sort_values("Day_Name")

print("\nSales by day:")
print(sales_by_day.to_string(index=False))
save_csv(sales_by_day, "sales_by_day_of_week.csv")

sales_by_hour = (
    sales_only.groupby("Hour", as_index=False)
    .agg(
        Gross_Sales=("Sales_Revenue", "sum"),
        Sales_Invoices=("InvoiceNo", "nunique"),
    )
    .sort_values("Hour")
)

print("\nSales by hour:")
print(sales_by_hour.to_string(index=False))
save_csv(sales_by_hour, "sales_by_hour.csv")

if not sales_by_day.empty:
    best_day = sales_by_day.loc[
        sales_by_day["Gross_Sales"].idxmax()
    ]

    print(
        f"\nHighest gross-sales day: {best_day['Day_Name']} "
        f"({money(best_day['Gross_Sales'])})"
    )


# ============================================================
# 18. CHART GENERATION
# ============================================================

print_section("GENERATING CHARTS")

# 01: Missing data
missing_data = df_raw.isna().sum()
missing_data = missing_data[missing_data > 0].sort_values()

if not missing_data.empty:
    save_bar_chart(
        missing_data,
        "Missing Values by Source Column",
        "Missing value count",
        "Source column",
        "01_missing_data.png",
        horizontal=True,
    )
else:
    print("Skipped missing-data chart: no missing values.")

# 02: Monthly sales trend
if not monthly_sales.empty:
    plt.figure()
    plt.plot(
        monthly_sales["Year_Month"],
        monthly_sales["Gross_Sales"],
        marker="o",
    )
    plt.title("Monthly Gross Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Gross sales")
    plt.xticks(rotation=70)
    save_chart("02_monthly_sales_trend.png")

# 03: Monthly invoice volume
if not monthly_sales.empty:
    plt.figure()
    plt.bar(
        monthly_sales["Year_Month"],
        monthly_sales["Sales_Invoices"],
    )
    plt.title("Monthly Sales Invoice Volume")
    plt.xlabel("Month")
    plt.ylabel("Distinct sales invoices")
    plt.xticks(rotation=70)
    save_chart("03_monthly_invoice_volume.png")

# 04: Top countries
if not country_sales.empty:
    save_bar_chart(
        country_sales.set_index("Country")["Gross_Sales"],
        "Top Countries by Gross Sales",
        "Gross sales",
        "Country",
        "04_top_countries_by_sales.png",
        horizontal=True,
        top_n=15,
    )

# 05: Top ranking-eligible merchandise
if not eligible_products.empty:
    top_products = eligible_products.head(TOP_N).copy()

    save_bar_chart(
        top_products.set_index("Description")["Gross_Sales"],
        "Top Merchandise Entries by Gross Sales",
        "Gross sales",
        "Product",
        "05_top_products_by_sales.png",
        horizontal=True,
    )

# 06: Product Pareto
if not eligible_products.empty:
    pareto_data = eligible_products.head(50).copy()

    plt.figure()
    plt.bar(
        range(len(pareto_data)),
        pareto_data["Sales_Contribution_Pct"],
    )
    plt.plot(
        range(len(pareto_data)),
        pareto_data["Cumulative_Sales_Pct"],
        marker=".",
        color="black",
    )
    plt.axhline(80, linestyle="--", color="grey")
    plt.title("Product Sales Concentration — Top 50 Eligible Products")
    plt.xlabel("Products ordered by sales")
    plt.ylabel("Sales contribution / cumulative percentage")
    save_chart("06_product_pareto_analysis.png")

# 07: Top customers
if not customer_sales.empty:
    save_bar_chart(
        customer_sales.head(TOP_N).set_index("CustomerID")["Gross_Sales"],
        "Top Customers by Gross Sales",
        "Gross sales",
        "Customer ID",
        "07_top_customers.png",
        horizontal=True,
    )

# 08: Customer order frequency
if not customer_sales.empty:
    frequency_counts = (
        customer_sales["Orders"]
        .value_counts()
        .sort_index()
        .head(30)
    )

    save_bar_chart(
        frequency_counts,
        "Customer Order Frequency Distribution",
        "Number of distinct sales invoices",
        "Number of customers",
        "08_customer_order_frequency.png",
    )

# 09: Customer segment distribution
if not segment_summary.empty:
    save_bar_chart(
        segment_summary.set_index("Customer_Segment")["Customers"],
        "Customer Segment Distribution",
        "Number of customers",
        "Segment",
        "09_customer_segment_distribution.png",
        horizontal=True,
    )

# 10: Customer count versus revenue share
if not segment_summary.empty:
    segment_plot = segment_summary.set_index("Customer_Segment")

    plt.figure()
    segment_plot[
        ["Customer_Share_Pct", "Revenue_Share_Pct"]
    ].plot(kind="bar")

    plt.title("Customer Share vs Revenue Share by RFM Segment")
    plt.xlabel("Customer segment")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45, ha="right")
    save_chart("10_customer_vs_revenue_share.png")

# 11: Monthly return and cancellation values
if not monthly_returns.empty:
    plt.figure()
    plt.plot(
        monthly_returns["Year_Month"],
        monthly_returns["Return_Value"],
        marker="o",
        label="Returns",
    )
    plt.plot(
        monthly_returns["Year_Month"],
        monthly_returns["Cancellation_Value"],
        marker="o",
        label="Cancellations",
    )
    plt.title("Monthly Return and Cancellation Adjustments")
    plt.xlabel("Month")
    plt.ylabel("Signed transaction value")
    plt.xticks(rotation=70)
    plt.legend()
    save_chart("11_monthly_return_value.png")

# 12: Sales by weekday
if not sales_by_day.empty:
    plt.figure()
    plt.bar(
        sales_by_day["Day_Name"].astype(str),
        sales_by_day["Gross_Sales"],
    )
    plt.title("Gross Sales by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Gross sales")
    save_chart("12_sales_by_day.png")

# 13: Sales by hour
if not sales_by_hour.empty:
    plt.figure()
    plt.bar(
        sales_by_hour["Hour"],
        sales_by_hour["Gross_Sales"],
    )
    plt.title("Gross Sales by Hour of Day")
    plt.xlabel("Hour (24-hour clock)")
    plt.ylabel("Gross sales")
    plt.xticks(range(0, 24))
    save_chart("13_sales_by_hour.png")


# ============================================================
# 19. BUSINESS FINDINGS
# ============================================================

print_section("BUSINESS FINDINGS")

findings = []

findings.append(f"1. Source rows: {source_rows:,}.")
findings.append(f"2. Cleaned rows retained: {len(df):,}.")
findings.append(f"3. Exact duplicate rows removed: {exact_duplicates:,}.")
findings.append(f"4. Eligible sales invoices: {sales_invoices:,}.")
findings.append(f"5. Identified customers in sales records: {identified_customers:,}.")
findings.append(f"6. Gross sales: {money(gross_sales)}.")
findings.append(f"7. Recorded return adjustment: {money(return_value)}.")
findings.append(
    f"8. Recorded cancellation adjustment: {money(cancellation_value)}."
)
findings.append(
    f"9. Provisional net transaction value: {money(recorded_net_value)}."
)
findings.append(
    "10. Note: provisional net transaction value is not audited "
    "accounting revenue; cancellation and return conventions need validation."
)

if not monthly_sales.empty:
    peak = monthly_sales.loc[monthly_sales["Gross_Sales"].idxmax()]
    findings.append(
        f"11. Peak reported sales month: {peak['Year_Month']} "
        f"({money(peak['Gross_Sales'])})."
    )

if not country_sales.empty:
    country = country_sales.iloc[0]
    findings.append(
        f"12. Leading country by gross sales: {country['Country']} "
        f"({country['Sales_Contribution_Pct']:.2f}% of gross sales)."
    )

if not eligible_products.empty:
    product = eligible_products.iloc[0]
    findings.append(
        f"13. Top ranking-eligible merchandise entry: "
        f"{product['Description']} ({money(product['Gross_Sales'])})."
    )

if not customer_sales.empty and pd.notna(top_10_customer_share):
    findings.append(
        "14. Top 10 customers' share of identified-customer gross sales: "
        f"{top_10_customer_share:.2f}%."
    )

findings.append(
    f"15. Largest RFM segment by customer count: {largest_segment}."
)

findings.append("")
findings.append("POTENTIAL BUSINESS ACTIONS")
findings.append(
    "1. Review high-contributing merchandise when planning inventory "
    "and promotions."
)
findings.append(
    "2. Investigate high-value customers with long recency before "
    "designing retention campaigns."
)
findings.append(
    "3. Compare country-level sales with shipping costs, margins, and "
    "customer acquisition costs before expansion decisions."
)
findings.append(
    "4. Investigate return and cancellation records separately; do not "
    "assume cancellation invoices always match original sales invoices."
)
findings.append(
    "5. Exclude or clearly flag incomplete months when comparing "
    "monthly performance."
)
findings.append(
    "6. Review the product anomaly report before interpreting product "
    "rankings as normal demand."
)
findings.append(
    "7. Add margin and cost data before making profitability claims."
)

findings_text = "\n".join(findings)

print(findings_text)

with open(
    OUTPUT_DIR / "business_findings.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(findings_text)

print("Saved: business_findings.txt")


# ============================================================
# 20. FINAL VALIDATION
# ============================================================

print_section("FINAL VALIDATION")

# Confirm that all transaction value components reconcile exactly
# under the script's stated reporting conventions.
recalculated_net = (
    df["Sales_Revenue"].sum()
    + df["Return_Revenue"].sum()
    + df["Cancellation_Revenue"].sum()
)

net_check_pass = np.isclose(
    recalculated_net,
    df["Recorded_Net_Transaction_Value"].sum(),
    atol=0.01,
)

print(f"Rows in cleaned dataset: {len(df):,}")
print(f"Sales transaction lines: {int(df['Is_Sales_Line'].sum()):,}")
print(f"Return transaction lines: {int(df['Is_Return_Line'].sum()):,}")
print(f"Cancellation transaction lines: {int(df['Is_Cancellation'].sum()):,}")

print("\nFinancial totals")
print(f"Gross sales:                 {money(gross_sales)}")
print(f"Return value:                {money(return_value)}")
print(f"Cancellation adjustment:     {money(cancellation_value)}")
print(f"Provisional net transaction: {money(recorded_net_value)}")
print(f"Net calculation check:       {'PASS' if net_check_pass else 'FAIL'}")

print("\nData checks")
print(f"Invalid dates remaining: {int(df['InvoiceDate'].isna().sum()):,}")
print(
    "Negative unit-price rows flagged: "
    f"{int(df['Has_Negative_UnitPrice'].sum()):,}"
)
print(f"Rows missing customer ID: {int(df['CustomerID'].isna().sum()):,}")
print(f"Return lines with zero calculated return value: {int(((df['Is_Return_Line']) & (df['Return_Revenue'] == 0)).sum()):,}")
print(f"Product entries flagged for review: {len(product_exceptions):,}")
print(f"Partial months flagged: {', '.join(sorted(partial_months)) or 'None'}")

# A reconciliation pass does not prove that the source's transaction
# conventions are correct; it only validates arithmetic consistency.
if not net_check_pass:
    raise AssertionError(
        "Financial reconciliation failed. Review transaction value logic."
    )

# List files actually generated.
report_files = sorted(
    path.name for path in OUTPUT_DIR.iterdir()
    if path.is_file()
)

chart_files = sorted(
    path.name for path in CHART_DIR.iterdir()
    if path.is_file() and path.suffix.lower() == ".png"
)

print("\nExported report files:")
for filename in report_files:
    print(f"- {filename}")

print("\nGenerated charts:")
for filename in chart_files:
    print(f"- {filename}")

print(f"\nTotal report files: {len(report_files)}")
print(f"Total chart files: {len(chart_files)}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Charts directory: {CHART_DIR}")

print_section("PROJECT EXECUTION COMPLETED")
