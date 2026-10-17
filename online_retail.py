
"""E-commerce Cohort Retention & Customer Analytics

Run:
    python ecommerce_cohort_retention_analysis.py --input online_retail.csv

Optional:
    python ecommerce_cohort_retention_analysis.py --input online_retail.csv --output-dir outputs

Accepted common column variants:
InvoiceNo / Invoice / Invoice Number, StockCode, Description, Quantity,
InvoiceDate / Invoice Date, UnitPrice / Price, CustomerID / Customer ID, Country.

Outputs are written to ./outputs by default.
Sales, returns, and cancellation invoices are classified separately.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
LOGGER = logging.getLogger("ecommerce_cohort_analysis")


COLUMN_ALIASES = {
    "Invoice": [
        "Invoice", "InvoiceNo", "Invoice No",
        "Invoice Number", "Invoice_ID"
    ],
    "StockCode": [
        "StockCode", "Stock Code",
        "ProductCode", "Product Code"
    ],
    "Description": [
        "Description", "Product Description",
        "Item Description"
    ],
    "Quantity": ["Quantity", "Qty"],
    "InvoiceDate": [
        "InvoiceDate", "Invoice Date",
        "OrderDate", "Order Date", "Transaction Date"
    ],
    "Price": [
        "Price", "UnitPrice", "Unit Price", "Unit_Price"
    ],
    "CustomerID": [
        "CustomerID", "Customer ID", "Customer_Id",
        "Customer Id", "Customer"
    ],
    "Country": ["Country", "Customer Country", "Market"],
}


def normalized_name(value: object) -> str:
    """Normalize column names for resilient schema matching."""
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).strip().lower()
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common source column variants to a consistent schema."""
    lookup = {
        normalized_name(column): column
        for column in df.columns
    }

    rename = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            source = lookup.get(normalized_name(alias))

            if source is not None:
                rename[source] = canonical
                break

    df = df.rename(columns=rename).copy()

    required = [
        "Invoice",
        "Quantity",
        "InvoiceDate",
        "Price",
        "CustomerID",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            "Check the CSV schema or add an alias to COLUMN_ALIASES."
        )

    optional_defaults = {
        "StockCode": "UNKNOWN",
        "Description": "Unknown",
        "Country": "Unknown",
    }

    for column, default_value in optional_defaults.items():
        if column not in df.columns:
            df[column] = default_value

    return df


def load_data(path: Path) -> pd.DataFrame:
    """Load the source CSV and standardize its column names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Pass the dataset path using --input, for example:\n"
            '--input "C:/Users/yourname/Documents/online_retail.csv"'
        )

    LOGGER.info("Loading dataset: %s", path)

    try:
        raw = pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False
        )
    except UnicodeDecodeError:
        raw = pd.read_csv(
            path,
            encoding="latin-1",
            low_memory=False
        )

    LOGGER.info(
        "Raw data: %s rows × %s columns",
        *raw.shape
    )

    return standardize_columns(raw)


def clean_data(
    raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean transaction data and separate sales from reversals."""
    df = raw.copy()

    df["Invoice"] = (
        df["Invoice"].astype("string").str.strip()
    )
    df["CustomerID"] = (
        df["CustomerID"].astype("string").str.strip()
    )

    # Support both ISO dates and day-first retail dataset dates.
    date_text = (
        df["InvoiceDate"].astype("string").str.strip()
    )

    iso_mask = date_text.str.match(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        na=False
    )

    parsed_dates = pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns]"
    )

    parsed_dates.loc[iso_mask] = pd.to_datetime(
        date_text.loc[iso_mask],
        errors="coerce",
        yearfirst=True
    )

    parsed_dates.loc[~iso_mask] = pd.to_datetime(
        date_text.loc[~iso_mask],
        errors="coerce",
        dayfirst=True
    )

    df["InvoiceDate"] = parsed_dates

    for column in ["Quantity", "Price"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    original_row_count = len(df)

    df = df.dropna(
        subset=[
            "Invoice",
            "InvoiceDate",
            "Quantity",
            "Price",
        ]
    ).copy()

    df = df[df["Invoice"].ne("")].copy()

    # Revenue is a line-level transaction amount.
    df["Revenue"] = df["Quantity"] * df["Price"]

    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day_of_Week"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    # C-prefixed invoices are commonly used for cancellations
    # in the UCI Online Retail dataset. Validate this convention
    # against the documentation for your particular dataset.
    df["Is_Cancellation_Invoice"] = (
        df["Invoice"].str.upper().str.startswith("C", na=False)
    )

    df["Transaction_Type"] = np.select(
        [
            df["Is_Cancellation_Invoice"],
            df["Quantity"] < 0,
        ],
        [
            "Cancellation",
            "Return",
        ],
        default="Sale"
    )

    # Gross sales exclude negative quantities and C-prefixed invoices.
    df_sales = df[
        (df["Quantity"] > 0)
        & (~df["Is_Cancellation_Invoice"])
        & (df["Price"] >= 0)
    ].copy()

    # Preserve reversals for separate diagnostics.
    df_reversals = df[
        (df["Quantity"] < 0)
        | df["Is_Cancellation_Invoice"]
    ].copy()

    # Retention analysis requires a known customer ID.
    df_customer_sales = df_sales.dropna(
        subset=["CustomerID"]
    ).copy()

    df_customer_sales = df_customer_sales[
        df_customer_sales["CustomerID"].ne("")
    ].copy()

    LOGGER.info(
        "Rows excluded for missing transaction fields: %s",
        original_row_count - len(df)
    )
    LOGGER.info(
        "Clean rows: %s | Sales lines: %s | Reversal/cancellation rows: %s",
        len(df),
        len(df_sales),
        len(df_reversals)
    )
    LOGGER.info(
        "Customer-linked sales rows used for retention: %s",
        len(df_customer_sales)
    )

    return df, df_sales, df_customer_sales


def safe_divide(
    numerator: float,
    denominator: float
) -> float:
    """Avoid division-by-zero errors in KPI calculations."""
    return float(numerator / denominator) if denominator else np.nan


def build_kpis(
    df_sales: pd.DataFrame,
    df_all: pd.DataFrame
) -> pd.DataFrame:
    """Calculate headline business performance metrics."""
    revenue = df_sales["Revenue"].sum()
    orders = df_sales["Invoice"].nunique()
    customers = df_sales["CustomerID"].nunique()
    products = df_sales["StockCode"].nunique()
    units = df_sales["Quantity"].sum()

    kpis = {
        "Gross Sales Revenue": revenue,
        "Units Sold": units,
        "Sales Orders": orders,
        "Identified Customers": customers,
        "Distinct Products": products,
        "Average Order Value": safe_divide(revenue, orders),
        "Average Revenue per Identified Customer": safe_divide(
            revenue, customers
        ),
        "Average Units per Order": safe_divide(units, orders),
        "Source Rows": len(df_all),
        "Sales Lines": len(df_sales),
    }

    return pd.DataFrame({
        "KPI": list(kpis.keys()),
        "Value": list(kpis.values())
    })


def build_monthly_metrics(
    df_sales: pd.DataFrame
) -> pd.DataFrame:
    """Summarize sales and activity by calendar month."""
    monthly = df_sales.groupby("YearMonth").agg(
        Revenue=("Revenue", "sum"),
        Orders=("Invoice", "nunique"),
        Units_Sold=("Quantity", "sum"),
        Active_Customers=("CustomerID", "nunique"),
    )

    monthly["Average_Order_Value"] = (
        monthly["Revenue"]
        / monthly["Orders"].replace(0, np.nan)
    )

    monthly.index = monthly.index.astype(str)

    return monthly.reset_index(names="YearMonth")


def build_cohort_analysis(
    df_customer_sales: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign customers to their first-purchase cohort.

    CohortIndex 1 represents the acquisition month.
    Retention is active customers in a cohort month divided
    by the original customer count in that cohort.
    """
    if df_customer_sales.empty:
        raise ValueError(
            "No customer-linked positive sales found. "
            "Cohort analysis cannot be computed."
        )

    tx = df_customer_sales[
        ["CustomerID", "Invoice", "InvoiceDate", "Revenue"]
    ].copy()

    tx["OrderMonth"] = tx["InvoiceDate"].dt.to_period("M")

    first_purchase = (
        tx.groupby("CustomerID")["OrderMonth"]
        .min()
        .rename("CohortMonth")
    )

    tx = tx.join(first_purchase, on="CustomerID")

    tx["CohortIndex"] = (
        (tx["OrderMonth"].dt.year - tx["CohortMonth"].dt.year) * 12
        + (
            tx["OrderMonth"].dt.month
            - tx["CohortMonth"].dt.month
        )
        + 1
    ).astype(int)

    cohort_sizes = (
        tx.groupby("CohortMonth")["CustomerID"]
        .nunique()
        .rename("Cohort_Size")
    )

    cohort_counts = (
        tx.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
        .nunique()
        .unstack(fill_value=0)
    )

    max_index = int(tx["CohortIndex"].max())

    cohort_counts = cohort_counts.reindex(
        columns=range(1, max_index + 1),
        fill_value=0
    )

    cohort_counts = cohort_counts.join(cohort_sizes).sort_index()

    retention = cohort_counts.drop(
        columns="Cohort_Size"
    ).div(
        cohort_counts["Cohort_Size"],
        axis=0
    )

    retention.index = retention.index.astype(str)
    cohort_counts.index = cohort_counts.index.astype(str)

    assignments = first_purchase.reset_index()
    assignments["CohortMonth"] = (
        assignments["CohortMonth"].astype(str)
    )

    return assignments, cohort_counts, retention


def build_customer_metrics(
    df_customer_sales: pd.DataFrame
) -> pd.DataFrame:
    """Build customer-level RFM-style metrics and rule-based segments."""
    if df_customer_sales.empty:
        raise ValueError(
            "No customer-linked sales available for customer metrics."
        )

    latest_date = (
        df_customer_sales["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    customer = df_customer_sales.groupby("CustomerID").agg(
        First_Purchase=("InvoiceDate", "min"),
        Last_Purchase=("InvoiceDate", "max"),
        Orders=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Units_Purchased=("Quantity", "sum"),
    )

    customer["Recency_Days"] = (
        latest_date - customer["Last_Purchase"]
    ).dt.days

    customer["Customer_Tenure_Days"] = (
        customer["Last_Purchase"]
        - customer["First_Purchase"]
    ).dt.days

    customer["Average_Order_Value"] = (
        customer["Monetary"]
        / customer["Orders"].replace(0, np.nan)
    )

    customer["Repeat_Customer"] = customer["Orders"] >= 2

    customer["Cohort_Month"] = (
        customer["First_Purchase"].dt.to_period("M").astype(str)
    )

    monetary_75 = customer["Monetary"].quantile(0.75)
    recency_75 = customer["Recency_Days"].quantile(0.75)

    # Transparent rule-based segmentation, not machine learning.
    customer["Customer_Segment"] = np.select(
        [
            (customer["Orders"] >= 2)
            & (customer["Monetary"] >= monetary_75),

            customer["Orders"] >= 2,

            customer["Recency_Days"] > recency_75,
        ],
        [
            "Loyal High-Value",
            "Repeat Customer",
            "At-Risk / Inactive",
        ],
        default="New / Occasional Customer"
    )

    return customer.reset_index()


def build_product_metrics(
    df_sales: pd.DataFrame
) -> pd.DataFrame:
    """Rank products by gross sales revenue."""
    products = df_sales.groupby(
        ["StockCode", "Description"],
        dropna=False
    ).agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Quantity", "sum"),
        Orders=("Invoice", "nunique"),
        Average_Unit_Price=("Price", "mean"),
    ).reset_index()

    return products.sort_values(
        "Revenue",
        ascending=False
    )


def build_country_metrics(
    df_sales: pd.DataFrame
) -> pd.DataFrame:
    """Summarize revenue, orders, and customers by country."""
    countries = df_sales.groupby(
        "Country",
        dropna=False
    ).agg(
        Revenue=("Revenue", "sum"),
        Orders=("Invoice", "nunique"),
        Customers=("CustomerID", "nunique"),
        Units_Sold=("Quantity", "sum"),
    ).reset_index()

    countries["Average_Order_Value"] = (
        countries["Revenue"]
        / countries["Orders"].replace(0, np.nan)
    )

    total_revenue = countries["Revenue"].sum()

    countries["Revenue_Share_Pct"] = (
        countries["Revenue"] / total_revenue * 100
        if total_revenue else np.nan
    )

    return countries.sort_values(
        "Revenue",
        ascending=False
    )


def build_return_metrics(
    df_all: pd.DataFrame,
    df_sales: pd.DataFrame
) -> pd.DataFrame:
    """Report return and cancellation indicators separately from sales."""
    cancellation_invoices = df_all.loc[
        df_all["Is_Cancellation_Invoice"], "Invoice"
    ].nunique()

    negative_quantity_invoices = df_all.loc[
        df_all["Quantity"] < 0, "Invoice"
    ].nunique()

    negative_units = abs(
        df_all.loc[df_all["Quantity"] < 0, "Quantity"].sum()
    )

    positive_units = df_sales["Quantity"].sum()

    return pd.DataFrame([
        {
            "Metric": "Cancellation invoice count (C-prefix heuristic)",
            "Value": cancellation_invoices,
        },
        {
            "Metric": "Invoices containing negative quantity lines",
            "Value": negative_quantity_invoices,
        },
        {
            "Metric": "Negative-quantity units (absolute value)",
            "Value": negative_units,
        },
        {
            "Metric": "Positive sales units",
            "Value": positive_units,
        },
        {
            "Metric": "Negative-quantity units / positive sales units (%)",
            "Value": safe_divide(
                negative_units, positive_units
            ) * 100,
        },
    ])


def save_plots(
    monthly: pd.DataFrame,
    retention: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    outdir: Path
) -> None:
    """Generate charts suitable for project analysis and reporting."""
    sns.set_theme(style="whitegrid")

    # Monthly revenue trend
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        monthly["YearMonth"],
        monthly["Revenue"],
        marker="o",
        linewidth=2
    )
    ax.set(
        title="Monthly Gross Sales Revenue",
        xlabel="Month",
        ylabel="Revenue"
    )
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(outdir / "monthly_revenue.png", dpi=160)
    plt.close(fig)

    # Cohort retention heatmap
    fig, ax = plt.subplots(
        figsize=(
            max(10, retention.shape[1] * 0.65),
            max(6, retention.shape[0] * 0.45)
        )
    )

    sns.heatmap(
        retention * 100,
        mask=retention.isna(),
        annot=True,
        fmt=".0f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        linewidths=0.3,
        cbar_kws={"label": "Customer retention (%)"},
        ax=ax
    )

    ax.set(
        title="Monthly Customer Cohort Retention",
        xlabel="Months Since First Purchase",
        ylabel="Acquisition Cohort"
    )

    fig.tight_layout()
    fig.savefig(
        outdir / "cohort_retention_heatmap.png",
        dpi=180
    )
    plt.close(fig)

    # Top 10 products
    top_products = products.head(10).sort_values("Revenue")

    labels = (
        top_products["StockCode"].astype(str)
        + " — "
        + top_products["Description"].fillna("").astype(str).str.slice(0, 35)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, top_products["Revenue"])
    ax.set(
        title="Top 10 Products by Gross Sales Revenue",
        xlabel="Revenue",
        ylabel="Product"
    )
    fig.tight_layout()
    fig.savefig(outdir / "top_products_revenue.png", dpi=160)
    plt.close(fig)

    # Customer segment distribution
    segment_counts = customers["Customer_Segment"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(segment_counts.index, segment_counts.values)
    ax.set(
        title="Customer Distribution by Rule-Based Segment",
        xlabel="Segment",
        ylabel="Customers"
    )
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(outdir / "customer_segments.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "E-commerce sales, cohort retention, "
            "and customer analytics"
        )
    )

    parser.add_argument(
        "--input",
        default="online_retail.csv",
        help="Path to source CSV (default: online_retail.csv)"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for reports and charts (default: outputs)"
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    outdir = Path(args.output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        raw = load_data(input_path)

        df_all, df_sales, df_customer_sales = clean_data(raw)

        if df_sales.empty:
            raise ValueError(
                "No positive-quantity sales rows remain after cleaning. "
                "Check the dataset and column mappings."
            )

        if df_customer_sales.empty:
            raise ValueError(
                "Sales exist, but no customer-linked sales remain. "
                "Customer cohort and segmentation analysis require "
                "valid customer IDs."
            )

        kpis = build_kpis(df_sales, df_all)
        monthly = build_monthly_metrics(df_sales)

        cohort_assignments, cohort_counts, retention = (
            build_cohort_analysis(df_customer_sales)
        )

        customer_metrics = build_customer_metrics(df_customer_sales)
        product_metrics = build_product_metrics(df_sales)
        country_metrics = build_country_metrics(df_sales)
        reversal_metrics = build_return_metrics(df_all, df_sales)

        reports = {
            "business_kpis.csv": kpis,
            "monthly_sales_metrics.csv": monthly,
            "customer_cohort_assignments.csv": cohort_assignments,
            "cohort_customer_counts.csv": (
                cohort_counts.reset_index(names="CohortMonth")
            ),
            "cohort_retention_matrix.csv": (
                retention.reset_index(names="CohortMonth")
            ),
            "customer_metrics_and_segments.csv": customer_metrics,
            "product_performance.csv": product_metrics,
            "country_performance.csv": country_metrics,
            "returns_cancellations_diagnostics.csv": reversal_metrics,
        }

        for filename, frame in reports.items():
            output_path = outdir / filename
            frame.to_csv(output_path, index=False)
            LOGGER.info("Saved report: %s", output_path)

        save_plots(
            monthly,
            retention,
            customer_metrics,
            product_metrics,
            outdir
        )

        print("\n========== E-COMMERCE ANALYTICS SUMMARY ==========")
        print(kpis.to_string(index=False))

        print("\nTop 10 products by gross sales revenue:")
        print(
            product_metrics[
                [
                    "StockCode",
                    "Description",
                    "Revenue",
                    "Units_Sold",
                    "Orders",
                ]
            ].head(10).to_string(index=False)
        )

        print("\nCustomer segment summary:")
        segment_summary = customer_metrics.groupby(
            "Customer_Segment"
        ).agg(
            Customers=("CustomerID", "nunique"),
            Revenue=("Monetary", "sum"),
            Average_Recency_Days=("Recency_Days", "mean"),
        )

        print(segment_summary.round(2).to_string())

        print(f"\nReports and charts saved to: {outdir.resolve()}")

        print(
            "\nNote: C-prefixed invoice cancellation detection is a common "
            "Online Retail convention. Validate it against your dataset "
            "documentation."
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
