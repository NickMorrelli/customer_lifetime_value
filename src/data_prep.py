"""
data_prep.py
------------
Loads and prepares the UCI Online Retail dataset for Customer Lifetime
Value (CLV) analysis.

What is Customer Lifetime Value?
---------------------------------
CLV is the total revenue a business can expect from a single customer
over the entire duration of their relationship. It answers:
  - Which customers are most valuable over the long term?
  - How much should we spend to acquire a new customer?
  - Which customers are worth investing in to retain?

Data Requirements for CLV
--------------------------
CLV models need customer-level summary statistics derived from transaction
history. Specifically we need the RFM+ summary:

  - recency   : days between first and last purchase (how long active)
  - frequency : number of REPEAT purchases (transactions - 1)
                Note: in CLV models, 'frequency' means repeat purchases,
                NOT total purchases as in standard RFM scoring
  - T         : age of the customer in days (days since first purchase
                to the end of the observation period)
  - monetary  : average order value (NOT total spend — the Gamma-Gamma
                model works with average transaction value)

Observation Period vs Prediction Period
----------------------------------------
We split the data into two windows:
  - Observation period : first 75% of weeks — used to FIT the model
  - Prediction period  : last 25% of weeks — used to VALIDATE the model

This simulates a real-world scenario where you train on historical data
and predict future behavior.
"""

import pandas as pd
import numpy as np
import os


# ── Constants ─────────────────────────────────────────────────────────────────

# Use 75% of the data to train, 25% to validate
TRAIN_FRACTION = 0.75


# ── Load & Clean ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load and clean the UCI Online Retail dataset."""
    print(f"Loading data from: {filepath}")
    df = pd.read_excel(filepath, dtype={"CustomerID": str})
    print(f"  Raw shape: {df.shape}")

    # Standard cleaning
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"]  = pd.to_datetime(df["InvoiceDate"])
    df["OrderRevenue"] = df["Quantity"] * df["UnitPrice"]

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Unique customers: {df['CustomerID'].nunique():,}")
    print(f"  Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    return df


# ── Train / Validation Split ──────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Split transactions into observation (train) and prediction (validation) periods.

    Parameters
    ----------
    df : pd.DataFrame  Cleaned transaction data.

    Returns
    -------
    df_train     : transactions in the observation period
    df_valid     : transactions in the prediction period
    split_date   : the date dividing train and validation
    end_date     : the last date in the dataset
    """
    min_date = df["InvoiceDate"].min()
    max_date = df["InvoiceDate"].max()
    total_days = (max_date - min_date).days

    split_date = min_date + pd.Timedelta(days=int(total_days * TRAIN_FRACTION))
    end_date   = max_date

    df_train = df[df["InvoiceDate"] <= split_date]
    df_valid  = df[df["InvoiceDate"] >  split_date]

    print(f"\n  Train period : {min_date.date()} → {split_date.date()} ({len(df_train):,} transactions)")
    print(f"  Valid period : {split_date.date()} → {end_date.date()} ({len(df_valid):,} transactions)")

    return df_train, df_valid, split_date, end_date


# ── Customer Summary Table ────────────────────────────────────────────────────

def build_customer_summary(df_train: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build the customer-level RFM+ summary table needed for CLV models.

    This produces the standard input format expected by the lifetimes library
    (used for BG/NBD modeling):

      CustomerID : unique customer identifier
      frequency  : number of REPEAT transactions (total orders - 1)
                   Customers with 0 repeat purchases are included but will
                   have frequency = 0
      recency    : days between first and last purchase within train period
      T          : days from first purchase to end of observation period
      monetary   : mean order revenue (for Gamma-Gamma model)

    Parameters
    ----------
    df_train   : pd.DataFrame  Training period transactions.
    split_date : pd.Timestamp  End of observation period.

    Returns
    -------
    pd.DataFrame  One row per customer.
    """
    # Order-level aggregation first (one row per invoice)
    orders = (
        df_train.groupby(["CustomerID", "InvoiceNo", "InvoiceDate"])
        ["OrderRevenue"].sum()
        .reset_index()
    )
    orders.rename(columns={"OrderRevenue": "OrderValue"}, inplace=True)

    # Customer-level summary
    summary = (
        orders.groupby("CustomerID")
        .agg(
            first_purchase = ("InvoiceDate", "min"),
            last_purchase  = ("InvoiceDate", "max"),
            n_orders       = ("InvoiceNo",   "nunique"),
            mean_order_value = ("OrderValue", "mean"),
        )
        .reset_index()
    )

    # CLV model inputs
    summary["frequency"] = summary["n_orders"] - 1     # repeat purchases
    summary["recency"]   = (summary["last_purchase"]  - summary["first_purchase"]).dt.days
    summary["T"]         = (split_date - summary["first_purchase"]).dt.days
    summary["monetary"]  = summary["mean_order_value"]

    # Remove customers with T = 0 (purchased only on the last day)
    summary = summary[summary["T"] > 0]

    # Remove customers with negative or zero monetary value
    summary = summary[summary["monetary"] > 0]

    print(f"\n  Customer summary table:")
    print(f"    Total customers    : {len(summary):,}")
    print(f"    One-time buyers    : {(summary['frequency'] == 0).sum():,} ({(summary['frequency'] == 0).mean():.1%})")
    print(f"    Repeat buyers      : {(summary['frequency'] > 0).sum():,} ({(summary['frequency'] > 0).mean():.1%})")
    print(f"    Avg frequency      : {summary['frequency'].mean():.2f} repeat purchases")
    print(f"    Avg monetary       : ${summary['monetary'].mean():.2f} per order")
    print(f"    Avg T (age)        : {summary['T'].mean():.0f} days")

    return summary


# ── Validation Summary ────────────────────────────────────────────────────────

def build_validation_summary(df_valid: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build actual purchase counts in the validation period for model evaluation.

    Parameters
    ----------
    df_valid : pd.DataFrame  Validation period transactions.
    summary  : pd.DataFrame  Training period customer summary.

    Returns
    -------
    pd.DataFrame  Customer summary with actual_future_purchases column added.
    """
    actual = (
        df_valid[df_valid["CustomerID"].isin(summary["CustomerID"])]
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "actual_future_purchases"})
    )

    summary_valid = summary.merge(actual, on="CustomerID", how="left")
    summary_valid["actual_future_purchases"] = summary_valid["actual_future_purchases"].fillna(0)

    return summary_valid


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_prep_pipeline(filepath: str) -> tuple:
    """
    Full data preparation pipeline.

    Returns
    -------
    df           : full cleaned transaction data
    df_train     : training period transactions
    df_valid     : validation period transactions
    summary      : customer summary (train period)
    summary_valid: customer summary with actual future purchases
    split_date   : train/validation split date
    end_date     : last date in dataset
    """
    print("=" * 50)
    print("  DATA PREPARATION")
    print("=" * 50)

    df                       = load_data(filepath)
    df_train, df_valid, split_date, end_date = split_data(df)
    summary                  = build_customer_summary(df_train, split_date)
    summary_valid            = build_validation_summary(df_valid, summary)

    # Save summary to CSV
    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "customer_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary_valid.to_csv(out_path, index=False)
    print(f"\n  Customer summary saved to: {out_path}")

    return df, df_train, df_valid, summary, summary_valid, split_date, end_date


if __name__ == "__main__":
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "OnlineRetail.xlsx")
    df, df_train, df_valid, summary, summary_valid, split_date, end_date = run_prep_pipeline(path)
    print(summary.head())
