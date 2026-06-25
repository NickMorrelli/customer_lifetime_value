"""
clv_simple.py
-------------
Simple RFM-based Customer Lifetime Value model.

What is this approach?
-----------------------
Before sophisticated probabilistic models existed, analysts estimated CLV
using a straightforward formula based on historical averages:

    CLV = Average Order Value
        × Purchase Frequency (orders per year)
        × Customer Lifespan (years)
        × Gross Margin

This is intuitive, easy to explain to stakeholders, and still used widely
in companies that don't have data science teams.

Why show this first?
--------------------
Starting with the simple model and then moving to BG/NBD shows:
  1. You understand the business intuition behind CLV
  2. You know when to use a simple vs complex approach
  3. You can explain trade-offs — simple is more transparent, BG/NBD
     is more accurate for heterogeneous customer bases

Limitations of this approach
-----------------------------
  - Assumes all customers have the same lifespan (not true)
  - Doesn't account for customers who have already churned
  - Purchase frequency is backward-looking, not predictive
  - Doesn't model the probability a customer is still active

These limitations motivate the BG/NBD model in clv_bgnbd.py.
"""

import pandas as pd
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

GROSS_MARGIN     = 0.40    # Assumed 40% gross margin (typical e-commerce)
DISCOUNT_RATE    = 0.10    # Annual discount rate for NPV calculation (10%)
PREDICTION_YEARS = 3       # CLV horizon (3 years)
DAYS_PER_YEAR    = 365.25


# ── Simple CLV Formula ────────────────────────────────────────────────────────

def calculate_simple_clv(summary: pd.DataFrame,
                          prediction_years: int = PREDICTION_YEARS) -> pd.DataFrame:
    """
    Calculate CLV using the simple historical average formula.

    Formula
    -------
    CLV = AOV × purchase_frequency_per_year × lifespan_years × gross_margin

    Where:
      AOV                     = average order value (monetary column)
      purchase_frequency/year = (n_orders / customer_age_years)
      lifespan_years          = estimated remaining lifespan
                                (we use prediction_years as a fixed horizon
                                since we don't model churn here)

    Net Present Value Adjustment
    ----------------------------
    Money received in the future is worth less than money received today
    (due to inflation and opportunity cost). We apply a discount rate to
    convert future cash flows to today's dollars:

        NPV_CLV = CLV × [1 - (1 + discount_rate)^(-years)] / discount_rate

    Parameters
    ----------
    summary          : pd.DataFrame  Customer summary from data_prep.
    prediction_years : int           CLV time horizon.

    Returns
    -------
    pd.DataFrame with added columns:
        age_years, orders_per_year, simple_clv, simple_clv_npv
    """
    df = summary.copy()

    # Customer age in years (based on observation period)
    df["age_years"] = df["T"] / DAYS_PER_YEAR

    # Purchase frequency per year
    # Use n_orders (not frequency) since this is total historical rate
    df["orders_per_year"] = df["n_orders"] / df["age_years"].clip(lower=0.1)

    # Raw CLV over prediction horizon
    df["simple_clv"] = (
        df["monetary"]
        * df["orders_per_year"]
        * prediction_years
        * GROSS_MARGIN
    )

    # Net Present Value adjustment
    # Annuity formula: converts future cash flows to present value
    if DISCOUNT_RATE > 0:
        npv_factor = (1 - (1 + DISCOUNT_RATE) ** (-prediction_years)) / DISCOUNT_RATE
    else:
        npv_factor = prediction_years

    df["simple_clv_npv"] = df["simple_clv"] * npv_factor / prediction_years

    # CLV segments based on percentiles
    df["clv_segment"] = pd.qcut(
        df["simple_clv_npv"],
        q=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low Value", "Mid Value", "High Value", "Top Value"]
    )

    return df


# ── Summary Statistics ────────────────────────────────────────────────────────

def summarize_simple_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return a segment-level summary of simple CLV results.

    Parameters
    ----------
    df : pd.DataFrame  Output of calculate_simple_clv().

    Returns
    -------
    pd.DataFrame  Segment-level summary.
    """
    segment_summary = (
        df.groupby("clv_segment", observed=True)
        .agg(
            customers     = ("CustomerID",    "count"),
            avg_clv       = ("simple_clv_npv","mean"),
            total_clv     = ("simple_clv_npv","sum"),
            avg_aov       = ("monetary",       "mean"),
            avg_frequency = ("orders_per_year","mean"),
        )
        .reset_index()
    )

    print("\n  Simple CLV Results:")
    print(f"  {'Segment':<15} {'Customers':>10} {'Avg CLV':>12} {'Total CLV':>14} {'Avg AOV':>10}")
    print("  " + "-" * 65)
    for _, row in segment_summary.iterrows():
        print(f"  {row['clv_segment']:<15} {int(row['customers']):>10,} "
              f"${row['avg_clv']:>11,.2f} ${row['total_clv']:>13,.2f} "
              f"${row['avg_aov']:>9,.2f}")

    total_clv = df["simple_clv_npv"].sum()
    print(f"\n  Total portfolio CLV ({PREDICTION_YEARS}-year NPV): ${total_clv:,.2f}")

    return segment_summary


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_simple_clv_pipeline(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full simple CLV pipeline.

    Returns
    -------
    df_clv          : customer-level CLV data
    segment_summary : segment-level summary
    """
    print("\n" + "=" * 50)
    print("  SIMPLE CLV MODEL (RFM-Based)")
    print("=" * 50)
    print(f"  Gross margin     : {GROSS_MARGIN:.0%}")
    print(f"  Discount rate    : {DISCOUNT_RATE:.0%}")
    print(f"  Prediction years : {PREDICTION_YEARS}")

    df_clv          = calculate_simple_clv(summary)
    segment_summary = summarize_simple_clv(df_clv)

    return df_clv, segment_summary


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.data_prep import run_prep_pipeline
    path = os.path.join(os.path.dirname(__file__), "..", "data", "OnlineRetail.xlsx")
    _, _, _, summary, summary_valid, split_date, _ = run_prep_pipeline(path)
    df_clv, seg = run_simple_clv_pipeline(summary)
    print(df_clv[["CustomerID", "monetary", "orders_per_year", "simple_clv_npv", "clv_segment"]].head(10))
