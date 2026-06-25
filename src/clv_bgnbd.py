"""
clv_bgnbd.py
------------
Industry-standard CLV model using BG/NBD + Gamma-Gamma models.

What is BG/NBD?
---------------
BG/NBD stands for Beta-Geometric / Negative Binomial Distribution.
It was introduced by Fader, Hardie & Lee (2005) and is the industry
standard for non-contractual CLV modeling (e-commerce, retail).

It models TWO processes simultaneously for each customer:
  1. TRANSACTION process (NBD): while a customer is active, they make
     purchases at a rate λ (individual purchase rate), where λ varies
     across customers following a Gamma distribution.

  2. DROPOUT process (BG): after each transaction, a customer has a
     probability p of becoming permanently inactive ("dying"). This
     probability p varies across customers following a Beta distribution.

Why is this better than the simple model?
-----------------------------------------
The simple model assumes all customers will keep buying forever at their
historical rate. BG/NBD explicitly models the probability that each
customer is still alive (hasn't churned), which leads to much more
accurate predictions — especially for customers who haven't bought recently.

What is Gamma-Gamma?
--------------------
The Gamma-Gamma model (Fader & Hardie, 2013) estimates the monetary value
of future transactions. It models:
  - Each customer has an average transaction value drawn from a Gamma distribution
  - The population of customers has transaction values that follow a Gamma-Gamma
    mixture distribution

Combined pipeline
-----------------
  Step 1: Fit BG/NBD model on frequency/recency/T data
           → predicts expected number of future purchases per customer
  Step 2: Fit Gamma-Gamma model on frequency/monetary data
           → predicts expected average transaction value per customer
  Step 3: CLV = predicted purchases × predicted avg value × gross margin
           → discounted to Net Present Value

Library
-------
We use the `lifetimes` library which implements both models:
    pip install lifetimes
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False
    print("  ⚠️  lifetimes library not found. Run: pip install lifetimes")


# ── Constants ─────────────────────────────────────────────────────────────────

GROSS_MARGIN     = 0.40    # 40% gross margin
DISCOUNT_RATE    = 0.10    # Annual discount rate
PREDICTION_DAYS  = 90      # Predict purchases over next 90 days
PREDICTION_YEARS = 3       # CLV horizon for full NPV calculation


# ── BG/NBD Model ──────────────────────────────────────────────────────────────

def fit_bgnbd(summary: pd.DataFrame) -> "BetaGeoFitter":
    """
    Fit the BG/NBD model to customer transaction summary data.

    The model is fit using Maximum Likelihood Estimation (MLE) — it finds
    the parameters (r, α, a, b) that make the observed frequency/recency/T
    data most probable under the BG/NBD assumptions.

    Parameters
    ----------
    summary : pd.DataFrame
        Must contain columns: frequency, recency, T
        (from data_prep.build_customer_summary)

    Returns
    -------
    BetaGeoFitter  Fitted BG/NBD model.
    """
    if not LIFETIMES_AVAILABLE:
        raise ImportError("lifetimes library required. Run: pip install lifetimes")

    print("\n  Fitting BG/NBD model...")

    # Only customers with frequency >= 1 are used for fitting
    # (customers with 0 repeat purchases don't inform the transaction rate)
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(
        summary["frequency"],
        summary["recency"],
        summary["T"],
        verbose=False,
    )

    print(f"  BG/NBD fitted successfully")
    print(f"  Parameters: r={bgf.params_['r']:.4f}, α={bgf.params_['alpha']:.4f}, "
          f"a={bgf.params_['a']:.4f}, b={bgf.params_['b']:.4f}")

    return bgf


def predict_purchases(bgf: "BetaGeoFitter", summary: pd.DataFrame,
                      days: int = PREDICTION_DAYS) -> pd.DataFrame:
    """
    Predict expected number of purchases in the next `days` days per customer.

    This is the key output of the BG/NBD model: E[X(t) | frequency, recency, T]
    — the expected number of transactions in the next t days, given what
    we've observed about this customer so far.

    Parameters
    ----------
    bgf     : BetaGeoFitter  Fitted model.
    summary : pd.DataFrame   Customer summary.
    days    : int            Prediction horizon in days.

    Returns
    -------
    pd.DataFrame with predicted_purchases column added.
    """
    df = summary.copy()
    df["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        days,
        df["frequency"],
        df["recency"],
        df["T"],
    )

    # Probability customer is still alive (hasn't churned)
    df["prob_alive"] = bgf.conditional_probability_alive(
        df["frequency"],
        df["recency"],
        df["T"],
    )

    print(f"\n  Purchase predictions ({days}-day horizon):")
    print(f"    Avg predicted purchases : {df['predicted_purchases'].mean():.3f}")
    print(f"    Avg probability alive   : {df['prob_alive'].mean():.3f}")
    print(f"    Customers likely alive  : {(df['prob_alive'] > 0.5).sum():,} "
          f"({(df['prob_alive'] > 0.5).mean():.1%})")

    return df


# ── Gamma-Gamma Model ─────────────────────────────────────────────────────────

def fit_gamma_gamma(summary: pd.DataFrame) -> "GammaGammaFitter":
    """
    Fit the Gamma-Gamma model to estimate expected transaction value.

    Important: the Gamma-Gamma model requires customers with at least
    ONE repeat purchase (frequency >= 1). Customers with frequency = 0
    are excluded from fitting but still get CLV predictions using the
    population average transaction value.

    Parameters
    ----------
    summary : pd.DataFrame  Must contain frequency and monetary columns.

    Returns
    -------
    GammaGammaFitter  Fitted model.
    """
    if not LIFETIMES_AVAILABLE:
        raise ImportError("lifetimes library required.")

    print("\n  Fitting Gamma-Gamma model...")

    # Gamma-Gamma requires frequency > 0
    repeat_buyers = summary[summary["frequency"] > 0]
    print(f"  Using {len(repeat_buyers):,} repeat buyers for fitting")

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(
        repeat_buyers["frequency"],
        repeat_buyers["monetary"],
        verbose=False,
    )

    print(f"  Gamma-Gamma fitted successfully")
    print(f"  Parameters: p={ggf.params_['p']:.4f}, q={ggf.params_['q']:.4f}, "
          f"v={ggf.params_['v']:.4f}")

    return ggf


def predict_clv(bgf: "BetaGeoFitter", ggf: "GammaGammaFitter",
                summary: pd.DataFrame) -> pd.DataFrame:
    """
    Combine BG/NBD and Gamma-Gamma to calculate full CLV.

    CLV = E[future purchases] × E[avg transaction value] × gross margin
          discounted to net present value over PREDICTION_YEARS.

    Parameters
    ----------
    bgf     : BetaGeoFitter    Fitted purchase model.
    ggf     : GammaGammaFitter Fitted monetary model.
    summary : pd.DataFrame     Customer summary.

    Returns
    -------
    pd.DataFrame with bgnbd_clv and clv_segment columns added.
    """
    df = summary.copy()

    # Expected average transaction value per customer
    df["predicted_avg_value"] = ggf.conditional_expected_average_profit(
        df["frequency"],
        df["monetary"],
    )

    # Full CLV using lifetimes built-in method
    # This handles the NPV discounting internally
    df["bgnbd_clv"] = ggf.customer_lifetime_value(
        bgf,
        df["frequency"],
        df["recency"],
        df["T"],
        df["monetary"],
        time           = PREDICTION_YEARS * 12,   # in months
        discount_rate  = DISCOUNT_RATE / 12,       # monthly discount rate
        freq           = "D",
    ) * GROSS_MARGIN

    # CLV segments
    df["clv_segment"] = pd.qcut(
        df["bgnbd_clv"],
        q=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low Value", "Mid Value", "High Value", "Top Value"]
    )

    print(f"\n  BG/NBD CLV Results ({PREDICTION_YEARS}-year horizon):")
    print(f"    Avg CLV per customer : ${df['bgnbd_clv'].mean():,.2f}")
    print(f"    Total portfolio CLV  : ${df['bgnbd_clv'].sum():,.2f}")
    print(f"    Median CLV           : ${df['bgnbd_clv'].median():,.2f}")

    return df


# ── Model Validation ──────────────────────────────────────────────────────────

def validate_model(bgf: "BetaGeoFitter", summary_valid: pd.DataFrame,
                   days: int = PREDICTION_DAYS) -> dict:
    """
    Validate the BG/NBD model by comparing predicted vs actual purchases
    in the held-out validation period.

    Parameters
    ----------
    bgf           : BetaGeoFitter  Fitted model.
    summary_valid : pd.DataFrame   Customer summary with actual_future_purchases.
    days          : int            Validation horizon in days.

    Returns
    -------
    dict  Validation metrics.
    """
    df = summary_valid.copy()
    df["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        days,
        df["frequency"],
        df["recency"],
        df["T"],
    )

    actual    = df["actual_future_purchases"].values
    predicted = df["predicted_purchases"].values

    mae  = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    corr = np.corrcoef(actual, predicted)[0, 1]

    metrics = {
        "mae"        : mae,
        "rmse"       : rmse,
        "correlation": corr,
        "actual_mean": actual.mean(),
        "pred_mean"  : predicted.mean(),
    }

    print(f"\n  Model Validation ({days}-day prediction):")
    print(f"    MAE (mean abs error) : {mae:.4f} purchases")
    print(f"    RMSE                 : {rmse:.4f} purchases")
    print(f"    Correlation          : {corr:.4f}")
    print(f"    Actual mean purchases: {actual.mean():.4f}")
    print(f"    Predicted mean       : {predicted.mean():.4f}")

    df["actual_future_purchases_val"] = actual
    return metrics, df


# ── Segment Summary ───────────────────────────────────────────────────────────

def summarize_bgnbd_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return segment-level BG/NBD CLV summary."""
    segment_summary = (
        df.groupby("clv_segment", observed=True)
        .agg(
            customers        = ("CustomerID",         "count"),
            avg_clv          = ("bgnbd_clv",          "mean"),
            total_clv        = ("bgnbd_clv",          "sum"),
            avg_prob_alive   = ("prob_alive",          "mean"),
            avg_pred_purchases = ("predicted_purchases","mean"),
        )
        .reset_index()
    )

    print("\n  BG/NBD CLV by Segment:")
    print(f"  {'Segment':<15} {'Customers':>10} {'Avg CLV':>12} {'Total CLV':>14} {'P(Alive)':>10}")
    print("  " + "-" * 65)
    for _, row in segment_summary.iterrows():
        print(f"  {row['clv_segment']:<15} {int(row['customers']):>10,} "
              f"${row['avg_clv']:>11,.2f} ${row['total_clv']:>13,.2f} "
              f"{row['avg_prob_alive']:>9.1%}")

    return segment_summary


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_bgnbd_pipeline(summary: pd.DataFrame, summary_valid: pd.DataFrame) -> tuple:
    """
    Full BG/NBD + Gamma-Gamma CLV pipeline.

    Returns
    -------
    df_clv          : customer-level CLV predictions
    segment_summary : segment-level summary
    bgf             : fitted BG/NBD model
    ggf             : fitted Gamma-Gamma model
    val_metrics     : validation metrics dict
    val_df          : validation DataFrame with predicted vs actual
    """
    print("\n" + "=" * 50)
    print("  BG/NBD + GAMMA-GAMMA CLV MODEL")
    print("=" * 50)
    print(f"  Gross margin     : {GROSS_MARGIN:.0%}")
    print(f"  Discount rate    : {DISCOUNT_RATE:.0%}")
    print(f"  Prediction years : {PREDICTION_YEARS}")

    if not LIFETIMES_AVAILABLE:
        raise ImportError("Install lifetimes: pip install lifetimes")

    bgf    = fit_bgnbd(summary)
    df_tmp = predict_purchases(bgf, summary)
    ggf    = fit_gamma_gamma(summary)
    df_clv = predict_clv(bgf, ggf, df_tmp)
    segment_summary = summarize_bgnbd_clv(df_clv)
    val_metrics, val_df = validate_model(bgf, summary_valid)

    return df_clv, segment_summary, bgf, ggf, val_metrics, val_df
