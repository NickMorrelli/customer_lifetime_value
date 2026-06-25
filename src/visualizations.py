"""
visualizations.py
-----------------
Generates all plots for the CLV analysis.

Charts
------
1. CLV Distribution          – histogram of BG/NBD CLV values
2. Simple vs BG/NBD CLV      – scatter comparison of both models
3. Probability Alive          – distribution of P(customer still active)
4. Predicted vs Actual        – model validation chart
5. Strategic Segments         – 2x2 matrix visualization
6. Top 20 Customers           – bar chart of highest CLV customers
7. Summary Dashboard          – key metrics in one view
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────────────

SEGMENT_COLORS = {
    "Champions"  : "#2ECC71",
    "At Risk"    : "#E74C3C",
    "Rising Stars": "#F39C12",
    "Dormant"    : "#95A5A6",
}

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
    "figure.dpi"       : 120,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save(fig, filename):
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def _usd(x, pos):
    return f"${x:,.0f}"


# ── Plot 1: CLV Distribution ──────────────────────────────────────────────────

def plot_clv_distribution(df_combined: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Customer Lifetime Value Distribution", fontsize=14, fontweight="bold")

    # BG/NBD CLV histogram
    clv_cap = df_combined["bgnbd_clv"].quantile(0.95)
    data    = df_combined[df_combined["bgnbd_clv"] <= clv_cap]["bgnbd_clv"]
    ax1.hist(data, bins=50, color="#3498DB", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax1.axvline(df_combined["bgnbd_clv"].median(), color="black", linestyle="--",
                linewidth=1.5, label=f"Median: ${df_combined['bgnbd_clv'].median():,.0f}")
    ax1.axvline(df_combined["bgnbd_clv"].mean(), color="#E74C3C", linestyle="--",
                linewidth=1.5, label=f"Mean: ${df_combined['bgnbd_clv'].mean():,.0f}")
    ax1.set_title("BG/NBD CLV Distribution (95th pct cap)")
    ax1.set_xlabel("3-Year CLV ($)")
    ax1.set_ylabel("Customers")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax1.legend()

    # Segment breakdown pie
    seg_clv = df_combined.groupby("strategic_segment")["bgnbd_clv"].sum()
    colors  = [SEGMENT_COLORS.get(s, "#999") for s in seg_clv.index]
    ax2.pie(seg_clv.values, labels=seg_clv.index, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax2.set_title("CLV Share by Strategic Segment")

    plt.tight_layout()
    _save(fig, "01_clv_distribution.png")


# ── Plot 2: Simple vs BG/NBD Comparison ──────────────────────────────────────

def plot_model_comparison(df_combined: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7))

    for segment, group in df_combined.groupby("strategic_segment"):
        ax.scatter(
            group["simple_clv_npv"], group["bgnbd_clv"],
            c=SEGMENT_COLORS.get(segment, "#999"),
            label=segment, alpha=0.5, s=15, edgecolors="none"
        )

    # Diagonal reference line (y=x means both models agree)
    max_val = min(df_combined["simple_clv_npv"].quantile(0.95),
                  df_combined["bgnbd_clv"].quantile(0.95))
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="Models agree (y=x)")

    ax.set_xlabel("Simple CLV ($)")
    ax.set_ylabel("BG/NBD CLV ($)")
    ax.set_title("Simple CLV vs BG/NBD CLV\nPoints above line: BG/NBD is more optimistic",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    _save(fig, "02_model_comparison.png")


# ── Plot 3: Probability Alive ─────────────────────────────────────────────────

def plot_probability_alive(df_combined: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Probability Customer is Still Active (BG/NBD)", fontsize=14, fontweight="bold")

    # Overall distribution
    ax1.hist(df_combined["prob_alive"], bins=40, color="#9B59B6", alpha=0.8,
             edgecolor="white", linewidth=0.3)
    ax1.axvline(0.5, color="#E74C3C", linestyle="--", linewidth=1.5, label="50% threshold")
    ax1.set_xlabel("P(Customer Still Active)")
    ax1.set_ylabel("Customers")
    ax1.set_title("Distribution of P(Alive)")
    ax1.legend()

    # By segment
    for segment, group in df_combined.groupby("strategic_segment"):
        ax2.hist(group["prob_alive"], bins=30, alpha=0.6,
                 color=SEGMENT_COLORS.get(segment, "#999"), label=segment)
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("P(Customer Still Active)")
    ax2.set_ylabel("Customers")
    ax2.set_title("P(Alive) by Strategic Segment")
    ax2.legend()

    plt.tight_layout()
    _save(fig, "03_probability_alive.png")


# ── Plot 4: Predicted vs Actual Purchases ─────────────────────────────────────

def plot_validation(val_df: pd.DataFrame, val_metrics: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BG/NBD Model Validation — Predicted vs Actual Purchases",
                 fontsize=13, fontweight="bold")

    actual    = val_df["actual_future_purchases"].values
    predicted = val_df["predicted_purchases"].values

    # Drop NaN values from both arrays
    mask      = ~(np.isnan(actual) | np.isnan(predicted))
    actual    = actual[mask]
    predicted = predicted[mask]

    # Scatter
    ax1.scatter(actual, predicted, alpha=0.3, s=10, color="#3498DB", edgecolors="none")
    max_val = max(actual.max(), predicted.max())
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    ax1.set_xlabel("Actual Purchases (Validation Period)")
    ax1.set_ylabel("Predicted Purchases")
    ax1.set_title(f"Predicted vs Actual\nCorrelation: {val_metrics['correlation']:.3f}")
    ax1.legend()

    # Distribution comparison
    max_purchases = int(np.percentile(np.concatenate([actual, predicted]), 95))
    bins = np.arange(0, max_purchases + 2) - 0.5
    ax2.hist(actual,    bins=bins, alpha=0.6, color="#E74C3C",  label=f"Actual (mean={actual.mean():.2f})")
    ax2.hist(predicted, bins=bins, alpha=0.6, color="#3498DB",  label=f"Predicted (mean={predicted.mean():.2f})")
    ax2.set_xlabel("Number of Purchases")
    ax2.set_ylabel("Customers")
    ax2.set_title(f"Distribution of Purchases\nMAE: {val_metrics['mae']:.4f}")
    ax2.legend()

    plt.tight_layout()
    _save(fig, "04_model_validation.png")


# ── Plot 5: Strategic Segment Matrix ──────────────────────────────────────────

def plot_strategic_segments(df_combined: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))

    simple_median = df_combined["simple_clv_npv"].median()
    bgnbd_median  = df_combined["bgnbd_clv"].median()

    for segment, group in df_combined.groupby("strategic_segment"):
        ax.scatter(
            group["simple_clv_npv"], group["bgnbd_clv"],
            c=SEGMENT_COLORS.get(segment, "#999"),
            label=f"{segment} (n={len(group):,})",
            alpha=0.5, s=20, edgecolors="none"
        )

    # Quadrant lines
    ax.axvline(simple_median, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(bgnbd_median,  color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Quadrant labels
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    label_props = dict(fontsize=9, alpha=0.6, fontweight="bold")
    ax.text(simple_median * 1.05, bgnbd_median * 1.8,  "CHAMPIONS\n↑ Historical + Future Value",  color=SEGMENT_COLORS["Champions"],   **label_props)
    ax.text(simple_median * 1.05, bgnbd_median * 0.2,  "AT RISK\n↑ Historical, ↓ Future",         color=SEGMENT_COLORS["At Risk"],      **label_props)
    ax.text(simple_median * 0.05, bgnbd_median * 1.8,  "RISING STARS\n↓ Historical, ↑ Future",    color=SEGMENT_COLORS["Rising Stars"], **label_props)
    ax.text(simple_median * 0.05, bgnbd_median * 0.2,  "DORMANT\n↓ Historical + Future Value",    color=SEGMENT_COLORS["Dormant"],      **label_props)

    ax.set_xlabel("Simple CLV ($) — Historical Value")
    ax.set_ylabel("BG/NBD CLV ($) — Predicted Future Value")
    ax.set_title("Strategic CLV Segmentation Matrix", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    _save(fig, "05_strategic_segments.png")


# ── Plot 6: Top 20 Customers ──────────────────────────────────────────────────

def plot_top_customers(df_combined: pd.DataFrame):
    top20 = df_combined.nlargest(20, "bgnbd_clv").sort_values("bgnbd_clv")

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [SEGMENT_COLORS.get(s, "#999") for s in top20["strategic_segment"]]
    bars   = ax.barh(range(len(top20)), top20["bgnbd_clv"], color=colors, alpha=0.85)

    for bar, val in zip(bars, top20["bgnbd_clv"]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", fontsize=8)

    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"Customer {cid}" for cid in top20["CustomerID"]], fontsize=8)
    ax.set_xlabel("3-Year BG/NBD CLV ($)")
    ax.set_title("Top 20 Customers by Predicted CLV", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))

    legend_patches = [mpatches.Patch(color=c, label=s)
                      for s, c in SEGMENT_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    _save(fig, "06_top_customers.png")


# ── Plot 7: Summary Dashboard ─────────────────────────────────────────────────

def plot_summary_dashboard(df_combined: pd.DataFrame, val_metrics: dict):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Customer Lifetime Value — Summary Dashboard",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top-left: CLV distribution ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    clv_cap = df_combined["bgnbd_clv"].quantile(0.95)
    data    = df_combined[df_combined["bgnbd_clv"] <= clv_cap]["bgnbd_clv"]
    ax1.hist(data, bins=40, color="#3498DB", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax1.axvline(df_combined["bgnbd_clv"].median(), color="#E74C3C", linestyle="--", linewidth=1.5)
    ax1.set_title("BG/NBD CLV Distribution")
    ax1.set_xlabel("CLV ($)")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))

    # ── Top-center: Strategic segment scatter ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for segment, group in df_combined.groupby("strategic_segment"):
        ax2.scatter(group["simple_clv_npv"], group["bgnbd_clv"],
                    c=SEGMENT_COLORS.get(segment, "#999"), label=segment,
                    alpha=0.4, s=8, edgecolors="none")
    simple_med = df_combined["simple_clv_npv"].median()
    bgnbd_med  = df_combined["bgnbd_clv"].median()
    ax2.axvline(simple_med, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.axhline(bgnbd_med,  color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_title("CLV Segmentation Matrix")
    ax2.set_xlabel("Simple CLV ($)")
    ax2.set_ylabel("BG/NBD CLV ($)")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_usd))
    ax2.legend(fontsize=7)

    # ── Top-right: Segment sizes ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    seg_counts = df_combined["strategic_segment"].value_counts()
    colors3    = [SEGMENT_COLORS.get(s, "#999") for s in seg_counts.index]
    ax3.pie(seg_counts.values, labels=seg_counts.index, colors=colors3,
            autopct="%1.0f%%", startangle=140,
            textprops={"fontsize": 8},
            wedgeprops={"edgecolor": "white"})
    ax3.set_title("Segment Distribution")

    # ── Bottom-left: P(Alive) distribution ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(df_combined["prob_alive"], bins=40, color="#9B59B6", alpha=0.8,
             edgecolor="white", linewidth=0.3)
    ax4.axvline(0.5, color="#E74C3C", linestyle="--", linewidth=1.5)
    ax4.set_title("P(Customer Still Active)")
    ax4.set_xlabel("Probability Alive")

    # ── Bottom-center: Top 10 customers ───────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    top10  = df_combined.nlargest(10, "bgnbd_clv").sort_values("bgnbd_clv")
    colors5 = [SEGMENT_COLORS.get(s, "#999") for s in top10["strategic_segment"]]
    ax5.barh(range(10), top10["bgnbd_clv"], color=colors5, alpha=0.85)
    ax5.set_yticks(range(10))
    ax5.set_yticklabels([f"C-{cid[-4:]}" for cid in top10["CustomerID"]], fontsize=8)
    ax5.set_title("Top 10 Customers by CLV")
    ax5.xaxis.set_major_formatter(mticker.FuncFormatter(_usd))

    # ── Bottom-right: Key metrics ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = [
        ["Metric", "Value"],
        ["Total Customers",    f"{len(df_combined):,}"],
        ["Avg BG/NBD CLV",     f"${df_combined['bgnbd_clv'].mean():,.2f}"],
        ["Portfolio CLV",      f"${df_combined['bgnbd_clv'].sum():,.0f}"],
        ["Avg P(Alive)",       f"{df_combined['prob_alive'].mean():.1%}"],
        ["Model MAE",          f"{val_metrics['mae']:.4f}"],
        ["Model Correlation",  f"{val_metrics['correlation']:.3f}"],
    ]
    tbl = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax6.set_title("Key Metrics", pad=12)

    plt.tight_layout()
    _save(fig, "07_summary_dashboard.png")


# ── Run All ───────────────────────────────────────────────────────────────────

def generate_all_plots(df_combined, val_df, val_metrics):
    print("\nGenerating visualizations...")
    plot_clv_distribution(df_combined)
    plot_model_comparison(df_combined)
    plot_probability_alive(df_combined)
    plot_validation(val_df, val_metrics)
    plot_strategic_segments(df_combined)
    plot_top_customers(df_combined)
    plot_summary_dashboard(df_combined, val_metrics)
    print("All plots saved to /outputs/\n")
