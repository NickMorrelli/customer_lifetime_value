"""
main.py
-------
End-to-end pipeline for the Customer Lifetime Value (CLV) analysis.

Usage
-----
    python main.py          (run from the customer_lifetime_value/ folder)

Steps
-----
1. Load and clean the UCI Online Retail dataset.
2. Build customer RFM+ summary table.
3. Calculate Simple (RFM-based) CLV.
4. Fit BG/NBD + Gamma-Gamma model and predict CLV.
5. Validate model against held-out data.
6. Assign strategic segments and generate executive summary.
7. Generate and save all visualizations to /outputs/.
"""

import os
import sys

from src.data_prep      import run_prep_pipeline
from src.clv_simple     import run_simple_clv_pipeline
from src.clv_bgnbd      import run_bgnbd_pipeline
from src.segments       import run_segments_pipeline
from src.visualizations import generate_all_plots

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH    = os.path.join(os.path.dirname(__file__), "data", "OnlineRetail.xlsx")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "outputs", "clv_executive_summary.txt")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CUSTOMER LIFETIME VALUE (CLV) ANALYSIS")
    print("  Simple RFM + BG/NBD + Gamma-Gamma")
    print("  UCI Online Retail Dataset")
    print("=" * 65)

    # ── Step 1: Data Preparation ───────────────────────────────────────────
    print("\n[1/5] Data Preparation")
    if not os.path.exists(DATA_PATH):
        print(f"\n  Dataset not found at: {DATA_PATH}")
        print("  Please download 'OnlineRetail.xlsx' from:")
        print("  https://archive.ics.uci.edu/dataset/352/online+retail")
        print("  and place it in the /data/ folder.\n")
        sys.exit(1)

    df, df_train, df_valid, summary, summary_valid, split_date, end_date = (
        run_prep_pipeline(DATA_PATH)
    )

    # ── Step 2: Simple CLV ─────────────────────────────────────────────────
    print("\n[2/5] Simple CLV Model")
    df_simple, simple_segments = run_simple_clv_pipeline(summary)

    # ── Step 3: BG/NBD + Gamma-Gamma CLV ──────────────────────────────────
    print("\n[3/5] BG/NBD + Gamma-Gamma CLV Model")
    df_bgnbd, bgnbd_segments, bgf, ggf, val_metrics, val_df = (
        run_bgnbd_pipeline(summary, summary_valid)
    )

    # ── Step 4: Strategic Segmentation ────────────────────────────────────
    print("\n[4/5] Strategic Segmentation")
    df_combined, exec_summary = run_segments_pipeline(df_simple, df_bgnbd)

    # Save executive summary
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(exec_summary)
    print(f"  Executive summary saved to: {SUMMARY_PATH}")

    # Save full CLV table
    clv_out = os.path.join(os.path.dirname(__file__), "outputs", "clv_predictions.csv")
    df_combined.to_csv(clv_out, index=False)
    print(f"  CLV predictions saved to  : {clv_out}")

    # ── Step 5: Visualizations ─────────────────────────────────────────────
    print("\n[5/5] Generating Visualizations")
    generate_all_plots(df_combined, val_df, val_metrics)

    # ── Done ───────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs saved to: {os.path.join(os.path.dirname(__file__), 'outputs')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
