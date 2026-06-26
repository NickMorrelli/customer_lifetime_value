
# 💰 Customer Lifetime Value (CLV) Analysis

An end-to-end Customer Lifetime Value pipeline using the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail). This project demonstrates **two approaches to CLV** — progressing from a simple RFM-based model to the industry-standard **BG/NBD + Gamma-Gamma** probabilistic model — and combines both to create actionable strategic customer segments.

---

## 🎯 Business Questions

> - *Which customers are most valuable over the long term?*
> - *How much should we spend to acquire or retain a customer?*
> - *Which customers are at risk of churning despite high historical spend?*
> - *Who are our rising stars worth investing in now?*

---

## 📁 Project Structure

```
customer_lifetime_value/
├── data/                        # Place OnlineRetail.xlsx here
├── outputs/                     # Charts, CLV predictions, executive summary
├── src/
│   ├── data_prep.py             # Load, clean, build RFM+ customer summary
│   ├── clv_simple.py            # Simple RFM-based CLV (historical average method)
│   ├── clv_bgnbd.py             # BG/NBD + Gamma-Gamma (industry standard)
│   ├── segments.py              # Strategic 2×2 segmentation + recommendations
│   └── visualizations.py       # All charts and summary dashboard
├── main.py                      # Run the full pipeline
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**UCI Online Retail Dataset** — real-world UK e-commerce transactions (Dec 2010 – Dec 2011).

**Download:** https://archive.ics.uci.edu/dataset/352/online+retail  
Place `OnlineRetail.xlsx` in the `/data/` folder before running.

The data is split into:
- **Observation period** (75%): used to fit the models
- **Validation period** (25%): used to evaluate prediction accuracy

---

## 📐 Methodology

### Approach 1: Simple RFM-Based CLV

```
CLV = Average Order Value × Orders per Year × Years × Gross Margin
```

**Pros:** Intuitive, easy to explain, no special libraries required.  
**Cons:** Assumes all customers will keep buying at their historical rate — ignores churn.

### Approach 2: BG/NBD + Gamma-Gamma (Industry Standard)

**BG/NBD** (Beta-Geometric / Negative Binomial Distribution) models two simultaneous processes:
1. **Transaction process**: while active, a customer purchases at rate λ (Gamma-distributed across customers)
2. **Dropout process**: after each transaction, a customer has probability *p* of churning permanently (Beta-distributed across customers)

**Gamma-Gamma** model estimates expected average transaction value per customer.

**Combined CLV:**
```
CLV = E[future purchases | BG/NBD] × E[avg order value | Gamma-Gamma] × gross margin
```
Discounted to Net Present Value over a 3-year horizon.

### Strategic 2×2 Segmentation

|  | **High BG/NBD CLV** | **Low BG/NBD CLV** |
|---|---|---|
| **High Simple CLV** | 🏆 Champions | ⚠️ At Risk |
| **Low Simple CLV** | 🌟 Rising Stars | 😴 Dormant |

| Segment | Strategy |
|---|---|
| **Champions** | VIP rewards, early access, loyalty programs |
| **At Risk** | Personalized win-back campaigns, time-limited offers |
| **Rising Stars** | Onboarding sequences, product discovery, habit building |
| **Dormant** | Low-cost reactivation; suppress if unresponsive |

---

## 📈 Output Files

| File | Description |
|---|---|
| `01_clv_distribution.png` | CLV histogram + segment pie chart |
| `02_model_comparison.png` | Simple vs BG/NBD CLV scatter plot |
| `03_probability_alive.png` | P(customer still active) distribution |
| `04_model_validation.png` | Predicted vs actual purchases (validation) |
| `05_strategic_segments.png` | 2×2 strategic segmentation matrix |
| `06_top_customers.png` | Top 20 customers by predicted CLV |
| `07_summary_dashboard.png` | Full summary dashboard |
| `clv_predictions.csv` | Full customer-level CLV predictions |
| `clv_executive_summary.txt` | Business-ready summary with strategies |

---

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/customer_lifetime_value.git
cd customer_lifetime_value
pip install -r requirements.txt
# Add OnlineRetail.xlsx to /data/
python main.py
```

---

## 🛠 Tech Stack

- **Python 3.14+**
- `lifetimes` — BG/NBD and Gamma-Gamma model fitting
- `pandas` / `numpy` — data wrangling
- `scipy` — statistical utilities
- `matplotlib` — visualizations
- `scikit-learn` — preprocessing utilities

---

## 💡 Key Concepts Demonstrated

- RFM feature engineering from raw transaction data
- Simple CLV with NPV discounting
- Probabilistic CLV modeling (BG/NBD + Gamma-Gamma)
- Train/validation split for time-series model evaluation
- P(alive) estimation and churn probability
- Strategic 2×2 customer segmentation
- Executive-ready reporting and business recommendations

---

## 👤 Author

Built as part of a data science portfolio project.  
Background: 15+ years in Marketing Analytics | SQL | Python | Statistical Modeling
=======
# customer_lifetime_value
A CLV model that uses public data and compares the simple method based on AOV with the industry standard BG/NBD + Gamma-Gamma models.
>>>>>>> 413ce37a5c6498dd747d510bbc4c0597e0504d72
