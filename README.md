# Practicum II: Predictive Analysis on Geopolitical Risk Affects U.S. CPI, Energy, and Macroeconomics

**Student:** Munkhbayasgalan Ganaa  
**Program:** MS Data Science  
**University:** Regis University, Denver, CO  
**Email:** mganaa@regis.edu

---

## Project Overview

This project studies whether geopolitical events around the world  like wars, conflicts, and crises  affect everyday consumer prices in the United States. I collected monthly data from January 2000 to February 2026 from three sources, built four machine learning models, and analyzed how geopolitical risk travels through oil and gas prices before eventually reaching consumer inflation (CPI).

**The main question:** Does geopolitical risk drive US inflation, and through what transmission channels?

**The answer I found:** Yes but it is indirect and delayed. A geopolitical shock hits oil prices first (1-3 months), oil moves gas prices almost immediately, and then CPI adjusts 3-6 months later.

---

## Live Dashboard

I built an interactive web dashboard to visualize all project results:

🔗 **[Launch Dashboard](https://db3jybx6evdfqvvskkhmh5.streamlit.app)**

The dashboard has 5 tabs:
- **Data Health** — checks all CSV files are loaded correctly
- **Models** — RMSE and R² comparison across all 4 models
- **Categories** — CPI category sensitivity to geopolitical events
- **Final** — best model per target variable
- **AI Analyst** — rule-based project brief generated automatically

---

## Data Sources

| Source | Raw Records | Description |
|--------|------------|-------------|
| BLS CPI | 394,002 | Consumer Price Index  6 categories (All Items, Food, Housing, Apparel, Transportation, Medical) |
| FRED Macro | 20,929 | Oil WTI, Gas Retail, Fed Funds Rate, Unemployment, USD Index, VIX |
| GPR Index | 1,514 | Geopolitical Risk Index (Caldara & Iacoviello, 2022) |

After cleaning and merging, the final panel has **313 monthly observations** from 2000 to 2026.

---

## Models Tested

| Model | Description | Best on |
|-------|-------------|---------|
| **LagLinear** | Linear regression using own past values (lag 1/3/6/12 months) | 6 out of 7 targets |
| **GPRLinear** | LagLinear + GPR index and its lags | Fed Funds Rate |
| **GPRBoosting** | Same features as GPRLinear + Gradient Boosting algorithm | — |
| **StackingEnsemble** | Combines LagLinear + GPRBoosting via meta linear model (cv=5) | — |

---

## Key Results

| Target | Best Model | R² | RMSE |
|--------|-----------|-----|------|
| CPI All Items | LagLinear | 0.994 | 0.78 pts |
| Fed Funds | GPRLinear | 0.982 | 0.19 pts |
| Oil WTI | LagLinear | 0.794 | $5.75/bbl |
| Gas Retail | LagLinear | 0.747 | $0.22/gal |
| USD Index | LagLinear | 0.720 | 1.52 pts |
| Unemployment | LagLinear | 0.637 | 0.18 pts |
| VIX | LagLinear | 0.556 | 3.35 pts |

**CPI predicted within 0.18% average error** during a test period that included the highest inflation in 40 years (June 2022, 9.1%).

---

## Transmission Chain

```
GPR spike → Oil reacts (1-3 months) → Gas follows immediately → CPI adjusts (3-6 months)
```

| Channel | Correlation | Best Lag |
|---------|------------|---------|
| Oil → Gas | 0.885 | 0 months |
| USD → Oil | 0.518 | 3 months |
| Gas → CPI | 0.271 | 6 months |
| GPR → VIX | 0.163 | 1 month |
| GPR → Oil | 0.130 | 3 months |

---

## Category Sensitivity

Transportation prices react the most to geopolitical events:

| Category | Avg Abs Change | Worst Event |
|----------|---------------|-------------|
| Transportation | 8.40 pts | Russia-Ukraine 2022 (+25.2) |
| Food | 6.54 pts | Russia-Ukraine 2022 (+17.8) |
| Shelter | 5.46 pts | Middle East 2023 (+8.8) |

---

## 6 Major Events Analyzed

| Event | Date | Oil Effect |
|-------|------|-----------|
| 9/11 Attacks | Sep 2001 | ↓ Fell (demand shock) |
| Iraq War | Mar 2003 | ↑ Rose (supply uncertainty) |
| Crimea Conflict | Feb 2014 | ↑ Moderate rise |
| COVID-19 | Mar 2020 | ↓ Crashed (demand collapse) |
| Russia-Ukraine Invasion | Feb 2022 | ↑ Surged to $130 (supply shock) |
| Middle East Escalation | Oct 2023 | ↑ Moderate rise |

---

## Repository Structure

```
Practicum-II/
├── Practicum_II_Clean_Final.ipynb     ← Main analysis notebook (clean, organized)
├── practicum2_macro_ai_app.py         ← Streamlit dashboard app
├── requirements.txt                   ← Python dependencies
├── clean_step5_results.csv            ← Model performance results
├── clean_step5_incremental_gpr.csv    ← GPR incremental value vs LagLinear
├── clean_step7_links.csv              ← Transmission channel correlations
├── clean_step7_event.csv              ← Event window analysis results
├── clean_step7_summary.csv            ← Category sensitivity summary
├── clean_step8_final_summary.csv      ← Best model per target
└── README.md                          ← This file
```

---

## How to Run the Notebook

1. Make sure your Raw Data folder contains the BLS, FRED, and GPR files
2. Update `DATA_ROOT` in the first code cell to point to your data folder
3. Run all cells from top to bottom each step depends on the previous one

**Required Python libraries:**
```bash
pip install pandas numpy plotly scikit-learn streamlit
```

---

## How to Run the Dashboard Locally

```bash
pip install -r requirements.txt
streamlit run practicum2_macro_ai_app.py
```

Then open `(https://5df9bjbppwrdnvl8kd77bs.streamlit.app/)` in your browser.

---

## Key Findings

1. **Geopolitical risk is a real inflation driver** but it works through a chain reaction, not a direct hit on consumer prices
2. **Simplicity wins** LagLinear beat all three more complex models on 6 out of 7 targets. The stacking ensemble made things worse on every target
3. **Transportation is most exposed** it swings the hardest during every major event, confirming that energy dependent prices are the first to feel geopolitical shocks
4. **GPR adds value for interest rates** GPRLinear won on fed_funds because the Federal Reserve directly responds to global uncertainty when setting policy

---

## References

- Caldara, D. & Iacoviello, M. (2022). Measuring Geopolitical Risk. *American Economic Review*, 112(4), 1194–1225.
- Caldara, D., Conlisk, S., Iacoviello, M., & Penn, M. (2026). Do Geopolitical Risks Raise or Lower Inflation? *Journal of International Economics*, 159, 104188.
- Hassan, M.S. et al. (2023). The Consumer Price Index Prediction Using Machine Learning Approaches. *Heliyon*, 9(10), e20325.
- Kohlscheen, E. (2021). What Does Machine Learning Say About the Drivers of Inflation? *BIS Working Paper* No. 980.

---

*Practicum II — MS Data Science — Regis University — 2025/2026*
