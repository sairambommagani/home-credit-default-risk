# Home Credit Default Risk: Multi-Table Credit Risk Modeling

Predict the probability that a loan applicant will have payment difficulties, using all **7 relational
tables** of the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset (~307,500 applicants, ~8% default rate, 300+ raw columns), and serve the model as a **scoring API**
that explains every decision.

| Deliverable | Where |
|---|---|
| Analysis notebook: EDA → baseline → features → CV model → evaluation → explainability | `notebooks/home_credit_risk.ipynb` |
| Feature pipeline (7 tables → one row per applicant) | `hcrisk/features.py` |
| Training and evaluation (5-fold LightGBM, metrics, risk bands, threshold, fairness) | `hcrisk/train.py` |
| Scoring API (probability, risk band, decision, per-applicant reasons, what-if) | `hcrisk/api.py` |
| Tests (end-to-end on schema-identical synthetic data) | `tests/`, `scripts/make_synthetic.py` |

---

## 1. The problem

```
application_{train,test}  SK_ID_CURR   1 row per applicant: the loan asked for + applicant profile (TARGET in train)
├── bureau                SK_ID_BUREAU loans at OTHER institutions, reported to the credit bureau
│   └── bureau_balance                 monthly status of each of those loans (27M rows)
└── previous_application  SK_ID_PREV   earlier applications at Home Credit (approved / refused)
    ├── POS_CASH_balance               monthly snapshots of POS and cash loans
    ├── installments_payments          every instalment: amount and day due vs. amount and day paid
    └── credit_card_balance            monthly credit card snapshots
```

The difficulty compared with a single-table problem:
- **One-to-many history.** The six history tables have many rows per applicant, so they must be aggregated
  to one row per applicant before modeling, without leaking the target.
- **Size on a laptop.** About 2.5 GB of CSV, which needs memory-aware loading on an 8 GB machine.
- **Imbalance.** Only ~8% default, so accuracy is useless. We report **ROC-AUC** (the competition metric),
  **PR-AUC**, **KS**, **Brier score** and calibration.

## 2. Approach

**Memory (8 GB laptops).**
- Numbers are downcast to float32 and small ints, and text is read directly as categories.
- Each table is cached as Parquet on first load.
- Tables are processed one at a time.
- `--sample 0.3` trains on a stratified 30% of applicants.

**Feature engineering** (`hcrisk/features.py`, several hundred features; training prints the exact count):

| Source | Examples |
|---|---|
| application | credit/income, annuity/income, payment rate and loan term, goods/credit, income per person, age, employment/age; `DAYS_EMPLOYED = 365243` turned into a flag + missing; external-score mean/min/max/product/weighted; document and bureau-enquiry counts; missing count |
| bureau + bureau_balance | monthly status → share of months past due and severe DPD; debt/credit, overdue/debt; active vs closed loans separately; total active debt/credit |
| previous_application | asked vs granted amount, down-payment ratio, approved vs refused separately, refusal share, applications in the last year |
| POS_CASH | DPD max/mean, remaining-instalment share, late share in the last 12 months |
| installments | days late / early, underpaid amount, payment ratio, share of late and 30+ days late payments, last-12-month window |
| credit_card | utilisation (balance/limit), payment vs minimum, drawings/limit, last-6-month utilisation and lateness |
| cross-table | this loan vs. existing bureau debt, total annuity burden / income, this loan vs. previous approved loans |

**Leakage control.**
- Aggregations use only each applicant's own history, which is dated before the application.
- `TARGET` is never an input.
- Train and test rows go through identical code.

**Model.**
- LightGBM gradient boosting with 5-fold stratified CV and early stopping.
- Predictions are out-of-fold, so every metric is measured on data the model didn't train on.
- No class weights: AUC doesn't need them, and unweighted training keeps probabilities **calibrated**, which
  the risk bands and the business threshold rely on.

**Evaluation** (`artifacts/metrics.json`):
- baseline (application table only) vs. the full model
- per-fold AUC
- PR-AUC vs. the random baseline, KS, Gini, Brier, log-loss
- 10-bin calibration and decile lift table
- risk bands at score percentiles 60/85/95, with observed default rates
- a cost-based approval threshold (a missed default costs 5, a rejected good customer costs 1; the ratio is a
  business assumption, and the full trade-off table is saved)
- fairness monitoring: AUC and approval rate by gender and age band

**Explainability.**
- Exact TreeSHAP values (LightGBM `pred_contrib`) give the global importance and the share of importance per
  source table.
- Per applicant, the API returns the top factors raising and lowering the risk.

## 3. Results (full data, 307,511 applicants, 5-fold out-of-fold)

| Model | Features | ROC-AUC | Fold AUCs |
|---|---|---|---|
| Baseline: application table only | 146 | 0.7690 | 0.7647 · 0.7759 · 0.7659 · 0.7727 · 0.7659 |
| All 7 tables | 672 | 0.7948 | 0.7915 · 0.8017 · 0.7924 · 0.7973 · 0.7914 |
| **All 7 tables, without `CODE_GENDER` (recommended)** | **671** | **0.7930** | 0.7906 · 0.7994 · 0.7903 · 0.7947 · 0.7905 |

**The history tables add +0.026 AUC.** In credit scoring, gains of a few thousandths already matter.

**Gender removed for fair lending.** `CODE_GENDER` was the #2 feature by SHAP. Removing it costs only 0.0018 AUC
(PR-AUC 0.289 vs 0.290, KS 0.443 vs 0.446), so the gender-free model is the one to serve:
`$env:HC_ARTIFACTS="artifacts_nogender"` before starting the API. Its full report is in `RESULTS_nogender.md`.

| Metric | Value | Reading |
|---|---|---|
| Gini | 0.590 | 2·AUC−1; above 0.5 is considered a good scorecard |
| PR-AUC | 0.290 | random = 0.081, so 3.6× better than chance at finding defaulters |
| KS | 0.446 | separation of the defaulter and payer score distributions; above 0.4 is strong |
| Brier | 0.0654 | 12% better than always predicting the 8.07% base rate (0.0742) |

**What the model uses** (share of mean |SHAP|, by source table):

| Table | Share |
|---|---|
| application | 49.0% |
| previous_application | 20.2% |
| bureau + bureau_balance | 15.5% |
| installments_payments | 8.0% |
| POS_CASH_balance | 3.7% |
| credit_card_balance | 3.6% |

More than half of the model's signal (51%) comes from the six history tables.

The complete numbers are written by `python -m hcrisk.report` into **`RESULTS.md`**, straight from
`artifacts/metrics.json`. They cover the top features, decile lift, calibration, risk bands, the cost-based
threshold and fairness by group.

**Honest caveats.**
- Early stopping used each validation fold, so the out-of-fold AUC may be optimistic by about 0.001–0.002.
  Kaggle's hidden test set (`artifacts/submission.csv`) gives the fully independent number.
- Scores for *training* applicants in the API come from models that saw them in 4 of 5 folds. Use
  `application_test` IDs for demos.

For context, public work on this dataset typically reports about 0.76 from the application table alone and
0.78–0.79 with good multi-table features and a single LightGBM; winning Kaggle ensembles reached about 0.80.

## 4. Run it (Windows PowerShell)

```powershell
cd home-credit-risk
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. **Data.** On Kaggle, join the competition, download all files, and unzip the CSVs into `data\raw\`.
2. **Train.**
   ```powershell
   python -m hcrisk.train --raw data\raw --sample 0.3     # first run: ~30% of applicants, quick check
   python -m hcrisk.train --raw data\raw                  # full data
   python -m hcrisk.train --raw data\raw --quality high   # lower learning rate, best AUC, slower
   ```
   The first run converts the CSVs to Parquet in `data\cache\`. Later runs reuse it.
   Outputs go to `artifacts\`: `models\fold*.txt`, `metrics.json`, `deciles.csv`, `thresholds.csv`,
   `feature_importance.csv`, `oof.parquet`, `features.parquet` (the API's feature store) and `submission.csv`
   (Kaggle format).
3. **Notebook.** Run `jupyter notebook notebooks\home_credit_risk.ipynb` and Run All. It calls the same
   training code, then adds the plots and the discussion. Set `HC_SAMPLE=0.3` to run it on a sample.
4. **API.**
   ```powershell
   python -m uvicorn hcrisk.api:app --port 8001
   ```
   Open http://localhost:8001/docs for the interactive Swagger UI.
5. **Results report.** Run `python -m hcrisk.report` to write `RESULTS.md` from `artifacts\metrics.json`.
6. **Tests.** Run `python -m pytest`. This builds a synthetic dataset with the real schema and checks features,
   training and the API end to end in about 10 seconds.

## 5. API

| Endpoint | Purpose |
|---|---|
| `GET /health` | model and feature store loaded |
| `GET /model-info` | validation metrics, baseline vs. full AUC, risk bands, threshold, importance by table, top features |
| `POST /predict` | score an applicant by `sk_id_curr` (history from the feature store); optional `overrides` for what-if analysis |
| `POST /predict/new` | score a brand-new applicant from application fields only (no history) |
| `POST /predict/batch` | score up to 1000 applicants |

Example request:
```json
POST /predict
{"sk_id_curr": 100002, "overrides": {"AMT_CREDIT": 300000}}
```
The response has this shape (values are illustrative):
```json
{
  "sk_id_curr": 100002,
  "default_probability": 0.31,
  "risk_band": "very high",
  "decision": "refer to underwriter",
  "threshold": 0.12,
  "factors_increasing_risk": [
    {"feature": "EXT_MEAN", "description": "average of the 3 external credit-bureau scores",
     "value": 0.16, "impact_log_odds": 0.62}
  ],
  "factors_decreasing_risk": [],
  "has_credit_history": true
}
```
- **Probability.** Averaged over the 5 fold models: the mean log-odds, then the sigmoid.
- **Factors.** SHAP contributions in log-odds, averaged over the same models.
- **Decision.** `approve` below the cost-optimal threshold, otherwise `refer to underwriter`. A human makes the
  final call on referrals.

## 6. Limitations and next steps

- **Cost ratio.** The 5:1 ratio behind the threshold is an assumption; agree on real loss and margin figures with
  the business.
- **Gender (`--exclude CODE_GENDER`).** On the full data, `CODE_GENDER` is the #2 feature by SHAP. Fair-lending
  rules generally don't allow it in credit decisions, so train a compliant variant with
  `python -m hcrisk.train --raw <data> --out artifacts_nogender --exclude CODE_GENDER` and compare the AUC.
  Gender stays in the fairness *monitoring* table either way.
- **Gender and age.** These are used by the model as provided in the dataset. They are shown for fairness
  *monitoring*, but many lending regulations restrict using them in decisions. A production model should be
  reviewed for this, and possibly retrained without them.
- **Model improvements.** Hyper-parameter search (Optuna), more time-window and trend features, and a second model
  family (CatBoost) for an ensemble.
- **Production.** A real feature store (the history aggregates recomputed as new payments arrive), monitoring for
  data and score drift, and periodic recalibration.
- **Platform reuse.** The same multi-table aggregation and imbalanced-classification pipeline applies to our
  education platform later (candidate history tables → one row per candidate → fail or integrity-risk scoring).
