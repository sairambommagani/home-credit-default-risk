# Results: Home Credit Default Risk

Generated from `artifacts_nogender/metrics.json`. 307,511 training applicants (8.07% default), 671 features, 5-fold stratified CV, all metrics on out-of-fold predictions. Excluded features: CODE_GENDER.

## Model quality

| Model | Features | ROC-AUC | Fold AUCs |
|---|---|---|---|
| Baseline: application table only | 145 | 0.7674 | 0.7634, 0.7747, 0.7648, 0.7704, 0.7640 |
| All 7 tables | 671 | **0.7930** | 0.7906, 0.7994, 0.7903, 0.7947, 0.7905 |

The six history tables add **+0.0257 AUC** over the application table alone.

| Metric | Value | Reading |
|---|---|---|
| Gini | 0.586 | 2·AUC−1, the usual scorecard measure (>0.5 is good) |
| PR-AUC | 0.289 | random = 0.081, so 3.6× better than chance |
| KS | 0.443 | max separation of defaulter vs payer score distributions (>0.4 is strong) |
| Brier | 0.0655 | 12% better than always predicting the base rate (0.0742) |
| Log-loss | 0.2350 | |

## What the model uses

Share of total mean |SHAP| by source table:

| Table | Share |
|---|---|
| APPLICATION | 46.7% |
| PREV | 21.5% |
| BURO | 16.0% |
| INST | 8.2% |
| CC | 3.8% |
| POS | 3.8% |

Top 20 features (mean |SHAP|, log-odds):

| Feature | mean abs SHAP |
|---|---|
| `EXT_MEAN` | 0.2697 |
| `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | 0.1007 |
| `EXT_MIN` | 0.0996 |
| `ORGANIZATION_TYPE` | 0.0919 |
| `EXT_MAX` | 0.0908 |
| `AMT_ANNUITY` | 0.0819 |
| `OCCUPATION_TYPE` | 0.0728 |
| `CREDIT_MINUS_GOODS` | 0.0682 |
| `NAME_FAMILY_STATUS` | 0.0643 |
| `DAYS_EMPLOYED` | 0.0623 |
| `NAME_EDUCATION_TYPE` | 0.0620 |
| `INST_AMT_PAYMENT_SUM` | 0.0534 |
| `BURO_DEBT_CREDIT_RATIO_MAX` | 0.0515 |
| `GOODS_CREDIT_RATIO` | 0.0503 |
| `CREDIT_TERM_MONTHS` | 0.0477 |
| `EXT_SOURCE_3` | 0.0445 |
| `EXT_SOURCE_1` | 0.0436 |
| `PAYMENT_RATE` | 0.0432 |
| `BURO_ACTIVE_DEBT_CREDIT_RATIO_MAX` | 0.0410 |
| `EXT_PROD` | 0.0400 |

## Lift by score decile

The riskiest 10% of applicants hold **38% of all defaults** (default rate 30.8%, 3.8× average); the safest 10% have 0.8%.

| Decile (1 = safest) | Applicants | Default rate | Lift | Share of all defaults |
|---|---|---|---|---|
| 1 | 30,752 | 0.8% | 0.10 | 1.0% |
| 2 | 30,751 | 1.4% | 0.17 | 1.7% |
| 3 | 30,751 | 2.2% | 0.27 | 2.7% |
| 4 | 30,751 | 2.9% | 0.35 | 3.5% |
| 5 | 30,751 | 4.0% | 0.49 | 4.9% |
| 6 | 30,751 | 5.2% | 0.65 | 6.5% |
| 7 | 30,751 | 7.3% | 0.91 | 9.1% |
| 8 | 30,751 | 10.5% | 1.30 | 13.0% |
| 9 | 30,751 | 15.7% | 1.94 | 19.4% |
| 10 | 30,751 | 30.8% | 3.81 | 38.1% |

## Calibration (predicted vs observed default rate, 10 bins)

| Predicted | Observed | Applicants |
|---|---|---|
| 0.86% | 0.79% | 30,752 |
| 1.54% | 1.40% | 30,751 |
| 2.18% | 2.17% | 30,751 |
| 2.91% | 2.85% | 30,751 |
| 3.83% | 3.98% | 30,751 |
| 5.07% | 5.22% | 30,751 |
| 6.86% | 7.34% | 30,751 |
| 9.69% | 10.51% | 30,751 |
| 14.97% | 15.69% | 30,751 |
| 31.66% | 30.78% | 30,751 |

## Risk bands (score percentiles 60 / 85 / 95)

| Band | Score range | Share of applicants | Observed default rate |
|---|---|---|---|
| low | ≥ 0.0000 | 60.0% | 2.7% |
| medium | ≥ 0.0585 | 25.0% | 9.8% |
| high | ≥ 0.1470 | 10.0% | 20.7% |
| very high | ≥ 0.2806 | 5.0% | 38.1% |

## Approval threshold

Assumed costs: a missed default = 5.0, a rejected good customer = 1.0 (to be agreed with the business; full trade-off in `artifacts/thresholds.csv`).

Cost-optimal rule: **approve if score < 0.16**, which approves 86.6% of applicants, catches 46.1% of defaults, and lowers the default rate among approved loans to 5.02% (from 8.07% if all were approved).

## Fairness monitoring

Ranking quality and approval rate by group at the threshold above. Gender and age are used by the model as provided in the dataset; this is monitoring only (see README, limitations).

**gender**

| Group | Applicants | Default rate | AUC | Mean score | Approval rate |
|---|---|---|---|---|---|
| M | 105,059 | 10.1% | 0.784 | 0.093 | 82.8% |
| F | 202,448 | 7.0% | 0.794 | 0.073 | 88.6% |

**age**

| Group | Applicants | Default rate | AUC | Mean score | Approval rate |
|---|---|---|---|---|---|
| <30 | 45,186 | 11.4% | 0.769 | 0.115 | 76.5% |
| 45-60 | 103,302 | 6.6% | 0.788 | 0.066 | 90.6% |
| 30-45 | 123,737 | 9.0% | 0.796 | 0.087 | 84.5% |
| 60+ | 35,286 | 4.9% | 0.754 | 0.048 | 95.6% |
