# Results: Home Credit Default Risk

Generated from `artifacts/metrics.json`. 307,511 training applicants (8.07% default), 672 features, 5-fold stratified CV, all metrics on out-of-fold predictions.

## Model quality

| Model | Features | ROC-AUC | Fold AUCs |
|---|---|---|---|
| Baseline: application table only | 146 | 0.7690 | 0.7647, 0.7759, 0.7659, 0.7727, 0.7659 |
| All 7 tables | 672 | **0.7948** | 0.7915, 0.8017, 0.7924, 0.7973, 0.7914 |

The six history tables add **+0.0258 AUC** over the application table alone.

| Metric | Value | Reading |
|---|---|---|
| Gini | 0.590 | 2·AUC−1, the usual scorecard measure (>0.5 is good) |
| PR-AUC | 0.290 | random = 0.081, so 3.6× better than chance |
| KS | 0.446 | max separation of defaulter vs payer score distributions (>0.4 is strong) |
| Brier | 0.0654 | 12% better than always predicting the base rate (0.0742) |
| Log-loss | 0.2345 | |

## What the model uses

Share of total mean |SHAP| by source table:

| Table | Share |
|---|---|
| APPLICATION | 49.0% |
| PREV | 20.2% |
| BURO | 15.5% |
| INST | 8.0% |
| POS | 3.7% |
| CC | 3.6% |

Top 20 features (mean |SHAP|, log-odds):

| Feature | mean abs SHAP |
|---|---|
| `EXT_MEAN` | 0.3026 |
| `CODE_GENDER` | 0.1039 |
| `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | 0.0960 |
| `ORGANIZATION_TYPE` | 0.0869 |
| `EXT_MAX` | 0.0860 |
| `EXT_MIN` | 0.0810 |
| `AMT_ANNUITY` | 0.0727 |
| `CREDIT_MINUS_GOODS` | 0.0695 |
| `NAME_FAMILY_STATUS` | 0.0691 |
| `NAME_EDUCATION_TYPE` | 0.0601 |
| `DAYS_EMPLOYED` | 0.0515 |
| `GOODS_CREDIT_RATIO` | 0.0509 |
| `OCCUPATION_TYPE` | 0.0482 |
| `BURO_ACTIVE_DEBT_CREDIT_RATIO_MAX` | 0.0481 |
| `INST_AMT_PAYMENT_SUM` | 0.0474 |
| `BURO_DEBT_CREDIT_RATIO_MAX` | 0.0435 |
| `INST_LATE_MEAN` | 0.0434 |
| `OWN_CAR_AGE` | 0.0423 |
| `CREDIT_TERM_MONTHS` | 0.0422 |
| `PAYMENT_RATE` | 0.0394 |

## Lift by score decile

The riskiest 10% of applicants hold **38% of all defaults** (default rate 30.9%, 3.8× average); the safest 10% have 0.8%.

| Decile (1 = safest) | Applicants | Default rate | Lift | Share of all defaults |
|---|---|---|---|---|
| 1 | 30,752 | 0.8% | 0.10 | 1.0% |
| 2 | 30,751 | 1.4% | 0.17 | 1.7% |
| 3 | 30,751 | 2.0% | 0.25 | 2.5% |
| 4 | 30,751 | 2.9% | 0.36 | 3.6% |
| 5 | 30,751 | 3.9% | 0.48 | 4.8% |
| 6 | 30,751 | 5.2% | 0.64 | 6.4% |
| 7 | 30,751 | 7.4% | 0.92 | 9.2% |
| 8 | 30,751 | 10.3% | 1.27 | 12.7% |
| 9 | 30,751 | 15.9% | 1.97 | 19.7% |
| 10 | 30,751 | 30.9% | 3.83 | 38.3% |

## Calibration (predicted vs observed default rate, 10 bins)

| Predicted | Observed | Applicants |
|---|---|---|
| 0.88% | 0.81% | 30,752 |
| 1.55% | 1.36% | 30,751 |
| 2.19% | 2.03% | 30,751 |
| 2.92% | 2.91% | 30,751 |
| 3.85% | 3.89% | 30,751 |
| 5.09% | 5.18% | 30,751 |
| 6.89% | 7.42% | 30,751 |
| 9.71% | 10.29% | 30,751 |
| 14.98% | 15.94% | 30,751 |
| 31.69% | 30.91% | 30,751 |

## Risk bands (score percentiles 60 / 85 / 95)

| Band | Score range | Share of applicants | Observed default rate |
|---|---|---|---|
| low | ≥ 0.0000 | 60.0% | 2.7% |
| medium | ≥ 0.0586 | 25.0% | 9.9% |
| high | ≥ 0.1471 | 10.0% | 20.5% |
| very high | ≥ 0.2810 | 5.0% | 38.4% |

## Approval threshold

Assumed costs: a missed default = 5.0, a rejected good customer = 1.0 (to be agreed with the business; full trade-off in `artifacts/thresholds.csv`).

Cost-optimal rule: **approve if score < 0.16**, which approves 86.7% of applicants, catches 46.0% of defaults, and lowers the default rate among approved loans to 5.03% (from 8.07% if all were approved).

## Fairness monitoring

Ranking quality and approval rate by group at the threshold above. Gender and age are used by the model as provided in the dataset; this is monitoring only (see README, limitations).

**gender**

| Group | Applicants | Default rate | AUC | Mean score | Approval rate |
|---|---|---|---|---|---|
| M | 105,059 | 10.1% | 0.785 | 0.099 | 81.4% |
| F | 202,448 | 7.0% | 0.795 | 0.070 | 89.4% |

**age**

| Group | Applicants | Default rate | AUC | Mean score | Approval rate |
|---|---|---|---|---|---|
| <30 | 45,186 | 11.4% | 0.768 | 0.115 | 76.5% |
| 45-60 | 103,302 | 6.6% | 0.791 | 0.066 | 90.6% |
| 30-45 | 123,737 | 9.0% | 0.798 | 0.087 | 84.5% |
| 60+ | 35,286 | 4.9% | 0.757 | 0.048 | 95.6% |
