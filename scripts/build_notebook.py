"""Writes notebooks/home_credit_risk.ipynb (kept as code so the notebook stays reviewable in git)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Home Credit Default Risk: multi-table credit risk modeling

**Goal:** predict the probability that an applicant will have payment difficulties (`TARGET = 1`, about 8% of
applicants) using 7 relational tables: the current application plus the applicant's history at other banks
(credit bureau) and at Home Credit (previous loans, monthly balances, every instalment payment).

**Why it's harder than a single table:** the history tables have *many rows per applicant* (up to 27M rows in
`bureau_balance`), so they must be aggregated to one row per applicant without leaking the target, and the
~300 raw columns grow to 700+ engineered features. The classes are imbalanced, so accuracy is meaningless:
we use **ROC-AUC** (the competition metric), **PR-AUC**, **KS** and calibration.

**Plan**
1. Look at the data and the table relationships
2. Baseline: application table only
3. Feature engineering across all 7 tables (`hcrisk/features.py`)
4. LightGBM with 5-fold stratified CV (`hcrisk/train.py`)
5. Evaluation: AUC / PR-AUC / KS, lift by decile, calibration, risk bands, business threshold, fairness
6. Explainability: global (SHAP) and for one applicant (the same code the API uses)
""")

code("""
import os, sys, json, gc
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from hcrisk.data import load_table, memory_mb
from hcrisk import train as hc_train

RAW = Path(os.environ.get("HC_RAW", ROOT / "data" / "raw"))        # the Kaggle CSVs
CACHE = Path(os.environ.get("HC_CACHE", ROOT / "data" / "cache"))  # Parquet cache (fast reloads, less RAM)
ART = Path(os.environ.get("HC_ARTIFACTS", ROOT / "artifacts"))
SAMPLE = float(os.environ.get("HC_SAMPLE", "1.0"))   # 0.3 = 30% of applicants (8 GB RAM / quick run)
QUALITY = os.environ.get("HC_QUALITY", "fast")       # "high" = lower learning rate, best AUC, slower
pd.set_option("display.max_columns", 50)
print(RAW, "| sample", SAMPLE, "| quality", QUALITY)
""")

md("## 1. The data\n### Table sizes and memory (after downcasting to float32 / categories)")
code("""
sizes = []
for name in ["application_train", "application_test", "bureau", "bureau_balance", "previous_application",
             "pos_cash", "installments", "credit_card"]:
    t = load_table(name, RAW, CACHE)
    sizes.append({"table": name, "rows": len(t), "columns": t.shape[1], "memory_MB": round(memory_mb(t), 1)})
    del t; gc.collect()
pd.DataFrame(sizes)
""")
md("""
**Keys.** Everything hangs off the applicant `SK_ID_CURR`:

```
application (SK_ID_CURR) ─┬─ bureau (SK_ID_BUREAU) ── bureau_balance        loans at OTHER banks
                          └─ previous_application (SK_ID_PREV) ─┬─ POS_CASH_balance
                                                                ├─ installments_payments
                                                                └─ credit_card_balance   loans at HOME CREDIT
```
""")
code("""
app = load_table("application_train", RAW, CACHE)
print(f"{len(app):,} applicants, default rate {app.TARGET.mean():.2%}")
miss = app.isna().mean().sort_values(ascending=False)
print(f"{(miss > 0.5).sum()} columns are more than 50% missing (mostly building / housing descriptors)")
miss.head(10)
""")
md("### What separates defaulters? (default rate by decile of a few raw signals)")
code("""
fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
for ax, col in zip(axes, ["EXT_SOURCE_3", "EXT_SOURCE_2", "DAYS_BIRTH", "AMT_CREDIT"]):
    d = app[[col, "TARGET"]].dropna()
    d["bin"] = pd.qcut(d[col], 10, duplicates="drop")
    r = d.groupby("bin", observed=True)["TARGET"].mean()
    ax.bar(range(len(r)), r.values); ax.set_title(col); ax.set_xlabel("decile (low -> high)"); ax.set_ylabel("default rate")
plt.tight_layout(); plt.show()
emp = app["DAYS_EMPLOYED"] == 365243
print(f"DAYS_EMPLOYED = 365243 ('1000 years', i.e. not employed) for {emp.mean():.1%} of applicants; "
      f"their default rate is {app.loc[emp, 'TARGET'].mean():.2%} vs {app.loc[~emp, 'TARGET'].mean():.2%}. "
      "We turn it into a flag + missing value instead of a fake number.")
del app; gc.collect()
""")

md("""
## 2-4. Features, baseline and the full model

`hcrisk.train` does it all (the same code the API is trained with):

* **features** (`hcrisk/features.py`) - per table: count / min / max / mean / sum aggregations, one-hot shares
  for categories, last-12-month windows, plus domain features: credit/income, annuity/income, loan term,
  external-score combinations, bureau debt/credit and overdue, refused-application share, days-past-due and
  underpayment on instalments, credit-card utilisation, and cross-table ratios (this loan vs. existing debt).
* **baseline** - LightGBM on the application table only.
* **full model** - LightGBM on all tables, 5-fold stratified CV with early stopping, no class weights
  (keeps probabilities calibrated).
""")
code("""
report = hc_train.main(["--raw", str(RAW), "--cache", str(CACHE), "--out", str(ART),
                        "--sample", str(SAMPLE), "--quality", QUALITY])
""")
code("""
res = pd.DataFrame([
    {"model": "baseline: application table only", "features": report["baseline"]["n_features"],
     "ROC-AUC (OOF)": report["baseline"]["oof_auc"]},
    {"model": "all 7 tables", "features": report["n_features"], "ROC-AUC (OOF)": report["oof_auc"]},
])
res["ROC-AUC (OOF)"] = res["ROC-AUC (OOF)"].round(4)
print(f"Gain from the 6 history tables: +{report['oof_auc'] - report['baseline']['oof_auc']:.4f} AUC")
print(f"fold AUCs {np.round(report['fold_auc'], 4)}  (std {report['fold_auc_std']:.4f})")
print(f"PR-AUC {report['pr_auc']:.3f} (random = {report['pr_auc_random']:.3f}) | KS {report['ks']:.3f} | "
      f"Gini {report['gini']:.3f} | Brier {report['brier']:.4f}")
res
""")

md("## 5. Evaluation\n### ROC and precision-recall curves (out-of-fold)")
code("""
from sklearn.metrics import roc_curve, precision_recall_curve
oof = pd.read_parquet(ART / "oof.parquet")
y, p = oof["TARGET"].to_numpy(), oof["oof"].to_numpy()
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.5))
fpr, tpr, _ = roc_curve(y, p); a1.plot(fpr, tpr); a1.plot([0, 1], [0, 1], "--", c="grey")
a1.set_title(f"ROC (AUC {report['oof_auc']:.3f})"); a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate")
pr, rc, _ = precision_recall_curve(y, p); a2.plot(rc, pr); a2.axhline(y.mean(), ls="--", c="grey")
a2.set_title(f"Precision-recall (AP {report['pr_auc']:.3f})"); a2.set_xlabel("recall"); a2.set_ylabel("precision")
cal = pd.DataFrame(report["calibration"])
a3.plot(cal.predicted, cal.observed, "o-"); m = max(cal.predicted.max(), cal.observed.max())
a3.plot([0, m], [0, m], "--", c="grey"); a3.set_title("Calibration (10 bins)")
a3.set_xlabel("predicted default rate"); a3.set_ylabel("observed default rate")
plt.tight_layout(); plt.show()
""")
md("### Lift by decile: how concentrated are the defaults in the riskiest scores?")
code("""
dec = pd.read_csv(ART / "deciles.csv")
top = dec.iloc[-1]
print(f"The riskiest 10% of applicants hold {top.share_of_all_defaults:.0%} of all defaults "
      f"(default rate {top.default_rate:.1%}, {top.lift:.1f}x the average).")
dec.round(4)
""")
md("### Risk bands and a business threshold")
code("""
display(pd.DataFrame(report["risk_bands"]))
thr = pd.read_csv(ART / "thresholds.csv"); best = report["threshold"]["best"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(thr.threshold, thr.cost); ax[0].axvline(best["threshold"], ls="--", c="red")
ax[0].set_title("expected cost per applicant (missed default = 5, rejected good customer = 1)"); ax[0].set_xlabel("approve if score <")
ax[1].plot(thr.threshold, thr.approval_rate, label="approval rate"); ax[1].plot(thr.threshold, thr.defaults_caught, label="defaults caught")
ax[1].plot(thr.threshold, thr.default_rate_in_approved, label="default rate among approved"); ax[1].legend(); ax[1].set_xlabel("threshold")
plt.tight_layout(); plt.show()
print(f"Cost-optimal threshold {best['threshold']}: approve {best['approval_rate']:.1%} of applicants, "
      f"catch {best['defaults_caught']:.1%} of defaults, default rate among approved {best['default_rate_in_approved']:.2%} "
      f"(vs {report['default_rate']:.2%} if everyone were approved). The 5:1 cost ratio is an assumption to agree with the business.")
""")
md("### Fairness check (monitoring only)")
code("""
for g, rows in report["fairness"].items():
    print(g); display(pd.DataFrame(rows))
""")

md("## 6. Explainability\n### What drives the model (mean |SHAP|), and which tables matter")
code("""
imp = pd.read_csv(ART / "feature_importance.csv", index_col=0)["mean_abs_shap"].head(25)[::-1]
fig, ax = plt.subplots(1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [2, 1]})
ax[0].barh(imp.index, imp.values); ax[0].set_title("top 25 features (mean |SHAP|, log-odds)")
bt = pd.Series(report["importance_by_table"]); ax[1].bar(bt.index, bt.values); ax[1].set_title("share of importance by source table")
plt.tight_layout(); plt.show()
""")
md("### One applicant, explained (the same function the API's `/predict` uses)")
code("""
os.environ["HC_ARTIFACTS"], os.environ["HC_CACHE"] = str(ART), str(CACHE)
from hcrisk.api import Scorer
scorer = Scorer(ART, CACHE)
risky_id = int(oof.sort_values("oof").iloc[-1]["SK_ID_CURR"])
out = scorer.score(scorer.by_id(risky_id))[0]
print(f"applicant {risky_id}: default probability {out['default_probability']:.1%}, band {out['risk_band']}, {out['decision']}")
pd.DataFrame(out["factors_increasing_risk"] + out["factors_decreasing_risk"])
""")
code("""
# what-if: the same applicant asking for half the credit
raw_credit = float(scorer.raw_app.loc[risky_id, "AMT_CREDIT"]) if scorer.raw_app is not None else None
if raw_credit:
    w = scorer.score(scorer.by_id(risky_id, {"AMT_CREDIT": raw_credit / 2, "AMT_ANNUITY":
                                             float(scorer.raw_app.loc[risky_id, "AMT_ANNUITY"]) / 2}))[0]
    print(f"credit {raw_credit:,.0f} -> {raw_credit/2:,.0f}: probability {out['default_probability']:.1%} -> {w['default_probability']:.1%}")
""")

md("""
## Summary

Run the cell below for the numbers of this run. Things to discuss:

* **The history tables add real signal** on top of the application form (baseline vs. full AUC), led by
  instalment behaviour (late / underpaid payments), bureau debt and previous refusals; the external scores
  remain the strongest single features.
* **Ranking quality:** OOF ROC-AUC with a small fold-to-fold spread; the riskiest decile concentrates
  defaults several times above average.
* **Probabilities are calibrated** (no class re-weighting), so risk bands and the cost-based threshold
  are meaningful as rates.
* **Limitations / next steps:** the threshold cost ratio is an assumption; fairness is monitored, not
  enforced; next would be hyper-parameter search (Optuna), more time-window features, a second model type
  (CatBoost) for an ensemble, and monitoring for drift once deployed.
""")
code("""
print(json.dumps({k: report[k] for k in ("n_train", "default_rate", "n_features", "oof_auc", "pr_auc", "ks", "brier")}
                 | {"baseline_auc": report["baseline"]["oof_auc"], "importance_by_table": report["importance_by_table"]},
                 indent=1))
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "home_credit_risk.ipynb"
nbf.write(nb, out)
print("wrote", out)
