"""Write RESULTS.md from the training artifacts (no numbers typed by hand).

    python -m hcrisk.report                # reads artifacts/, writes RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def table(rows: list[dict], cols: list[tuple[str, str, callable]]) -> str:
    out = ["| " + " | ".join(h for h, _, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f(r.get(k)) for _, k, f in cols) + " |")
    return "\n".join(out)


def build(art: Path) -> str:
    m = json.loads((art / "metrics.json").read_text())
    base = m.get("baseline") or {}
    num = lambda d: (lambda v: "–" if v is None else f"{v:.{d}f}")
    s = ["# Results: Home Credit Default Risk", "",
         f"Generated from `{art}/metrics.json`. {m['n_train']:,} training applicants "
         f"({pct(m['default_rate'], 2)} default), {m['n_features']} features, 5-fold stratified CV, "
         "all metrics on out-of-fold predictions."
         + (f" Excluded features: {', '.join(m['excluded_features'])}." if m.get("excluded_features") else ""), "",
         "## Model quality", "",
         "| Model | Features | ROC-AUC | Fold AUCs |", "|---|---|---|---|"]
    if base:
        s.append(f"| Baseline: application table only | {base['n_features']} | {base['oof_auc']:.4f} | "
                 f"{', '.join(f'{v:.4f}' for v in base['fold_auc'])} |")
    s += [f"| All 7 tables | {m['n_features']} | **{m['oof_auc']:.4f}** | "
          f"{', '.join(f'{v:.4f}' for v in m['fold_auc'])} |", ""]
    if base:
        s.append(f"The six history tables add **+{m['oof_auc'] - base['oof_auc']:.4f} AUC** over the application table alone.")
    brier_ref = m["default_rate"] * (1 - m["default_rate"])
    s += ["", "| Metric | Value | Reading |", "|---|---|---|",
          f"| Gini | {m['gini']:.3f} | 2·AUC−1, the usual scorecard measure (>0.5 is good) |",
          f"| PR-AUC | {m['pr_auc']:.3f} | random = {m['pr_auc_random']:.3f}, so {m['pr_auc'] / m['pr_auc_random']:.1f}× better than chance |",
          f"| KS | {m['ks']:.3f} | max separation of defaulter vs payer score distributions (>0.4 is strong) |",
          f"| Brier | {m['brier']:.4f} | {100 * (1 - m['brier'] / brier_ref):.0f}% better than always predicting the base rate ({brier_ref:.4f}) |",
          f"| Log-loss | {m['log_loss']:.4f} | |", ""]

    s += ["## What the model uses", "", "Share of total mean |SHAP| by source table:", "",
          table([{"t": k, "v": v} for k, v in m["importance_by_table"].items()],
                [("Table", "t", str), ("Share", "v", lambda v: pct(v))]), "",
          "Top 20 features (mean |SHAP|, log-odds):", "",
          table(m["top_features"][:20], [("Feature", "feature", lambda v: f"`{v}`"),
                                         ("mean abs SHAP", "mean_abs_shap", num(4))]), ""]

    if (art / "deciles.csv").exists():
        d = pd.read_csv(art / "deciles.csv").to_dict("records")
        top = d[-1]
        s += ["## Lift by score decile", "",
              f"The riskiest 10% of applicants hold **{pct(top['share_of_all_defaults'], 0)} of all defaults** "
              f"(default rate {pct(top['default_rate'])}, {top['lift']:.1f}× average); the safest 10% have "
              f"{pct(d[0]['default_rate'])}.", "",
              table(d, [("Decile (1 = safest)", "decile", lambda v: str(int(v))), ("Applicants", "applicants", lambda v: f"{int(v):,}"),
                        ("Default rate", "default_rate", lambda v: pct(v)), ("Lift", "lift", num(2)),
                        ("Share of all defaults", "share_of_all_defaults", lambda v: pct(v))]), ""]

    s += ["## Calibration (predicted vs observed default rate, 10 bins)", "",
          table(m["calibration"], [("Predicted", "predicted", lambda v: pct(v, 2)), ("Observed", "observed", lambda v: pct(v, 2)),
                                   ("Applicants", "n", lambda v: f"{v:,}")]), "",
          "## Risk bands (score percentiles 60 / 85 / 95)", "",
          table(m["risk_bands"], [("Band", "band", str), ("Score range", "min_score", lambda v: f"≥ {v:.4f}"),
                                  ("Share of applicants", "share_of_applicants", lambda v: pct(v)),
                                  ("Observed default rate", "observed_default_rate", lambda v: pct(v))]), ""]
    t = m["threshold"]
    b = t["best"]
    s += ["## Approval threshold", "",
          f"Assumed costs: a missed default = {t['cost_default']}, a rejected good customer = {t['cost_reject_good']} "
          "(to be agreed with the business; full trade-off in `artifacts/thresholds.csv`).", "",
          f"Cost-optimal rule: **approve if score < {b['threshold']}**, which approves {pct(b['approval_rate'])} of applicants, "
          f"catches {pct(b['defaults_caught'])} of defaults, and lowers the default rate among approved loans to "
          f"{pct(b['default_rate_in_approved'], 2)} (from {pct(m['default_rate'], 2)} if all were approved).", ""]
    s += ["## Fairness monitoring", "",
          "Ranking quality and approval rate by group at the threshold above. Gender and age are used by the model as "
          "provided in the dataset; this is monitoring only (see README, limitations).", ""]
    for g, rows in m["fairness"].items():
        s += [f"**{g}**", "", table(rows, [("Group", "group", str), ("Applicants", "n", lambda v: f"{v:,}"),
                                         ("Default rate", "default_rate", lambda v: pct(v)), ("AUC", "auc", num(3)),
                                         ("Mean score", "mean_score", num(3)), ("Approval rate", "approval_rate", lambda v: pct(v))]), ""]
    return "\n".join(s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="RESULTS.md")
    a = ap.parse_args(argv)
    Path(a.out).write_text(build(Path(a.artifacts)), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
