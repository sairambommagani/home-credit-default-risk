"""Train and evaluate the default-risk model.

    python -m hcrisk.train --raw data/raw                 # full data (≈15-25 min on a laptop)
    python -m hcrisk.train --raw data/raw --sample 0.3    # 30% of applicants, for 8 GB RAM / quick runs
    python -m hcrisk.train --raw data/raw --quality high  # slower learning rate, best AUC

Steps
  1. features      all 7 tables -> one row per applicant (hcrisk.features), cached
  2. baseline      LightGBM on the application table only (what one table gives you)
  3. full model    LightGBM on all features, 5-fold stratified CV, early stopping
  4. evaluation    out-of-fold ROC-AUC, PR-AUC, KS, Brier, calibration, decile lift,
                   risk bands, a cost-based approval threshold, fairness check
  5. explainability mean |SHAP| (LightGBM's exact TreeSHAP via pred_contrib)
  6. artifacts     fold models, feature list, category levels, metrics, OOF predictions,
                   Kaggle submission, feature store for the API
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data import load_table, sample_ids
from .features import HISTORY_PREFIXES, build_features, clean_names, feature_columns

SEED = 42

PARAMS = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "learning_rate": 0.05, "num_leaves": 34, "max_depth": 8, "min_child_weight": 40,
    "min_split_gain": 0.02, "reg_alpha": 0.04, "reg_lambda": 0.07,
    "subsample": 0.87, "subsample_freq": 1, "colsample_bytree": 0.5,
    "max_bin": 255, "verbose": -1, "seed": SEED, "num_threads": 0,
}
# No class weights on purpose: AUC doesn't need them, and unweighted training keeps the
# predicted probabilities calibrated (a 0.10 really means ~10% of such applicants default),
# which the risk bands and the cost-based threshold rely on.


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: max gap between the score distributions of defaulters and payers
    (a standard credit-scoring metric)."""
    order = np.argsort(p)
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def decile_table(y, p, n=10) -> pd.DataFrame:
    """Applicants split into 10 equal groups by score: the riskiest decile should hold many
    times the average default rate (lift)."""
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), n, labels=False) + 1        # 1 = safest
    t = d.groupby("decile").agg(applicants=("y", "size"), defaults=("y", "sum"),
                                default_rate=("y", "mean"), mean_score=("p", "mean"),
                                min_score=("p", "min"), max_score=("p", "max"))
    t["lift"] = t["default_rate"] / d["y"].mean()
    t["share_of_all_defaults"] = t["defaults"] / d["y"].sum()
    return t.reset_index()


RISK_BANDS = (("low", 0.60), ("medium", 0.85), ("high", 0.95), ("very high", 1.0))   # by score percentile


def risk_bands(y, p) -> list[dict]:
    """Cut-offs at score percentiles 60 / 85 / 95 of the training population."""
    out, lo = [], 0.0
    qs = np.quantile(p, [q for _, q in RISK_BANDS[:-1]])
    edges = [0.0, *qs.tolist(), 1.0]
    for (name, _), a, b in zip(RISK_BANDS, edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1.0 else (p >= a)
        out.append({"band": name, "min_score": round(float(a), 4), "max_score": round(float(b), 4),
                    "share_of_applicants": round(float(m.mean()), 4),
                    "observed_default_rate": round(float(np.asarray(y)[m].mean()), 4) if m.any() else None})
    return out


def threshold_analysis(y, p, cost_default: float = 5.0, cost_reject_good: float = 1.0) -> dict:
    """Approve when score < t. A missed default costs `cost_default` (lost principal), rejecting a
    good customer costs `cost_reject_good` (lost margin). The ratio is a business assumption to
    agree with the risk team; the table shows the trade-off for every threshold."""
    y = np.asarray(y)
    rows = []
    for t in np.round(np.arange(0.02, 0.51, 0.01), 2):
        approve = p < t
        defaults_approved = int((approve & (y == 1)).sum())
        good_rejected = int((~approve & (y == 0)).sum())
        rows.append({"threshold": float(t), "approval_rate": float(approve.mean()),
                     "default_rate_in_approved": float(y[approve].mean()) if approve.any() else 0.0,
                     "defaults_caught": float(1 - defaults_approved / max(y.sum(), 1)),
                     "cost": float(defaults_approved * cost_default + good_rejected * cost_reject_good) / len(y)})
    best = min(rows, key=lambda r: r["cost"])
    return {"cost_default": cost_default, "cost_reject_good": cost_reject_good, "best": best, "table": rows}


def calibration_table(y, p, n=10) -> list[dict]:
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n, labels=False)
    g = d.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return [{k: round(float(v), 4) if k != "n" else int(v) for k, v in r.items()} for r in g.to_dict("records")]


def fairness(df: pd.DataFrame, y, p, threshold: float) -> dict:
    """Does the model rank equally well, and approve at similar rates, across groups?
    (Gender and age are shown for monitoring; in many jurisdictions they may not be used for
    decisions at all - see README.)"""
    out = {}
    groups = {}
    if "CODE_GENDER" in df:
        groups["gender"] = df["CODE_GENDER"].astype("string").fillna("unknown").to_numpy()
    if "AGE_YEARS" in df:
        groups["age"] = pd.cut(df["AGE_YEARS"], [0, 30, 45, 60, 120],
                               labels=["<30", "30-45", "45-60", "60+"]).astype("string").fillna("unknown").to_numpy()
    y = np.asarray(y)
    for gname, g in groups.items():
        rows = []
        for lvl in pd.unique(g):
            m = g == lvl
            if m.sum() < 50 or len(np.unique(y[m])) < 2:
                continue
            rows.append({"group": str(lvl), "n": int(m.sum()), "default_rate": round(float(y[m].mean()), 4),
                         "auc": round(float(roc_auc_score(y[m], p[m])), 4),
                         "mean_score": round(float(p[m].mean()), 4),
                         "approval_rate": round(float((p[m] < threshold).mean()), 4)})
        out[gname] = rows
    return out


def run_cv(X: pd.DataFrame, y: np.ndarray, params: dict, rounds: int, early: int, log=print, tag=""):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    models, fold_auc = [], []
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        dtr = lgb.Dataset(X.iloc[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        fold_auc.append(roc_auc_score(y[va], oof[va]))
        log(f"  {tag} fold {k}: AUC {fold_auc[-1]:.4f}  ({m.best_iteration} trees)")
        models.append(m)
    return oof, models, fold_auc


def mean_abs_shap(models, X: pd.DataFrame, n: int = 20000) -> pd.Series:
    """Global importance = mean |SHAP value| per feature (exact TreeSHAP built into LightGBM)."""
    Xs = X.sample(min(n, len(X)), random_state=SEED)
    contrib = np.mean([m.predict(Xs, pred_contrib=True, num_iteration=m.best_iteration) for m in models[:2]], axis=0)
    return pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=X.columns).sort_values(ascending=False)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", help="folder with the Kaggle CSV files")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--cache", default="data/cache", help="Parquet cache of the raw tables")
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of applicants (e.g. 0.3 on 8 GB RAM)")
    ap.add_argument("--quality", choices=["fast", "high"], default="fast")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--exclude", default="", help="comma-separated features to leave out, e.g. CODE_GENDER "
                                                   "(protected attributes that fair-lending rules restrict)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)

    # 1. features
    fpath, fmeta = out / "features.parquet", out / "features_meta.json"
    same = fmeta.exists() and json.loads(fmeta.read_text()).get("sample") == a.sample
    if fpath.exists() and same and not a.rebuild_features:
        log("loading cached features", fpath)
        df = pd.read_parquet(fpath)
        before = list(df.columns)
        df = clean_names(df)
        if list(df.columns) != before:                      # cache from an older version: fix and re-save
            df.to_parquet(fpath, index=False)
    else:
        ids = sample_ids(a.raw, a.sample, cache_dir=a.cache)
        log("building features" + (f" on a {a.sample:.0%} sample" if ids else ""))
        df = build_features(a.raw, a.cache, ids, log=log)
        df.to_parquet(fpath, index=False)
        fmeta.write_text(json.dumps({"sample": a.sample}))
    excluded = [c.strip() for c in a.exclude.split(",") if c.strip()]
    feats = [c for c in feature_columns(df) if c not in excluded]
    if excluded:
        log(f"excluding {excluded}")
    cats = {c: [str(x) for x in df[c].cat.categories] for c in feats if isinstance(df[c].dtype, pd.CategoricalDtype)}
    train = df[df["TARGET"].notna()].reset_index(drop=True)
    test = df[df["TARGET"].isna()].reset_index(drop=True)
    y = train["TARGET"].astype(int).to_numpy()
    X = train[feats]
    log(f"train {len(train)} applicants ({y.mean():.2%} default), test {len(test)}, {len(feats)} features")

    params = dict(PARAMS)
    rounds, early = 3000, 100
    if a.quality == "high":
        params.update(learning_rate=0.02)
        rounds, early = 10000, 200

    # 2. baseline: one table only
    report = {"n_train": int(len(train)), "n_test": int(len(test)), "default_rate": float(y.mean()),
              "n_features": len(feats), "excluded_features": excluded, "sample": a.sample, "quality": a.quality, "params": params}
    if not a.skip_baseline:
        base_cols = [c for c in feats if not c.startswith(HISTORY_PREFIXES) and c not in
                     ("CREDIT_TO_BURO_ACTIVE_DEBT", "ANNUITY_PLUS_BURO_ANNUITY_TO_INCOME",
                      "CREDIT_TO_PREV_APPROVED_MEAN", "HAS_ANY_HISTORY")]
        log(f"baseline: application table only ({len(base_cols)} features)")
        boof, _, bfold = run_cv(X[base_cols], y, params, rounds, early, log, "baseline")
        report["baseline"] = {"n_features": len(base_cols), "oof_auc": float(roc_auc_score(y, boof)),
                              "fold_auc": [float(v) for v in bfold]}
        log(f"baseline OOF AUC {report['baseline']['oof_auc']:.4f}")

    # 3. full model
    log(f"full model: all 7 tables ({len(feats)} features)")
    oof, models, fold_auc = run_cv(X, y, params, rounds, early, log, "full")

    # 4. evaluation
    auc = roc_auc_score(y, oof)
    band_info = risk_bands(y, oof)
    thr = threshold_analysis(y, oof)
    report.update({
        "oof_auc": float(auc), "gini": float(2 * auc - 1), "fold_auc": [float(v) for v in fold_auc],
        "fold_auc_std": float(np.std(fold_auc)), "pr_auc": float(average_precision_score(y, oof)),
        "pr_auc_random": float(y.mean()), "ks": ks_statistic(y, oof), "brier": float(brier_score_loss(y, oof)),
        "log_loss": float(log_loss(y, oof)), "best_iterations": [int(m.best_iteration) for m in models],
        "calibration": calibration_table(y, oof), "risk_bands": band_info,
        "threshold": {k: v for k, v in thr.items() if k != "table"},
        "fairness": fairness(train, y, oof, thr["best"]["threshold"]),
    })
    deciles = decile_table(y, oof)
    log(f"FULL OOF AUC {auc:.4f} (Gini {2 * auc - 1:.3f}, PR-AUC {report['pr_auc']:.3f}, KS {report['ks']:.3f})")

    # 5. explainability
    shap_imp = mean_abs_shap(models, X)
    report["top_features"] = [{"feature": f, "mean_abs_shap": round(float(v), 5)} for f, v in shap_imp.head(30).items()]
    by_table = {}
    for f, v in shap_imp.items():
        src = next((p[:-1] for p in HISTORY_PREFIXES if f.startswith(p)), "APPLICATION")
        by_table[src] = by_table.get(src, 0.0) + float(v)
    tot = sum(by_table.values()) or 1.0
    report["importance_by_table"] = {k: round(v / tot, 4) for k, v in sorted(by_table.items(), key=lambda kv: -kv[1])}

    # 6. artifacts
    mdir = out / "models"
    mdir.mkdir(exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(str(mdir / f"fold{i}.txt"), num_iteration=m.best_iteration)
    test_pred = np.mean([m.predict(test[feats], num_iteration=m.best_iteration) for m in models], axis=0) \
        if len(test) else np.array([])
    pd.DataFrame({"SK_ID_CURR": test["SK_ID_CURR"], "TARGET": test_pred}).to_csv(out / "submission.csv", index=False)
    pd.DataFrame({"SK_ID_CURR": train["SK_ID_CURR"], "TARGET": y, "oof": oof}).to_parquet(out / "oof.parquet", index=False)
    deciles.to_csv(out / "deciles.csv", index=False)
    pd.DataFrame(thr["table"]).to_csv(out / "thresholds.csv", index=False)
    shap_imp.rename("mean_abs_shap").to_csv(out / "feature_importance.csv")
    (out / "model_meta.json").write_text(json.dumps({
        "features": feats, "categories": cats, "n_models": len(models), "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "risk_bands": band_info, "threshold": thr["best"]["threshold"], "oof_auc": float(auc),
        "baseline_auc": report.get("baseline", {}).get("oof_auc")}, indent=1))
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    report["seconds"] = round(time.time() - t0)
    log(f"done. artifacts in {out}/ (models, metrics.json, deciles.csv, thresholds.csv, submission.csv)")
    return report


if __name__ == "__main__":
    main()
