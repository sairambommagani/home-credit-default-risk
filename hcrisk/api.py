"""Credit-risk scoring API (FastAPI).

    uvicorn hcrisk.api:app --port 8001

Endpoints
  GET  /health             model loaded?
  GET  /model-info         validation metrics, risk bands, threshold, what drives the model
  POST /predict            score an existing applicant by SK_ID_CURR (history comes from the
                           feature store); optional "overrides" for what-if analysis, e.g.
                           {"sk_id_curr": 100002, "overrides": {"AMT_CREDIT": 300000}}
  POST /predict/new        score a brand-new applicant from application fields only (no credit
                           history yet - history features are missing, which the model handles)
  POST /predict/batch      score up to 1000 applicants by id

Every score returns: default probability, risk band, decision (approve / refer), and the
top factors that raised or lowered the risk for THIS applicant (exact TreeSHAP values from
LightGBM, averaged over the 5 fold models, in log-odds).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import application_features

ART = Path(os.environ.get("HC_ARTIFACTS", "artifacts"))
CACHE = Path(os.environ.get("HC_CACHE", "data/cache"))

DESCRIPTIONS = {
    "EXT_MEAN": "average of the 3 external credit-bureau scores",
    "EXT_MIN": "lowest external credit score", "EXT_MAX": "highest external credit score",
    "EXT_PROD": "product of external credit scores", "EXT_WEIGHTED": "weighted external credit score",
    "EXT_SOURCE_1": "external credit score 1", "EXT_SOURCE_2": "external credit score 2",
    "EXT_SOURCE_3": "external credit score 3",
    "CREDIT_TERM_MONTHS": "loan length (credit / annuity)", "PAYMENT_RATE": "annuity / credit",
    "CREDIT_INCOME_RATIO": "loan amount / income", "ANNUITY_INCOME_RATIO": "yearly repayment / income",
    "GOODS_CREDIT_RATIO": "goods price / loan amount", "AGE_YEARS": "age in years",
    "DAYS_EMPLOYED": "days employed (negative = before application)", "EMPLOYED_TO_AGE": "share of life employed",
    "AMT_ANNUITY": "loan annuity", "AMT_CREDIT": "loan amount", "AMT_GOODS_PRICE": "price of the goods",
    "CODE_GENDER": "gender", "NAME_EDUCATION_TYPE": "education level",
    "INST_DPD_MEAN": "average days late on past instalments", "INST_DPD_MAX": "most days late on a past instalment",
    "INST_LATE_MEAN": "share of past instalments paid late", "INST_PAYMENT_DIFF_SUM": "total underpaid on past instalments",
    "INST_LAST365_DPD_MEAN": "average days late in the last 12 months",
    "BURO_DAYS_CREDIT_MAX": "most recent credit at another bank (days ago)",
    "BURO_ACTIVE_COUNT": "active loans at other banks", "BURO_TOTAL_DEBT_TO_CREDIT": "debt / credit on active external loans",
    "PREV_REFUSED_SHARE": "share of previous applications refused", "PREV_COUNT": "previous Home Credit applications",
    "CC_UTILIZATION_MEAN": "average credit card utilisation", "CC_LAST6M_UTILIZATION_MEAN": "card utilisation, last 6 months",
    "POS_SK_DPD_MAX": "most days past due on POS loans", "DAYS_EMPLOYED_ANOM": "no employment (pensioner / unemployed)",
}
PREFIX_NAMES = {"BURO_": "credit bureau", "PREV_": "previous applications", "POS_": "POS/cash loans",
                "INST_": "instalment payments", "CC_": "credit card"}


def describe(f: str) -> str:
    if f in DESCRIPTIONS:
        return DESCRIPTIONS[f]
    src = next((v for k, v in PREFIX_NAMES.items() if f.startswith(k)), "application")
    return f"{src}: {f}"


class Scorer:
    def __init__(self, art: Path = ART, cache: Path = CACHE):
        meta_path = art / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} missing - run `python -m hcrisk.train` first")
        self.meta = json.loads(meta_path.read_text())
        self.metrics = json.loads((art / "metrics.json").read_text()) if (art / "metrics.json").exists() else {}
        self.features: list[str] = self.meta["features"]
        self.categories: dict[str, list[str]] = self.meta["categories"]
        self.models = [lgb.Booster(model_file=str(p)) for p in sorted((art / "models").glob("fold*.txt"))]
        store = pd.read_parquet(art / "features.parquet")
        self.store = store.set_index("SK_ID_CURR")
        self.history_cols = [c for c in self.features if c.startswith(("BURO_", "PREV_", "POS_", "INST_", "CC_"))]
        self.raw_app: Optional[pd.DataFrame] = None
        parts = [cache / "application_train.parquet", cache / "application_test.parquet"]
        if all(p.exists() for p in parts):                  # raw rows, needed for what-if overrides
            self.raw_app = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).set_index("SK_ID_CURR")

    # ---- shaping ----------------------------------------------------------------------
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.features)
        for c in self.features:
            if c in self.categories:
                X[c] = pd.Categorical(X[c].astype("string").to_numpy(dtype=object), categories=self.categories[c])
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
        return X

    def _from_application(self, raw: dict[str, Any]) -> pd.DataFrame:
        row = pd.DataFrame([raw])
        for c in row.columns:
            if row[c].dtype == object:
                row[c] = row[c].astype("category")
        return application_features(row)

    # ---- scoring ----------------------------------------------------------------------
    def score(self, X: pd.DataFrame, top_k: int = 5) -> list[dict]:
        X = self._prepare(X)
        raw = np.mean([m.predict(X, raw_score=True) for m in self.models], axis=0)
        contrib = np.mean([m.predict(X, pred_contrib=True) for m in self.models], axis=0)
        prob = 1.0 / (1.0 + np.exp(-raw))
        out = []
        for i in range(len(X)):
            c = contrib[i, :-1]
            order = np.argsort(c)
            up = [self._factor(X, i, j, c[j]) for j in order[::-1][:top_k] if c[j] > 0]
            down = [self._factor(X, i, j, c[j]) for j in order[:top_k] if c[j] < 0]
            p = float(prob[i])
            out.append({"default_probability": round(p, 4), "risk_band": self.band(p),
                        "decision": "approve" if p < self.meta["threshold"] else "refer to underwriter",
                        "threshold": self.meta["threshold"],
                        "factors_increasing_risk": up, "factors_decreasing_risk": down,
                        "has_credit_history": bool(X.iloc[i][self.history_cols].notna().any()) if self.history_cols else None})
        return out

    def _factor(self, X, i, j, v) -> dict:
        f = self.features[j]
        val = X.iloc[i, j]
        if isinstance(val, (float, np.floating)):
            val = None if np.isnan(val) else round(float(val), 4)
        elif val is not None and not isinstance(val, (int, str)):
            val = None if pd.isna(val) else str(val)
        return {"feature": f, "description": describe(f), "value": val, "impact_log_odds": round(float(v), 4)}

    def band(self, p: float) -> str:
        for b in self.meta["risk_bands"]:
            if p < b["max_score"] or b is self.meta["risk_bands"][-1]:
                return b["band"]
        return "unknown"

    def by_id(self, sk_id: int, overrides: Optional[dict] = None) -> pd.DataFrame:
        if sk_id not in self.store.index:
            raise KeyError(sk_id)
        row = self.store.loc[[sk_id]].copy()
        if overrides:
            if self.raw_app is None or sk_id not in self.raw_app.index:
                raise RuntimeError("what-if needs the raw application cache (data/cache)")
            raw = self.raw_app.loc[sk_id].to_dict()
            raw.update(overrides)
            raw["SK_ID_CURR"] = sk_id
            app = self._from_application(raw).set_index("SK_ID_CURR")
            for c in app.columns:
                if c in row.columns:
                    row[c] = app[c].to_numpy()
        return row


# ---------------------------------------------------------------------------------- API
app = FastAPI(title="Home Credit Default Risk API", version="1.0",
              description="LightGBM on 7 relational tables; returns default probability, risk band, "
                          "decision and per-applicant explanations.")
_scorer: Optional[Scorer] = None


def scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        try:
            _scorer = Scorer()
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _scorer


class PredictById(BaseModel):
    sk_id_curr: int = Field(..., description="applicant id from application_train / application_test")
    overrides: dict[str, Any] = Field(default_factory=dict,
                                      description="what-if: application fields to change, e.g. AMT_CREDIT")
    model_config = {"json_schema_extra": {"examples": [
        {"sk_id_curr": 100002, "overrides": {}},
        {"sk_id_curr": 100002, "overrides": {"AMT_CREDIT": 200000, "AMT_ANNUITY": 12000}}]}}


class PredictNew(BaseModel):
    application: dict[str, Any] = Field(..., description="application_train columns, e.g. AMT_INCOME_TOTAL, "
                                                         "AMT_CREDIT, AMT_ANNUITY, DAYS_BIRTH, EXT_SOURCE_2 ...")
    model_config = {"json_schema_extra": {"examples": [{"application": {
        "NAME_CONTRACT_TYPE": "Cash loans", "CODE_GENDER": "F", "AMT_INCOME_TOTAL": 180000,
        "AMT_CREDIT": 600000, "AMT_ANNUITY": 30000, "AMT_GOODS_PRICE": 540000, "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2500, "NAME_EDUCATION_TYPE": "Higher education", "CNT_FAM_MEMBERS": 2,
        "EXT_SOURCE_2": 0.62, "EXT_SOURCE_3": 0.55}}]}}


class PredictBatch(BaseModel):
    ids: list[int] = Field(..., max_length=1000)
    model_config = {"json_schema_extra": {"examples": [{"ids": [100002, 100003, 100004]}]}}


@app.get("/health")
def health():
    try:
        s = scorer()
        return {"status": "ok", "models": len(s.models), "features": len(s.features), "applicants": len(s.store)}
    except HTTPException as e:
        return {"status": "not ready", "detail": e.detail}


@app.get("/model-info")
def model_info():
    s = scorer()
    m = s.metrics
    return {"validation": "5-fold stratified cross-validation, out-of-fold predictions",
            "trained_at": s.meta.get("trained_at"), "n_train": m.get("n_train"), "default_rate": m.get("default_rate"),
            "features": len(s.features), "roc_auc": m.get("oof_auc"), "roc_auc_baseline_application_only":
            (m.get("baseline") or {}).get("oof_auc"), "gini": m.get("gini"), "pr_auc": m.get("pr_auc"),
            "ks": m.get("ks"), "brier": m.get("brier"), "risk_bands": s.meta["risk_bands"],
            "approval_threshold": s.meta["threshold"], "threshold_analysis": m.get("threshold"),
            "importance_by_table": m.get("importance_by_table"), "top_features": (m.get("top_features") or [])[:15]}


@app.post("/predict")
def predict(body: PredictById):
    s = scorer()
    try:
        X = s.by_id(body.sk_id_curr, body.overrides)
    except KeyError:
        raise HTTPException(404, f"applicant {body.sk_id_curr} not in the feature store")
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"sk_id_curr": body.sk_id_curr, "what_if": body.overrides or None, **s.score(X)[0]}


@app.post("/predict/new")
def predict_new(body: PredictNew):
    s = scorer()
    if not body.application:
        raise HTTPException(400, "application fields are required")
    X = s._from_application({**body.application, "SK_ID_CURR": -1})
    return {"note": "new applicant: no credit history in the bureau / previous-application tables", **s.score(X)[0]}


@app.post("/predict/batch")
def predict_batch(body: PredictBatch):
    s = scorer()
    known = [i for i in body.ids if i in s.store.index]
    missing = [i for i in body.ids if i not in s.store.index]
    res = s.score(s.store.loc[known], top_k=3) if known else []
    return {"results": [{"sk_id_curr": i, **r} for i, r in zip(known, res)], "not_found": missing}
