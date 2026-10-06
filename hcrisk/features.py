"""Feature engineering: 7 relational tables -> one row per applicant.

Every history table has many rows per applicant (one per old loan, per month, per
payment). A model needs ONE row per applicant, so each table is aggregated to
SK_ID_CURR with count / min / max / mean / sum (and some last-N-months windows),
plus domain features a credit analyst would compute (debt ratios, late payments,
utilisation). Feature names are prefixed by source table:

    (none)  application        (current loan + applicant profile)
    BURO_   bureau + bureau_balance  (loans at other banks, credit-bureau status)
    PREV_   previous_application     (earlier applications at Home Credit)
    POS_    POS_CASH_balance          (monthly snapshots of POS / cash loans)
    INST_   installments_payments     (each instalment: due vs paid, on time vs late)
    CC_     credit_card_balance       (monthly credit card snapshots)

Leakage: aggregations only use the applicant's OWN history rows (all dated before the
application); TARGET is never used in any feature. Train and test applicants go through
exactly the same code.
"""
from __future__ import annotations

import gc
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data import load_table

DAYS_ANOMALY = 365243            # "1000 years" placeholder Home Credit uses for "not applicable"


# ----------------------------------------------------------------------------- helpers
def _safe_div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=(b != 0) & ~np.isnan(b))
    return out.astype(np.float32)


def one_hot(df: pd.DataFrame, max_levels: int = 30, prefix: str = "", by: str = "SK_ID_CURR"
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Categorical columns -> per-applicant SHARE of rows in each level (the most frequent
    `max_levels` levels). Computed with a group-by count, NOT dense one-hot columns: on the real
    previous_application table (1.7M rows) dense dummies would need ~1-2 GB of RAM.
    Returns (frame without the categorical columns, shares indexed by `by`)."""
    cat_cols = [c for c in df.columns if c != by and (isinstance(df[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))]
    parts = []
    n = df.groupby(by).size()
    for c in cat_cols:
        top = df[c].value_counts().index[:max_levels]
        counts = (df.loc[df[c].isin(top), [by, c]].astype({c: "string"})
                  .groupby([by, c]).size().unstack(fill_value=0))
        counts = counts.reindex(index=n.index, fill_value=0).div(n, axis=0)
        counts.columns = [f"{prefix}{c}_{l}_MEAN".upper() for l in counts.columns]
        parts.append(counts.astype(np.float32))
    shares = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=n.index)
    return df.drop(columns=cat_cols), shares


def aggregate(df: pd.DataFrame, by: str, prefix: str, num_aggs: dict[str, list[str]],
              cat_cols: Iterable[str] = ()) -> pd.DataFrame:
    """Group by `by`; numeric columns get the listed aggregations, one-hot columns get mean
    (the share of rows in that category). Column names: PREFIX_COL_AGG."""
    spec = {c: aggs for c, aggs in num_aggs.items() if c in df.columns}
    spec.update({c: ["mean"] for c in cat_cols if c in df.columns})
    g = df.groupby(by).agg(spec)
    g.columns = [f"{prefix}{c}_{a}".upper() for c, a in g.columns]
    return g.astype(np.float32)


def numeric_spec(df: pd.DataFrame, exclude: Iterable[str], aggs=("min", "max", "mean")) -> dict:
    ex = set(exclude)
    return {c: list(aggs) for c in df.columns
            if c not in ex and pd.api.types.is_numeric_dtype(df[c])}


# ----------------------------------------------------------------------------- application
APP_CATEGORICAL_MAX_LEVELS = 60


def application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Current application: clean anomalies + ratios. Categorical columns stay categorical
    (LightGBM handles them natively). Works on 1 row (API) or 356k rows (training)."""
    df = app.copy()
    # anomalies
    if "DAYS_EMPLOYED" in df:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_ANOMALY).astype(np.float32)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != DAYS_ANOMALY)
    if "CODE_GENDER" in df:
        df["CODE_GENDER"] = df["CODE_GENDER"].astype("string").replace("XNA", pd.NA)
    if "DAYS_LAST_PHONE_CHANGE" in df:
        df["DAYS_LAST_PHONE_CHANGE"] = df["DAYS_LAST_PHONE_CHANGE"].where(df["DAYS_LAST_PHONE_CHANGE"] != 0)

    g = lambda c: df[c] if c in df else pd.Series(np.nan, index=df.index)
    # affordability: how big is the loan / instalment compared with income and the goods
    df["CREDIT_INCOME_RATIO"] = _safe_div(g("AMT_CREDIT"), g("AMT_INCOME_TOTAL"))
    df["ANNUITY_INCOME_RATIO"] = _safe_div(g("AMT_ANNUITY"), g("AMT_INCOME_TOTAL"))
    df["PAYMENT_RATE"] = _safe_div(g("AMT_ANNUITY"), g("AMT_CREDIT"))          # ~ 1 / loan term
    df["CREDIT_TERM_MONTHS"] = _safe_div(g("AMT_CREDIT"), g("AMT_ANNUITY"))
    df["GOODS_CREDIT_RATIO"] = _safe_div(g("AMT_GOODS_PRICE"), g("AMT_CREDIT"))
    df["CREDIT_MINUS_GOODS"] = g("AMT_CREDIT") - g("AMT_GOODS_PRICE")
    df["INCOME_PER_PERSON"] = _safe_div(g("AMT_INCOME_TOTAL"), g("CNT_FAM_MEMBERS"))
    df["CHILDREN_RATIO"] = _safe_div(g("CNT_CHILDREN"), g("CNT_FAM_MEMBERS"))
    # life stage / stability
    df["AGE_YEARS"] = -g("DAYS_BIRTH") / 365.25
    df["EMPLOYED_TO_AGE"] = _safe_div(g("DAYS_EMPLOYED"), g("DAYS_BIRTH"))
    df["ID_PUBLISH_TO_AGE"] = _safe_div(g("DAYS_ID_PUBLISH"), g("DAYS_BIRTH"))
    df["REGISTRATION_TO_AGE"] = _safe_div(g("DAYS_REGISTRATION"), g("DAYS_BIRTH"))
    df["PHONE_CHANGE_TO_AGE"] = _safe_div(g("DAYS_LAST_PHONE_CHANGE"), g("DAYS_BIRTH"))
    df["CAR_TO_AGE"] = _safe_div(g("OWN_CAR_AGE"), -g("DAYS_BIRTH") / 365.25)
    # external credit scores: the strongest single signals in this dataset
    ext = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df]
    if ext:
        e = df[ext].astype("float64")
        df["EXT_MEAN"] = e.mean(axis=1)
        df["EXT_STD"] = e.std(axis=1)
        df["EXT_MIN"] = e.min(axis=1)
        df["EXT_MAX"] = e.max(axis=1)
        df["EXT_PROD"] = e.prod(axis=1, min_count=len(ext))
        df["EXT_NAN_COUNT"] = e.isna().sum(axis=1)
        if len(ext) == 3:
            df["EXT_WEIGHTED"] = e["EXT_SOURCE_1"] * 2 + e["EXT_SOURCE_2"] + e["EXT_SOURCE_3"] * 3
        df["EXT_MEAN_X_CREDIT_TERM"] = df["EXT_MEAN"] * df["CREDIT_TERM_MONTHS"]
    docs = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if docs:
        df["DOCUMENT_COUNT"] = df[docs].sum(axis=1)
    bureau_req = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if bureau_req:
        df["BUREAU_REQUESTS_TOTAL"] = df[bureau_req].sum(axis=1)
    df["APP_MISSING_COUNT"] = app.isna().sum(axis=1)

    for c in df.columns:                               # text -> categorical for LightGBM
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("category")
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
    return df


# ----------------------------------------------------------------------------- bureau
def bureau_features(bureau: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    """Loans at other financial institutions, reported to the credit bureau."""
    # bureau_balance: monthly status per bureau loan (C closed, X unknown, 0 = no DPD, 1..5 = DPD buckets)
    # (27M rows on the real data: compare the categorical column directly - no string copy)
    st = bb["STATUS"]
    bb = bb[["SK_ID_BUREAU", "MONTHS_BALANCE"]].copy()
    bb["BB_DPD"] = st.isin(["1", "2", "3", "4", "5"]).to_numpy(dtype=np.float32)
    bb["BB_DPD_SEVERE"] = st.isin(["3", "4", "5"]).to_numpy(dtype=np.float32)
    del st
    bb["BB_RECENT_DPD"] = bb["BB_DPD"].where(bb["MONTHS_BALANCE"] >= -12)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "size"), BB_FIRST_MONTH=("MONTHS_BALANCE", "min"),
        BB_DPD_SHARE=("BB_DPD", "mean"), BB_DPD_SEVERE_ANY=("BB_DPD_SEVERE", "max"),
        BB_RECENT_DPD_SHARE=("BB_RECENT_DPD", "mean"))
    del bb
    b = bureau.merge(bb_agg, left_on="SK_ID_BUREAU", right_index=True, how="left")
    for c in ("DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"):
        if c in b:
            b[c] = b[c].where(b[c] > -40000)
    b["DEBT_CREDIT_RATIO"] = _safe_div(b.get("AMT_CREDIT_SUM_DEBT"), b.get("AMT_CREDIT_SUM"))
    b["OVERDUE_DEBT_RATIO"] = _safe_div(b.get("AMT_CREDIT_SUM_OVERDUE"), b.get("AMT_CREDIT_SUM_DEBT"))
    b["CREDIT_DURATION"] = b["DAYS_CREDIT_ENDDATE"] - b["DAYS_CREDIT"] if "DAYS_CREDIT_ENDDATE" in b else np.nan
    b["ENDDATE_DIFF"] = (b["DAYS_CREDIT_ENDDATE"] - b["DAYS_ENDDATE_FACT"]
                         if "DAYS_ENDDATE_FACT" in b else np.nan)   # closed earlier/later than planned
    b["IS_OVERDUE"] = (b.get("CREDIT_DAY_OVERDUE", 0) > 0).astype(np.float32)

    status = b["CREDIT_ACTIVE"].astype("string") if "CREDIT_ACTIVE" in b else None
    b, shares = one_hot(b.drop(columns=["SK_ID_BUREAU"]), max_levels=12, prefix="BURO_")
    spec = numeric_spec(b, exclude=["SK_ID_CURR"], aggs=("min", "max", "mean"))
    for c in ("AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY",
              "CNT_CREDIT_PROLONG", "IS_OVERDUE"):
        if c in spec:
            spec[c] = spec[c] + ["sum"]
    out = aggregate(b, "SK_ID_CURR", "BURO_", spec).join(shares)
    out["BURO_COUNT"] = b.groupby("SK_ID_CURR").size().astype(np.float32)
    if status is not None:
        for name, mask in (("ACTIVE", status == "Active"), ("CLOSED", status == "Closed")):
            sub = b[mask.fillna(False).to_numpy()]
            keep = {c: ["sum", "mean", "max"] for c in
                    ("AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT",
                     "DEBT_CREDIT_RATIO", "CREDIT_DAY_OVERDUE", "BB_DPD_SHARE") if c in sub}
            a = aggregate(sub, "SK_ID_CURR", f"BURO_{name}_", keep)
            a[f"BURO_{name}_COUNT"] = sub.groupby("SK_ID_CURR").size().astype(np.float32)
            out = out.join(a, how="left")
        out["BURO_ACTIVE_SHARE"] = _safe_div(out.get("BURO_ACTIVE_COUNT"), out["BURO_COUNT"])
        out["BURO_TOTAL_DEBT_TO_CREDIT"] = _safe_div(out.get("BURO_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM"),
                                                     out.get("BURO_ACTIVE_AMT_CREDIT_SUM_SUM"))
    return out


# ----------------------------------------------------------------------------- previous applications
def previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Earlier applications at Home Credit: approved / refused, amounts, terms."""
    p = prev            # modified in place: the caller drops the raw table right after (saves RAM)
    for c in ("DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE",
              "DAYS_TERMINATION"):
        if c in p:
            p[c] = p[c].where(p[c] != DAYS_ANOMALY)
    p["APP_CREDIT_RATIO"] = _safe_div(p.get("AMT_APPLICATION"), p.get("AMT_CREDIT"))  # asked vs granted
    p["CREDIT_GOODS_RATIO"] = _safe_div(p.get("AMT_CREDIT"), p.get("AMT_GOODS_PRICE"))
    p["DOWN_PAYMENT_RATIO"] = _safe_div(p.get("AMT_DOWN_PAYMENT"), p.get("AMT_CREDIT"))
    p["PREV_PAYMENT_RATE"] = _safe_div(p.get("AMT_ANNUITY"), p.get("AMT_CREDIT"))
    if "DAYS_LAST_DUE_1ST_VERSION" in p and "DAYS_LAST_DUE" in p:
        p["LAST_DUE_SHIFT"] = p["DAYS_LAST_DUE"] - p["DAYS_LAST_DUE_1ST_VERSION"]
    status = p["NAME_CONTRACT_STATUS"].astype("string") if "NAME_CONTRACT_STATUS" in p else None
    p, shares = one_hot(p.drop(columns=["SK_ID_PREV"]), max_levels=20, prefix="PREV_")
    spec = numeric_spec(p, exclude=["SK_ID_CURR"])
    out = aggregate(p, "SK_ID_CURR", "PREV_", spec).join(shares)
    out["PREV_COUNT"] = p.groupby("SK_ID_CURR").size().astype(np.float32)
    if status is not None:
        for name, lvl in (("APPROVED", "Approved"), ("REFUSED", "Refused")):
            sub = p[(status == lvl).fillna(False).to_numpy()]
            keep = {c: ["min", "max", "mean"] for c in
                    ("AMT_CREDIT", "AMT_ANNUITY", "APP_CREDIT_RATIO", "DAYS_DECISION", "CNT_PAYMENT",
                     "PREV_PAYMENT_RATE") if c in sub}
            a = aggregate(sub, "SK_ID_CURR", f"PREV_{name}_", keep)
            a[f"PREV_{name}_COUNT"] = sub.groupby("SK_ID_CURR").size().astype(np.float32)
            out = out.join(a, how="left")
        out["PREV_REFUSED_SHARE"] = _safe_div(out.get("PREV_REFUSED_COUNT"), out["PREV_COUNT"])
        recent = p[(p["DAYS_DECISION"] >= -365).to_numpy()] if "DAYS_DECISION" in p else p.iloc[:0]
        out["PREV_LAST_YEAR_COUNT"] = recent.groupby("SK_ID_CURR").size().astype(np.float32)
    return out


# ----------------------------------------------------------------------------- POS / cash
def pos_features(pos: pd.DataFrame) -> pd.DataFrame:
    p = pos
    p["POS_LATE"] = (p.get("SK_DPD", 0) > 0).astype(np.float32)
    p["POS_REMAINING_SHARE"] = _safe_div(p.get("CNT_INSTALMENT_FUTURE"), p.get("CNT_INSTALMENT"))
    p, shares = one_hot(p.drop(columns=["SK_ID_PREV"]), max_levels=10, prefix="POS_")
    spec = numeric_spec(p, exclude=["SK_ID_CURR"], aggs=("max", "mean"))
    if "MONTHS_BALANCE" in spec:
        spec["MONTHS_BALANCE"] = ["min", "max", "size"]
    out = aggregate(p, "SK_ID_CURR", "POS_", spec).join(shares)
    recent = p[(p["MONTHS_BALANCE"] >= -12).to_numpy()]
    out["POS_RECENT_LATE_SHARE"] = recent.groupby("SK_ID_CURR")["POS_LATE"].mean().astype(np.float32)
    return out


# ----------------------------------------------------------------------------- installments
def installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    """Payment behaviour: the most direct evidence of whether someone repays on time."""
    i = inst            # 13.6M rows on the real data: no copy
    i["PAYMENT_RATIO"] = _safe_div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])
    i["PAYMENT_DIFF"] = i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]                    # underpaid amount
    i["DPD"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)      # days late
    i["DBD"] = (i["DAYS_INSTALMENT"] - i["DAYS_ENTRY_PAYMENT"]).clip(lower=0)      # days early
    i["LATE"] = (i["DPD"] > 0).astype(np.float32)
    i["LATE_30"] = (i["DPD"] > 30).astype(np.float32)
    i["UNDERPAID"] = (i["PAYMENT_DIFF"] > 0).astype(np.float32)
    spec = {"NUM_INSTALMENT_VERSION": ["nunique"], "DPD": ["max", "mean", "sum"], "DBD": ["max", "mean"],
            "PAYMENT_RATIO": ["min", "mean", "var"], "PAYMENT_DIFF": ["max", "mean", "sum"],
            "AMT_INSTALMENT": ["max", "mean", "sum"], "AMT_PAYMENT": ["min", "max", "mean", "sum"],
            "DAYS_ENTRY_PAYMENT": ["max", "mean"], "LATE": ["mean", "sum"], "LATE_30": ["mean", "sum"],
            "UNDERPAID": ["mean"]}
    out = aggregate(i, "SK_ID_CURR", "INST_", spec)
    out["INST_COUNT"] = i.groupby("SK_ID_CURR").size().astype(np.float32)
    recent = i[(i["DAYS_INSTALMENT"] >= -365).to_numpy()]                         # last 12 months
    r = aggregate(recent, "SK_ID_CURR", "INST_LAST365_",
                  {"DPD": ["max", "mean"], "LATE": ["mean"], "PAYMENT_RATIO": ["mean"], "PAYMENT_DIFF": ["sum"]})
    return out.join(r, how="left")


# ----------------------------------------------------------------------------- credit cards
def credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    c = cc
    c["UTILIZATION"] = _safe_div(c.get("AMT_BALANCE"), c.get("AMT_CREDIT_LIMIT_ACTUAL"))
    c["PAYMENT_MIN_RATIO"] = _safe_div(c.get("AMT_PAYMENT_CURRENT"), c.get("AMT_INST_MIN_REGULARITY"))
    c["DRAWINGS_LIMIT_RATIO"] = _safe_div(c.get("AMT_DRAWINGS_CURRENT"), c.get("AMT_CREDIT_LIMIT_ACTUAL"))
    c["CC_LATE"] = (c.get("SK_DPD", 0) > 0).astype(np.float32)
    c, shares = one_hot(c.drop(columns=["SK_ID_PREV"]), max_levels=8, prefix="CC_")
    spec = numeric_spec(c, exclude=["SK_ID_CURR"], aggs=("min", "max", "mean"))
    out = aggregate(c, "SK_ID_CURR", "CC_", spec).join(shares)
    out["CC_COUNT"] = c.groupby("SK_ID_CURR").size().astype(np.float32)
    recent = c[(c["MONTHS_BALANCE"] >= -6).to_numpy()]
    r = aggregate(recent, "SK_ID_CURR", "CC_LAST6M_", {"UTILIZATION": ["mean", "max"], "CC_LATE": ["mean"]})
    return out.join(r, how="left")


# ----------------------------------------------------------------------------- full build
HISTORY_PREFIXES = ("BURO_", "PREV_", "POS_", "INST_", "CC_")


def build_features(raw_dir, cache_dir=None, ids: Optional[set] = None, log=print) -> pd.DataFrame:
    """All tables -> one row per applicant (train + test). Returns a frame with SK_ID_CURR,
    TARGET (NaN for test) and features. Tables are loaded one at a time to limit RAM."""
    train = load_table("application_train", raw_dir, cache_dir, ids)
    test = load_table("application_test", raw_dir, cache_dir, ids)
    app = pd.concat([train, test], ignore_index=True)
    for c in app.columns:                                  # concat can un-categorise; restore
        if pd.api.types.is_object_dtype(app[c]):
            app[c] = app[c].astype("category")
    del train, test
    df = application_features(app).set_index("SK_ID_CURR")
    del app
    log(f"application: {df.shape[1]} columns")

    bureau = load_table("bureau", raw_dir, cache_dir, ids)
    bb = load_table("bureau_balance", raw_dir, cache_dir)
    bb = bb[bb["SK_ID_BUREAU"].isin(set(bureau["SK_ID_BUREAU"]))]
    blocks = [("bureau", lambda: bureau_features(bureau, bb))]
    for name, fn in blocks:
        f = fn()
        df = df.join(f, how="left")
        log(f"{name}: +{f.shape[1]} columns")
    del bureau, bb, f
    gc.collect()

    for table, fn in (("previous_application", previous_features), ("pos_cash", pos_features),
                      ("installments", installments_features), ("credit_card", credit_card_features)):
        t = load_table(table, raw_dir, cache_dir, ids)
        f = fn(t)
        df = df.join(f, how="left")
        log(f"{table}: +{f.shape[1]} columns")
        del t, f
        gc.collect()

    # cross-table features: current request vs the applicant's history
    get = lambda c: df[c] if c in df else pd.Series(np.nan, index=df.index)
    hist_counts = [c for c in df.columns if c.endswith("_COUNT") and c.startswith(HISTORY_PREFIXES)]
    cross = pd.DataFrame({
        "CREDIT_TO_BURO_ACTIVE_DEBT": _safe_div(df["AMT_CREDIT"], get("BURO_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM")),
        "ANNUITY_PLUS_BURO_ANNUITY_TO_INCOME": _safe_div(
            df["AMT_ANNUITY"].astype("float64") + get("BURO_AMT_ANNUITY_SUM").fillna(0), df["AMT_INCOME_TOTAL"]),
        "CREDIT_TO_PREV_APPROVED_MEAN": _safe_div(df["AMT_CREDIT"], get("PREV_APPROVED_AMT_CREDIT_MEAN")),
        "HAS_ANY_HISTORY": df[hist_counts].notna().any(axis=1).astype(np.float32),
    }, index=df.index)
    df = pd.concat([df, cross], axis=1)                    # one concat = no fragmented frame
    num = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    df[num] = df[num].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    df = df.copy()
    df = clean_names(df.reset_index())
    log(f"total: {df.shape[0]} applicants x {df.shape[1] - 1} columns")
    return df


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """Feature names may only use letters, digits and _ (LightGBM rejects , : " [ ] { } and
    category values like 'Secondary / secondary special' end up in one-hot names)."""
    import re
    seen: dict[str, int] = {}
    new = []
    for c in df.columns:
        n = re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z_]", "_", str(c))).strip("_")
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        new.append(n)
    df.columns = new
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]
