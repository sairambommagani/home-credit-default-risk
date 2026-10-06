"""Generate a small SYNTHETIC dataset with the same 8 files, keys and columns as the Kaggle
Home Credit data. Used only by the tests and for a 1-minute smoke run of the whole pipeline
(features -> training -> API) when the real data isn't available. The default signal is
planted in EXT_SOURCE, late instalments and bureau overdue, so a working pipeline must score
clearly above AUC 0.5 on it. Numbers from this data mean nothing about real credit risk.

    python scripts/make_synthetic.py --out data/synthetic --n 4000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(out: Path, n: int = 4000, n_test: int = 600, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    N = n + n_test
    ids = np.arange(100001, 100001 + N)
    risk = rng.normal(0, 1, N)                                  # hidden "true" riskiness

    def ext(noise):
        v = 1 / (1 + np.exp(risk * 1.2 + rng.normal(0, noise, N)))
        v[rng.random(N) < 0.3] = np.nan
        return v.round(4)

    income = np.exp(rng.normal(11.9, 0.5, N)).round(-2)
    credit = (income * rng.uniform(1, 6, N)).round(-2)
    annuity = (credit / rng.choice([12, 24, 36, 48, 60], N) * 1.15).round(1)
    days_emp = -rng.integers(30, 12000, N)
    days_emp[rng.random(N) < 0.15] = 365243
    app = pd.DataFrame({
        "SK_ID_CURR": ids,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], N, p=[0.9, 0.1]),
        "CODE_GENDER": rng.choice(["F", "M", "XNA"], N, p=[0.65, 0.3499, 0.0001]),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"], N), "FLAG_OWN_REALTY": rng.choice(["Y", "N"], N),
        "CNT_CHILDREN": rng.poisson(0.4, N), "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": (credit * rng.uniform(0.8, 1.0, N)).round(-2),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "Pensioner", "State servant"], N),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education",
                                           "Incomplete higher", "Lower secondary"], N),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married", "Civil marriage", "Widow"], N),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Sales staff", "Core staff", "Managers", None], N),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity Type 3", "XNA", "Self-employed", "Medicine"], N),
        "DAYS_BIRTH": -rng.integers(7500, 25000, N), "DAYS_EMPLOYED": days_emp,
        "DAYS_REGISTRATION": -rng.uniform(0, 20000, N).round(), "DAYS_ID_PUBLISH": -rng.integers(0, 7000, N),
        "OWN_CAR_AGE": np.where(rng.random(N) < 0.65, np.nan, rng.integers(0, 30, N)),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, N).astype(float),
        "REGION_RATING_CLIENT": rng.choice([1, 2, 3], N, p=[0.1, 0.75, 0.15]),
        "EXT_SOURCE_1": ext(0.8), "EXT_SOURCE_2": ext(0.6), "EXT_SOURCE_3": ext(0.7),
        "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, N).astype(float),
        "FLAG_DOCUMENT_3": rng.integers(0, 2, N), "FLAG_DOCUMENT_6": rng.integers(0, 2, N),
        "AMT_REQ_CREDIT_BUREAU_YEAR": np.where(rng.random(N) < 0.13, np.nan, rng.poisson(2, N)),
        "AMT_REQ_CREDIT_BUREAU_MON": np.where(rng.random(N) < 0.13, np.nan, rng.poisson(0.3, N)),
    })

    # bureau: 0-8 loans per applicant
    nb = rng.poisson(np.clip(4 - risk, 0.2, 8))
    b_cur = np.repeat(ids, nb)
    b_risk = np.repeat(risk, nb)
    m = len(b_cur)
    b_amt = np.exp(rng.normal(12, 1, m)).round(-2)
    bureau = pd.DataFrame({
        "SK_ID_CURR": b_cur, "SK_ID_BUREAU": np.arange(5000000, 5000000 + m),
        "CREDIT_ACTIVE": rng.choice(["Closed", "Active", "Sold", "Bad debt"], m, p=[0.6, 0.38, 0.015, 0.005]),
        "CREDIT_CURRENCY": rng.choice(["currency 1", "currency 2"], m, p=[0.99, 0.01]),
        "DAYS_CREDIT": -rng.integers(1, 2900, m), "CREDIT_DAY_OVERDUE": np.where(
            rng.random(m) < 0.02 + 0.05 * (b_risk > 1), rng.integers(1, 300, m), 0),
        "DAYS_CREDIT_ENDDATE": rng.integers(-2900, 30000, m).astype(float),
        "DAYS_ENDDATE_FACT": np.where(rng.random(m) < 0.4, np.nan, -rng.integers(1, 2900, m)),
        "AMT_CREDIT_MAX_OVERDUE": np.where(rng.random(m) < 0.6, np.nan, rng.exponential(2000, m)),
        "CNT_CREDIT_PROLONG": rng.poisson(0.01, m), "AMT_CREDIT_SUM": b_amt,
        "AMT_CREDIT_SUM_DEBT": (b_amt * rng.uniform(0, 1, m) * (b_risk > 0)).round(1),
        "AMT_CREDIT_SUM_LIMIT": np.where(rng.random(m) < 0.4, np.nan, 0.0),
        "AMT_CREDIT_SUM_OVERDUE": np.where(rng.random(m) < 0.01 + 0.03 * (b_risk > 1), rng.exponential(5000, m), 0),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan", "Mortgage"], m, p=[.7, .25, .03, .02]),
        "DAYS_CREDIT_UPDATE": -rng.integers(0, 3000, m), "AMT_ANNUITY": np.where(rng.random(m) < 0.7, np.nan,
                                                                                 rng.exponential(10000, m)),
    })
    # bureau_balance: monthly status
    nm = rng.integers(0, 24, m)
    bb_id = np.repeat(bureau["SK_ID_BUREAU"].to_numpy(), nm)
    bb_risk = np.repeat(b_risk, nm)
    k = len(bb_id)
    late = rng.random(k) < 0.03 + 0.06 * (bb_risk > 1)
    status = np.where(late, rng.choice(["1", "2", "3", "5"], k), rng.choice(["0", "C", "X"], k, p=[.5, .4, .1]))
    bb = pd.DataFrame({"SK_ID_BUREAU": bb_id, "MONTHS_BALANCE": -np.concatenate([np.arange(x) for x in nm]),
                       "STATUS": status})

    # previous applications: 0-6 per applicant
    npv = rng.poisson(2.5, N)
    p_cur = np.repeat(ids, npv)
    p_risk = np.repeat(risk, npv)
    q = len(p_cur)
    p_credit = np.exp(rng.normal(11.5, 1, q)).round(-2)
    prev = pd.DataFrame({
        "SK_ID_PREV": np.arange(2000000, 2000000 + q), "SK_ID_CURR": p_cur,
        "NAME_CONTRACT_TYPE": rng.choice(["Consumer loans", "Cash loans", "Revolving loans"], q),
        "AMT_ANNUITY": (p_credit / 24).round(1), "AMT_APPLICATION": (p_credit * rng.uniform(0.9, 1.2, q)).round(-2),
        "AMT_CREDIT": p_credit, "AMT_DOWN_PAYMENT": np.where(rng.random(q) < 0.5, np.nan, p_credit * 0.1),
        "AMT_GOODS_PRICE": p_credit * 0.95,
        "NAME_CONTRACT_STATUS": np.where(rng.random(q) < 0.15 + 0.15 * (p_risk > 0.8), "Refused",
                                         rng.choice(["Approved", "Canceled", "Unused offer"], q, p=[.8, .17, .03])),
        "DAYS_DECISION": -rng.integers(1, 2900, q), "NAME_PAYMENT_TYPE": rng.choice(["Cash through the bank", "XNA"], q),
        "CODE_REJECT_REASON": rng.choice(["XAP", "HC", "LIMIT", "SCO"], q, p=[.8, .1, .05, .05]),
        "NAME_YIELD_GROUP": rng.choice(["middle", "high", "low_normal", "XNA"], q),
        "NAME_TYPE_SUITE": rng.choice(["Unaccompanied", "Spouse, partner", "Family", "Other_A"], q),
        "NAME_GOODS_CATEGORY": rng.choice(["Mobile", "Audio/Video", "Computers", "XNA"], q),
        "CNT_PAYMENT": rng.choice([6, 12, 24, 36, np.nan], q),
        "DAYS_FIRST_DRAWING": np.where(rng.random(q) < 0.6, 365243, -rng.integers(1, 2900, q)),
        "DAYS_FIRST_DUE": -rng.integers(1, 2900, q), "DAYS_LAST_DUE_1ST_VERSION": rng.integers(-2900, 2000, q),
        "DAYS_LAST_DUE": np.where(rng.random(q) < 0.4, 365243, -rng.integers(1, 2900, q)),
        "DAYS_TERMINATION": np.where(rng.random(q) < 0.4, 365243, -rng.integers(1, 2900, q)),
        "NFLAG_INSURED_ON_APPROVAL": rng.choice([0, 1, np.nan], q),
    })
    approved = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"]
    a_risk = risk[approved["SK_ID_CURR"].to_numpy() - 100001]

    # installments: 1-12 payments per approved loan
    ni = rng.integers(1, 12, len(approved))
    i_prev = np.repeat(approved["SK_ID_PREV"].to_numpy(), ni)
    i_cur = np.repeat(approved["SK_ID_CURR"].to_numpy(), ni)
    i_risk = np.repeat(a_risk, ni)
    r = len(i_prev)
    due = -rng.integers(1, 2900, r).astype(float)
    late_days = np.where(rng.random(r) < 0.05 + 0.15 * (i_risk > 0.7), rng.integers(1, 60, r), -rng.integers(0, 10, r))
    inst_amt = rng.exponential(8000, r).round(2)
    inst = pd.DataFrame({
        "SK_ID_PREV": i_prev, "SK_ID_CURR": i_cur, "NUM_INSTALMENT_VERSION": rng.choice([0, 1, 2], r),
        "NUM_INSTALMENT_NUMBER": np.concatenate([np.arange(1, x + 1) for x in ni]), "DAYS_INSTALMENT": due,
        "DAYS_ENTRY_PAYMENT": due + late_days, "AMT_INSTALMENT": inst_amt,
        "AMT_PAYMENT": np.where(rng.random(r) < 0.03 + 0.1 * (i_risk > 1), inst_amt * 0.5, inst_amt),
    })

    # POS / cash monthly snapshots
    npos = rng.integers(1, 10, len(approved))
    pos_prev = np.repeat(approved["SK_ID_PREV"].to_numpy(), npos)
    pos_cur = np.repeat(approved["SK_ID_CURR"].to_numpy(), npos)
    pos_risk = np.repeat(a_risk, npos)
    s = len(pos_prev)
    pos = pd.DataFrame({
        "SK_ID_PREV": pos_prev, "SK_ID_CURR": pos_cur, "MONTHS_BALANCE": -rng.integers(1, 96, s),
        "CNT_INSTALMENT": rng.choice([6.0, 12.0, 24.0], s), "CNT_INSTALMENT_FUTURE": rng.integers(0, 12, s).astype(float),
        "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed", "Signed"], s, p=[.8, .15, .05]),
        "SK_DPD": np.where(rng.random(s) < 0.02 + 0.05 * (pos_risk > 1), rng.integers(1, 100, s), 0),
        "SK_DPD_DEF": np.where(rng.random(s) < 0.01, rng.integers(1, 30, s), 0),
    })

    # credit card monthly snapshots (for ~30% of applicants)
    cc_owner = approved.drop_duplicates("SK_ID_CURR").sample(frac=0.3, random_state=seed)
    ncc = rng.integers(1, 12, len(cc_owner))
    c_prev = np.repeat(cc_owner["SK_ID_PREV"].to_numpy(), ncc)
    c_cur = np.repeat(cc_owner["SK_ID_CURR"].to_numpy(), ncc)
    c_risk = risk[c_cur - 100001]
    t = len(c_prev)
    limit = rng.choice([45000.0, 90000.0, 135000.0, 270000.0], t)
    bal = limit * np.clip(rng.beta(2, 5, t) + 0.3 * (c_risk > 0.8), 0, 1.2)
    cc = pd.DataFrame({
        "SK_ID_PREV": c_prev, "SK_ID_CURR": c_cur, "MONTHS_BALANCE": -rng.integers(1, 96, t),
        "AMT_BALANCE": bal.round(2), "AMT_CREDIT_LIMIT_ACTUAL": limit,
        "AMT_DRAWINGS_ATM_CURRENT": np.where(rng.random(t) < 0.2, np.nan, rng.exponential(3000, t)),
        "AMT_DRAWINGS_CURRENT": rng.exponential(4000, t), "AMT_INST_MIN_REGULARITY": (bal * 0.05).round(2),
        "AMT_PAYMENT_CURRENT": np.where(rng.random(t) < 0.2, np.nan, bal * rng.uniform(0, 0.3, t)),
        "AMT_PAYMENT_TOTAL_CURRENT": bal * rng.uniform(0, 0.3, t), "AMT_RECEIVABLE_PRINCIPAL": bal * 0.9,
        "AMT_TOTAL_RECEIVABLE": bal, "CNT_DRAWINGS_CURRENT": rng.poisson(1, t),
        "NAME_CONTRACT_STATUS": rng.choice(["Active", "Completed"], t, p=[.95, .05]),
        "SK_DPD": np.where(rng.random(t) < 0.02 + 0.05 * (c_risk > 1), rng.integers(1, 60, t), 0),
        "SK_DPD_DEF": np.where(rng.random(t) < 0.01, rng.integers(1, 10, t), 0),
    })

    # TARGET: ~8% defaults driven by the hidden risk + observable behaviour
    logit = -3.1 + 1.3 * risk + rng.normal(0, 0.6, N)
    target = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
    train = app.iloc[:n].copy()
    train.insert(1, "TARGET", target[:n])
    train.to_csv(out / "application_train.csv", index=False)
    app.iloc[n:].to_csv(out / "application_test.csv", index=False)
    bureau.to_csv(out / "bureau.csv", index=False)
    bb.to_csv(out / "bureau_balance.csv", index=False)
    prev.to_csv(out / "previous_application.csv", index=False)
    pos.to_csv(out / "POS_CASH_balance.csv", index=False)
    inst.to_csv(out / "installments_payments.csv", index=False)
    cc.to_csv(out / "credit_card_balance.csv", index=False)
    print(f"synthetic data in {out}: {n} train ({target[:n].mean():.1%} default), {n_test} test, "
          f"{m} bureau, {k} bureau_balance, {q} previous, {r} installments, {s} POS, {t} credit-card rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=600)
    a = ap.parse_args()
    make(Path(a.out), a.n, a.n_test)
