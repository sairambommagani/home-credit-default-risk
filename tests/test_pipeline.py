"""End-to-end checks on a small SYNTHETIC dataset with the real Kaggle schema
(scripts/make_synthetic.py): features -> 5-fold training -> artifacts -> scoring API."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hcrisk.features import application_features, build_features, feature_columns  # noqa: E402


@pytest.fixture(scope="session")
def workdir(tmp_path_factory):
    import make_synthetic
    d = tmp_path_factory.mktemp("hc")
    make_synthetic.make(d / "raw", n=3000, n_test=400)
    return d


@pytest.fixture(scope="session")
def features(workdir):
    return build_features(workdir / "raw", workdir / "cache", log=lambda *a: None)


@pytest.fixture(scope="session")
def trained(workdir, features):
    from hcrisk import train
    rep = train.main(["--raw", str(workdir / "raw"), "--cache", str(workdir / "cache"),
                      "--out", str(workdir / "art")])
    return workdir / "art", rep


def test_one_row_per_applicant_and_no_target_leak(features):
    assert features["SK_ID_CURR"].is_unique and len(features) == 3400
    feats = feature_columns(features)
    assert "TARGET" not in feats and "SK_ID_CURR" not in feats
    assert features["TARGET"].notna().sum() == 3000                     # test rows have no label
    for prefix in ("BURO_", "PREV_", "POS_", "INST_", "CC_"):
        assert any(c.startswith(prefix) for c in feats), prefix
    import re
    assert all(re.fullmatch(r"[0-9A-Za-z_]+", c) for c in feats)           # LightGBM-safe names
    assert any("SPOUSE_PARTNER" in c for c in feats) and any("AUDIO_VIDEO" in c for c in feats)
    num = features[feats].select_dtypes("number")
    assert not np.isinf(num.to_numpy(dtype="float64")).any()


def test_days_employed_anomaly_and_ratios():
    app = pd.DataFrame({"SK_ID_CURR": [1, 2], "DAYS_EMPLOYED": [365243, -1000], "DAYS_BIRTH": [-20000, -10000],
                        "AMT_CREDIT": [100.0, 200.0], "AMT_ANNUITY": [10.0, 0.0], "AMT_INCOME_TOTAL": [50.0, 100.0],
                        "CODE_GENDER": ["XNA", "F"]})
    f = application_features(app)
    assert np.isnan(f.loc[0, "DAYS_EMPLOYED"]) and f.loc[0, "DAYS_EMPLOYED_ANOM"] == 1
    assert f.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(2.0)
    assert np.isnan(f.loc[1, "CREDIT_TERM_MONTHS"])                         # division by zero -> NaN
    assert pd.isna(f.loc[0, "CODE_GENDER"])


def test_installment_features_measure_lateness(features, workdir):
    inst = pd.read_csv(workdir / "raw" / "installments_payments.csv")
    late = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    expect = late.groupby(inst["SK_ID_CURR"]).max()
    got = features.set_index("SK_ID_CURR")["INST_DPD_MAX"].dropna()
    common = got.index.intersection(expect.index)[:200]
    assert np.allclose(got.loc[common].to_numpy(), expect.loc[common].to_numpy(), atol=1e-3)


def test_training_learns_signal_and_writes_artifacts(trained):
    art, rep = trained
    assert rep["oof_auc"] > 0.70                                           # synthetic signal is learnable
    assert len(rep["fold_auc"]) == 5 and "baseline" in rep
    for f in ("models/fold0.txt", "metrics.json", "model_meta.json", "deciles.csv", "thresholds.csv",
              "submission.csv", "features.parquet", "oof.parquet"):
        assert (art / f).exists(), f
    bands = rep["risk_bands"]
    rates = [b["observed_default_rate"] for b in bands]
    assert rates[-1] > rates[0]                                            # riskier band, more defaults
    sub = pd.read_csv(art / "submission.csv")
    assert len(sub) == 400 and sub["TARGET"].between(0, 1).all()
    assert abs(np.mean([c["predicted"] for c in rep["calibration"]]) - rep["default_rate"]) < 0.03


def test_api(trained, workdir, monkeypatch):
    art, _ = trained
    monkeypatch.setenv("HC_ARTIFACTS", str(art))
    monkeypatch.setenv("HC_CACHE", str(workdir / "cache"))
    import hcrisk.api as api
    api = importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    info = c.get("/model-info").json()
    assert info["roc_auc"] > 0.7 and info["risk_bands"]
    sid = int(pd.read_parquet(art / "oof.parquet")["SK_ID_CURR"].iloc[0])
    r = c.post("/predict", json={"sk_id_curr": sid}).json()
    assert 0 <= r["default_probability"] <= 1 and r["risk_band"] in ("low", "medium", "high", "very high")
    assert r["factors_increasing_risk"] or r["factors_decreasing_risk"]
    worse = c.post("/predict", json={"sk_id_curr": sid, "overrides": {
        "EXT_SOURCE_1": 0.01, "EXT_SOURCE_2": 0.01, "EXT_SOURCE_3": 0.01}}).json()
    assert worse["default_probability"] > r["default_probability"]        # what-if moves the score
    new = c.post("/predict/new", json={"application": {"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 500000,
                                                      "AMT_ANNUITY": 25000, "DAYS_BIRTH": -14000,
                                                      "EXT_SOURCE_2": 0.5, "CODE_GENDER": "M"}}).json()
    assert new["has_credit_history"] is False and 0 <= new["default_probability"] <= 1
    b = c.post("/predict/batch", json={"ids": [sid, 1]}).json()
    assert len(b["results"]) == 1 and b["not_found"] == [1]
    assert c.post("/predict", json={"sk_id_curr": 1}).status_code == 404


def test_results_report(trained, tmp_path):
    from hcrisk import report
    art, rep = trained
    out = tmp_path / "RESULTS.md"
    report.main(["--artifacts", str(art), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert f"{rep['oof_auc']:.4f}" in text and "Risk bands" in text and "Lift by score decile" in text


def test_exclude_protected_feature(workdir, trained):
    from hcrisk import train
    art = workdir / "art_nogender"
    rep = train.main(["--raw", str(workdir / "raw"), "--cache", str(workdir / "cache"), "--out", str(art),
                      "--skip-baseline", "--exclude", "CODE_GENDER"])
    meta = json.loads((art / "model_meta.json").read_text())
    assert "CODE_GENDER" not in meta["features"] and rep["excluded_features"] == ["CODE_GENDER"]
    assert rep["oof_auc"] > 0.70
