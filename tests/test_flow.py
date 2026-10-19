from __future__ import annotations

import numpy as np
import pandas as pd

from financetoken.direction.flow_experiment import fit_calibrators, hit_deciles, rank_ic
from financetoken.direction.flows import merge_flow_features


def _price_frame(n=400):
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "close": close,
            "quote_volume": np.full(n, 1000.0),
            "split": "train",
        },
        index=idx,
    )


def test_merge_flow_features_asof_and_causality():
    price = _price_frame()
    idx = price.index
    spot = pd.DataFrame(
        {
            "flow_tbr_4h": np.linspace(0.4, 0.6, len(idx)),
            "flow_cvd_4h": np.zeros(len(idx)),
        },
        index=idx,
    )
    funding = pd.DataFrame({"funding_rate": [0.0001, 0.0002]}, index=pd.to_datetime(["2022-12-31 16:00", "2023-01-01 00:00"], utc=True))
    n_metrics = len(idx) * 3
    metrics = pd.DataFrame(
        {
            "sum_open_interest": np.arange(100, 100 + n_metrics),
            "sum_open_interest_value": np.arange(100, 100 + n_metrics) * 100.0,
            "sum_taker_long_short_vol_ratio": np.ones(n_metrics),
        },
        index=pd.date_range(idx[0] - pd.Timedelta(minutes=10), periods=n_metrics, freq="5min", tz="UTC"),
    )
    out = merge_flow_features(price, spot, funding, metrics)
    assert "flow_tbr_4h" in out.columns
    assert out.index.equals(price.index)
    # Funding as-of: a las 00:00 usa la tasa de 00:00, no la futura.
    t = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    assert np.isclose(out.loc[t, "flow_funding_rate"], 0.0002)
    # OI as-of: 15m usa el ultimo 5m <= t.
    t2 = pd.Timestamp("2023-01-01 00:15", tz="UTC")
    assert out.loc[t2, "flow_oi"] > 0
    assert out.loc[t2, "flow_has_oi"] == 1.0


def test_rank_ic_and_hit_deciles():
    rng = np.random.default_rng(0)
    p = np.linspace(0.2, 0.8, 500)
    fwd = p - 0.5 + rng.normal(0, 0.01, 500)
    rho, pval, n = rank_ic(p, fwd)
    assert rho > 0.9
    assert n == 500
    y = (fwd > 0).astype(int)
    rows = hit_deciles(p, y)
    assert len(rows) == 10
    assert rows[-1]["mean_p"] > rows[0]["mean_p"]


def test_calibration_monotonic_and_bounded():
    rng = np.random.default_rng(1)
    p_val = rng.uniform(0, 1, 1000)
    y_val = (rng.uniform(0, 1, 1000) < p_val).astype(int)
    p_test = rng.uniform(0, 1, 300)
    cal, fits, scores = fit_calibrators(p_val, y_val, ["isotonic", "platt"])
    q = cal.predict(p_test)
    assert q.shape == p_test.shape
    assert ((q >= 0) & (q <= 1)).all()
    # Isotonic es monotona no decreciente.
    iso = fits["isotonic"]
    grid = np.linspace(0, 1, 100)
    qi = iso.predict(grid)
    assert (np.diff(qi) >= -1e-12).all()
    assert "isotonic" in scores and "platt" in scores
