from __future__ import annotations

import numpy as np

from financetoken.eval.monte_carlo_barriers import (
    calibration_deciles,
    first_touch_np,
    scan_barriers_np,
    _tail,
)


def test_first_touch_only_upper_and_lower():
    open_ = np.array([100.0, 100.0])
    high = np.array([106.0, 100.5])
    low = np.array([99.0, 94.0])
    close = np.array([101.0, 99.0])
    up, dn = first_touch_np(open_, high, low, close, 105.0, 95.0)
    assert up.tolist() == [True, False]
    assert dn.tolist() == [False, True]


def test_first_touch_both_uses_morphology():
    # Alcista: asumimos low antes que high.
    up, dn = first_touch_np(
        np.array([100.0]), np.array([106.0]), np.array([94.0]), np.array([102.0]), 105.0, 95.0
    )
    assert not up[0] and dn[0]
    # Bajista: asumimos high antes que low.
    up, dn = first_touch_np(
        np.array([100.0]), np.array([106.0]), np.array([94.0]), np.array([98.0]), 105.0, 95.0
    )
    assert up[0] and not dn[0]


def test_scan_barriers_long_tp():
    # Entry 100, ATR 2: long TP=103, long SL=98.
    open_ = np.array([[100.0]])
    high = np.array([[103.5]])
    low = np.array([[99.5]])
    close = np.array([[103.0]])
    out = scan_barriers_np(open_, high, low, close, 100.0, 2.0)
    assert out["long_tp"][0] and not out["long_sl"][0]
    assert out["long_time"][0] == 1


def test_scan_barriers_short_tp():
    # Entry 100, ATR 2: short TP=97, short SL=102.
    open_ = np.array([[100.0]])
    high = np.array([[100.5]])
    low = np.array([[96.5]])
    close = np.array([[97.0]])
    out = scan_barriers_np(open_, high, low, close, 100.0, 2.0)
    assert out["short_tp"][0] and not out["short_sl"][0]
    assert out["short_time"][0] == 1


def test_scan_barriers_timeout():
    open_ = np.full((48, 1), 100.0)
    high = np.full((48, 1), 100.5)
    low = np.full((48, 1), 99.5)
    close = np.full((48, 1), 100.0)
    out = scan_barriers_np(open_, high, low, close, 100.0, 2.0)
    assert out["timeout_long"][0] and out["timeout_short"][0]
    assert not out["long_tp"][0] and not out["long_sl"][0]


def test_calibration_deciles_and_tail_ev():
    p = np.repeat([0.1, 0.3, 0.5, 0.7, 0.9], 20)
    y = np.concatenate(
        [
            np.r_[np.ones(2), np.zeros(18)],
            np.r_[np.ones(6), np.zeros(14)],
            np.r_[np.ones(10), np.zeros(10)],
            np.r_[np.ones(14), np.zeros(6)],
            np.r_[np.ones(18), np.zeros(2)],
        ]
    )
    rows, summary = calibration_deciles(p, y)
    assert len(rows) == 10
    assert summary["n"] == 100
    assert summary["ece"] < 0.01
    assert summary["brier"] < 0.20

    n = 20
    pp = np.linspace(0.0, 0.99, n)
    tp = np.zeros(n, dtype=bool)
    sl = np.ones(n, dtype=bool)
    timeout = np.zeros(n, dtype=bool)
    ret = np.zeros(n)
    atr = np.ones(n) * 100.0
    r = _tail(pp, tp, sl, timeout, ret, atr, cost_bps=10.0, side=1)
    assert r["n"] == 2
    assert np.isclose(r["ev_bps"], -110.0)
