from __future__ import annotations

import numpy as np
import pandas as pd

from financetoken.data.synth import synthesize
from financetoken.eval import metrics
from financetoken.eval.report import build_report
from financetoken.pipeline import encode, fit_tokenizer, load_feature_artifact, load_tokens, prepare
from financetoken.tokenizer import vocab


def test_end_to_end(config_factory):
    cfg = config_factory()
    out = synthesize(cfg.raw_dir, cfg.data.symbols, "2023-01-01", 60_000, seed=1)
    assert out["AAAUSDT"] >= 1

    art = prepare(cfg)
    assert art["fingerprint"] == cfg.fingerprint()
    assert art["embargo_bars"] > cfg.features.atr_long_window
    assert (cfg.features_dir / "AAAUSDT.parquet").exists()

    pq = fit_tokenizer(cfg)
    assert pq.k_s == cfg.codebook.shape.k
    assert pq.k_v == cfg.codebook.magnitude.k
    assert (cfg.artifacts_dir / "tokenizer.npz").exists()
    assert (cfg.artifacts_dir / "tokenizer.json").exists()

    summary = encode(cfg)
    assert summary["AAAUSDT"]["encoded"] > 0

    for symbol in cfg.data.symbols:
        df = load_tokens(cfg, symbol)
        valid = df["token"] >= 0
        assert valid.any()
        assert df.loc[valid, "token"].max() < pq.vocab_size
        assert (df.loc[df["token"] >= vocab.MORPH_OFFSET, "shape_id"] >= 0).all()
        # El sintetico inyecta rachas planas: [FLAT] debe existir a 15m.
        assert (df["token"] == vocab.FLAT).sum() > 0

    # La reconstruccion debe producir velas validas y respetar la forma.
    df = load_tokens(cfg, "AAAUSDT")
    rec = metrics.reconstruction(df, pq)
    good = rec["token"] >= vocab.MORPH_OFFSET
    assert good.sum() > 100
    sub = rec[good]
    assert (sub["h_hat"] >= np.maximum(sub["o_hat"], sub["c_hat"]) - 1e-8).all()
    assert (sub["l_hat"] <= np.minimum(sub["o_hat"], sub["c_hat"]) + 1e-8).all()
    assert np.isfinite(sub["c_hat"]).all()

    report = build_report(cfg)
    text = report.read_text(encoding="utf-8")
    assert "Reporte del tokenizador" in text
    assert "Patrones canonicos" in text
    assert (cfg.reports_dir / "figures" / "shape_map.png").exists()
    assert (cfg.reports_dir / "figures" / "reconstruction.png").exists()


def test_splits_have_embargo(config_factory):
    cfg = config_factory()
    idx = pd.date_range("2023-01-01", "2023-03-10", freq="15min", tz="UTC")
    from financetoken.pipeline import assign_splits

    s = assign_splits(idx, cfg)
    assert (s == "train").sum() > 0
    assert (s == "val").sum() > 0
    # Al inicio de val hay un hueco de embargo, no una transicion directa.
    first_val = idx[s == "val"][0]
    assert first_val >= pd.Timestamp("2023-02-01", tz="UTC") + cfg.embargo_bars * cfg.bar_delta
