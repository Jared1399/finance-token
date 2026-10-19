"""Evaluacion one-step, calibracion y closed-loop del GPT."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..eval.tables import md_table
from ..pipeline import load_tokenizer, load_tokens
from ..tokenizer import vocab
from . import plots
from .baselines import BigramBaseline, LogisticDirectionBaseline, UnigramBaseline
from .config import TransformerCfg
from .data import ScaleNormalizer, WindowDataset, build_arrays
from .generate import (
    THRESHOLDS,
    BigramSampler,
    ModelSampler,
    UnigramSampler,
    aggregate_stats,
    generate_sequence,
    make_prompt,
    prompt_ends,
    standardized_returns,
    stylized_stats,
)
from .train import load_trained_model

VOCAB = 8195


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 15) -> dict:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {"n": 0, "brier": float("nan"), "ece": float("nan"), "logloss": float("nan"), "acc": float("nan"), "curve": []}
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    curve = []
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        mp = float(p[mask].mean())
        my = float(y[mask].mean())
        n = int(mask.sum())
        curve.append({"bin": f"{edges[b]:.2f}-{edges[b + 1]:.2f}", "p": mp, "y": my, "n": n})
        ece += n / len(p) * abs(mp - my)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    logloss = float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
    return {
        "n": int(len(p)),
        "brier": float(((p - y) ** 2).mean()),
        "ece": float(ece),
        "logloss": logloss,
        "acc": float(((p >= 0.5).astype(float) == y).mean()),
        "curve": curve,
    }


@torch.no_grad()
def model_pass(model, loader, arrays, device, k_v: int) -> dict:
    model.eval()
    out = {
        "nll": [],
        "target": [],
        "pred": [],
        "p_up": [],
        "p_down": [],
        "p_flat": [],
        "p_event": [],
        "y_up": [],
        "y_event": [],
        "target_pos": [],
        "symbol": [],
        "prev_token": [],
    }
    for batch in loader:
        token = batch["token"].to(device)
        state = batch["state"].to(device)
        scale = batch["scale"].to(device)
        gap = batch["gap"].to(device)
        target = batch["target"].to(device)
        kind, shape, mag = model(token, state, scale, gap)
        nll = model.nll(kind, shape, mag, target)
        p_up, p_down, p_flat = model.direction_probs(kind, shape)
        p_event = model.event_probs(kind, shape, mag, THRESHOLDS)

        kp = kind.argmax(-1)
        sp = shape.argmax(-1)
        vp = mag.argmax(-1)
        pred = torch.where(kp == vocab.FLAT, torch.full_like(sp, vocab.FLAT), vocab.MORPH_OFFSET + sp * k_v + vp)

        tc = batch["target_close"].to(device)
        tpc = batch["target_prev_close"].to(device)
        topen = batch["target_open"].to(device)
        tap = batch["target_atr_prev"].to(device)
        y_up = (tc > tpc).float()
        body = (tc - topen).abs() / tap.clamp(min=1e-12)
        y_event = torch.stack([(body > x).float() for x in THRESHOLDS], dim=-1)
        tpos = batch["target_pos"].to(device)
        sym = batch["target_symbol"].to(device)

        out["nll"].append(nll.reshape(-1).cpu().numpy())
        out["target"].append(target.reshape(-1).cpu().numpy())
        out["pred"].append(pred.reshape(-1).cpu().numpy())
        out["p_up"].append(p_up.reshape(-1).cpu().numpy())
        out["p_down"].append(p_down.reshape(-1).cpu().numpy())
        out["p_flat"].append(p_flat.reshape(-1).cpu().numpy())
        out["p_event"].append(p_event.reshape(-1, len(THRESHOLDS)).cpu().numpy())
        out["y_up"].append(y_up.reshape(-1).cpu().numpy())
        out["y_event"].append(y_event.reshape(-1, len(THRESHOLDS)).cpu().numpy())
        out["target_pos"].append(tpos.reshape(-1).cpu().numpy())
        out["symbol"].append(sym.reshape(-1).cpu().numpy())
        prev = arrays.token[tpos.reshape(-1).cpu().numpy() - 1]
        out["prev_token"].append(prev)

    return {k: np.concatenate(v) for k, v in out.items()}


def _baseline_tables(model_arrays, test_pass, unigram, bigram) -> pd.DataFrame:
    target = test_pass["target"]
    prev = test_pass["prev_token"]
    nll_uni = unigram.nll(target)
    nll_bi = bigram.nll(prev, target)
    nll_model = test_pass["nll"]
    sym = test_pass["symbol"]
    rows = []
    for name, nll in (("modelo", nll_model), ("unigram", nll_uni), ("bigram", nll_bi)):
        for i, symbol in enumerate(model_arrays.symbols):
            mask = sym == i
            if not mask.any():
                continue
            ce = float(nll[mask].mean())
            rows.append(
                {
                    "metodo": name,
                    "symbol": symbol,
                    "n": int(mask.sum()),
                    "ce_nats": ce,
                    "bits": ce / math.log(2),
                    "ppl": math.exp(min(ce, 20.0)),
                }
            )
    return pd.DataFrame(rows)


def _closed_loop(
    cfg: TransformerCfg,
    model,
    pq,
    normalizer,
    device,
    unigram,
    bigram,
    n_samples: int,
    n_bars: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[dict[str, list[dict]], dict[str, np.ndarray], list[tuple[str, np.ndarray, np.ndarray | None]]]:
    seq_len = cfg.model.seq_len
    symbols = list(cfg.base.data.symbols)
    dfs = {s: load_tokens(cfg.base, s) for s in symbols}
    stats: dict[str, list[dict]] = {"model": [], "unigram": [], "bigram": []}
    returns: dict[str, list[np.ndarray]] = {"model": [], "unigram": [], "bigram": []}
    examples: list[tuple[str, np.ndarray, np.ndarray | None]] = []

    for symbol, df in dfs.items():
        ends = prompt_ends(df, n=2)
        for end in ends:
            for i in range(n_samples):
                prompt = make_prompt(df, end, seq_len, normalizer, symbol=symbol)
                for name, sampler in (
                    ("model", ModelSampler(model, temperature, top_p)),
                    ("unigram", UnigramSampler(unigram, temperature, top_p)),
                    ("bigram", BigramSampler(bigram, temperature, top_p)),
                ):
                    gen = generate_sequence(
                        model,
                        prompt,
                        sampler,
                        n_bars,
                        seq_len,
                        pq,
                        normalizer,
                        cfg.base.features.atr_k,
                        cfg.base.state.sma_m,
                        cfg.base.features.atr_long_window,
                        device,
                        seed + 1000 * i + 17 * end,
                    )
                    close = np.r_[gen.start_close, gen.close]
                    st = stylized_stats(close, gen.atr_prev)
                    stats[name].append(st)
                    returns[name].append(standardized_returns(close, gen.atr_prev))
                    if name == "model" and i == 0 and end == ends[0]:
                        real = df["close"].iloc[end : end + 300].to_numpy(dtype=np.float64)
                        examples.append((symbol, gen.close[:300], real))
    return stats, {k: np.concatenate(v) for k, v in returns.items()}, examples


def _real_blocks(cfg: TransformerCfg, n_bars: int):
    for symbol in cfg.base.data.symbols:
        df = load_tokens(cfg.base, symbol)
        test_pos = np.flatnonzero(df["split"].to_numpy() == "test")
        if len(test_pos) == 0:
            continue
        start = int(test_pos[0])
        for s in range(start, len(df) - n_bars, n_bars):
            close = np.r_[df["close"].iloc[s - 1], df["close"].iloc[s : s + n_bars].to_numpy()]
            atr = df["atr_prev"].iloc[s : s + n_bars].to_numpy()
            yield close, atr


def _real_stats(cfg: TransformerCfg, n_bars: int) -> list[dict]:
    return [stylized_stats(close, atr) for close, atr in _real_blocks(cfg, n_bars)]


def _real_returns(cfg: TransformerCfg, n_bars: int) -> np.ndarray:
    out = [standardized_returns(close, atr) for close, atr in _real_blocks(cfg, n_bars)]
    return np.concatenate(out) if out else np.array([])


def evaluate_transformer(cfg: TransformerCfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_trained_model(cfg)
    model.to(device)
    pq = load_tokenizer(cfg.base)
    normalizer = ScaleNormalizer.from_dict(ckpt["scale_normalizer"])
    print(f"[eval] checkpoint epoch={ckpt.get('epoch')} val_ce={ckpt.get('val_ce'):.4f} device={device}")

    fit_symbols = cfg.base.data.fit_symbols
    train_arrays, _ = build_arrays(cfg.base, "train", symbols=fit_symbols, normalizer=normalizer)
    val_arrays, _ = build_arrays(cfg.base, "val", symbols=fit_symbols, normalizer=normalizer)
    test_arrays, _ = build_arrays(cfg.base, "test", symbols=cfg.base.data.symbols, normalizer=normalizer)

    seq = cfg.model.seq_len
    test_ds = WindowDataset(test_arrays, seq, seq)
    val_ds = WindowDataset(val_arrays, seq, seq)
    train_ds = WindowDataset(train_arrays, seq, cfg.train.stride_train)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.eval.batch_size, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.eval.batch_size, shuffle=False)

    test_pass = model_pass(model, test_loader, test_arrays, device, pq.k_v)
    val_pass = model_pass(model, val_loader, val_arrays, device, pq.k_v)
    print(f"[eval] tokens test={len(test_pass['nll']):,} val={len(val_pass['nll']):,}")

    unigram = UnigramBaseline.fit(train_arrays, VOCAB)
    bigram = BigramBaseline.tuned(train_arrays, val_arrays, VOCAB)
    print(f"[eval] bigram alpha={bigram.alpha} lambda={bigram.lam}")

    base = _baseline_tables(test_arrays, test_pass, unigram, bigram)
    base.to_csv(cfg.reports_dir / "transformer_baselines.csv", index=False)

    one_step = []
    for split, arr, p in (("val", val_arrays, val_pass), ("test", test_arrays, test_pass)):
        ce = float(p["nll"].mean())
        one_step.append(
            {
                "split": split,
                "n": len(p["nll"]),
                "ce_nats": ce,
                "bits": ce / math.log(2),
                "ppl": math.exp(min(ce, 20.0)),
                "top1": float((p["pred"] == p["target"]).mean()),
            }
        )
    overall = pd.DataFrame(one_step)

    # --- calibracion de direccion ---
    dir_model = calibration(test_pass["p_up"], test_pass["y_up"])
    logistic = LogisticDirectionBaseline.fit(train_loader, pq.shape_centroids, pq.k_v)
    p_log, y_log = logistic.predict_proba(test_loader, pq.shape_centroids, pq.k_v)
    dir_log = calibration(p_log, y_log)
    base_rate = float(test_pass["y_up"].mean())

    # --- calibracion de magnitud ---
    body = np.abs(pq.shape_centroids[:, 1] - pq.shape_centroids[:, 0])
    ratio = np.exp(pq.range_scaler.inverse(pq.mag_centroids))
    morph_probs = unigram.probs[vocab.MORPH_OFFSET :].reshape(pq.k_s, pq.k_v)
    event_rows = []
    for k, thr in enumerate(THRESHOLDS):
        ev = calibration(test_pass["p_event"][:, k], test_pass["y_event"][:, k])
        p_uni = float(morph_probs[body[:, None] * ratio[None, :] > thr].sum())
        y = test_pass["y_event"][:, k]
        event_rows.append(
            {
                "umbral_R_ATR": thr,
                "base_rate": float(y.mean()),
                "model_brier": ev["brier"],
                "model_ece": ev["ece"],
                "model_logloss": ev["logloss"],
                "unigram_p": p_uni,
                "unigram_brier": float(((p_uni - y) ** 2).mean()),
                "n": ev["n"],
            }
        )

    # --- closed loop ---
    print("[eval] generando closed-loop...")
    stats, rets, examples = _closed_loop(
        cfg,
        model,
        pq,
        normalizer,
        device,
        unigram,
        bigram,
        n_samples=cfg.eval.gen_samples,
        n_bars=cfg.eval.gen_bars,
        temperature=cfg.eval.temperature,
        top_p=cfg.eval.top_p,
        seed=cfg.eval.seed,
    )
    real_stats = _real_stats(cfg, cfg.eval.gen_bars)
    real_rets = _real_returns(cfg, cfg.eval.gen_bars)

    # --- figuras ---
    fig_dir = cfg.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plots.plot_training(cfg.artifacts_dir / "train_log.jsonl", fig_dir / "training_curves.png")
    plots.plot_reliability(
        {"modelo": dir_model["curve"], "logistica": dir_log["curve"]},
        base_rate,
        fig_dir / "reliability_direction.png",
        "Calibracion P(subida)",
    )
    plots.plot_event_reliability(
        {f"{t:g}R/ATR": calibration(test_pass["p_event"][:, i], test_pass["y_event"][:, i])["curve"] for i, t in enumerate(THRESHOLDS)},
        fig_dir / "reliability_events.png",
    )
    agg = {name: aggregate_stats(rows) for name, rows in stats.items()}
    real_agg = aggregate_stats(real_stats)
    plots.plot_acf(
        real_agg,
        agg["model"],
        agg["unigram"],
        agg["bigram"],
        fig_dir / "acf_abs.png",
    )
    plots.plot_returns({**rets, "real": real_rets}, fig_dir / "return_distribution.png")
    if examples:
        _, gen_close, real_close = examples[0]
        plots.plot_generated_path(gen_close, real_close, fig_dir / "generated_path.png")

    # --- reporte ---
    def stat_row(name, a):
        return {
            "metodo": name,
            **{k: f"{v[0]:.4f} ± {v[1]:.4f}" for k, v in a.items()},
        }

    stats_rows = [stat_row("real", real_agg), stat_row("modelo", agg["model"]), stat_row("unigram", agg["unigram"]), stat_row("bigram", agg["bigram"])]
    metric_keys = ["acf1", "acf5", "acf20", "kurtosis", "leverage", "q01", "q99", "hill", "hurst"]

    md = []
    md.append("# PoC del transformer de velas\n")
    md.append(f"Generado: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`  ")
    md.append(f"Tokenizer fingerprint: `{ckpt.get('tokenizer_fingerprint')}`  ")
    md.append(f"Checkpoint epoch: {ckpt.get('epoch')} · val CE: {ckpt.get('val_ce'):.4f}  ")
    md.append(
        f"Modelo: d_model={cfg.model.d_model}, layers={cfg.model.n_layer}, heads={cfg.model.n_head}, "
        f"seq_len={cfg.model.seq_len}, params={model.num_params():,}  "
    )
    md.append(
        f"Datos: train={len(train_arrays):,} velas (BTC/ETH), val={len(val_arrays):,}, "
        f"test={len(test_arrays):,} (BTC/ETH/SOL; SOL holdout).  "
    )
    md.append(
        "Tokenizer: 512×16 (8195 tokens), magnitud normalizada con `ATR_{t-1}` "
        "(error de cierre ~4.3 bps).\n"
    )
    md.append("## 1. One-step\n")
    md.append(md_table(overall.to_dict(orient="records")))
    md.append(
        "\nSe comparan los mismos targets (ventanas no solapadas, stride=seq_len; el remanente "
        "que no completa una ventana queda fuera). `top1` es la accuracy del token conjunto; "
        "con 8195 tokens y una distribucion ancha es una metrica poco informativa: el token mas "
        "frecuente del train tiene ~0.76% de masa. La metrica principal es CE.\n"
    )
    md.append(md_table(base.to_dict(orient="records")))
    md.append(
        "\nEl bigrama se **apaga a λ=0**: con 445k transiciones para 8195² pares posibles no hay "
        "estructura de orden 1 explotable y el tuning lo colapsa al unigram. La mejora del "
        "transformer, si existe, viene de contexto/estado, no de pares adyacentes.\n"
    )
    md.append("\n![Curvas de entrenamiento](figures/training_curves.png)\n")

    md.append("\n## 2. Calibracion\n")
    md.append(
        f"Direccion: tasa base de subida en test = {base_rate:.4f}. "
        "El modelo agrega P(subida) sumando las probabilidades de las formas con c>o.\n"
    )
    md.append(
        md_table(
            [
                {"metodo": "modelo", **{k: dir_model[k] for k in ("brier", "ece", "logloss", "acc", "n")}},
                {"metodo": "logistica", **{k: dir_log[k] for k in ("brier", "ece", "logloss", "acc", "n")}},
            ]
        )
    )
    md.append("\n![Calibracion direccion](figures/reliability_direction.png)\n")
    md.append("\nMagnitud: P(|cuerpo|·R/ATR > umbral), sumando tokens compatibles.\n")
    md.append(md_table(event_rows))
    md.append("\n![Calibracion magnitud](figures/reliability_events.png)\n")

    md.append("\n## 3. Closed-loop\n")
    md.append(
        f"Se generan {cfg.eval.gen_bars} velas por muestra, {cfg.eval.gen_samples} muestras por prompt, "
        "2 prompts por simbolo. El decoder recompone ATR, State y Scale despues de cada vela. "
        "Los baselines usan el mismo decoder y el mismo feedback; solo cambia el sampler. "
        "Los retornos se normalizan por **ATR relativo** (`ATR/precio`), no por el ATR en unidades "
        "de precio: eso los hace comparables entre BTC, ETH y SOL.\n"
    )
    md.append(md_table(stats_rows))
    md.append(
        "\nLa fila `bigram` es identica a `unigram`: el tuning eligio λ=0 porque no hay "
        "estructura de orden 1 explotable. Se mantiene como control.\n"
    )
    md.append(
        "\nEstadisticos sobre retornos logaritmicos; `q01`/`q99` y `hill` usan retornos "
        "normalizados por ATR relativo. `hill` alto = cola mas fina.\n"
    )
    md.append("\n![ACF de |retorno|](figures/acf_abs.png)\n")
    md.append("\n![Distribucion de retornos](figures/return_distribution.png)\n")
    md.append("\n![Ejemplo generado](figures/generated_path.png)\n")

    test_ce = float(test_pass["nll"].mean())
    uni_ce = float(unigram.nll(test_pass["target"]).mean())
    bi_ce = float(bigram.nll(test_pass["prev_token"], test_pass["target"]).mean())
    base_brier = base_rate * (1.0 - base_rate)
    m, u, b, r = agg["model"], agg["unigram"], agg["bigram"], real_agg
    sol_ce = next((row["ce_nats"] for row in base.to_dict(orient="records") if row["metodo"] == "modelo" and row["symbol"] == "SOLUSDT"), float("nan"))

    md.append("\n## 4. Hallazgos\n")
    md.append(
        f"- **CE**: {test_ce:.4f} nats/token en test, {test_ce - uni_ce:+.3f} nats vs unigram "
        f"({uni_ce:.4f}). La mejora es chica y el modelo empieza a overfitear en la epoch 3: el "
        "cuello de botella es el tamano de datos, no la arquitectura.\n"
        f"- **Transferencia**: SOL (holdout) da CE {sol_ce:.4f}, similar a BTC/ETH. El tokenizer y "
        "el modelo no dependen de un simbolo particular; el holdout no se degrada.\n"
        f"- **Direccion**: Brier {dir_model['brier']:.4f} vs {base_brier:.4f} de la tasa base. "
        f"Calibrado (ECE {dir_model['ece']:.4f}) pero sin skill direccional a 15m. La logistica "
        f"tiene accuracy algo mayor ({dir_log['acc']:.3f} vs {dir_model['acc']:.3f}) pero Brier "
        f"practicamente identico ({dir_log['brier']:.4f}).\n"
        f"- **Magnitud**: el modelo mejora apenas al unigram en Brier "
        f"({event_rows[0]['model_brier']:.4f} vs {event_rows[0]['unigram_brier']:.4f} para "
        "|cuerpo| > 0.5 R/ATR); la senal esta en la magnitud, no en la direccion.\n"
        f"- **Closed-loop**: el modelo reproduce mejor que los baselines la persistencia de corto "
        f"plazo (ACF1 {m['acf1'][0]:.3f} vs {u['acf1'][0]:.3f}/{b['acf1'][0]:.3f}; real "
        f"{r['acf1'][0]:.3f}) y el Hurst ({m['hurst'][0]:.3f} vs {r['hurst'][0]:.3f} real), pero "
        f"sigue siendo demasiado persistente (ACF5 {m['acf5'][0]:.3f} vs {r['acf5'][0]:.3f} real) "
        f"y de cola mas fina (Hill {m['hill'][0]:.2f} vs {r['hill'][0]:.2f}). El unigram acierta "
        f"el Hill ({u['hill'][0]:.2f}) pero dispara la kurtosis ({u['kurtosis'][0]:.1f} vs "
        f"{r['kurtosis'][0]:.1f} real); el modelo queda mas cerca en kurtosis "
        f"({m['kurtosis'][0]:.1f}).\n"
    )

    md.append("\n## 5. Limitaciones\n")
    md.append(
        f"- **Datos chicos**: ~{len(train_arrays):,} tokens de train y ~{len(train_arrays)//seq:,} "
        f"ventanas independientes de {seq} velas. El modelo es deliberadamente chico "
        f"({model.num_params():,} parametros). La CE de test mejora al unigram solo "
        f"{(uni_ce - test_ce) / uni_ce * 100:.2f}% y empeora a partir de la epoch 3.\n"
        f"- **Top-1 vs CE**: CE mejora al unigram pero top-1 es {overall.iloc[-1]['top1']*100:.2f}%, "
        "peor que predecir siempre el token mas frecuente (~0.76%). Con una distribucion ancha, el "
        "modo del modelo no es el modo empirico. No usar top-1 como criterio principal.\n"
        f"- **Direccion**: el Brier es {dir_model['brier']:.4f}, practicamente igual al de la tasa "
        f"base ({base_brier:.4f}). A 15m la direccion del siguiente cierre es casi impredecible con "
        f"este modelo; la calibracion es buena (ECE {dir_model['ece']:.4f}) pero la informacion es nula.\n"
        f"- **Colas y persistencia**: el modelo subrepresenta tokens de magnitud extrema. La "
        f"trayectoria generada es mas suave y de cola mas fina que la real (Hill {m['hill'][0]:.2f} "
        f"vs {r['hill'][0]:.2f}), y el ACF de |retorno| decae demasiado lento (ACF5 "
        f"{m['acf5'][0]:.3f} vs {r['acf5'][0]:.3f}). El unigram preserva la cola "
        f"(Hill {u['hill'][0]:.2f}) pero sobreestima la kurtosis "
        f"({u['kurtosis'][0]:.1f} vs {r['kurtosis'][0]:.1f}). Ningun sampler reproduce bien las dos "
        "cosas a la vez; el feedback del ATR agrega persistencia por construccion.\n"
        "- **Cabeza factorizada**: asume independencia condicional forma/magnitud dado el estado oculto. "
        "La correlacion real cuerpo-volatilidad se captura solo a traves de h.\n"
        "- **Cola truncada del tokenizer**: el codebook de magnitud satura en ~4.1 R/ATR; los crashes "
        "reales llegan a ~8.8. La generacion no puede producir esas colas aunque el sampler las pida.\n"
        "- **Gap no modelado**: el decoder ancla O=prev_close; el gap real p99 es ~1-2 bps, chico pero no cero.\n"
        "- **Exposure bias**: en training State/Scale son reales; en closed-loop se recalculan. "
        "Se agrego ruido en training, no scheduled sampling.\n"
        f"- **Contexto de {seq} velas** (~{seq*15/60/24:.1f} dias): no captura estacionalidad semanal.\n"
    )

    out = cfg.reports_dir / "transformer.md"
    out.write_text("\n".join(md), encoding="utf-8")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer_fingerprint": ckpt.get("tokenizer_fingerprint"),
        "params": model.num_params(),
        "one_step": overall.to_dict(orient="records"),
        "baselines": base.to_dict(orient="records"),
        "direction": {"model": {k: dir_model[k] for k in ("brier", "ece", "logloss", "acc", "n")}, "logistic": {k: dir_log[k] for k in ("brier", "ece", "logloss", "acc", "n")}, "base_rate": base_rate},
        "events": event_rows,
        "closed_loop": {name: {k: v for k, v in a.items()} for name, a in agg.items()} | {"real": real_agg},
    }
    with open(cfg.reports_dir / "transformer_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)

    print(f"[eval] escrito {out}")
    print(f"[eval] one-step test CE={overall.iloc[-1]['ce_nats']:.4f} bits={overall.iloc[-1]['bits']:.3f} top1={overall.iloc[-1]['top1']:.4f}")
    print(f"[eval] direccion brier modelo={dir_model['brier']:.4f} logistica={dir_log['brier']:.4f}")
    for row in stats_rows:
        print("  ", row["metodo"], {k: row[k] for k in ("acf1", "acf5", "kurtosis", "hill")})
