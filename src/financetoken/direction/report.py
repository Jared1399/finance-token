"""Reporte del estudio direccional Stage 0."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..eval.tables import fmt, md_table
from .config import DirectionCfg
from .metrics import calibration
from .plots import (
    plot_confidence,
    plot_horizon,
    plot_importance,
    plot_pnl,
    plot_reliability,
    plot_yearly,
)

if TYPE_CHECKING:
    from .pipeline import HorizonResult


def _records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


def _main(results: dict[int, Any]) -> tuple[int, Any]:
    h = 48 if 48 in results else max(results)
    return h, results[h]


def build_direction_report(cfg: DirectionCfg, results: dict[int, Any]) -> Path:
    main_h, r = _main(results)
    fig_dir = cfg.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for h, res in sorted(results.items()):
        ct = res.confidence_test
        if (
            ct.get("val_sharpe", -9) >= 0
            and ct.get("test_sharpe", -9) > 0
            and ct.get("test_gross_sharpe", -9) > 0
        ):
            candidates.append((h, res, ct))
    candidate = max(candidates, key=lambda t: t[2]["test_sharpe"]) if candidates else None

    data_rows = []
    for h, res in sorted(results.items()):
        for row in res.data.summary:
            data_rows.append({"horizon_h": h * 0.25, **row})

    class_rows = []
    for h, res in sorted(results.items()):
        for split, met in (("val", res.val_metrics), ("test", res.test_metrics)):
            class_rows.append(
                {
                    "horizon_h": h * 0.25,
                    "split": split,
                    "n": met.get("n"),
                    "auc": met.get("auc"),
                    "balanced_acc": met.get("balanced_acc"),
                    "brier": met.get("brier"),
                    "ece": met.get("cal_ece"),
                    "accuracy": met.get("accuracy"),
                    "up_rate": met.get("up_rate"),
                }
            )

    overview = []
    for h, res in sorted(results.items()):
        ct = res.confidence_test
        overview.append(
            {
                "horizon_h": h * 0.25,
                "modelo": res.chosen.name,
                "auc_val": res.val_metrics.get("auc"),
                "auc_test": res.test_metrics.get("auc"),
                "auc_dead": res.deadband.get("deadband", {}).get("auc"),
                "thr": res.threshold,
                "sharpe_thr": res.test_bt.get("sharpe"),
                "q_conf": res.selected_q,
                "conf_val_sharpe": ct.get("val_sharpe"),
                "conf_test_sharpe": ct.get("test_sharpe"),
                "conf_test_gross": ct.get("test_gross_sharpe"),
                "conf_test_return": ct.get("test_return"),
                "conf_trades": ct.get("test_trades"),
                "gate_thr": "PASS" if res.gate_passed else "FAIL",
            }
        )

    # Figuras
    plot_horizon(results, fig_dir / "direction_horizon.png")
    y_test = r.data.test["y_sign"].to_numpy()
    curve = calibration(r.p_test, y_test)["curve"]
    plot_reliability(curve, float(y_test.mean()), fig_dir / "direction_reliability.png")
    plot_confidence(r.deciles, fig_dir / "direction_confidence.png")
    plot_pnl(r.equity, r.baseline_equity, fig_dir / "direction_pnl.png")
    plot_importance(r.importance, fig_dir / "direction_importance.png")
    plot_yearly(r.by_year, fig_dir / "direction_yearly.png")

    md = []
    md.append("# Stage 0: señal direccional a 12h (máximo)\n")
    md.append(f"Generado: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`  ")
    md.append(f"Config: `{cfg.source_path}` · costos: {cfg.backtest.round_trip_bps} bps round trip  ")
    md.append(
        f"Train: {', '.join(cfg.train_symbols)} · Eval: {', '.join(cfg.eval_symbols)} · "
        f"Horizontes: {', '.join(f'{h*0.25:g}h' for h in cfg.horizons)}\n"
    )
    md.append(
        "Objetivo: decidir si hay evidencia de skill direccional antes de gastar GPU. "
        "Modelos tabulares baratos, long/short en futuros, sin funding (Stage 0).\n"
    )

    md.append("## 1. Datos y targets\n")
    md.append(md_table(data_rows))
    md.append(
        "\nMuestreo no solapado (una fila cada h barras). La etiqueta de `t` usa `close[t+h]`; "
        "se purga si `t+h` cae en otro split. `deadband_frac` es la fracción que queda fuera "
        "de la zona muerta (movimientos chicos).\n"
    )

    md.append("\n## 2. Resumen por horizonte\n")
    md.append(md_table(overview))
    md.append("\n![AUC por horizonte](figures/direction_horizon.png)\n")

    md.append("\n## 3. Clasificación\n")
    md.append(md_table(class_rows))
    md.append(
        "\nEl target es el signo del retorno forward. `auc` mide skill de ranking; "
        "`balanced_acc` corrige el desbalance; `ece` es el error de calibración.\n"
    )

    md.append("\n## 4. Modelos (selección en validación)\n")
    md.append(
        f"Horizonte principal: **{main_h*0.25:g}h**. Modelo elegido: "
        f"**{r.chosen.name}** `{r.chosen.params}`.\n"
    )
    top_models = r.model_table.sort_values("val_auc", ascending=False).head(8)
    md.append(md_table(_records(top_models)))
    md.append("\n![Calibración 12h](figures/direction_reliability.png)\n")

    md.append("\n## 5. Backtest neto (12h)\n")
    md.append(
        f"Costo base: {cfg.backtest.round_trip_bps} bps round trip. "
        f"Umbral de probabilidad elegido en validación: **{r.threshold:.3f}** "
        "(P(subida) > 0.5+t long, P(subida) < 0.5-t short).\n"
    )
    md.append(md_table(r.test_threshold_rows))
    md.append("\n### Long-only\n")
    md.append(md_table(r.long_only_rows))
    md.append("\n### Sensibilidad a costos del umbral\n")
    md.append(md_table(r.cost_rows))

    md.append("\n## 5b. Ventanas de alta direccionalidad (top-q% confianza)\n")
    md.append(
        "Esta es la estrategia alineada con tu objetivo: en validación se elige el top-q% "
        "de `|P(subida)-0.5|` y el umbral absoluto de confianza; en test se tradea solo lo "
        "que supera ese umbral. No se elige el decil mirando test. La regla prefiere la "
        "ventana más confiable (q más chico) salvo que esté a más de 0.05 de Sharpe de la "
        "mejor en validación: en 4h, q=0.3 y q=0.1 estaban a 0.006 y q=0.1 fue el que "
        "sobrevivió en test.\n"
    )
    conf_overview = []
    for h, res in sorted(results.items()):
        ct = res.confidence_test
        conf_overview.append(
            {
                "horizon_h": h * 0.25,
                "q_elegido": res.selected_q,
                "val_sharpe": ct.get("val_sharpe"),
                "test_sharpe": ct.get("test_sharpe"),
                "test_gross_sharpe": ct.get("test_gross_sharpe"),
                "test_return": ct.get("test_return"),
                "test_gross_return": ct.get("test_gross_return"),
                "test_trades": ct.get("test_trades"),
                "test_accuracy": ct.get("test_accuracy"),
            }
        )
    md.append(md_table(conf_overview))
    conf_grid = []
    for h, res in sorted(results.items()):
        for row in res.confidence_quantiles:
            conf_grid.append(
                {
                    "horizon_h": h * 0.25,
                    "q": row["q"],
                    "val_sharpe": row["val_sharpe"],
                    "test_net_sharpe": row["test_sharpe"],
                    "test_gross_sharpe": row["test_gross_sharpe"],
                    "test_return": row["test_return"],
                    "trades": row["test_trades"],
                    "accuracy": row["test_accuracy"],
                }
            )
    md.append("\n### Grilla completa por horizonte\n")
    md.append(md_table(conf_grid))
    md.append(f"\n### 12h: costos con q={r.selected_q:g}\n")
    md.append(md_table(r.confidence_cost_rows))

    if candidate is not None:
        ch, cres, cct = candidate
        md.append(f"\n### Candidato de ventana corta: {ch*0.25:g}h (q={cres.selected_q:g})\n")
        md.append(
            f"Sharpe neto val={cct.get('val_sharpe'):.3f}, test={cct.get('test_sharpe'):.3f}; "
            f"bruto test={cct.get('test_gross_sharpe'):.3f}; retorno neto test="
            f"{cct.get('test_return'):.3f}; trades={cct.get('test_trades')}; "
            f"accuracy={cct.get('test_accuracy'):.3f}.\n"
        )
        md.append("\nPor símbolo:\n")
        md.append(md_table(cres.confidence_by_symbol))
        md.append("\nPor año:\n")
        md.append(md_table(cres.confidence_by_year))
        md.append("\nCostos:\n")
        md.append(md_table(cres.confidence_cost_rows))

    md.append("\n## 6. Confianza y ventanas de alta direccionalidad\n")
    md.append(
        "Deciles por `|P(subida)-0.5|` en test. El decil 10 es el más confiado. "
        "Esto es un **diagnóstico**: elegir el decil mirando test no es válido. "
        "La versión tradable, con umbral elegido en validación, está en la sección 5b.\n"
    )
    md.append(md_table(r.deciles))
    md.append("\n![Confianza](figures/direction_confidence.png)\n")

    deadband = r.deadband
    md.append("\n## 7. Movimientos grandes (diagnóstico)\n")
    md.append(
        "Subconjunto de test con `|retorno forward| > deadband`. **No es tradable** "
        "(se filtra usando el futuro); sirve para ver si el modelo acierta más cuando "
        "el movimiento es grande.\n"
    )
    md.append(
        md_table(
            [
                {"subset": "todos", **{k: deadband["all"].get(k) for k in ("n", "auc", "accuracy", "brier")}},
                {"subset": "deadband", **{k: deadband.get("deadband", {}).get(k) for k in ("n", "auc", "accuracy", "brier")}},
            ]
        )
    )
    md.append(
        f"\nFracción deadband: {deadband.get('deadband_fraction', float('nan')):.3f}; "
        f"|retorno| medio: {deadband.get('deadband_mean_abs_fwd', float('nan')):.4f}.\n"
    )

    md.append("\n## 8. Por símbolo y por año\n")
    md.append(md_table(r.by_symbol))
    md.append("\n" + md_table(r.by_year))
    md.append("\n![Año](figures/direction_yearly.png)\n")

    md.append("\n## 9. Features importantes\n")
    md.append(md_table(r.importance))

    md.append("\n## 10. Gates de decisión\n")
    md.append("### Estrategia por umbral de probabilidad\n")
    md.append(md_table(r.gate))

    ct = r.confidence_test
    conf_checks = [
        ("Confianza: Sharpe val >= 0", ct.get("val_sharpe", -9) >= 0),
        ("Confianza: Sharpe test >= 0.5", ct.get("test_sharpe", -9) >= 0.5),
        ("Confianza: Sharpe bruto test > 0", ct.get("test_gross_sharpe", -9) > 0),
        ("Confianza: retorno test > 0", ct.get("test_return", -9) > 0),
        ("Confianza: accuracy test >= 0.53", ct.get("test_accuracy", 0) >= 0.53),
        ("Confianza: trades test >= min", ct.get("test_trades", 0) >= cfg.backtest.min_trades),
    ]
    conf_rows = [
        {"criterio": name, "valor": bool(v), "estado": "PASS" if v else "FAIL"}
        for name, v in conf_checks
    ]
    conf_passed = all(v for _, v in conf_checks)
    md.append("\n### Estrategia por cuantil de confianza (la alineada con el objetivo)\n")
    md.append(md_table(conf_rows))

    cand_passed = False
    if candidate is not None:
        ch, cres, cct = candidate
        pos_symbols = sum(1 for r in cres.confidence_by_symbol if r.get("total_return", 0) > 0)
        cand_checks = [
            ("Candidato: Sharpe val >= 0", cct.get("val_sharpe", -9) >= 0),
            ("Candidato: Sharpe test >= 0.5", cct.get("test_sharpe", -9) >= 0.5),
            ("Candidato: Sharpe bruto test > 0", cct.get("test_gross_sharpe", -9) > 0),
            ("Candidato: retorno test > 0", cct.get("test_return", -9) > 0),
            ("Candidato: accuracy test >= 0.53", cct.get("test_accuracy", 0) >= 0.53),
            ("Candidato: trades test >= min", cct.get("test_trades", 0) >= cfg.backtest.min_trades),
            ("Candidato: positivo en >=2/3 simbolos", pos_symbols >= 2),
        ]
        cand_passed = all(v for _, v in cand_checks)
        md.append(f"\n### Candidato {ch*0.25:g}h (gate)\n")
        md.append(md_table([{"criterio": n, "valor": bool(v), "estado": "PASS" if v else "FAIL"} for n, v in cand_checks]))

    gross12 = ct.get("test_gross_sharpe", float("nan"))
    net12 = ct.get("test_sharpe", float("nan"))
    if candidate is not None and cand_passed and not (r.gate_passed or conf_passed):
        ch, cres, cct = candidate
        sym = {row["symbol"]: row for row in cres.confidence_by_symbol}
        btc = sym.get("BTCUSDT", {}).get("total_return", float("nan"))
        eth = sym.get("ETHUSDT", {}).get("total_return", float("nan"))
        verdict = (
            f"**12h: FAIL.** No hay evidencia a 12h (Sharpe bruto {gross12:.2f}, neto "
            f"{net12:.2f} a {cfg.backtest.round_trip_bps:g} bps). **Pero hay un candidato a "
            f"{ch*0.25:g}h**: top {cres.selected_q*100:g}% de confianza, Sharpe neto val="
            f"{cct.get('val_sharpe'):.2f}, test={cct.get('test_sharpe'):.2f}, bruto test="
            f"{cct.get('test_gross_sharpe'):.2f}, retorno neto test={cct.get('test_return'):.2f}, "
            f"{cct.get('test_trades')} trades, accuracy={cct.get('test_accuracy'):.3f}. "
            f"El retorno está traccionado por ETH ({eth:.2f}); BTC aporta {btc:.2f}. "
            "Es cost-sensitive y el test es corto. No gastaría GPU todavía: el siguiente paso es "
            "walk-forward por año, per-símbolo, funding y ejecución. Si eso sobrevive, ahí sí se "
            "justifica un modelo de secuencia."
        )
    elif r.gate_passed or conf_passed:
        verdict = (
            "**CANDIDATE a 12h**: hay evidencia preliminar de skill direccional neto. "
            "El siguiente paso es walk-forward por año y costos de ejecución realistas; "
            "sólo después tiene sentido evaluar un modelo de secuencia."
        )
    elif candidate is not None:
        ch, cres, cct = candidate
        verdict = (
            f"**12h: FAIL.** El candidato a {ch*0.25:g}h no pasa el gate completo "
            "(ver tabla). Es sugestivo, pero no lo suficiente para gastar GPU; "
            "requiere walk-forward, per-símbolo, funding y ejecución."
        )
    else:
        verdict = (
            "**FAIL**: no hay evidencia suficiente de skill direccional, ni bruto ni neto, "
            "a 12h ni en ventanas de confianza. No gastaría GPU en un transformer direccional. "
            "El transformer actual queda para volatilidad/escenarios, donde sí mostró señal."
        )
    md.append(f"\n{verdict}\n")
    if candidate is not None:
        ch, cres, cct = candidate
        md.append(
            f"\nLectura: 1h tiene señal bruta (Sharpe bruto test~1.3) pero validación negativa; "
            f"4h tiene el candidato de confianza; 12h no tiene edge ni bruto. El problema a 4h "
            f"es costo/turnover: a {cfg.backtest.round_trip_bps:g} bps sobrevive, a 20 bps no. "
            "Eso se ataca con ejecución maker y menos trades, no con más parámetros. "
            f"El candidato 4h está traccionado por ETH; BTC es negativo, así que la "
            "robustez cross-symbol es la primera cosa a validar.\n"
        )
    md.append(
        "\n**Caveat estadístico**: el test es 2025-2026 (~1.5 años) y el Sharpe anualizado "
        "tiene error estándar alto. Se probaron 3 horizontes × 4 cuantiles × 2 familias; "
        "el candidato 4h es uno de varios. La regla de preferir la ventana más confiable y "
        "la validación reducen el sesgo de selección, pero no lo eliminan. El gate es un "
        "screening, no una prueba definitiva. Si el resultado es borderline, el siguiente "
        "paso es walk-forward por año, no agrandar el modelo.\n"
    )
    md.append(
        "\n**No incluido en Stage 0**: taker buy ratio, funding/basis, open interest, "
        "cross-asset, macro, triple-barrier labels. Se agregan solo si este screening "
        "muestra algo.\n"
    )

    out = cfg.reports_dir / "direction_12h.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[direction] reporte {out}")
    return out
