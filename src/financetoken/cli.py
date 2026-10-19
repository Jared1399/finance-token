"""CLI: `ft <comando>`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import load_config
from .data.binance import download
from .data.synth import synthesize
from .eval.report import build_report
from .eval.sweep import run_sweep
from .pipeline import encode, fit_tokenizer, prepare


def _cfg(args):
    path = Path(args.config)
    if not path.exists():
        raise FileNotFoundError(f"no existe el config: {path}")
    return load_config(path)


def cmd_download(args) -> None:
    cfg = _cfg(args)
    print(f"[download] {cfg.describe()}")
    t0 = time.time()
    res = download(cfg, workers=args.workers)
    for symbol, counts in res.items():
        print(f"  {symbol:10s} {dict(sorted(counts.items()))}")
    total = sum(sum(c.values()) for c in res.values())
    print(f"[download] {total} archivos procesados ({time.time()-t0:.1f}s)")


def cmd_synth(args) -> None:
    cfg = _cfg(args)
    minutes = int(args.years * 365.25 * 24 * 60)
    print(f"[synth] {len(cfg.data.symbols)} simbolos, {minutes:,} minutos desde {args.start}")
    t0 = time.time()
    out = synthesize(cfg.raw_dir, cfg.data.symbols, args.start, minutes, seed=args.seed)
    for symbol, n in out.items():
        print(f"  {symbol:10s} {n} archivos mensuales")
    print(f"[synth] listo ({time.time()-t0:.1f}s)")


def cmd_prepare(args) -> None:
    prepare(_cfg(args))


def cmd_fit(args) -> None:
    fit_tokenizer(_cfg(args))


def cmd_encode(args) -> None:
    encode(_cfg(args))


def cmd_report(args) -> None:
    build_report(_cfg(args))


def cmd_sweep(args) -> None:
    cfg = _cfg(args)
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    kvs = tuple(int(x) for x in args.kvs.split(",") if x.strip())
    run_sweep(cfg, ks=ks, kvs=kvs, budget=args.budget)


def cmd_all(args) -> None:
    cfg = _cfg(args)
    cmd_download(args)
    prepare(cfg)
    fit_tokenizer(cfg)
    encode(cfg)
    build_report(cfg)


def cmd_demo(args) -> None:
    cfg = _cfg(args)
    # Aislado del dataset real: correr el demo no debe mezclar velas sinteticas
    # con las descargadas ni pisar los artefactos de produccion.
    from dataclasses import replace

    cfg.data = replace(cfg.data, raw_dir="data/demo/raw", processed_dir="data/demo/processed")
    cfg.artifacts_path = "artifacts/demo"
    cfg.reports_path = "reports/demo"
    print("=" * 72)
    print("DEMO end-to-end con datos sinteticos (mismo formato que Binance 1m)")
    print(f"Salidas aisladas en {cfg.raw_dir}, {cfg.artifacts_path}, {cfg.reports_path}")
    print("=" * 72)
    minutes = int(args.years * 365.25 * 24 * 60)
    print(f"[synth] {len(cfg.data.symbols)} simbolos, {minutes:,} minutos desde {args.start}")
    t0 = time.time()
    out = synthesize(cfg.raw_dir, cfg.data.symbols, args.start, minutes, seed=args.seed)
    for symbol, n in out.items():
        print(f"  {symbol:10s} {n} archivos mensuales")
    print(f"[synth] listo ({time.time()-t0:.1f}s)")
    prepare(cfg)
    fit_tokenizer(cfg)
    encode(cfg)
    build_report(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ft", description="Tokenizador de morfologia de velas")
    parser.add_argument("--version", action="version", version="finance-token 0.1.0")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default="configs/tokenizer.yaml",
        help="ruta al YAML de configuracion (default: configs/tokenizer.yaml)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", parents=[common], help="descarga klines 1m de Binance Vision")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("synth", parents=[common], help="genera datos 1m sinteticos")
    p.add_argument("--years", type=float, default=3.0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("prepare", parents=[common], help="resamplea, valida y calcula features")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("fit-tokenizer", parents=[common], help="ajusta el codebook")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("encode", parents=[common], help="tokeniza todas las velas")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("report", parents=[common], help="genera reporte y figuras")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("sweep", parents=[common], help="barrido de K_s y K_v")
    p.add_argument("--ks", default="128,256,512,1024")
    p.add_argument("--kvs", default="8,16,32,64")
    p.add_argument("--budget", type=int, default=10240, help="presupuesto de vocabulario")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("all", parents=[common], help="download + prepare + fit + encode + report")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("demo", parents=[common], help="pipeline completo con datos sinteticos")
    p.add_argument("--years", type=float, default=3.0)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
