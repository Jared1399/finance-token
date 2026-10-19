"""Formateo de tablas Markdown sin dependencias extra."""

from __future__ import annotations

import numpy as np


def fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "si" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "—"
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_sin datos_\n"
    cols = cols or list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"
