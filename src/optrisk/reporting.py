"""Small helpers for Markdown reports and environment capture."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Sequence


def fmt(x: Any, digits: int = 4) -> str:
    """Compact number formatting for tables."""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if math.isnan(x):
            return "n/a"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x == 0.0:
            return "0"
        if abs(x) >= 1e6 or abs(x) < 10 ** (-digits):
            return f"{x:.{max(digits - 2, 1)}e}"
        return f"{x:,.{digits}f}"
    return str(x)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], digits: int = 4, align_right_from: int = 1) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" if i < align_right_from else "---:" for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(fmt(c, digits) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def dict_table(rows: Sequence[dict[str, Any]], digits: int = 4, columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "_no rows_"
    cols = list(columns) if columns else list(rows[0].keys())
    return md_table(cols, [[r.get(c, "") for c in cols] for r in rows], digits)


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict[str, Any]:
    """Hardware and software versions, for reproducibility statements in reports."""
    import numpy
    import scipy

    from . import __version__
    from .validation.quantlib_ref import quantlib_version

    cpu = _sysctl("machdep.cpu.brand_string") if sys.platform == "darwin" else None
    if cpu is None and sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                cpu = next((line.split(":", 1)[1].strip() for line in fh if line.startswith("model name")), None)
        except OSError:
            cpu = None
    mem = _sysctl("hw.memsize") if sys.platform == "darwin" else None
    return {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.machine(),
        "cpu": cpu or platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "memory_gb": round(int(mem) / 2**30, 1) if mem else None,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "quantlib": quantlib_version(),
        "optrisk": __version__,
    }


def environment_table(env: dict[str, Any]) -> str:
    rows = [
        ("CPU", env["cpu"]),
        ("Logical CPUs", env["logical_cpus"]),
        ("Memory", f"{env['memory_gb']} GB" if env["memory_gb"] else "unknown"),
        ("OS", env["os"]),
        ("Python", env["python"]),
        ("NumPy", env["numpy"]),
        ("SciPy", env["scipy"]),
        ("QuantLib", env["quantlib"]),
        ("optrisk", env["optrisk"]),
        ("Run date (UTC)", env["date_utc"]),
    ]
    return md_table(["Item", "Value"], rows, align_right_from=9)
