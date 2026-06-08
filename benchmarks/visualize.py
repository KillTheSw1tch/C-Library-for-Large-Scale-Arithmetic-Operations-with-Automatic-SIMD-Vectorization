#!/usr/bin/env python3
"""
Visualize Google Benchmark results for the SIMD array library.

Reads a JSON file produced by benchmark_suite with:
    --benchmark_out=results.json --benchmark_out_format=json

Produces three PNG figures:
    throughput.png  — Throughput (GB/s) per backend at N=1 000 000 for each op
    speedup.png     — Speedup vs scalar at N=1 000 000 for each backend and op
    scaling.png     — Throughput vs array size (N) for each backend (Add op)

Usage:
    python visualize.py results.json [--outdir .]
"""

import sys
import json
import re
import os
import argparse
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    sys.exit("matplotlib and numpy are required: pip install matplotlib numpy")

# ---------------------------------------------------------------------------
# Constants matching benchmark_suite.cpp
# ---------------------------------------------------------------------------
LARGE_N  = 1_000_000
BACKENDS = ["Scalar", "SSE", "AVX", "AVX512", "Array"]
OPS      = ["Add", "Mul", "Sum", "Dot"]

BACKEND_COLORS = {
    "Scalar": "#5e81ac",
    "SSE":    "#81a1c1",
    "AVX":    "#88c0d0",
    "AVX512": "#8fbcbb",
    "Array":  "#bf616a",
}

BACKEND_LABELS = {
    "Scalar": "Scalar",
    "SSE":    "SSE2",
    "AVX":    "AVX2",
    "AVX512": "AVX-512F",
    "Array":  "simd::Array\n(SIMD+threads)",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Benchmark name format: BM_Suite_<Backend>_<Op>/<N>
_NAME_RE = re.compile(
    r"BM_Suite_(?P<backend>Scalar|SSE|AVX512|AVX|Array)_(?P<op>Add|Mul|Sum|Dot)/(?P<n>\d+)"
)


def parse_results(path):
    """Return a nested dict: data[op][backend][n] = bytes_per_second."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    data = defaultdict(lambda: defaultdict(dict))
    skipped = []

    for bench in raw.get("benchmarks", []):
        if bench.get("run_type") == "aggregate":
            continue
        name = bench["name"]
        m = _NAME_RE.match(name)
        if not m:
            continue
        backend = m.group("backend")
        op      = m.group("op")
        n       = int(m.group("n"))

        if bench.get("skipped"):
            skipped.append(name)
            continue

        bps = bench.get("bytes_per_second")
        if bps is None or bps == 0:
            continue
        data[op][backend][n] = bps

    if skipped:
        print(f"[info] Skipped {len(skipped)} benchmark(s) (CPU feature not available):")
        for s in skipped:
            print(f"       {s}")

    return data


# ---------------------------------------------------------------------------
# Figure 1: Throughput at N=1M
# ---------------------------------------------------------------------------
def plot_throughput(data, outdir):
    ops_present = [op for op in OPS if op in data]
    backends_present = [b for b in BACKENDS if any(b in data[op] for op in ops_present)]

    n_ops = len(ops_present)
    n_be  = len(backends_present)
    x     = np.arange(n_ops)
    width = 0.8 / n_be

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, backend in enumerate(backends_present):
        heights = []
        for op in ops_present:
            bps = data[op].get(backend, {}).get(LARGE_N)
            heights.append(bps / 1e9 if bps else 0.0)
        bars = ax.bar(
            x + (i - n_be / 2 + 0.5) * width,
            heights,
            width,
            label=BACKEND_LABELS[backend],
            color=BACKEND_COLORS[backend],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Operation")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title(f"Throughput by backend — N = {LARGE_N:,} floats")
    ax.set_xticks(x)
    ax.set_xticklabels(ops_present)
    ax.legend(loc="upper left", fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.4)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.2)
    fig.tight_layout()

    path = os.path.join(outdir, "throughput.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Figure 2: Speedup vs Scalar at N=1M
# ---------------------------------------------------------------------------
def plot_speedup(data, outdir):
    ops_present = [op for op in OPS if op in data]
    # All backends except Scalar itself; skip those with no scalar baseline.
    compare_backends = [b for b in BACKENDS if b != "Scalar"]

    n_ops = len(ops_present)
    n_be  = len(compare_backends)
    x     = np.arange(n_ops)
    width = 0.8 / n_be

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, backend in enumerate(compare_backends):
        speedups = []
        for op in ops_present:
            scalar_bps  = data[op].get("Scalar", {}).get(LARGE_N)
            backend_bps = data[op].get(backend, {}).get(LARGE_N)
            if scalar_bps and backend_bps and scalar_bps > 0:
                speedups.append(backend_bps / scalar_bps)
            else:
                speedups.append(0.0)

        bars = ax.bar(
            x + (i - n_be / 2 + 0.5) * width,
            speedups,
            width,
            label=BACKEND_LABELS[backend],
            color=BACKEND_COLORS[backend],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, sp in zip(bars, speedups):
            if sp > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{sp:.1f}×",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Scalar (1×)")
    ax.set_xlabel("Operation")
    ax.set_ylabel("Speedup vs scalar")
    ax.set_title(f"Speedup vs scalar — N = {LARGE_N:,} floats")
    ax.set_xticks(x)
    ax.set_xticklabels(ops_present)
    ax.legend(loc="upper left", fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(outdir, "speedup.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Figure 3: Throughput vs N for the Add operation
# ---------------------------------------------------------------------------
def plot_scaling(data, outdir):
    op = "Add"
    if op not in data:
        print("[info] No 'Add' data — skipping scaling chart")
        return

    backends_present = [b for b in BACKENDS if b in data[op]]
    all_ns = sorted({n for bdata in data[op].values() for n in bdata})
    if not all_ns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for backend in backends_present:
        ns  = sorted(data[op][backend].keys())
        bps = [data[op][backend][n] / 1e9 for n in ns]
        ax.plot(
            ns, bps,
            marker="o", linewidth=1.8,
            label=BACKEND_LABELS[backend],
            color=BACKEND_COLORS[backend],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Array size N (elements)")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title("Throughput scaling with array size — float Add")
    ax.legend(fontsize=8)
    ax.grid(which="major", linestyle="--", alpha=0.4)
    ax.grid(which="minor", linestyle=":", alpha=0.2)

    # Label x-axis with powers of 10 and cache level hints.
    def fmt_n(n, _):
        if n >= 1_000_000:
            return f"{n // 1_000_000}M"
        if n >= 1_000:
            return f"{n // 1_000}K"
        return str(n)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_n))
    fig.tight_layout()

    path = os.path.join(outdir, "scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Summary table (stdout)
# ---------------------------------------------------------------------------
def print_summary(data):
    print()
    print(f"{'Operation':<8} {'Backend':<10} {'N':>10}  {'GB/s':>8}  {'Speedup':>8}")
    print("-" * 54)
    for op in OPS:
        if op not in data:
            continue
        scalar_bps = data[op].get("Scalar", {}).get(LARGE_N)
        for backend in BACKENDS:
            bps = data[op].get(backend, {}).get(LARGE_N)
            if bps is None:
                continue
            speedup = f"{bps / scalar_bps:.1f}×" if scalar_bps else "—"
            print(f"{op:<8} {backend:<10} {LARGE_N:>10,}  {bps / 1e9:>8.2f}  {speedup:>8}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_json", help="JSON file from --benchmark_out")
    parser.add_argument("--outdir", default=".", help="Directory for output PNGs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = parse_results(args.results_json)

    if not data:
        sys.exit("[error] No benchmark_suite results found in the JSON file.\n"
                 "        Make sure you ran benchmark_suite (not benchmark_add/multiply).")

    print_summary(data)
    plot_throughput(data, args.outdir)
    plot_speedup(data, args.outdir)
    plot_scaling(data, args.outdir)


if __name__ == "__main__":
    main()
