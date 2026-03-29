#!/usr/bin/env python3
"""
Benchmark script for the parallel Game of Life MPI project.
Runs WITHOUT pygame (no display) to measure pure computation + communication performance.

Usage:
    python3 benchmark.py
    python3 benchmark.py --quick          # Reduced grid sizes for fast testing
    python3 benchmark.py --test-only      # Only run correctness tests
    python3 benchmark.py --no-plots       # Skip plot generation

The MPI scripts (game_of_life_mpi_row.py, game_of_life_mpi_col.py,
game_of_life_mpi_block.py) must support a --benchmark mode:
    mpirun -n X python3 game_of_life_mpi_row.py --benchmark NY NX N_ITER [SEED]
which disables pygame, uses a random grid with the given seed, and prints:
    TIME_COMPUTE=xxx TIME_COMM=yyy TIME_TOTAL=zzz
    GRID=<base64-encoded numpy array>   (only for correctness tests)
"""

import argparse
import base64
import io
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.signal import convolve2d

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Serial Game of Life (reference implementation)
# ---------------------------------------------------------------------------

def serial_game_of_life(cells, n_iterations):
    """Run the Game of Life serially and return the final grid."""
    ny, nx = cells.shape
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    for _ in range(n_iterations):
        voisins = convolve2d(cells, kernel, mode="same", boundary="wrap")
        new = np.zeros_like(cells)
        new[(cells == 1) & ((voisins == 2) | (voisins == 3))] = 1
        new[(cells == 0) & (voisins == 3)] = 1
        cells = new
    return cells


def serial_game_of_life_timed(cells, n_iterations):
    """Run serial GoL and return (final_grid, elapsed_time)."""
    t0 = time.perf_counter()
    result = serial_game_of_life(cells, n_iterations)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def make_grid(ny, nx, seed=42):
    """Create a random binary grid with a fixed seed."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=(ny, nx), dtype=np.uint8)


# ---------------------------------------------------------------------------
# MPI subprocess launcher
# ---------------------------------------------------------------------------

MPI_SCRIPTS = {
    "row":   "row.py",
    "col":   "col.py",
    "block": "block.py",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_mpi(strategy, n_procs, ny, nx, n_iter, seed=42, need_grid=False,
            timeout=600):
    """
    Launch an MPI script in --benchmark mode and parse output.

    Parameters
    ----------
    strategy : str
        One of 'row', 'col', 'block'.
    n_procs : int
        Total number of MPI processes (including rank 0 / display rank).
    ny, nx : int
        Grid dimensions.
    n_iter : int
        Number of generations.
    seed : int
        RNG seed for reproducibility.
    need_grid : bool
        If True, also request the final grid (for correctness checks).
    timeout : int
        Maximum time in seconds before killing the process.

    Returns
    -------
    dict with keys:
        'time_compute', 'time_comm', 'time_total' (floats, in seconds)
        'grid' (np.ndarray or None)
        'success' (bool)
        'error' (str or None)
    """
    script = MPI_SCRIPTS.get(strategy)
    if script is None:
        return {"success": False, "error": f"Unknown strategy: {strategy}"}

    script_path = os.path.join(SCRIPT_DIR, script)
    if not os.path.isfile(script_path):
        return {"success": False,
                "error": f"Script not found: {script_path}"}

    cmd = [
        "mpirun", "-n", str(n_procs), "--oversubscribe",
        sys.executable, script_path, "--benchmark",
        str(ny), str(nx), str(n_iter), str(seed),
    ]
    if need_grid:
        cmd.append("--dump-grid")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=SCRIPT_DIR,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout"}
    except FileNotFoundError:
        return {"success": False,
                "error": "mpirun not found. Is MPI installed?"}

    if result.returncode != 0:
        stderr_short = result.stderr[:500] if result.stderr else "(empty)"
        return {"success": False,
                "error": f"Exit code {result.returncode}: {stderr_short}"}

    # Parse stdout for timing and optional grid
    timings = {}
    grid = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("TIME_COMPUTE="):
            timings["time_compute"] = float(line.split("=", 1)[1])
        elif line.startswith("TIME_COMM="):
            timings["time_comm"] = float(line.split("=", 1)[1])
        elif line.startswith("TIME_TOTAL="):
            timings["time_total"] = float(line.split("=", 1)[1])
        elif line.startswith("GRID="):
            try:
                raw = base64.b64decode(line.split("=", 1)[1])
                grid = np.load(io.BytesIO(raw))
            except Exception:
                grid = None

    if "time_total" not in timings:
        stdout_short = result.stdout[:500] if result.stdout else "(empty)"
        return {"success": False,
                "error": f"Could not parse timing from stdout: {stdout_short}"}

    return {
        "success": True,
        "error": None,
        "time_compute": timings.get("time_compute", 0.0),
        "time_comm": timings.get("time_comm", 0.0),
        "time_total": timings["time_total"],
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# Test 1: Correctness
# ---------------------------------------------------------------------------

def test_correctness(ny=100, nx=100, n_iter=20, seed=42):
    """Compare parallel results against serial reference."""
    print("=" * 70)
    print("TEST 1: Correctness verification")
    print("=" * 70)

    cells = make_grid(ny, nx, seed=seed)
    ref = serial_game_of_life(cells.copy(), n_iter)
    print(f"  Serial reference computed ({ny}x{nx}, {n_iter} iterations)")

    results = {}
    all_pass = True

    for strategy in ["row", "col", "block"]:
        n_procs = 3  # rank 0 + 2 workers
        res = run_mpi(strategy, n_procs, ny, nx, n_iter, seed=seed,
                      need_grid=True)
        if not res["success"]:
            print(f"  [{strategy:>5}] FAIL - could not run: {res['error']}")
            results[strategy] = False
            all_pass = False
            continue

        if res["grid"] is None:
            print(f"  [{strategy:>5}] SKIP - no grid returned "
                  "(--dump-grid not implemented?)")
            results[strategy] = None
            continue

        match = np.array_equal(ref, res["grid"])
        tag = "PASS" if match else "FAIL"
        print(f"  [{strategy:>5}] {tag}")
        results[strategy] = match
        if not match:
            diff = np.sum(ref != res["grid"])
            print(f"          {diff} cells differ out of {ny * nx}")
            all_pass = False

    print()
    return all_pass, results


# ---------------------------------------------------------------------------
# Test 2: Performance & Speedup
# ---------------------------------------------------------------------------

def test_performance(ny=1000, nx=1000, n_iter=100,
                     proc_counts=None, seed=42):
    """
    Measure execution time and speedup for each decomposition strategy.
    proc_counts: list of total MPI process counts (including rank 0).
                 Default: [2, 3, 5, 9] i.e. 1, 2, 4, 8 workers.
    """
    if proc_counts is None:
        proc_counts = [2, 3, 5, 9]

    print("=" * 70)
    print("TEST 2: Performance & Speedup")
    print(f"  Grid: {ny}x{nx}, Iterations: {n_iter}")
    print("=" * 70)

    # Serial baseline
    cells = make_grid(ny, nx, seed=seed)
    _, t_serial = serial_game_of_life_timed(cells.copy(), n_iter)
    print(f"  Serial baseline: {t_serial:.4f} s")

    # strategy -> {n_procs -> time_total}
    data = defaultdict(dict)

    for strategy in ["row", "col", "block"]:
        print(f"\n  Strategy: {strategy}")
        for np_ in proc_counts:
            n_workers = np_ - 1
            res = run_mpi(strategy, np_, ny, nx, n_iter, seed=seed)
            if res["success"]:
                t = res["time_total"]
                s = t_serial / t if t > 0 else float("inf")
                print(f"    {np_:>2} procs ({n_workers} workers): "
                      f"{t:.4f} s  speedup={s:.2f}x")
                data[strategy][np_] = t
            else:
                print(f"    {np_:>2} procs: FAILED - {res['error']}")

    print()

    # Generate speedup plots
    if HAS_MATPLOTLIB:
        for strategy in ["row", "col", "block"]:
            if not data[strategy]:
                continue
            procs = sorted(data[strategy].keys())
            times = [data[strategy][p] for p in procs]
            speedups = [t_serial / t for t in times]
            workers = [p - 1 for p in procs]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(workers, times, "o-", linewidth=2, markersize=8)
            ax1.set_xlabel("Number of workers")
            ax1.set_ylabel("Time (s)")
            ax1.set_title(f"Execution Time - {strategy} decomposition")
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(workers)

            ax2.plot(workers, speedups, "o-", linewidth=2, markersize=8,
                     label="Measured")
            ax2.plot(workers, workers, "--", color="gray", alpha=0.5,
                     label="Ideal (linear)")
            ax2.set_xlabel("Number of workers")
            ax2.set_ylabel("Speedup")
            ax2.set_title(f"Speedup - {strategy} decomposition")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(workers)

            fig.tight_layout()
            fname = os.path.join(SCRIPT_DIR, f"speedup_{strategy}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  Plot saved: {fname}")

    return t_serial, data


# ---------------------------------------------------------------------------
# Test 3: Decomposition Strategy Comparison
# ---------------------------------------------------------------------------

def test_strategy_comparison(n_iter=100, n_procs=5, seed=42,
                             square_size=2000, rect_ny=200, rect_nx=4000):
    """
    Compare row vs column vs block decomposition on different grid shapes.
    """
    print("=" * 70)
    print("TEST 3: Decomposition Strategy Comparison")
    print("=" * 70)

    configs = [
        ("Square grid", square_size, square_size),
        ("Rectangular grid (wide)", rect_ny, rect_nx),
        ("Rectangular grid (tall)", rect_nx, rect_ny),
    ]

    # config_name -> {strategy -> time}
    all_data = {}

    for label, ny, nx in configs:
        print(f"\n  {label}: {ny}x{nx}, {n_procs} procs, {n_iter} iter")
        times = {}
        for strategy in ["row", "col", "block"]:
            res = run_mpi(strategy, n_procs, ny, nx, n_iter, seed=seed)
            if res["success"]:
                t = res["time_total"]
                print(f"    {strategy:>5}: {t:.4f} s")
                times[strategy] = t
            else:
                print(f"    {strategy:>5}: FAILED - {res['error']}")
        all_data[label] = times

    print()

    # Generate comparison plots
    if HAS_MATPLOTLIB:
        for label, times in all_data.items():
            if not times:
                continue

            strategies = list(times.keys())
            vals = [times[s] for s in strategies]
            colors = ["#2196F3", "#4CAF50", "#FF9800"]

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(strategies, vals, color=colors[:len(strategies)],
                          edgecolor="black", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.3f}s", ha="center", va="bottom", fontsize=10)
            ax.set_ylabel("Time (s)")
            ax.set_title(f"Strategy Comparison - {label}")
            ax.grid(True, axis="y", alpha=0.3)

            fig.tight_layout()
            if "Square" in label:
                fname = os.path.join(SCRIPT_DIR, "comparison_square.png")
            elif "wide" in label:
                fname = os.path.join(SCRIPT_DIR, "comparison_rect.png")
            else:
                fname = os.path.join(SCRIPT_DIR, "comparison_rect_tall.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  Plot saved: {fname}")

    return all_data


# ---------------------------------------------------------------------------
# Test 4: Scalability (Strong & Weak Scaling)
# ---------------------------------------------------------------------------

def test_strong_scaling(ny=4000, nx=4000, n_iter=100,
                        proc_counts=None, seed=42, strategy="row"):
    """
    Strong scaling: fixed problem size, increase core count.
    """
    if proc_counts is None:
        proc_counts = [2, 3, 5, 9, 13, 17]  # 1,2,4,8,12,16 workers

    print("=" * 70)
    print(f"TEST 4a: Strong Scaling ({strategy} decomposition)")
    print(f"  Grid: {ny}x{nx}, Iterations: {n_iter}")
    print("=" * 70)

    cells = make_grid(ny, nx, seed=seed)
    _, t_serial = serial_game_of_life_timed(cells.copy(), n_iter)
    print(f"  Serial baseline: {t_serial:.4f} s")

    data = {}
    for np_ in proc_counts:
        n_workers = np_ - 1
        res = run_mpi(strategy, np_, ny, nx, n_iter, seed=seed)
        if res["success"]:
            t = res["time_total"]
            eff = (t_serial / t) / n_workers * 100 if n_workers > 0 else 0
            print(f"    {np_:>2} procs ({n_workers:>2} workers): "
                  f"{t:.4f} s  speedup={t_serial / t:.2f}x  "
                  f"efficiency={eff:.1f}%")
            data[np_] = t
        else:
            print(f"    {np_:>2} procs: FAILED - {res['error']}")

    print()

    if HAS_MATPLOTLIB and data:
        procs = sorted(data.keys())
        times = [data[p] for p in procs]
        workers = [p - 1 for p in procs]
        speedups = [t_serial / t for t in times]
        efficiencies = [s / w * 100 for s, w in zip(speedups, workers)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(workers, times, "o-", linewidth=2, markersize=8)
        axes[0].set_xlabel("Number of workers")
        axes[0].set_ylabel("Time (s)")
        axes[0].set_title("Strong Scaling - Time")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(workers, speedups, "o-", linewidth=2, markersize=8,
                     label="Measured")
        axes[1].plot(workers, workers, "--", color="gray", alpha=0.5,
                     label="Ideal")
        axes[1].set_xlabel("Number of workers")
        axes[1].set_ylabel("Speedup")
        axes[1].set_title("Strong Scaling - Speedup")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(workers, efficiencies, "o-", linewidth=2, markersize=8)
        axes[2].axhline(y=100, color="gray", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Number of workers")
        axes[2].set_ylabel("Efficiency (%)")
        axes[2].set_title("Strong Scaling - Efficiency")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f"Strong Scaling ({ny}x{nx}, {strategy})", fontsize=14)
        fig.tight_layout()
        fname = os.path.join(SCRIPT_DIR, "strong_scaling.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {fname}")

    return t_serial, data


def test_weak_scaling(base_size=1000, n_iter=100,
                      proc_counts=None, seed=42, strategy="row"):
    """
    Weak scaling: problem size grows proportionally with core count.
    1 worker  -> base_size x base_size
    2 workers -> base_size x (2*base_size)
    4 workers -> (2*base_size) x (2*base_size)
    8 workers -> (2*base_size) x (4*base_size)
    etc.
    """
    if proc_counts is None:
        proc_counts = [2, 3, 5, 9]  # 1, 2, 4, 8 workers

    print("=" * 70)
    print(f"TEST 4b: Weak Scaling ({strategy} decomposition)")
    print(f"  Base size: {base_size}x{base_size}, Iterations: {n_iter}")
    print("=" * 70)

    # Map n_workers to grid size so each worker has ~base_size^2 cells
    def grid_for_workers(n_workers):
        total_cells = n_workers * base_size * base_size
        # Keep roughly square-ish
        side = int(np.sqrt(total_cells))
        ny = side
        nx = total_cells // ny
        return ny, nx

    data = {}
    for np_ in proc_counts:
        n_workers = np_ - 1
        if n_workers < 1:
            continue
        ny, nx = grid_for_workers(n_workers)
        res = run_mpi(strategy, np_, ny, nx, n_iter, seed=seed)
        if res["success"]:
            t = res["time_total"]
            print(f"    {np_:>2} procs ({n_workers:>2} workers), "
                  f"grid {ny}x{nx}: {t:.4f} s")
            data[np_] = {"time": t, "ny": ny, "nx": nx}
        else:
            print(f"    {np_:>2} procs, grid {ny}x{nx}: "
                  f"FAILED - {res['error']}")

    print()

    if HAS_MATPLOTLIB and data:
        procs = sorted(data.keys())
        workers = [p - 1 for p in procs]
        times = [data[p]["time"] for p in procs]
        labels = [f"{data[p]['ny']}x{data[p]['nx']}" for p in procs]

        # Ideal weak scaling: time stays constant
        t_ref = times[0] if times else 1.0
        scaled = [t / t_ref for t in times]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(workers, times, "o-", linewidth=2, markersize=8)
        for w, t, lbl in zip(workers, times, labels):
            ax1.annotate(lbl, (w, t), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=8)
        ax1.set_xlabel("Number of workers")
        ax1.set_ylabel("Time (s)")
        ax1.set_title("Weak Scaling - Time")
        ax1.grid(True, alpha=0.3)

        ax2.plot(workers, scaled, "o-", linewidth=2, markersize=8,
                 label="Measured")
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5,
                     label="Ideal")
        ax2.set_xlabel("Number of workers")
        ax2.set_ylabel("Normalized time (T / T_1)")
        ax2.set_title("Weak Scaling - Normalized")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Weak Scaling ({strategy})", fontsize=14)
        fig.tight_layout()
        fname = os.path.join(SCRIPT_DIR, "weak_scaling.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {fname}")

    return data


# ---------------------------------------------------------------------------
# Test 5: Communication vs Computation Overhead
# ---------------------------------------------------------------------------

def test_comm_vs_compute(ny=2000, nx=2000, n_iter=100,
                         proc_counts=None, seed=42):
    """
    Measure and compare computation time vs MPI communication time.
    """
    if proc_counts is None:
        proc_counts = [2, 3, 5, 9]

    print("=" * 70)
    print("TEST 5: Communication vs Computation Overhead")
    print(f"  Grid: {ny}x{nx}, Iterations: {n_iter}")
    print("=" * 70)

    # strategy -> {n_procs -> (compute, comm)}
    data = defaultdict(dict)

    for strategy in ["row", "col", "block"]:
        print(f"\n  Strategy: {strategy}")
        for np_ in proc_counts:
            n_workers = np_ - 1
            res = run_mpi(strategy, np_, ny, nx, n_iter, seed=seed)
            if res["success"]:
                tc = res["time_compute"]
                tm = res["time_comm"]
                tt = res["time_total"]
                ratio = tm / tc * 100 if tc > 0 else float("inf")
                print(f"    {np_:>2} procs: compute={tc:.4f}s  "
                      f"comm={tm:.4f}s  total={tt:.4f}s  "
                      f"comm/compute={ratio:.1f}%")
                data[strategy][np_] = (tc, tm)
            else:
                print(f"    {np_:>2} procs: FAILED - {res['error']}")

    print()

    if HAS_MATPLOTLIB and any(data.values()):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors_comp = "#2196F3"
        colors_comm = "#FF5722"

        for idx, strategy in enumerate(["row", "col", "block"]):
            ax = axes[idx]
            if not data[strategy]:
                ax.set_title(f"{strategy} - No data")
                continue

            procs = sorted(data[strategy].keys())
            workers = [p - 1 for p in procs]
            computes = [data[strategy][p][0] for p in procs]
            comms = [data[strategy][p][1] for p in procs]

            x = np.arange(len(workers))
            width = 0.35

            ax.bar(x - width / 2, computes, width, label="Computation",
                   color=colors_comp, edgecolor="black", linewidth=0.5)
            ax.bar(x + width / 2, comms, width, label="Communication",
                   color=colors_comm, edgecolor="black", linewidth=0.5)

            ax.set_xlabel("Number of workers")
            ax.set_ylabel("Time (s)")
            ax.set_title(f"{strategy} decomposition")
            ax.set_xticks(x)
            ax.set_xticklabels(workers)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Communication vs Computation Overhead", fontsize=14)
        fig.tight_layout()
        fname = os.path.join(SCRIPT_DIR, "comm_vs_compute.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {fname}")

    return data


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(t_serial, perf_data, proc_counts):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    header = f"{'Strategy':<10} | {'Procs':>5} | {'Workers':>7} | " \
             f"{'Time (s)':>10} | {'Speedup':>8} | {'Efficiency':>10}"
    print(header)
    print("-" * len(header))

    print(f"{'serial':<10} | {'1':>5} | {'1':>7} | "
          f"{t_serial:>10.4f} | {'1.00':>8} | {'100.0%':>10}")

    for strategy in ["row", "col", "block"]:
        if strategy not in perf_data:
            continue
        for np_ in sorted(perf_data[strategy].keys()):
            t = perf_data[strategy][np_]
            n_workers = np_ - 1
            speedup = t_serial / t if t > 0 else 0
            eff = speedup / n_workers * 100 if n_workers > 0 else 0
            print(f"{strategy:<10} | {np_:>5} | {n_workers:>7} | "
                  f"{t:>10.4f} | {speedup:>8.2f} | {eff:>9.1f}%")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark for parallel Game of Life MPI project")
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller grids for faster testing")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run correctness tests (Test 1)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--grid-ny", type=int, default=None,
                        help="Override grid height for performance tests")
    parser.add_argument("--grid-nx", type=int, default=None,
                        help="Override grid width for performance tests")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of GoL iterations (default: 100)")
    args = parser.parse_args()

    global HAS_MATPLOTLIB
    if args.no_plots:
        HAS_MATPLOTLIB = False

    if not HAS_MATPLOTLIB and not args.no_plots:
        print("WARNING: matplotlib not available, plots will be skipped.\n")

    seed = args.seed
    n_iter = args.iterations

    # Grid sizes
    if args.quick:
        perf_ny = args.grid_ny or 200
        perf_nx = args.grid_nx or 200
        strong_ny, strong_nx = 500, 500
        weak_base = 200
        comp_ny, comp_nx = 300, 300
        square_size = 400
        rect_ny, rect_nx = 100, 800
        proc_counts = [2, 3, 5]
        strong_procs = [2, 3, 5]
        n_iter = min(n_iter, 20)
    else:
        perf_ny = args.grid_ny or 1000
        perf_nx = args.grid_nx or 1000
        strong_ny, strong_nx = 4000, 4000
        weak_base = 1000
        comp_ny, comp_nx = 2000, 2000
        square_size = 2000
        rect_ny, rect_nx = 200, 4000
        proc_counts = [2, 3, 5, 9]
        strong_procs = [2, 3, 5, 9, 13, 17]

    print("Parallel Game of Life - MPI Benchmark Suite")
    print(f"  Quick mode: {args.quick}")
    print(f"  Seed: {seed}")
    print(f"  Iterations: {n_iter}")
    print()

    # ------------------------------------------------------------------
    # Test 1: Correctness
    # ------------------------------------------------------------------
    correctness_ny = 100 if not args.quick else 50
    correctness_nx = 100 if not args.quick else 50
    correctness_iter = 20 if not args.quick else 5

    passed, _ = test_correctness(
        ny=correctness_ny, nx=correctness_nx,
        n_iter=correctness_iter, seed=seed)

    if args.test_only:
        sys.exit(0 if passed else 1)

    # ------------------------------------------------------------------
    # Test 2: Performance & Speedup
    # ------------------------------------------------------------------
    t_serial, perf_data = test_performance(
        ny=perf_ny, nx=perf_nx, n_iter=n_iter,
        proc_counts=proc_counts, seed=seed)

    # ------------------------------------------------------------------
    # Test 3: Strategy Comparison
    # ------------------------------------------------------------------
    test_strategy_comparison(
        n_iter=n_iter, n_procs=5, seed=seed,
        square_size=square_size, rect_ny=rect_ny, rect_nx=rect_nx)

    # ------------------------------------------------------------------
    # Test 4: Scalability
    # ------------------------------------------------------------------
    test_strong_scaling(
        ny=strong_ny, nx=strong_nx, n_iter=n_iter,
        proc_counts=strong_procs, seed=seed, strategy="row")

    test_weak_scaling(
        base_size=weak_base, n_iter=n_iter,
        proc_counts=proc_counts, seed=seed, strategy="row")

    # ------------------------------------------------------------------
    # Test 5: Communication vs Computation
    # ------------------------------------------------------------------
    test_comm_vs_compute(
        ny=comp_ny, nx=comp_nx, n_iter=n_iter,
        proc_counts=proc_counts, seed=seed)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(t_serial, perf_data, proc_counts)

    print("Benchmark complete.")
    if HAS_MATPLOTLIB:
        print("Plots saved in:", SCRIPT_DIR)


if __name__ == "__main__":
    main()
