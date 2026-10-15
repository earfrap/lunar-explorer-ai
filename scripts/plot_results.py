"""
Lunar Explorer AI — Scientific Figures

Generate publication-style figures from the final
30-environment mission benchmark.

Run from repository root:
    python scripts/plot_results.py

Input:
    results/benchmarks/benchmark_coverage_results.csv

Output:
    docs/figures/*.png
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/benchmarks/benchmark_coverage_results.csv"
OUTPUT = ROOT / "docs/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["AI V3", "Systematic", "Systematic Coverage"]
LABELS = ["ML-Guided\nAI V3", "Systematic\nLocal", "Systematic\nCoverage"]

COLORS = {
    "AI V3": "#287D8E",
    "Systematic": "#9CA3AF",
    "Systematic Coverage": "#D5A34A",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 220,
})


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {INPUT}")

    with INPUT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Benchmark CSV is empty.")

    required = {
        "strategy",
        "confirmed",
        "measurements",
        "distance",
    }

    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available: {list(rows[0])}"
        )

    data = {}

    for strategy in STRATEGIES:
        subset = [r for r in rows if r["strategy"] == strategy]

        if not subset:
            raise ValueError(f"No data for strategy: {strategy}")

        data[strategy] = subset

    return data, rows


def values(data, strategy, metric):
    return np.array(
        [float(row[metric]) for row in data[strategy]],
        dtype=float,
    )


def save(fig, filename):
    path = OUTPUT / filename
    fig.savefig(path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"Saved: {path.relative_to(ROOT)}")


def plot_mission_performance(data):
    metric = "confirmed"

    means = [
        values(data, s, metric).mean()
        for s in STRATEGIES
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    bars = ax.bar(
        LABELS,
        means,
        color=[COLORS[s] for s in STRATEGIES],
        width=0.58,
    )

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.16,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title(
        "ML-guided exploration identifies more\n"
        "validated sampling sites",
        pad=18,
    )
    ax.set_ylabel("Mean validated confirmations per mission")
    ax.set_ylim(0, max(means) * 1.22)
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)

    fig.text(
        0.5, 0.015,
        "30 independently generated synthetic test environments",
        ha="center",
        fontsize=9,
        color="#666666",
    )

    save(fig, "01_mission_performance.png")


def plot_exploration_efficiency(data):
    """
    Three metrics with different units are displayed
    in separate panels to avoid misleading shared axes.
    """
    metrics = [
        ("confirmed", "Validated confirmations", "{:.2f}"),
        ("measurements", "Measurements", "{:.1f}"),
        ("distance", "Distance travelled (m)", "{:.0f}"),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.6),
        constrained_layout=True,
    )

    for ax, (metric, title, fmt) in zip(axes, metrics):
        means = [
            values(data, s, metric).mean()
            for s in STRATEGIES
        ]

        bars = ax.bar(
            range(3),
            means,
            color=[COLORS[s] for s in STRATEGIES],
            width=0.65,
        )

        ax.set_xticks(range(3))
        ax.set_xticklabels(
            ["AI V3", "Local", "Coverage"],
            fontsize=9,
        )
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_ylim(0, max(means) * 1.23)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)

        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(means) * 0.035,
                fmt.format(value),
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    fig.suptitle(
        "Mission performance under shared operational constraints",
        fontsize=15,
        fontweight="bold",
    )

    save(fig, "02_exploration_efficiency.png")


def paired_values(data, metric):
    """
    Pair environments by seed, not by row position.
    """
    if "seed" not in data["AI V3"][0]:
        raise ValueError(
            "Column 'seed' is required for paired comparisons. "
            "Inspect CSV headers before generating figures."
        )

    def by_seed(strategy):
        return {
            row["seed"]: float(row[metric])
            for row in data[strategy]
        }

    ai = by_seed("AI V3")
    coverage = by_seed("Systematic Coverage")

    common = sorted(set(ai) & set(coverage))

    if not common:
        raise ValueError("No matching environment seeds.")

    return np.array(
        [ai[s] - coverage[s] for s in common],
        dtype=float,
    )


def bootstrap_ci(differences, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(differences),
        size=(n_boot, len(differences)),
    )

    means = differences[indices].mean(axis=1)

    return np.percentile(means, [2.5, 97.5])


def plot_paired_comparison(data):
    metrics = [
        ("confirmed", "Validated confirmations"),
        ("measurements", "Measurements"),
        ("distance", "Distance (m)"),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.5),
        constrained_layout=True,
    )

    for ax, (metric, title) in zip(axes, metrics):
        diff = paired_values(data, metric)
        mean = diff.mean()
        low, high = bootstrap_ci(diff)

        ax.errorbar(
            [mean],
            [0],
            xerr=[
                [mean - low],
                [high - mean],
            ],
            fmt="o",
            markersize=9,
            color=COLORS["AI V3"],
            capsize=6,
            linewidth=2.5,
        )

        ax.axvline(
            0,
            color="#555555",
            linestyle="--",
            linewidth=1,
        )

        ax.set_yticks([])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("AI V3 − Systematic Coverage")
        ax.grid(axis="x", alpha=0.15)

        span = max(abs(low), abs(high), 1)
        ax.set_xlim(
            min(low, 0) - span * 0.25,
            max(high, 0) + span * 0.25,
        )

        ax.text(
            0.5, 0.12,
            f"Mean: {mean:+.2f}\n95% CI: [{low:+.2f}, {high:+.2f}]",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
        )

    fig.suptitle(
        "Paired mission comparison across 30 environments",
        fontsize=15,
        fontweight="bold",
    )

    save(fig, "03_paired_comparison.png")


def plot_mission_outcomes(data):
    if "seed" not in data["AI V3"][0]:
        raise ValueError("Column 'seed' is required.")

    def by_seed(strategy):
        return {
            row["seed"]: float(row["confirmed"])
            for row in data[strategy]
        }

    ai = by_seed("AI V3")
    local = by_seed("Systematic")
    coverage = by_seed("Systematic Coverage")

    seeds = sorted(set(ai) & set(local) & set(coverage))
    x = np.arange(len(seeds))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 5.5))

    ax.bar(
        x - width,
        [ai[s] for s in seeds],
        width,
        label="AI V3",
        color=COLORS["AI V3"],
    )
    ax.bar(
        x,
        [local[s] for s in seeds],
        width,
        label="Systematic Local",
        color=COLORS["Systematic"],
    )
    ax.bar(
        x + width,
        [coverage[s] for s in seeds],
        width,
        label="Systematic Coverage",
        color=COLORS["Systematic Coverage"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(s) for s in seeds],
        rotation=60,
        fontsize=8,
    )

    ax.set_title(
        "Mission outcomes vary across synthetic environments",
        pad=16,
    )
    ax.set_xlabel("Environment seed")
    ax.set_ylabel("Validated confirmations")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)

    save(fig, "04_mission_outcomes.png")


def main():
    data, rows = load_data()

    print(f"Loaded {len(rows)} benchmark rows")

    for strategy in STRATEGIES:
        print(f"  {strategy}: {len(data[strategy])} environments")

    plot_mission_performance(data)
    plot_exploration_efficiency(data)
    plot_paired_comparison(data)
    plot_mission_outcomes(data)

    print("\nAll scientific figures generated successfully.")


if __name__ == "__main__":
    main()
