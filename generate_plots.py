"""Generate publication-quality sensor data visualizations.

This script creates synthetic temperature sensor data and produces a single
PNG containing three subplots: scatter (time series), histogram, and boxplot.
The main() function saves the combined figure as 'sensor_analysis.png' at
150 DPI with a tight bounding box.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def generate_data(seed):
    """Generate reproducible synthetic sensor data for two temperature sensors.

    Parameters
    ----------
    seed : int or None
        Seed for NumPy's random number generator. If an integer is provided
        the results are deterministic and repeatable. If ``None``, the RNG is
        seeded nondeterministically and outputs vary each run.

    Returns
    -------
    sensor_a : numpy.ndarray, shape (200,), dtype float64
        Samples from N(loc=25.0, scale=3.0) representing Sensor A (°C).
    sensor_b : numpy.ndarray, shape (200,), dtype float64
        Samples from N(loc=27.0, scale=4.5) representing Sensor B (°C).
    timestamps : numpy.ndarray, shape (200,), dtype float64
        Sorted uniform samples in [0.0, 10.0] representing monotonic timestamps (s).
    """
    rng = np.random.default_rng(seed)
    n = 200
    sensor_a = rng.normal(loc=25.0, scale=3.0, size=n).astype(np.float64)
    sensor_b = rng.normal(loc=27.0, scale=4.5, size=n).astype(np.float64)
    timestamps = np.sort(rng.uniform(0.0, 10.0, size=n)).astype(np.float64)
    return sensor_a, sensor_b, timestamps


def plot_scatter(ax, timestamps, sensor_a, sensor_b):
    """Draw a time-series scatter plot of two sensors on an Axes.

    The function modifies the provided Axes in place and returns None.
    """
    ax.scatter(timestamps, sensor_a, c="tab:blue", s=30, alpha=0.7,
               label="Sensor A (μ=25,σ=3)")
    ax.scatter(timestamps, sensor_b, c="tab:orange", s=30, alpha=0.7, marker="s",
               label="Sensor B (μ=27,σ=4.5)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right", framealpha=0.9)
    all_min = float(min(np.min(sensor_a), np.min(sensor_b)))
    all_max = float(max(np.max(sensor_a), np.max(sensor_b)))
    pad = 0.05 * (all_max - all_min) if all_max > all_min else 1.0
    ax.set_ylim(all_min - pad, all_max + pad)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    return None


def plot_histogram(ax, sensor_a, sensor_b, bins=30):
    """Draw overlaid histograms for two sensors on an Axes.

    Modifies ax in place and returns None.
    """
    all_min = float(min(np.min(sensor_a), np.min(sensor_b)))
    all_max = float(max(np.max(sensor_a), np.max(sensor_b)))
    if isinstance(bins, int):
        bin_edges = np.linspace(all_min, all_max, bins + 1)
    else:
        bin_edges = np.asarray(bins)
    ax.hist(sensor_a, bins=bin_edges, alpha=0.6, color="tab:blue",
            label=f"Sensor A (n={sensor_a.size})", edgecolor="k")
    ax.hist(sensor_b, bins=bin_edges, alpha=0.5, color="tab:orange",
            label=f"Sensor B (n={sensor_b.size})", edgecolor="k")
    mean_a = float(np.mean(sensor_a))
    mean_b = float(np.mean(sensor_b))
    overall_mean = float(np.concatenate([sensor_a, sensor_b]).mean())
    ax.axvline(mean_a, color="tab:blue", linestyle="--", linewidth=1)
    ax.axvline(mean_b, color="tab:orange", linestyle="--", linewidth=1)
    ax.axvline(overall_mean, color="k", linestyle=":", linewidth=1, label="Overall mean")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Count")
    ax.legend(framealpha=0.9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    pad = 0.05 * (all_max - all_min) if all_max > all_min else 1.0
    ax.set_xlim(all_min - pad, all_max + pad)
    return None


def plot_boxplot(ax, sensor_a, sensor_b):
    """Draw side-by-side boxplots comparing two sensors on an Axes.

    Modifies ax in place and returns None.
    """
    data = [sensor_a, sensor_b]
    # Use tick_labels when available to avoid deprecation warnings
    try:
        bp = ax.boxplot(data, tick_labels=["Sensor A", "Sensor B"],
                        patch_artist=True, notch=True, showmeans=True,
                        meanprops=dict(marker="D", markeredgecolor="k",
                                       markerfacecolor="white", markersize=6),
                        widths=0.6)
    except TypeError:
        bp = ax.boxplot(data, labels=["Sensor A", "Sensor B"],
                        patch_artist=True, notch=True, showmeans=True,
                        meanprops=dict(marker="D", markeredgecolor="k",
                                       markerfacecolor="white", markersize=6),
                        widths=0.6)
    box_colors = ["tab:blue", "tab:orange"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("k")
        patch.set_alpha(0.6)
    for median in bp.get("medians", []):
        median.set(color="k", linewidth=1.25)
    for whisker in bp.get("whiskers", []):
        whisker.set(color="k", linewidth=0.8)
    for cap in bp.get("caps", []):
        cap.set(color="k", linewidth=0.8)
    for flier in bp.get("fliers", []):
        flier.set(marker="o", color="0.25", alpha=0.6, markersize=4)
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, axis="y")
    all_min = float(min(np.min(sensor_a), np.min(sensor_b)))
    all_max = float(max(np.max(sensor_a), np.max(sensor_b)))
    pad = 0.05 * (all_max - all_min) if all_max > all_min else 1.0
    ax.set_ylim(all_min - pad, all_max + pad)
    overall_mean = float(np.concatenate([sensor_a, sensor_b]).mean())
    ax.axhline(overall_mean, color="k", linestyle=":", linewidth=1, label="Overall mean")
    return None


def main(seed=5548, out_path='sensor_analysis.png'):
    """Generate data, create a 2x2 subplot figure (three plots + one empty),
    and save the combined figure to a single PNG file.

    Parameters
    ----------
    seed : int or None, optional
        RNG seed for reproducible synthetic data (default: 5548). If ``None``,
        RNG is nondeterministic.
    out_path : str or os.PathLike, optional
        Output PNG path for the combined figure (default: 'sensor_analysis.png').

    Returns
    -------
    None
        Writes the PNG to disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sensor_a, sensor_b, timestamps = generate_data(seed)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    ax_scatter = axes[0, 0]
    ax_hist = axes[0, 1]
    ax_box = axes[1, 0]
    ax_empty = axes[1, 1]

    plot_scatter(ax_scatter, timestamps, sensor_a, sensor_b)
    ax_scatter.set_title("Sensor readings over time")

    plot_histogram(ax_hist, sensor_a, sensor_b, bins=30)
    ax_hist.set_title("Temperature distribution")

    plot_boxplot(ax_box, sensor_a, sensor_b)
    ax_box.set_title("Distribution summary (boxplot)")
    try:
        ax_box.legend(framealpha=0.9)
    except Exception:
        pass

    ax_empty.axis('off')

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", str(out_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate a single PNG with a 2x2 grid (three plots + one empty).')
    parser.add_argument('--seed', type=int, default=5548, help='RNG seed (int) or omit for nondeterministic')
    parser.add_argument('--out-path', default='sensor_analysis.png', help='Output PNG path')
    args = parser.parse_args()
    main(seed=args.seed, out_path=args.out_path)
