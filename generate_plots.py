"""Generate publication-quality sensor data visualizations.

This script creates synthetic temperature sensor data and produces a
single PNG containing three subplots: scatter (time series), histogram,
and boxplot. The main() function saves the combined figure as
'sensor_analysis.png' at 150 DPI with a tight bounding box.
"""

import numpy as np


def generate_data(seed):
    """Generate reproducible synthetic sensor data for two temperature sensors.

    Parameters
    ----------
    seed : int or None
        Seed for NumPy's random number generator. If an integer is provided
        the results are deterministic and repeatable. If ``None``, the RNG is
        seeded nondeterministically (from OS entropy) and outputs vary each run.

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

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in-place.
    timestamps : numpy.ndarray, shape (N,)
        Monotonic timestamps (s).
    sensor_a : numpy.ndarray, shape (N,)
        Sensor A temperatures (°C).
    sensor_b : numpy.ndarray, shape (N,)
        Sensor B temperatures (°C).

    Returns
    -------
    None
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

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in-place.
    sensor_a, sensor_b : numpy.ndarray, shape (N,)
        Temperature samples for each sensor.
    bins : int or array-like, optional
        Number of bins or sequence of bin edges (default 30).

    Returns
    -------
    None
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

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in-place.
    sensor_a, sensor_b : numpy.ndarray
        Temperature samples for each sensor.

    Returns
    -------
    None
    """
    data = [sensor_a, sensor_b]
    # Use tick_labels to avoid Matplotlib deprecation warning when available
    try:
        bp = ax.boxplot(data, tick_labels=["Sensor A", "Sensor B"],
                        patch_artist=True, notch=True, showmeans=True,
                        meanprops=dict(marker="D", markeredgecolor="k",
                                       markerfacecolor="white", markersize=6),
                        widths=0.6)
    except TypeError:
        # Older Matplotlib expects 'labels' keyword
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


def main(seed=5548):
    """Generate data, create a 1x3 subplot figure, and save as sensor_analysis.png.

    Parameters
    ----------
    seed : int or None, optional
        RNG seed for reproducible data (defaults to 5548).

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_path = Path("sensor_analysis.png")
    sensor_a, sensor_b, timestamps = generate_data(seed)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    plot_scatter(axes[0], timestamps, sensor_a, sensor_b)
    axes[0].set_title("Sensor readings over time")
    plot_histogram(axes[1], sensor_a, sensor_b, bins=30)
    axes[1].set_title("Temperature distribution")
    plot_boxplot(axes[2], sensor_a, sensor_b)
    axes[2].set_title("Distribution summary (boxplot)")
    try:
        axes[2].legend(framealpha=0.9)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", str(out_path))


if __name__ == '__main__':
    main()
