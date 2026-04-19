"""Generate publication-quality sensor data visualizations.

This script creates synthetic temperature sensor data using NumPy
and produces scatter, histogram, and box plot visualizations saved
as PNG files.

Usage
-----
    python generate_plots.py
"""

import numpy as np


def generate_data(seed):
    """Generate reproducible synthetic temperature readings and timestamps.

    This function produces three aligned arrays suitable for plotting and
    statistical comparison between two synthetic temperature sensors.

    Parameters
    ----------
    seed : int or None
        Seed value passed to :func:`numpy.random.default_rng`. When an integer
        is provided the function produces deterministic, repeatable output
        suitable for tests and examples. When ``None``, the generator is
        seeded from system entropy and outputs will vary across calls.

    Returns
    -------
    sensor_a : numpy.ndarray, shape (200,), dtype float64
        Temperature measurements (degrees Celsius) emulating Sensor A. Values
        are sampled from a Gaussian distribution with mean 25.0 and standard
        deviation 3.0. The length is fixed at 200 to match the original
        notebook and downstream plotting code expectations.

    sensor_b : numpy.ndarray, shape (200,), dtype float64
        Temperature measurements (degrees Celsius) emulating Sensor B. Values
        are sampled from a Gaussian distribution with mean 27.0 and standard
        deviation 4.5. Returned dtype is ``float64``.

    timestamps : numpy.ndarray, shape (200,), dtype float64
        Monotonic measurement times (seconds). Values are sampled uniformly
        from the closed interval [0.0, 10.0] and sorted in ascending order to
        simulate realistic increasing timestamps for each measurement. The
        array length matches the sensor arrays so they can be used directly
        for time-series scatter plots.

    Notes
    -----
    - Uses :class:`numpy.random.Generator` via :func:`numpy.random.default_rng`
      for modern, reproducible random number generation.
    - All arrays are explicitly cast to ``float64`` to ensure consistent
      numeric types for downstream analysis and plotting libraries.
    - The function mirrors the behavior and parameters of the original
      Jupyter notebook so existing visualization code will work without
      modification.
    """
    rng = np.random.default_rng(seed)
    n = 200

    sensor_a = rng.normal(loc=25.0, scale=3.0, size=n).astype(np.float64)
    sensor_b = rng.normal(loc=27.0, scale=4.5, size=n).astype(np.float64)
    timestamps = np.sort(rng.uniform(0.0, 10.0, size=n)).astype(np.float64)

    return sensor_a, sensor_b, timestamps


def plot_scatter(ax, timestamps, sensor_a, sensor_b):
    """Draw a time-series scatter plot of two temperature sensors on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to modify in-place. The function uses this Axes for
        plotting and for setting labels, legend and limits.
    timestamps : numpy.ndarray, shape (N,), dtype float64
        Monotonic measurement times in seconds. Must be the same length as the
        sensor arrays.
    sensor_a : numpy.ndarray, shape (N,), dtype float64
        Temperature readings from Sensor A (degrees Celsius).
    sensor_b : numpy.ndarray, shape (N,), dtype float64
        Temperature readings from Sensor B (degrees Celsius).

    Returns
    -------
    None
        The function modifies the provided Axes in-place and returns None.

    Notes
    -----
    - Uses consistent marker/colour choices and labels to match the original
      notebook visualization: Sensor A as blue circles, Sensor B as orange squares.
    - Sets sensible y-limits with a small padding so plotted points are not
      clipped and enables a legend in the upper-right with slight background.
    """
    ax.scatter(timestamps, sensor_a, c="tab:blue", s=30, alpha=0.7,
               label="Sensor A (μ=25,σ=3)")
    ax.scatter(timestamps, sensor_b, c="tab:orange", s=30, alpha=0.7, marker="s",
               label="Sensor B (μ=27,σ=4.5)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right", framealpha=0.9)

    all_min = min(float(np.min(sensor_a)), float(np.min(sensor_b)))
    all_max = max(float(np.max(sensor_a)), float(np.max(sensor_b)))
    pad = 0.05 * (all_max - all_min) if all_max > all_min else 1.0
    ax.set_ylim(all_min - pad, all_max + pad)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    return None


def plot_histogram(ax, sensor_a, sensor_b, bins=30):
    """Draw overlaid histograms of two temperature sensors on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to modify in-place. The function plots onto this Axes
        and sets labels, legend and limits.
    sensor_a : numpy.ndarray, shape (N,), dtype float64
        Temperature readings (degrees Celsius) from Sensor A.
    sensor_b : numpy.ndarray, shape (N,), dtype float64
        Temperature readings (degrees Celsius) from Sensor B.
    bins : int or array-like, optional
        If an int, the number of equal-width bins to create across the combined
        range of both sensors. If array-like, the bin edges to use (passed
        directly to matplotlib.axes.Axes.hist). Default is 30.

    Returns
    -------
    None
        Modifies the provided Axes in-place and returns None.

    Notes
    -----
    - Uses identical bin edges for both datasets so the histograms are
      directly comparable.
    - Draws dashed vertical lines at each sensor's mean and a dotted line at
      the combined mean for quick visual reference.
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
    """Draw side-by-side box plots comparing two temperature sensors on the given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to modify in-place. The function draws the boxplots on
        this Axes and configures labels, grid and ticks.
    sensor_a : numpy.ndarray, shape (N,), dtype float64
        Temperature readings (degrees Celsius) from Sensor A.
    sensor_b : numpy.ndarray, shape (N,), dtype float64
        Temperature readings (degrees Celsius) from Sensor B.

    Returns
    -------
    None
        The function modifies the provided Axes in-place and returns None.

    Notes
    -----
    - Uses identical styling to the notebook: notch style, visible means, and
      colored boxes for each sensor. The function sets consistent y-limits with
      a small padding so whiskers and outliers are not clipped.
    - Expects the inputs to be 1-D arrays of equal length (or at least
      comparable datasets); length is not enforced but plotting assumes each
      entry is a sample from the sensor distribution.
    """
    data = [sensor_a, sensor_b]
    labels = ["Sensor A", "Sensor B"]

    bp = ax.boxplot(data,
                    labels=labels,
                    patch_artist=True,
                    notch=True,
                    showmeans=True,
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

    # Draw horizontal dotted line indicating overall mean across both sensors
    overall_mean = float(np.concatenate([sensor_a, sensor_b]).mean())
    ax.axhline(overall_mean, color="k", linestyle=":", linewidth=1, label="Overall mean")

    return None
