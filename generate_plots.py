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
