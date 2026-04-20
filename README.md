<!-- Create a README.md with these sections:
     1. Project title and one-sentence description
     2. Installation (activate ece105 conda env, pip install numpy matplotlib)
     3. Usage (python generate_plots.py)
     4. Example output (describe the three plots briefly)
     5. AI tools used and disclosure -->
ECE105 Lab 3 — Sensor Plots

A small script to generate synthetic temperature sensor data and produce a single publication-ready PNG
containing three subplots: a time-series scatter (two sensors), an overlaid histogram, and a side-by-side
boxplot comparison.

Installation

 1. Activate the course conda environment: conda activate ece105
 # or mamba activate ece105
 2. Install runtime dependencies (use conda or mamba): conda install numpy matplotlib
 # or
 mamba install numpy matplotlib

(If you prefer pip inside the environment: pip install numpy matplotlib.)

Usage

Run the script from the repository root:

 python generate_plots.py

This will create a single file sensor_analysis.png in the current directory.

Options (from the script):

 - --seed: integer RNG seed for reproducible output (default: 5548)
 - --out-path: path to the output PNG (when using the version that accepts this argument)

Example output

The generated PNG (sensor_analysis.png) contains three subplots arranged left-to-right:

 - Scatter (left): time-series scatter of Sensor A (blue circles) and Sensor B (orange squares) vs. time
(seconds). A legend, axis labels, and grid are included.
 - Histogram (center): overlaid histograms for Sensor A and Sensor B using consistent bin edges, with dashed
lines at each sensor mean and a dotted line at the combined mean.
 - Boxplot (right): notched, colored boxplots for each sensor with mean markers and a horizontal dotted line
showing the overall mean.

These figures are saved at 150 DPI with a tight bounding box and suitable for inclusion in reports.

AI tools used and disclosure

Copilot CLI has been used in generating code and documentation. Code has been manually tested for each function.
