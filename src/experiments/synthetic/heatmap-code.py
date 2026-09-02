import re
from glob import glob
from pathlib import Path

import matplotlib
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image



METRICS = ["F1"]
PERCENTAGES = [20, 40, 60, 80, 100]
TOTAL_VARIABLES = 50
SHOW_FIGURES = False
SAVE_FIGURES = True
SAVE_MATRICES = True
OUTPUT_DIR = Path(__file__).resolve().parent / "heatmaps"
COLOR_GAMMA = 0.6
HEATMAP_CMAP = "viridis"
TICK_LABEL_SIZE = 13
AXIS_LABEL_SIZE = 15
TITLE_SIZE = 16
COLORBAR_LABEL_SIZE = 14
COLORBAR_TICK_SIZE = 12

if not SHOW_FIGURES:
	matplotlib.use("Agg")

import matplotlib.pyplot as plt


def extract_pvalue(file_path):
	match = re.search(r"pVal([0-9]*\.?[0-9]+)", file_path)
	if match is None:
		return None
	return float(match.group(1))


def detect_algorithms(columns, metrics, percentages):
	algorithms = set()
	pct_pattern = "|".join(str(p) for p in percentages)
	regex = re.compile(rf"^({'|'.join(metrics)})_(.+?)_({pct_pattern})(?:_.+)?$")

	for col in columns:
		match = regex.match(col)
		if match is not None:
			algorithms.add(match.group(2))

	return sorted(algorithms)


def get_metric_alg_pct_value(df, metric, algorithm, percentage):
	prefix = f"{metric}_{algorithm}_{percentage}"
	cols = [c for c in df.columns if c.startswith(prefix)]

	if len(cols) == 0:
		return np.nan

	return df[cols].mean(axis=1).mean()


def safe_filename(text):
	return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def display_algorithm_name(algorithm):
	if algorithm == "FCI-FS":
		return "FCI-SF"
	return algorithm


def format_pvalue_label(p_value):
	# Map encoded p-value indices from filenames (0..5) to requested labels.
	if abs(p_value - round(p_value)) < 1e-9:
		idx = int(round(p_value))
		label_map = {
			0: r"$10^{-1}$",
			1: r"$10^{-1}\cdot 2^{-1}$",
			2: r"$10^{-1}\cdot 2^{-2}$",
			3: r"$10^{-1}\cdot 2^{-3}$",
			4: r"$10^{-1}\cdot 2^{-4}$",
			5: r"$10^{-1}\cdot 2^{-5}$",
		}
		if idx in label_map:
			return label_map[idx]

	return f"{p_value:g}"


def save_all_matrices_txt(out_path, matrix_exports, x_axis_name, format_x_label):
	if out_path.exists():
		out_path.unlink()

	with out_path.open("w", encoding="utf-8") as f:
		f.write("# Combined heatmap matrices\n")

		for idx, (metric, algorithm, x_labels, y_labels, matrix) in enumerate(matrix_exports):
			if idx > 0:
				f.write("\n")

			f.write(f"## {metric} | {algorithm}\n")
			f.write("# rows: number of observed variables\n")
			f.write(f"# cols: {x_axis_name}\n")
			f.write(f"# x_labels ({x_axis_name}): " + ", ".join(format_x_label(v) for v in x_labels) + "\n")
			f.write("# y_labels (observed vars): " + ", ".join(str(v) for v in y_labels) + "\n")
			np.savetxt(f, matrix, fmt="%.6f")


def save_figure_jpeg(fig, out_path, dpi=600, quality=200):
    """
    Save a Matplotlib figure as an optimized JPEG.

    The figure's physical dimensions are not changed. Pixel dimensions are
    determined by the figure size and the requested DPI.

    Args:
        fig: Matplotlib Figure.
        out_path: Destination JPEG path.
        dpi: Rendering resolution. Defaults to 600 DPI.
        quality: JPEG quality from 1 to 95. Lower values produce smaller files.

    Returns:
        Dictionary containing output information.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render_buffer = BytesIO()

    # Render losslessly first to avoid JPEG-to-JPEG recompression.
    fig.savefig(
        render_buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )

    render_buffer.seek(0)

    with Image.open(render_buffer) as source:
        image = source.convert("RGB")

        image.save(
            out_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=2,
            dpi=(dpi, dpi),
        )

    file_size = out_path.stat().st_size

    return {
        "path": out_path,
        "format": "jpeg",
        "size_kb": file_size / 1024,
        "dpi": dpi,
        "quality": quality,
        "width_px": image.width,
        "height_px": image.height,
    }

def main():
	data_dir = (
		Path(__file__).resolve().parents[3]
		/ "clean_datasets"
		/ "synthetic_data"
		/ "50_vart_all_alg"
	)
	path = str(data_dir / "*.csv")
	files = glob(path)

	if len(files) == 0:
		print(f"No files found for pattern: {path}")
		return

	data_by_file = []
	for file_path in files:
		p_val = extract_pvalue(file_path)
		if p_val is None:
			continue

		try:
			df = pd.read_csv(file_path)
		except Exception as exc:
			print(f"Skipping {file_path}: {exc}")
			continue

		data_by_file.append((p_val, df, file_path))

	if len(data_by_file) == 0:
		print("No valid CSVs with p-values were loaded.")
		return

	data_by_file.sort(key=lambda x: x[0])
	all_columns = pd.Index([])
	for _, df, _ in data_by_file:
		all_columns = all_columns.union(df.columns)

	algorithms = detect_algorithms(all_columns, METRICS, PERCENTAGES)
	if len(algorithms) == 0:
		print("No algorithm names could be inferred from columns like HD_<alg>_20.")
		return

	print(f"Loaded {len(data_by_file)} files with p-values.")
	print(f"Detected algorithms: {', '.join(algorithms)}")

	if SAVE_FIGURES or SAVE_MATRICES:
		OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	matrix_exports = []

	for metric in METRICS:
		metric_heatmaps = []

		for alg in algorithms:
			rows = []
			p_labels = []

			for p_val, df, _ in data_by_file:
				row = [
					get_metric_alg_pct_value(df, metric, alg, pct)
					for pct in PERCENTAGES
				]

				if np.all(np.isnan(row)):
					continue

				p_labels.append(p_val)
				rows.append(row)

			if len(rows) == 0:
				continue

			# rows are p-values and columns are percentages; transpose so:
			# x-axis -> p-value, y-axis -> observed variable percentage
			heatmap_data = np.array(rows, dtype=float).T
			metric_heatmaps.append((alg, p_labels, heatmap_data))

		if len(metric_heatmaps) == 0:
			continue

		all_values = np.concatenate([
			hm_data.ravel() for _, _, hm_data in metric_heatmaps
		])
		valid_values = all_values[~np.isnan(all_values)]

		if valid_values.size == 0:
			continue

		metric_vmin = float(np.min(valid_values))
		metric_vmax = float(np.max(valid_values))
		if np.isclose(metric_vmin, metric_vmax):
			norm = mcolors.Normalize(vmin=metric_vmin, vmax=metric_vmax)
		else:
			norm = mcolors.PowerNorm(gamma=COLOR_GAMMA, vmin=metric_vmin, vmax=metric_vmax)

		for alg, p_labels, heatmap_data in metric_heatmaps:
			observed_var_counts = [
				int(TOTAL_VARIABLES * pct / 100)
				for pct in PERCENTAGES
			]
			display_alg = display_algorithm_name(alg)

			fig, ax = plt.subplots(figsize=(8, 5))
			im = ax.imshow(
				heatmap_data,
				aspect="auto",
				origin="lower",
				cmap=HEATMAP_CMAP,
				norm=norm,
			)

			ax.set_xticks(range(len(p_labels)))
			ax.set_xticklabels([format_pvalue_label(p) for p in p_labels], fontsize=TICK_LABEL_SIZE)
			plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
			ax.set_yticks(range(len(PERCENTAGES)))
			ax.set_yticklabels([str(v) for v in observed_var_counts], fontsize=TICK_LABEL_SIZE)

			ax.set_xlabel("p-value", fontsize=AXIS_LABEL_SIZE)
			ax.set_ylabel("Number of observed variables", fontsize=AXIS_LABEL_SIZE)
			ax.set_title(f"{metric} | {display_alg}", fontsize=TITLE_SIZE)

			cbar = fig.colorbar(im, ax=ax)
			cbar.set_label("Mean value", fontsize=COLORBAR_LABEL_SIZE)
			cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

			plt.tight_layout()

			base_name = safe_filename(f"heatmap_{metric}_{alg}")

			if SAVE_MATRICES:
				matrix_exports.append((metric, display_alg, p_labels, observed_var_counts, heatmap_data))

			if SAVE_FIGURES:
				fig_path = OUTPUT_DIR / f"{base_name}.jpeg"
				save_figure_jpeg(fig, fig_path)
				print(f"Saved: {fig_path}")

			if SHOW_FIGURES:
				plt.show()
			else:
				plt.close(fig)

	if SAVE_MATRICES and len(matrix_exports) > 0:
		combined_txt_path = OUTPUT_DIR / "heatmap_all_matrices.txt"
		save_all_matrices_txt(combined_txt_path, matrix_exports, "p-value", format_pvalue_label)
		print(f"Saved: {combined_txt_path}")


if __name__ == "__main__":
	main()
