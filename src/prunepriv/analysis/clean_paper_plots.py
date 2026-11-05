from __future__ import annotations

import os
from pathlib import Path

from prunepriv.paths import results_dir as _results_dir

ROOT = Path(os.environ.get("PRUNEPRIV_ANALYSIS_ROOT", _results_dir("analysis"))).resolve()
ANALYSIS_ROOT = ROOT
OUTPUT_DIR = ANALYSIS_ROOT / "outputs" / "paper"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

MPLCONFIGDIR = ANALYSIS_ROOT / "_cache" / "matplotlib"
IPYTHONDIR = ANALYSIS_ROOT / "_cache" / "ipython"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
IPYTHONDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("IPYTHONDIR", str(IPYTHONDIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns

from prunepriv.analysis import report_plot_utils as rpu


DATASET_KEYS = ["cifar10"]
MEM_DATASET_KEYS = ["cifar10"]
RMIA_DATASET_KEYS = ["cifar10"]
METHOD_KEYS = ["std", "ib", "ib_prw"]
METHOD_LABELS = {"std": "Standard", "ib": "InfoBatch", "ib_prw": "WLIB (Ours)"}
METHOD_COLORS = {"std": "#3B6FB6", "ib": "#D95F02", "ib_prw": "#2F8F5B"}
METHOD_LINESTYLES = {"std": "-", "ib": "--", "ib_prw": "-"}
IMAGENET_METHOD_LINESTYLES = {"std": "-", "ib": (0, (5.5, 3.2)), "ib_prw": "-"}
DIFFICULTY_ORDER = ["Easy", "Medium", "Hard"]
DIFFICULTY_ORDER_DESC = ["Easy", "Medium", "Hard"]
DIFFICULTY_COLORS = {"Easy": "#5DA5DA", "Medium": "#F1C232", "Hard": "#D95F5F"}
MEM_EDGES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001])
MEM_LABELS = ["0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]


def set_ours_method_label(label: str) -> None:
    METHOD_LABELS["ib_prw"] = str(label)


def figure_dataset_label(label: str) -> str:
    return str(label).split(" / ")[0]


def lighten_color(color: str, amount: float = 0.58) -> tuple[float, float, float]:
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=np.float64)
    return tuple(rgb + (1.0 - rgb) * amount)


def endpoint_label_positions(values: dict[str, float], ymin: float, ymax: float, *, spacing_frac: float = 0.09) -> dict[str, float]:
    if not values:
        return {}
    span = max(float(ymax) - float(ymin), 1e-8)
    spacing = max(span * spacing_frac, 1e-3)
    ordered = sorted(values, key=values.get, reverse=True)
    raw_positions = {key: float(values[key]) for key in ordered}
    positions = {}
    prev = None
    for key in ordered:
        pos = raw_positions[key]
        if prev is not None and pos > prev - spacing:
            pos = prev - spacing
        positions[key] = pos
        prev = pos
    low_limit = ymin + 0.05 * span
    high_limit = ymax - 0.05 * span
    min_pos = min(positions.values())
    if min_pos < low_limit:
        shift = low_limit - min_pos
        positions = {key: val + shift for key, val in positions.items()}
    max_pos = max(positions.values())
    if max_pos > high_limit:
        shift = max_pos - high_limit
        positions = {key: val - shift for key, val in positions.items()}
    return positions


def add_endpoint_labels(ax: plt.Axes, end_values: dict[str, float], last_epoch: float, y_max: float, colors: dict[str, str] | None = None) -> None:
    if colors is None:
        colors = METHOD_COLORS
    positions = endpoint_label_positions(end_values, 0.0, y_max, spacing_frac=0.085)
    x0, x1 = ax.get_xlim()
    label_x = last_epoch + 0.025 * (x1 - x0)
    for method_key, y_value in end_values.items():
        y_label = positions[method_key]
        ax.plot([last_epoch, label_x - 0.006 * (x1 - x0)], [y_value, y_label], color=colors[method_key], linewidth=0.65, alpha=0.85, clip_on=False)
        ax.text(
            label_x,
            y_label,
            f"{y_value:.2f}",
            ha="left",
            va="center",
            fontsize=6.8,
            color=colors[method_key],
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.17,rounding_size=0.06",
                "facecolor": "white",
                "edgecolor": colors[method_key],
                "linewidth": 0.75,
                "alpha": 0.98,
            },
        )


def set_theme(font_size: int = 9) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Bitstream Charter", "Charter", "XCharter", "Nimbus Roman", "Times", "Times New Roman", "DejaVu Serif"],
            "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage[bitstream-charter]{mathdesign}",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Bitstream Charter",
            "mathtext.it": "Bitstream Charter:italic",
            "mathtext.bf": "Bitstream Charter:bold",
            "mathtext.fallback": "stix",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": max(font_size - 1, 7),
            "ytick.labelsize": max(font_size - 1, 7),
            "legend.fontsize": max(font_size - 1, 7),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CBD5E1",
        }
    )


def initialize(output_dir: str | Path = OUTPUT_DIR) -> pd.DataFrame:
    global OUTPUT_DIR, TABLE_DIR, FIG_DIR
    OUTPUT_DIR = Path(output_dir)
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = (ROOT / OUTPUT_DIR).resolve()
    TABLE_DIR = OUTPUT_DIR / "tables"
    FIG_DIR = OUTPUT_DIR / "figures"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_theme()
    summary = rpu.initialize_report(
        {
            "active_datasets": DATASET_KEYS,
            "pruning_datasets": ["cifar10"],
            "rmia_datasets": RMIA_DATASET_KEYS,
            "all_methods": METHOD_KEYS,
            "pruned_methods": ["ib", "ib_prw"],
            "save_figures": False,
            "save_vector": False,
            "output_dir": OUTPUT_DIR,
            "figure_dpi": 600,
            "font_size": 9,
            "rmia_mem_threshold": 0.5,
            "rmia_normalized_entropy_threshold": 0.3,
        }
    )
    set_theme()
    return summary


def save_table(df: pd.DataFrame, stem: str, *, latex: bool = False) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {"csv": TABLE_DIR / f"{stem}.csv"}
    df.to_csv(paths["csv"], index=False)
    if latex:
        paths["tex"] = TABLE_DIR / f"{stem}.tex"
        df.to_latex(paths["tex"], index=False, escape=False)
    return paths


def save_fig(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def _cap(values) -> np.ndarray:
    return rpu.cap_memorization(np.asarray(values, dtype=np.float64))


def actual_phase_boundaries(epochs) -> np.ndarray:
    epoch_values = np.sort(np.asarray(list(epochs), dtype=np.float64))
    epoch_values = epoch_values[np.isfinite(epoch_values)]
    if len(epoch_values) < 2:
        raise ValueError("At least two epochs are required to build phase boundaries.")
    ideal = np.linspace(float(epoch_values.min()), float(epoch_values.max()), 4)
    boundaries = [float(epoch_values[0])]
    for value in ideal[1:-1]:
        nearest = float(epoch_values[np.argmin(np.abs(epoch_values - value))])
        if nearest <= boundaries[-1]:
            later = epoch_values[epoch_values > boundaries[-1]]
            if len(later) == 0:
                nearest = float(boundaries[-1])
            else:
                nearest = float(later[0])
        boundaries.append(nearest)
    boundaries.append(float(epoch_values[-1]))
    return np.asarray(boundaries, dtype=np.float64)


def method_keys_for_dataset(dataset_key: str, *, rmia: bool = False) -> list[str]:
    return list(METHOD_KEYS)


def pruned_method_keys_for_dataset(dataset_key: str) -> list[str]:
    return ["ib", "ib_prw"]


def _group_positions(n_groups: int, n_series: int, span: float = 0.84) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.arange(n_groups, dtype=np.float64)
    width = span / max(n_series, 1)
    offsets = (np.arange(n_series, dtype=np.float64) - (n_series - 1) / 2.0) * width
    return x, offsets, width


def _method_epoch_mem(bundle: dict, method_key: str, epoch: int) -> pd.DataFrame:
    return rpu.get_method_epoch_mem(bundle, method_key, epoch)


def _method_epoch_frame(bundle: dict, method_key: str, epoch: int, *, rmia: bool = False) -> pd.DataFrame:
    return rpu.section_build_method_frame(bundle, int(epoch), method_key, include_rmia=rmia, require_rmia=rmia)


def _resolve_pruning_metric(bundle: dict, method_key: str, frame: pd.DataFrame) -> tuple[str | None, str]:
    return rpu.resolve_pruning_metric(bundle, method_key, frame)


def build_pruning_difficulty_anchor(dataset_key: str = "cifar10", epoch: int | None = None) -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    epoch = int(epoch if epoch is not None else bundle["cfg"]["end_epoch"])
    frames = []
    # for method_key in ["ib", "ib_prw"]:
    for method_key in ["ib"]:
        frame = _method_epoch_frame(bundle, method_key, epoch, rmia=False)
        pruning_col, _ = _resolve_pruning_metric(bundle, method_key, frame)
        if pruning_col is None:
            continue
        frames.append(frame[["sample_id", pruning_col]].rename(columns={pruning_col: method_key}))
    if not frames:
        raise ValueError(f"No pruning-rate fields found for {dataset_key}.")
    anchor = frames[0]
    for frame in frames[1:]:
        anchor = anchor.merge(frame, on="sample_id", how="outer")
    # pruning_cols = [col for col in ["ib", "ib_prw"] if col in anchor.columns]
    pruning_cols = [col for col in ["ib"] if col in anchor.columns]
    anchor["pruning_rate_anchor"] = anchor[pruning_cols].mean(axis=1)
    lo = float(anchor["pruning_rate_anchor"].min())
    hi = float(anchor["pruning_rate_anchor"].max())
    if np.isclose(lo, hi):
        hi = lo + 1e-8
    edges = np.linspace(lo, hi, 4)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    anchor["difficulty"] = pd.cut(
        anchor["pruning_rate_anchor"],
        bins=edges,
        labels=["Hard", "Medium", "Easy"],
        include_lowest=True,
        ordered=True,
    )
    anchor["difficulty"] = pd.Categorical(anchor["difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
    anchor.attrs["edges"] = edges
    return anchor[["sample_id", "pruning_rate_anchor", "difficulty"]].dropna().reset_index(drop=True)


def build_pruning_difficulty_selection_summary() -> pd.DataFrame:
    rows = []
    display_order = ["Hard", "Medium", "Easy"]
    for dataset_key in MEM_DATASET_KEYS:
        bundle = rpu.report_bundle(dataset_key)
        epoch = int(bundle["cfg"]["end_epoch"])
        anchor = build_pruning_difficulty_anchor(dataset_key, epoch)
        edges = anchor.attrs["edges"]
        range_by_difficulty = {
            "Hard": (float(edges[0]), float(edges[1])),
            "Medium": (float(edges[1]), float(edges[2])),
            "Easy": (float(edges[2]), float(edges[3])),
        }
        total = int(len(anchor))
        for difficulty in display_order:
            sub = anchor.loc[anchor["difficulty"] == difficulty]
            n = int(len(sub))
            lo, hi = range_by_difficulty[difficulty]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "epoch": epoch,
                    "difficulty": difficulty,
                    "pruning_rate_min": lo,
                    "pruning_rate_max": hi,
                    "range_label": f"{lo:.2f}-{hi:.2f}",
                    "n": n,
                    "total_n": total,
                    "share_pct": 100.0 * n / max(total, 1),
                }
            )
    return pd.DataFrame(rows)


def plot_pruning_difficulty_selection_summary() -> pd.DataFrame:
    table = build_pruning_difficulty_selection_summary()
    save_table(table, "pruning_difficulty_selection_summary")
    display_order = ["Easy", "Medium", "Hard"]
    for dataset_key in MEM_DATASET_KEYS:
        bundle = rpu.report_bundle(dataset_key)
        epoch = int(bundle["cfg"]["end_epoch"])
        anchor = build_pruning_difficulty_anchor(dataset_key, epoch)
        edges = anchor.attrs["edges"]
        x = anchor["pruning_rate_anchor"].to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(3.25, 1.95))
        ax.hist(x, bins=32, color="#94A3B8", edgecolor="white", linewidth=0.4)
        range_by_difficulty = {
            "Easy": (float(edges[2]), float(edges[3])),
            "Medium": (float(edges[1]), float(edges[2])),
            "Hard": (float(edges[0]), float(edges[1])),
        }
        y_top = max(ax.get_ylim()[1], 1.0)
        for difficulty in display_order:
            lo, hi = range_by_difficulty[difficulty]
            ax.axvspan(lo, hi, color=DIFFICULTY_COLORS[difficulty], alpha=0.14, linewidth=0)
            ax.text(
                (lo + hi) / 2,
                y_top * 0.94,
                difficulty,
                ha="center",
                va="top",
                fontsize=6.6,
                color="#334155",
            )
        ax.set_ylim(0.0, y_top)
        ax.set_xlim(float(edges[3]), float(edges[0]))
        ax.set_xlabel("Pruning rate")
        ax.set_ylabel("Samples")
        if dataset_key == "cifar10":
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
            ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, axis="y")
        fig.subplots_adjust(left=0.16, right=0.985, bottom=0.27, top=0.96)
        save_fig(fig, f"{dataset_key}_pruning_difficulty_selection_distribution")
        plt.show()
        plt.close(fig)
    return table


def _range_labels(edges: np.ndarray, precision: int = 2) -> list[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo:.{precision}f}-{hi:.{precision}f}")
    return labels


def build_hard_pruning_band_anchor(dataset_key: str, epoch: int | None = None, n_bins: int = 10) -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    epoch = int(epoch if epoch is not None else bundle["cfg"]["end_epoch"])
    anchor = build_pruning_difficulty_anchor(dataset_key, epoch)
    hard = anchor.loc[anchor["difficulty"] == "Hard", ["sample_id", "pruning_rate_anchor"]].dropna().copy()
    lo = float(hard["pruning_rate_anchor"].min())
    hi = float(hard["pruning_rate_anchor"].max())
    if np.isclose(lo, hi):
        hi = lo + 1e-8
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    labels = _range_labels(edges)
    hard["pruning_band"] = pd.cut(
        hard["pruning_rate_anchor"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    hard["pruning_band"] = pd.Categorical(hard["pruning_band"], categories=labels, ordered=True)
    hard["pruning_band_min"] = hard["pruning_band"].map(dict(zip(labels, edges[:-1]))).astype(float)
    hard["pruning_band_max"] = hard["pruning_band"].map(dict(zip(labels, edges[1:]))).astype(float)
    return hard.reset_index(drop=True)


def plot_tail_histogram_by_pruning_difficulty(dataset_key: str = "cifar10", epoch: int | None = None) -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    final_epoch = int(bundle["cfg"]["end_epoch"])
    epoch = int(final_epoch if epoch is None else epoch)
    anchor = build_pruning_difficulty_anchor(dataset_key, epoch)[["sample_id", "difficulty"]]
    method_keys = method_keys_for_dataset(dataset_key)
    rows = []
    for method_key in method_keys:
        mem = _method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]]
        frame = anchor.merge(mem, on="sample_id", how="inner")
        frame["memorization"] = _cap(frame["memorization"])
        frame = frame.loc[(frame["memorization"] > 0.1) & (frame["memorization"] <= 1.0)].copy()
        bin_ids = rpu.memorization_bin_index(frame["memorization"].to_numpy(dtype=np.float64))
        for bin_idx, label in enumerate(MEM_LABELS):
            mask = bin_ids == bin_idx
            for difficulty in DIFFICULTY_ORDER_DESC:
                rows.append(
                    {
                        "mem_bin": label,
                        "method_key": method_key,
                        "method": METHOD_LABELS[method_key],
                        "epoch": epoch,
                        "difficulty": difficulty,
                        "n": int(np.sum(mask & (frame["difficulty"].astype(str).to_numpy() == difficulty))),
                    }
                )
    table = pd.DataFrame(rows)
    stem = f"{dataset_key}_tail_histogram_by_pruning_difficulty"
    if epoch != final_epoch:
        stem = f"{stem}_epoch_{epoch}"
    save_table(table, stem)

    fig, ax = plt.subplots(figsize=(7.8, 2.9))
    x, offsets, width = _group_positions(len(MEM_LABELS), len(method_keys))
    hatch_by_difficulty = {"Easy": "---", "Medium": "...", "Hard": "///"}
    for idx, method_key in enumerate(method_keys):
        positions = x + offsets[idx]
        bottom = np.zeros(len(MEM_LABELS), dtype=np.float64)
        for difficulty in DIFFICULTY_ORDER_DESC:
            sub = (
                table.loc[(table["method_key"] == method_key) & (table["difficulty"] == difficulty)]
                .set_index("mem_bin")
                .reindex(MEM_LABELS)["n"]
                .fillna(0)
                .to_numpy(dtype=np.float64)
            )
            ax.bar(
                positions,
                sub,
                bottom=bottom,
                width=width,
                color=METHOD_COLORS[method_key],
                hatch=hatch_by_difficulty[difficulty],
                edgecolor="black",
                linewidth=0.45,
            )
            bottom += sub
    ax.set_xticks(x)
    ax.set_xticklabels(MEM_LABELS, rotation=0, ha="center", fontsize=11)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlabel("Memorization bin", fontsize=12)
    ax.set_ylabel("Samples", fontsize=12)
    ax.grid(True, axis="y")
    method_handles = [mpatches.Patch(facecolor=METHOD_COLORS[k], edgecolor="black", label=METHOD_LABELS[k]) for k in method_keys]
    hardness_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=hatch_by_difficulty[difficulty], label=difficulty)
        for difficulty in DIFFICULTY_ORDER_DESC
    ]
    method_legend = ax.legend(handles=method_handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=1, borderaxespad=0.2, fontsize=10.4)
    ax.add_artist(method_legend)
    ax.legend(handles=hardness_handles, loc="upper right", bbox_to_anchor=(0.78, 0.985), ncol=1, borderaxespad=0.2, fontsize=10.3)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.96)
    save_fig(fig, stem)
    plt.show()
    plt.close(fig)
    return table


def plot_cifar10_tail_histogram_by_pruning_difficulty() -> pd.DataFrame:
    return plot_tail_histogram_by_pruning_difficulty("cifar10")


def plot_cifar10_tail_histogram_by_pruning_difficulty_epoch_87() -> pd.DataFrame:
    return plot_tail_histogram_by_pruning_difficulty("cifar10", epoch=87)


def build_pruning_difficulty_trajectories(dataset_key: str = "cifar10") -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    anchor = build_pruning_difficulty_anchor(dataset_key, int(bundle["cfg"]["end_epoch"]))
    method_keys = method_keys_for_dataset(dataset_key)
    rows = []
    for method_key in method_keys:
        mem_long = bundle["mem_long"][method_key]
        for epoch, chunk in mem_long.groupby("epoch", sort=True):
            if dataset_key == "cifar10" and int(epoch) == 88:
                continue
            merged = anchor.merge(chunk[["sample_id", "memorization"]], on="sample_id", how="inner")
            merged["memorization"] = _cap(merged["memorization"])
            summary = (
                merged.groupby("difficulty", observed=False)
                .agg(mean_memorization=("memorization", "mean"), n=("sample_id", "size"))
                .reset_index()
            )
            summary["dataset_key"] = dataset_key
            summary["dataset"] = bundle["cfg"]["label"]
            summary["method_key"] = method_key
            summary["method"] = METHOD_LABELS[method_key]
            summary["epoch"] = int(epoch)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def plot_pruning_difficulty_trajectories() -> pd.DataFrame:
    table = pd.concat([build_pruning_difficulty_trajectories(key) for key in MEM_DATASET_KEYS], ignore_index=True)
    save_table(table, "pruning_difficulty_mean_trajectories")
    for dataset_key in MEM_DATASET_KEYS:
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3), sharey=True, squeeze=False)
        axes = axes[0]
        anchor = build_pruning_difficulty_anchor(dataset_key, int(rpu.report_bundle(dataset_key)["cfg"]["end_epoch"]))
        edges = anchor.attrs["edges"]
        difficulty_ranges = {
            "Easy": f"{edges[2]:.2f}-{edges[3]:.2f}",
            "Medium": f"{edges[1]:.2f}-{edges[2]:.2f}",
            "Hard": f"{edges[0]:.2f}-{edges[1]:.2f}",
        }
        method_keys = method_keys_for_dataset(dataset_key)
        for col_idx, difficulty in enumerate(DIFFICULTY_ORDER_DESC):
            ax = axes[col_idx]
            sub_d = table.loc[(table["dataset_key"] == dataset_key) & (table["difficulty"] == difficulty)]
            ymax = max(0.08, float(table.loc[table["dataset_key"] == dataset_key, "mean_memorization"].max()) * 1.22)
            ax.set_ylim(0, ymax)
            epochs_all = sub_d["epoch"].to_numpy(dtype=np.float64)
            if len(epochs_all):
                epoch_span = max(float(np.nanmax(epochs_all) - np.nanmin(epochs_all)), 1.0)
                ax.set_xlim(float(np.nanmin(epochs_all)) - 0.02 * epoch_span, float(np.nanmax(epochs_all)) + 0.16 * epoch_span)
            for method_key in method_keys:
                sub = sub_d.loc[sub_d["method_key"] == method_key].sort_values("epoch")
                ax.plot(
                    sub["epoch"],
                    sub["mean_memorization"],
                    color=METHOD_COLORS[method_key],
                    marker="o",
                    markersize=2.3,
                    linewidth=1.2,
                    label=METHOD_LABELS[method_key],
                )
            ax.text(
                0.02,
                0.95,
                f"{difficulty}\n$\\rho$: {difficulty_ranges[difficulty]}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                # fontweight="semibold",
                fontsize=7.4,
                linespacing=1.15,
            )
            if col_idx == 0:
                ax.set_ylabel("Mean memorization")
            ax.set_xlabel("Epoch")
            ax.grid(True, axis="y")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(labels))
        fig.subplots_adjust(left=0.11, right=0.965, bottom=0.20, top=0.84, wspace=0.22)
        save_fig(fig, f"{dataset_key}_pruning_difficulty_mean_trajectories")
        plt.show()
        plt.close(fig)
    return table


def plot_pruning_difficulty_trajectories_separate(table: pd.DataFrame | None = None) -> pd.DataFrame:
    if table is None:
        table = pd.concat([build_pruning_difficulty_trajectories(key) for key in MEM_DATASET_KEYS], ignore_index=True)
        save_table(table, "pruning_difficulty_mean_trajectories")
    for dataset_key in MEM_DATASET_KEYS:
        anchor = build_pruning_difficulty_anchor(dataset_key, int(rpu.report_bundle(dataset_key)["cfg"]["end_epoch"]))
        edges = anchor.attrs["edges"]
        difficulty_ranges = {
            "Easy": f"{edges[2]:.2f}-{edges[3]:.2f}",
            "Medium": f"{edges[1]:.2f}-{edges[2]:.2f}",
            "Hard": f"{edges[0]:.2f}-{edges[1]:.2f}",
        }
        method_keys = method_keys_for_dataset(dataset_key)
        dataset_table = table.loc[table["dataset_key"] == dataset_key]
        ymax = max(0.08, float(dataset_table["mean_memorization"].max()) * 1.16)
        for difficulty in DIFFICULTY_ORDER_DESC:
            fig, ax = plt.subplots(figsize=(2.75, 2.35))
            sub_d = dataset_table.loc[dataset_table["difficulty"] == difficulty]
            epochs_all = sub_d["epoch"].to_numpy(dtype=np.float64)
            if len(epochs_all):
                epoch_span = max(float(np.nanmax(epochs_all) - np.nanmin(epochs_all)), 1.0)
                ax.set_xlim(float(np.nanmin(epochs_all)) - 0.02 * epoch_span, float(np.nanmax(epochs_all)) + 0.05 * epoch_span)
            for method_key in method_keys:
                sub = sub_d.loc[sub_d["method_key"] == method_key].sort_values("epoch")
                ax.plot(
                    sub["epoch"],
                    sub["mean_memorization"],
                    color=METHOD_COLORS[method_key],
                    marker="o",
                    markersize=2.3,
                    linewidth=1.25,
                    label=METHOD_LABELS[method_key],
                )
            if difficulty != "Hard":
                ax.text(
                    0.02,
                    0.95,
                    f"{difficulty}\n$\\rho$: {difficulty_ranges[difficulty]}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.4,
                    linespacing=1.15,
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Mean memorization")
            ax.set_ylim(0, ymax)
            ax.grid(True, axis="y")
            if dataset_key == "cifar10" and difficulty == "Hard":
                ax.xaxis.label.set_size(6)
                ax.yaxis.label.set_size(6)
                ax.tick_params(axis="both", labelsize=5)
            handles, labels = ax.get_legend_handles_labels()
            legend_fontsize = 5 if dataset_key == "cifar10" and difficulty == "Hard" else None
            legend_kwargs = {"fontsize": legend_fontsize} if legend_fontsize is not None else {}
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(labels), **legend_kwargs)
            fig.subplots_adjust(left=0.18, right=0.96, bottom=0.22, top=0.82)
            save_fig(fig, f"{dataset_key}_{difficulty.lower()}_pruning_difficulty_mean_trajectory")
            plt.show()
            plt.close(fig)
    return table


def build_cifar10_pruning_difficulty_loss_trajectories() -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    anchor = build_pruning_difficulty_anchor(dataset_key, int(bundle["cfg"]["end_epoch"]))[["sample_id", "difficulty", "pruning_rate_anchor"]]
    rows = []
    for method_key in method_keys_for_dataset(dataset_key):
        mem_long = bundle["mem_long"][method_key]
        for epoch in sorted(mem_long["epoch"].dropna().unique()):
            frame = _method_epoch_frame(bundle, method_key, int(epoch), rmia=False)
            if "loss" not in frame.columns or frame["loss"].isna().all():
                continue
            merged = anchor.merge(frame[["sample_id", "loss"]], on="sample_id", how="inner").dropna(subset=["loss"])
            summary = (
                merged.groupby("difficulty", observed=False)
                .agg(
                    mean_loss=("loss", "mean"),
                    loss_std=("loss", "std"),
                    mean_pruning_rate=("pruning_rate_anchor", "mean"),
                    n=("sample_id", "size"),
                )
                .reindex(DIFFICULTY_ORDER)
                .reset_index()
            )
            summary["loss_sem"] = summary["loss_std"] / np.sqrt(summary["n"].clip(lower=1))
            summary["dataset_key"] = dataset_key
            summary["dataset"] = bundle["cfg"]["label"]
            summary["method_key"] = method_key
            summary["method"] = METHOD_LABELS[method_key]
            summary["epoch"] = int(epoch)
            rows.append(summary)
    table = pd.concat(rows, ignore_index=True)
    table["abs_delta_mean_loss"] = np.nan
    for method_key in method_keys_for_dataset(dataset_key):
        for difficulty in DIFFICULTY_ORDER:
            mask = (table["method_key"] == method_key) & (table["difficulty"] == difficulty)
            sub = table.loc[mask].sort_values("epoch")
            table.loc[sub.index, "abs_delta_mean_loss"] = sub["mean_loss"].diff().abs().to_numpy(dtype=np.float64)
    return table


def build_cifar10_hard_loss_phase_summary() -> pd.DataFrame:
    table = build_cifar10_pruning_difficulty_loss_trajectories()
    hard = table.loc[table["difficulty"] == "Hard"].copy()
    epochs = np.sort(hard["epoch"].dropna().astype(np.float64).unique())
    phase_edges = actual_phase_boundaries(epochs)
    rows = []
    for method_key in method_keys_for_dataset("cifar10"):
        sub = hard.loc[hard["method_key"] == method_key].sort_values("epoch")
        for phase_idx, phase in enumerate(["Early", "Middle", "Late"]):
            phase_start = float(phase_edges[phase_idx])
            phase_end = float(phase_edges[phase_idx + 1])
            phase_sub = sub.loc[(sub["epoch"].astype(float) >= phase_start) & (sub["epoch"].astype(float) <= phase_end)].sort_values("epoch")
            if phase_sub.empty:
                continue
            start_row = phase_sub.loc[phase_sub["epoch"].astype(float) == phase_start]
            end_row = phase_sub.loc[phase_sub["epoch"].astype(float) == phase_end]
            if start_row.empty or end_row.empty:
                continue
            start_loss = float(start_row.iloc[0]["mean_loss"])
            end_loss = float(end_row.iloc[0]["mean_loss"])
            rows.append(
                {
                    "dataset_key": "cifar10",
                    "dataset": sub["dataset"].iloc[0],
                    "method_key": method_key,
                    "method": METHOD_LABELS[method_key],
                    "difficulty": "Hard",
                    "phase": phase,
                    "phase_start_epoch": phase_start,
                    "phase_end_epoch": phase_end,
                    "mean_loss_phase": float(phase_sub["mean_loss"].mean()),
                    "median_loss_phase": float(phase_sub["mean_loss"].median()),
                    "mean_abs_delta_loss_phase": float(phase_sub["abs_delta_mean_loss"].dropna().mean()) if phase_sub["abs_delta_mean_loss"].notna().any() else np.nan,
                    "median_abs_delta_loss_phase": float(phase_sub["abs_delta_mean_loss"].dropna().median()) if phase_sub["abs_delta_mean_loss"].notna().any() else np.nan,
                    "phase_start_loss": start_loss,
                    "phase_end_loss": end_loss,
                    "phase_loss_change": end_loss - start_loss,
                    "n_checkpoints": int(len(phase_sub)),
                }
            )
    return pd.DataFrame(rows)


def plot_cifar10_hard_loss_phase_summary() -> pd.DataFrame:
    table = build_cifar10_hard_loss_phase_summary()
    save_table(table, "cifar10_hard_loss_phase_summary")
    phase_order = ["Early", "Middle", "Late"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.75))
    x, offsets, width = _group_positions(len(phase_order), len(METHOD_KEYS))
    metrics = [
        ("phase_end_loss", "Hard-sample loss at phase end", axes[0]),
        ("phase_loss_change", "Net hard-sample loss change", axes[1]),
    ]
    for metric, ylabel, ax in metrics:
        vals_all = table[metric].dropna().to_numpy(dtype=np.float64)
        if metric == "phase_loss_change":
            ymax = max(abs(float(np.nanmin(vals_all))), abs(float(np.nanmax(vals_all)))) if len(vals_all) else 1e-6
        else:
            ymax = float(np.nanmax(vals_all)) if len(vals_all) else 1e-6
        for idx, method_key in enumerate(METHOD_KEYS):
            sub = table.loc[table["method_key"] == method_key].set_index("phase").reindex(phase_order)
            vals = sub[metric].to_numpy(dtype=np.float64)
            ax.bar(x + offsets[idx], vals, width=width, color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
        ax.set_xticks(x)
        ax.set_xticklabels(phase_order)
        ax.set_ylabel(ylabel)
        if metric == "phase_loss_change":
            ax.axhline(0.0, color="#334155", linewidth=0.8)
            ax.set_ylim(-max(ymax * 1.20, 1e-6), max(ymax * 1.20, 1e-6))
        else:
            ax.set_ylim(0.0, max(ymax * 1.20, 1e-6))
        ax.grid(True, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=len(labels))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.80, wspace=0.28)
    save_fig(fig, "cifar10_hard_loss_phase_summary")
    plt.show()
    plt.close(fig)
    return table


def plot_hard_sample_pruning_response(dataset_key: str = "cifar10") -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    epoch = int(bundle["cfg"]["end_epoch"])
    hard_anchor = build_hard_pruning_band_anchor(dataset_key, epoch, n_bins=10)
    band_order = list(hard_anchor["pruning_band"].cat.categories)
    method_keys = pruned_method_keys_for_dataset(dataset_key)
    rows = []
    for method_key in method_keys:
        frame = _method_epoch_frame(bundle, method_key, epoch, rmia=False)
        sub = hard_anchor.merge(frame[["sample_id", "memorization"]], on="sample_id", how="inner").dropna()
        sub["memorization"] = _cap(sub["memorization"])
        summary = (
            sub.groupby("pruning_band", observed=False)
            .agg(
                mean_memorization=("memorization", "mean"),
                n=("sample_id", "size"),
                mean_pruning_rate=("pruning_rate_anchor", "mean"),
                pruning_band_min=("pruning_band_min", "first"),
                pruning_band_max=("pruning_band_max", "first"),
            )
            .reset_index()
        )
        summary["dataset_key"] = dataset_key
        summary["dataset"] = bundle["cfg"]["label"]
        summary["method_key"] = method_key
        summary["method"] = METHOD_LABELS[method_key]
        summary["epoch"] = epoch
        rows.append(summary)
    table = pd.concat(rows, ignore_index=True)
    save_table(table, f"{dataset_key}_hard_sample_pruning_response")

    fig, ax = plt.subplots(figsize=(4.25, 2.55))
    y_max = max(0.08, float(table["mean_memorization"].max()) * 1.18)
    x_min = float(table["pruning_band_min"].min())
    x_max = float(table["pruning_band_max"].max())
    x_pad = 0.04 * max(x_max - x_min, 1e-8)
    for method_key in method_keys:
        sub = table.loc[table["method_key"] == method_key].set_index("pruning_band").reindex(band_order)
        x = sub["mean_pruning_rate"].to_numpy(dtype=np.float64)
        y = sub["mean_memorization"].to_numpy(dtype=np.float64)
        ax.plot(x, y, color=METHOD_COLORS[method_key], linewidth=0.9, alpha=0.55)
        ax.scatter(x, y, s=22, color=METHOD_COLORS[method_key], edgecolor="white", linewidth=0.55, zorder=3, label=METHOD_LABELS[method_key])
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("InfoBatch pruning-rate anchor, hard samples")
    ax.set_ylabel("Final mean memorization")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", borderaxespad=0.35, fontsize=6.6)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.24, top=0.96)
    save_fig(fig, f"{dataset_key}_hard_sample_pruning_response")
    plt.show()
    plt.close(fig)
    return table


def _checkpoint_for_late_curve(available_epochs, *, low: int = 70, high: int = 80) -> int:
    epochs = np.sort(np.asarray(available_epochs, dtype=np.int64))
    in_window = epochs[(epochs >= int(low)) & (epochs <= int(high))]
    if len(in_window):
        return int(in_window[0])
    below = epochs[epochs < int(low)]
    if len(below):
        return int(below[-1])
    above = epochs[epochs > int(high)]
    if len(above):
        return int(above[0])
    return int(epochs[-1])


def build_hard_pruning_epoch_memorization(dataset_key: str = "cifar10") -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    final_epoch = int(bundle["cfg"]["end_epoch"])
    hard_anchor = build_hard_pruning_band_anchor(dataset_key, final_epoch, n_bins=10)
    method_keys = pruned_method_keys_for_dataset(dataset_key)
    available_epochs = np.sort(bundle["mem_long"]["ib"]["epoch"].dropna().astype(int).unique())
    target_epochs = [_checkpoint_for_late_curve(available_epochs, low=70, high=80)]
    candidates = available_epochs[available_epochs >= 85]
    if len(candidates):
        target_epochs.append(int(candidates[0]))
    target_epochs.append(final_epoch)
    target_epochs = list(dict.fromkeys(target_epochs))
    if dataset_key == "cifar10":
        target_epochs = [epoch for epoch in target_epochs if epoch != 73]
    rows = []
    for method_key in method_keys:
        for epoch in target_epochs:
            mem = _method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]]
            sub = hard_anchor.merge(mem, on="sample_id", how="inner").dropna()
            sub["memorization"] = _cap(sub["memorization"])
            summary = (
                sub.groupby("pruning_band", observed=False)
                .agg(
                    mean_memorization=("memorization", "mean"),
                    n=("sample_id", "size"),
                    mean_pruning_rate=("pruning_rate_anchor", "mean"),
                    pruning_band_min=("pruning_band_min", "first"),
                    pruning_band_max=("pruning_band_max", "first"),
                )
                .reset_index()
            )
            summary["dataset_key"] = dataset_key
            summary["dataset"] = bundle["cfg"]["label"]
            summary["method_key"] = method_key
            summary["method"] = METHOD_LABELS[method_key]
            summary["epoch"] = int(epoch)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def plot_hard_pruning_epoch_memorization(dataset_key: str = "cifar10") -> pd.DataFrame:
    table = build_hard_pruning_epoch_memorization(dataset_key)
    save_table(table, f"{dataset_key}_hard_pruning_epoch_memorization")
    band_meta = table[["pruning_band", "pruning_band_min"]].drop_duplicates().sort_values("pruning_band_min")
    bands = band_meta["pruning_band"].tolist()
    epochs = sorted(table["epoch"].unique())
    final_epoch = int(epochs[-1])
    shared_epochs = [int(epoch) for epoch in epochs if int(epoch) != final_epoch]
    shared_styles = [("#64748B", "-", "o"), ("#334155", "--", "s")]
    series = [
        ("ib", epoch, f"IB/WLIB epoch {epoch}", color, linestyle, marker)
        for epoch, (color, linestyle, marker) in zip(shared_epochs, shared_styles)
    ]
    series.extend(
        [
            ("ib", final_epoch, f"InfoBatch epoch {final_epoch}", METHOD_COLORS["ib"], ":", "^"),
            ("ib_prw", final_epoch, f"{METHOD_LABELS['ib_prw']} epoch {final_epoch}", METHOD_COLORS["ib_prw"], "-.", "D"),
        ]
    )
    fig, ax = plt.subplots(figsize=(4.45, 2.75))
    y_max = max(0.08, float(table["mean_memorization"].max()) * 1.18)
    x_min = float(table["pruning_band_min"].min())
    x_max = float(table["pruning_band_max"].max())
    x_pad = 0.04 * max(x_max - x_min, 1e-8)
    for method_key, epoch, label, color, linestyle, marker in series:
        sub = table.loc[(table["method_key"] == method_key) & (table["epoch"] == epoch)].set_index("pruning_band").reindex(bands)
        x = sub["mean_pruning_rate"].to_numpy(dtype=np.float64)
        y = sub["mean_memorization"].to_numpy(dtype=np.float64)
        ax.plot(x, y, color=color, linewidth=0.9, alpha=0.62, linestyle=linestyle)
        ax.scatter(x, y, s=22, marker=marker, color=color, edgecolor="white", linewidth=0.55, zorder=3, label=label)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel("InfoBatch pruning rate (hard samples)")
    ax.set_ylabel("Mean memorization")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", borderaxespad=0.35, fontsize=6.3)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.24, top=0.96)
    save_fig(fig, f"{dataset_key}_hard_pruning_epoch_memorization")
    plt.show()
    plt.close(fig)
    return table


def build_cifar10_hard_pruning_epoch_memorization() -> pd.DataFrame:
    return build_hard_pruning_epoch_memorization("cifar10")


def plot_cifar10_hard_pruning_epoch_memorization() -> pd.DataFrame:
    return plot_hard_pruning_epoch_memorization("cifar10")


def build_cifar10_hard_pruning_memorization_scatter() -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    final_epoch = int(bundle["cfg"]["end_epoch"])
    anchor = build_pruning_difficulty_anchor(dataset_key, final_epoch)
    hard = anchor.loc[anchor["difficulty"] == "Hard", ["sample_id", "pruning_rate_anchor", "difficulty"]].copy()
    rows = []
    for method_key in method_keys_for_dataset(dataset_key):
        mem_long = bundle["mem_long"][method_key]
        epochs = np.sort(mem_long["epoch"].dropna().astype(float).unique())
        late_start_epoch = int(actual_phase_boundaries(epochs)[2])
        start = _method_epoch_mem(bundle, method_key, late_start_epoch)[["sample_id", "memorization"]].rename(columns={"memorization": "late_start_memorization"})
        final = _method_epoch_mem(bundle, method_key, final_epoch)[["sample_id", "memorization"]].rename(columns={"memorization": "final_memorization"})
        sub = hard.merge(start, on="sample_id", how="inner").merge(final, on="sample_id", how="inner").dropna()
        sub["late_start_memorization"] = _cap(sub["late_start_memorization"])
        sub["final_memorization"] = _cap(sub["final_memorization"])
        sub["late_phase_memorization_increase"] = sub["final_memorization"] - sub["late_start_memorization"]
        sub["dataset_key"] = dataset_key
        sub["dataset"] = bundle["cfg"]["label"]
        sub["method_key"] = method_key
        sub["method"] = METHOD_LABELS[method_key]
        sub["final_epoch"] = final_epoch
        sub["late_start_epoch"] = late_start_epoch
        rows.append(
            sub[
                [
                    "dataset_key",
                    "dataset",
                    "method_key",
                    "method",
                    "sample_id",
                    "difficulty",
                    "pruning_rate_anchor",
                    "late_start_epoch",
                    "final_epoch",
                    "late_start_memorization",
                    "final_memorization",
                    "late_phase_memorization_increase",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def plot_cifar10_hard_pruning_memorization_scatter() -> pd.DataFrame:
    table = build_cifar10_hard_pruning_memorization_scatter()
    save_table(table, "cifar10_hard_pruning_memorization_scatter")
    method_keys = method_keys_for_dataset("cifar10")
    final_epoch = int(table["final_epoch"].iloc[0])
    late_start_epoch = int(table["late_start_epoch"].iloc[0])
    x_min = float(table["pruning_rate_anchor"].min())
    x_max = float(table["pruning_rate_anchor"].max())
    x_pad = 0.04 * max(x_max - x_min, 1e-8)
    y_final_max = max(0.08, float(table["final_memorization"].max()) * 1.08)
    increase_abs = float(np.nanmax(np.abs(table["late_phase_memorization_increase"].to_numpy(dtype=np.float64))))
    increase_lim = max(0.08, increase_abs * 1.12)

    metrics = [
        ("final_memorization", f"Final memorization\n(epoch {final_epoch})", (0.0, y_final_max)),
        ("late_phase_memorization_increase", f"Late-phase increase\n(epoch {late_start_epoch}-{final_epoch})", (-increase_lim, increase_lim)),
    ]
    fig, axes = plt.subplots(2, len(method_keys), figsize=(7.2, 4.05), sharex=True, squeeze=False)
    for row_idx, (metric, ylabel, ylim) in enumerate(metrics):
        for col_idx, method_key in enumerate(method_keys):
            ax = axes[row_idx, col_idx]
            sub = table.loc[table["method_key"] == method_key].copy()
            x = sub["pruning_rate_anchor"].to_numpy(dtype=np.float64)
            y = sub[metric].to_numpy(dtype=np.float64)
            color = METHOD_COLORS[method_key]
            ax.scatter(x, y, s=2.0, alpha=0.18, color=color, linewidths=0.0, rasterized=True)
            pearson = np.nan
            if len(sub) > 1 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                coeffs = np.polyfit(x, y, deg=1)
                x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
                y_line = coeffs[0] * x_line + coeffs[1]
                ax.plot(x_line, y_line, color="#111827", linewidth=1.0, linestyle="--")
                pearson = float(np.corrcoef(x, y)[0, 1])
            ax.text(
                0.04,
                0.94,
                f"$r$ = {pearson:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.8,
                color="#334155",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CBD5E1", "linewidth": 0.6, "alpha": 0.88},
            )
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[method_key], fontsize=7.8, pad=4)
            if row_idx == 1:
                ax.axhline(0.0, color="#334155", linewidth=0.75)
                ax.set_xlabel("Raw InfoBatch pruning-rate anchor")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(*ylim)
            ax.grid(True, axis="both", alpha=0.35)
    fig.suptitle("Hard samples only, grouped by InfoBatch pruning-rate anchor", y=0.985, fontsize=8.2)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.13, top=0.86, hspace=0.32, wspace=0.14)
    save_fig(fig, "cifar10_hard_pruning_memorization_scatter")
    plt.show()
    plt.close(fig)
    return table


def build_hard_sample_pruning_response_binned_table(dataset_key: str = "cifar10") -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    epoch = int(bundle["cfg"]["end_epoch"])
    hard_anchor = build_hard_pruning_band_anchor(dataset_key, epoch)
    method_keys = [method_key for method_key in ("std", "ib", "ib_prw") if method_key in bundle["cfg"]["methods"]]
    rows = []
    for method_key in method_keys:
        mem = _method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]].copy()
        sub = hard_anchor.merge(mem, on="sample_id", how="inner").dropna()
        sub["memorization"] = _cap(sub["memorization"])
        summary = (
            sub.groupby("pruning_band", observed=False)
            .agg(
                mean_memorization=("memorization", "mean"),
                n=("sample_id", "size"),
                mean_pruning_rate=("pruning_rate_anchor", "mean"),
                pruning_band_min=("pruning_band_min", "first"),
                pruning_band_max=("pruning_band_max", "first"),
            )
            .reset_index()
        )
        summary["method_key"] = method_key
        summary["method"] = METHOD_LABELS[method_key]
        summary["dataset_key"] = dataset_key
        summary["dataset"] = bundle["cfg"]["label"]
        summary["epoch"] = epoch
        rows.append(summary)
    table = pd.concat(rows, ignore_index=True)
    save_table(table, f"{dataset_key}_hard_sample_pruning_response_binned")
    return table


def plot_hard_sample_pruning_response_with_standard_bars(dataset_key: str = "cifar10") -> pd.DataFrame:
    table = build_hard_sample_pruning_response_binned_table(dataset_key)
    band_order = list(pd.Categorical(table["pruning_band"]).categories)
    if not band_order:
        band_order = sorted(table["pruning_band"].dropna().astype(str).unique().tolist())
    method_keys = [method_key for method_key in ("std", "ib", "ib_prw") if method_key in table["method_key"].unique()]

    fig, ax = plt.subplots(figsize=(4.95, 2.45))
    x, offsets, width = _group_positions(len(band_order), len(method_keys))
    y_max = float(table["mean_memorization"].max())
    for idx, method_key in enumerate(method_keys):
        sub = table.loc[table["method_key"] == method_key].set_index("pruning_band").reindex(band_order)
        vals = sub["mean_memorization"].to_numpy(dtype=np.float64)
        ax.bar(x + offsets[idx], vals, width=width, color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
    ax.set_ylim(0, y_max * 1.24)
    ax.set_xticks(x)
    ax.set_xticklabels(band_order, fontsize=6.7)
    ax.set_xlabel("Pruning-rate range within hard samples")
    ax.set_ylabel("Final mean memorization")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", borderaxespad=0.35, fontsize=6.6)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.23, top=0.96)
    save_fig(fig, f"{dataset_key}_hard_sample_pruning_response_with_standard_bars")
    plt.show()
    plt.close(fig)
    return table


def plot_hard_sample_pruning_response_pointview(dataset_key: str = "cifar10") -> pd.DataFrame:
    table = build_hard_sample_pruning_response_binned_table(dataset_key)
    band_order = list(pd.Categorical(table["pruning_band"]).categories)
    if not band_order:
        band_order = sorted(table["pruning_band"].dropna().astype(str).unique().tolist())
    method_keys = [method_key for method_key in ("std", "ib", "ib_prw") if method_key in table["method_key"].unique()]

    fig, ax = plt.subplots(figsize=(4.9, 2.4))
    x = np.arange(len(band_order), dtype=np.float64)
    y_max = float(table["mean_memorization"].max())
    for method_key in method_keys:
        sub = table.loc[table["method_key"] == method_key].set_index("pruning_band").reindex(band_order)
        vals = sub["mean_memorization"].to_numpy(dtype=np.float64)
        ax.plot(
            x,
            vals,
            color=METHOD_COLORS[method_key],
            marker="o",
            markersize=4.0,
            linewidth=1.45,
            label=METHOD_LABELS[method_key],
        )
    ax.set_ylim(0, y_max * 1.24)
    ax.set_xticks(x)
    ax.set_xticklabels(band_order, fontsize=6.7)
    ax.set_xlabel("Pruning-rate range within hard samples")
    ax.set_ylabel("Final mean memorization")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", borderaxespad=0.35, fontsize=6.6)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.23, top=0.96)
    save_fig(fig, f"{dataset_key}_hard_sample_pruning_response_pointview")
    plt.show()
    plt.close(fig)
    return table


def plot_hard_sample_pruning_response_scatter(dataset_key: str = "cifar10") -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    epoch = int(bundle["cfg"]["end_epoch"])
    hard_anchor = build_hard_pruning_band_anchor(dataset_key, epoch).copy()
    method_keys = [method_key for method_key in ("std", "ib", "ib_prw") if method_key in bundle["cfg"]["methods"]]
    if not method_keys:
        raise ValueError(f"No supported methods were available for {dataset_key}.")

    edge_values = sorted(
        {
            float(value)
            for value in np.concatenate(
                [
                    hard_anchor["pruning_band_min"].dropna().to_numpy(dtype=np.float64),
                    hard_anchor["pruning_band_max"].dropna().to_numpy(dtype=np.float64),
                ]
            )
        }
    )
    band_lines = edge_values[1:-1]

    rows = []
    for method_key in method_keys:
        mem = _method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]].copy()
        sub = hard_anchor.merge(mem, on="sample_id", how="inner").dropna()
        sub["memorization"] = _cap(sub["memorization"])
        sub["method_key"] = method_key
        sub["method"] = METHOD_LABELS[method_key]
        sub["dataset_key"] = dataset_key
        sub["dataset"] = bundle["cfg"]["label"]
        sub["epoch"] = epoch
        rows.append(sub)

    table = pd.concat(rows, ignore_index=True)
    save_table(table, f"{dataset_key}_hard_sample_pruning_response_scatter")

    fig, axes = plt.subplots(1, len(method_keys), figsize=(3.15 * len(method_keys), 2.7), sharey=True, squeeze=False)
    axes = axes[0]
    x_min = min(-0.02, float(table["memorization"].min()) - 0.01)
    x_max = max(0.22, float(table["memorization"].max()) + 0.01)
    y_min = float(min(edge_values))
    y_max = float(max(edge_values))

    for ax, method_key in zip(axes, method_keys):
        sub = table.loc[table["method_key"] == method_key]
        ax.scatter(
            sub["memorization"],
            sub["pruning_rate_anchor"],
            s=7,
            alpha=0.22,
            linewidths=0.0,
            color=METHOD_COLORS[method_key],
            rasterized=True,
        )
        for y in band_lines:
            ax.axhline(y, color="#64748B", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_title(METHOD_LABELS[method_key], fontsize=8.1)
        ax.set_xlabel("Memorization")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="both", alpha=0.28)

    axes[0].set_ylabel("Pruning rate")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.22, top=0.86, wspace=0.18)
    save_fig(fig, f"{dataset_key}_hard_sample_pruning_response_scatter")
    plt.show()
    plt.close(fig)
    return table


def build_dataset_mean_trajectories() -> pd.DataFrame:
    rows = []
    for dataset_key in MEM_DATASET_KEYS:
        bundle = rpu.report_bundle(dataset_key)
        for method_key in method_keys_for_dataset(dataset_key):
            mem_long = bundle["mem_long"][method_key]
            for epoch, chunk in mem_long.groupby("epoch", sort=True):
                if dataset_key == "cifar10" and int(epoch) == 87:
                    continue
                scores = _cap(chunk["memorization"])
                tail_mask = scores > 0.1
                tail_scores = scores[tail_mask]
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": bundle["cfg"]["label"],
                        "method_key": method_key,
                        "method": METHOD_LABELS[method_key],
                        "epoch": int(epoch),
                        "mean_memorization": float(scores.mean()),
                        "tail_mean_memorization": float(tail_scores.mean()) if len(tail_scores) else np.nan,
                        "tail_share_pct": float(100.0 * tail_mask.mean()),
                        "tail_n": int(tail_mask.sum()),
                        "plotted_mean_memorization": float(scores.mean()),
                        "plotted_mean_rule": "mean over all samples",
                    }
                )
    return pd.DataFrame(rows)


def plot_dataset_mean_trajectories() -> pd.DataFrame:
    table = build_dataset_mean_trajectories()
    save_table(table, "dataset_mean_memorization_trajectories")
    for dataset_key in MEM_DATASET_KEYS:
        fig, ax = plt.subplots(figsize=(2.6, 2.35))
        sub_d = table.loc[table["dataset_key"] == dataset_key]
        method_keys = method_keys_for_dataset(dataset_key)
        ymax = max(0.12, float(sub_d["plotted_mean_memorization"].max()) * 1.18)
        epochs_all = sub_d["epoch"].to_numpy(dtype=np.float64)
        if len(epochs_all):
            epoch_span = max(float(np.nanmax(epochs_all) - np.nanmin(epochs_all)), 1.0)
            ax.set_xlim(float(np.nanmin(epochs_all)) - 0.02 * epoch_span, float(np.nanmax(epochs_all)) + 0.18 * epoch_span)
        for method_key in method_keys:
            sub = sub_d.loc[sub_d["method_key"] == method_key].sort_values("epoch")
            ax.plot(sub["epoch"], sub["plotted_mean_memorization"], color=METHOD_COLORS[method_key], marker="o", markersize=2.3, linewidth=1.25, label=METHOD_LABELS[method_key])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean memorization")
        ax.set_ylim(0, ymax)
        ax.grid(True, axis="y")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(labels))
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.22, top=0.83)
        save_fig(fig, f"{dataset_key}_dataset_mean_memorization_trajectories")
        plt.show()
        plt.close(fig)
    return table


def plot_combined_dataset_mean_trajectories(table: pd.DataFrame | None = None) -> pd.DataFrame:
    if table is None:
        table = build_dataset_mean_trajectories()
        save_table(table, "dataset_mean_memorization_trajectories")

    fig, axes = plt.subplots(1, len(MEM_DATASET_KEYS), figsize=(7.35, 2.45), sharey=False, squeeze=False)
    axes = axes[0]
    for ax, dataset_key in zip(axes, MEM_DATASET_KEYS):
        sub_d = table.loc[table["dataset_key"] == dataset_key]
        dataset_label = figure_dataset_label(sub_d["dataset"].iloc[0])
        method_keys = method_keys_for_dataset(dataset_key)
        ymax = max(0.12, float(sub_d["plotted_mean_memorization"].max()) * 1.18)
        epochs_all = sub_d["epoch"].to_numpy(dtype=np.float64)
        if len(epochs_all):
            epoch_span = max(float(np.nanmax(epochs_all) - np.nanmin(epochs_all)), 1.0)
            ax.set_xlim(float(np.nanmin(epochs_all)) - 0.02 * epoch_span, float(np.nanmax(epochs_all)) + 0.18 * epoch_span)
        for method_key in method_keys:
            sub = sub_d.loc[sub_d["method_key"] == method_key].sort_values("epoch")
            ax.plot(
                sub["epoch"],
                sub["plotted_mean_memorization"],
                color=METHOD_COLORS[method_key],
                marker="o",
                markersize=2.1,
                linewidth=1.15,
                label=METHOD_LABELS[method_key],
            )
        ax.set_title(dataset_label, fontsize=8.0, pad=4)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, ymax)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Mean memorization")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=len(labels))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.78, wspace=0.32)
    save_fig(fig, "dataset_mean_memorization_trajectories_combined")
    save_fig(fig, "cumulative_memorization_exposure_reduction")
    plt.show()
    plt.close(fig)
    return table


def plot_dataset_auc_reduction_companions(table: pd.DataFrame) -> pd.DataFrame:
    comparison_method_keys = ["ib", "ib_prw"]
    for dataset_key in MEM_DATASET_KEYS:
        sub = table.loc[(table["dataset_key"] == dataset_key) & (table["method_key"].isin(comparison_method_keys))].copy()
        sub = sub.set_index("method_key").reindex(comparison_method_keys)
        vals = sub["reduction_vs_standard_pct"].to_numpy(dtype=np.float64)
        y_max = max(1.0, float(np.nanmax(vals)) if len(vals) else 1.0)
        fig, ax = plt.subplots(figsize=(2.6, 2.35))
        x = np.arange(len(comparison_method_keys), dtype=np.float64) * 0.68
        colors = [METHOD_COLORS[method_key] for method_key in comparison_method_keys]
        bars = ax.bar(x, vals, width=0.48, color=colors)
        for bar, val in zip(bars, vals):
            if not np.isfinite(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.035 * max(y_max, 1.0),
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[method_key] for method_key in comparison_method_keys])
        ax.set_xlim(x[0] - 0.42, x[-1] + 0.42)
        ax.set_ylabel("AUC reduction vs Standard (%)")
        ax.set_ylim(0.0, max(y_max * 1.28, y_max + 2.0))
        ax.grid(True, axis="y")
        fig.subplots_adjust(left=0.20, right=0.97, bottom=0.22, top=0.94)
        save_fig(fig, f"{dataset_key}_cumulative_memorization_exposure_reduction")
        plt.show()
        plt.close(fig)
    return table


def plot_cifar10_tail_mean_trajectory(tail_threshold: float = 0.1) -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    method_keys = method_keys_for_dataset(dataset_key)
    rows = []
    for method_key in method_keys:
        mem_long = bundle["mem_long"][method_key]
        for epoch, chunk in mem_long.groupby("epoch", sort=True):
            scores = _cap(chunk["memorization"])
            tail_scores = scores[scores >= float(tail_threshold)]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": METHOD_LABELS[method_key],
                    "epoch": int(epoch),
                    "tail_threshold": float(tail_threshold),
                    "tail_n": int(len(tail_scores)),
                    "tail_share_pct": float(100.0 * len(tail_scores) / max(len(scores), 1)),
                    "tail_mean_memorization": float(tail_scores.mean()) if len(tail_scores) else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    save_table(table, "cifar10_tail_mean_memorization_trajectory")

    fig, ax = plt.subplots(figsize=(2.75, 2.35))
    sub_d = table.sort_values("epoch")
    ymax = max(0.5, float(sub_d["tail_mean_memorization"].max()) * 1.16)
    epochs_all = sub_d["epoch"].to_numpy(dtype=np.float64)
    if len(epochs_all):
        epoch_span = max(float(np.nanmax(epochs_all) - np.nanmin(epochs_all)), 1.0)
        ax.set_xlim(float(np.nanmin(epochs_all)) - 0.02 * epoch_span, float(np.nanmax(epochs_all)) + 0.18 * epoch_span)
    for method_key in method_keys:
        sub = sub_d.loc[sub_d["method_key"] == method_key].sort_values("epoch")
        ax.plot(
            sub["epoch"],
            sub["tail_mean_memorization"],
            color=METHOD_COLORS[method_key],
            marker="o",
            markersize=2.3,
            linewidth=1.25,
            label=METHOD_LABELS[method_key],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tail mean memorization")
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(labels))
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.22, top=0.82)
    save_fig(fig, "cifar10_tail_mean_memorization_trajectory")
    plt.show()
    plt.close(fig)
    return table


def plot_cifar10_late_change_by_difficulty(start_epoch: int = 87, end_epoch: int = 99) -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    anchor = build_pruning_difficulty_anchor(dataset_key, int(bundle["cfg"]["end_epoch"]))[["sample_id", "difficulty"]]
    method_keys = method_keys_for_dataset(dataset_key)
    rows = []
    for method_key in method_keys:
        start = _method_epoch_mem(bundle, method_key, int(start_epoch))[["sample_id", "memorization"]].rename(columns={"memorization": "start_memorization"})
        end = _method_epoch_mem(bundle, method_key, int(end_epoch))[["sample_id", "memorization"]].rename(columns={"memorization": "end_memorization"})
        frame = anchor.merge(start, on="sample_id", how="inner").merge(end, on="sample_id", how="inner")
        frame["start_memorization"] = _cap(frame["start_memorization"])
        frame["end_memorization"] = _cap(frame["end_memorization"])
        for difficulty in DIFFICULTY_ORDER_DESC:
            sub = frame.loc[frame["difficulty"] == difficulty]
            start_mean = float(sub["start_memorization"].mean())
            end_mean = float(sub["end_memorization"].mean())
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": METHOD_LABELS[method_key],
                    "difficulty": difficulty,
                    "start_epoch": int(start_epoch),
                    "end_epoch": int(end_epoch),
                    "n": int(len(sub)),
                    "start_mean_memorization": start_mean,
                    "end_mean_memorization": end_mean,
                    "late_phase_change": end_mean - start_mean,
                }
            )
    table = pd.DataFrame(rows)
    save_table(table, "cifar10_late_change_87_99_by_difficulty")

    fig, ax = plt.subplots(figsize=(3.07, 2.51))
    x, offsets, width = _group_positions(len(DIFFICULTY_ORDER_DESC), len(method_keys), span=0.56)
    vals_all = table["late_phase_change"].to_numpy(dtype=np.float64)
    y_min = min(float(np.nanmin(vals_all)), 0.0)
    y_max = max(float(np.nanmax(vals_all)), 1e-6)
    for idx, method_key in enumerate(method_keys):
        sub = table.loc[table["method_key"] == method_key].set_index("difficulty").reindex(DIFFICULTY_ORDER_DESC)
        vals = sub["late_phase_change"].to_numpy(dtype=np.float64)
        ax.bar(x + offsets[idx], vals, width=width * 0.90, color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
    ax.axhline(0.0, color="#334155", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(DIFFICULTY_ORDER_DESC)
    ax.set_xlabel("Pruning-difficulty group")
    ax.set_ylabel("Mean memorization surge\n(in the annealing phase)")
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_ylim(y_min - 0.10 * max(y_max - y_min, 1e-6), y_max * 1.20)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=1, borderaxespad=0.0, fontsize=5)
    fig.subplots_adjust(left=0.26, right=0.96, bottom=0.25, top=0.94)
    save_fig(fig, "cifar10_late_change_87_99_by_difficulty")
    plt.show()
    plt.close(fig)
    return table


def build_cifar10_pruning_rate_vs_late_memorization_rise(start_epoch: int = 87, end_epoch: int = 99) -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    anchor = build_pruning_difficulty_anchor(dataset_key, int(start_epoch))[
        ["sample_id", "pruning_rate_anchor", "difficulty"]
    ].copy()
    method_keys = pruned_method_keys_for_dataset(dataset_key)
    common_sample_ids = set(anchor["sample_id"].dropna().tolist())
    for method_key in method_keys:
        start_ids = set(_method_epoch_mem(bundle, method_key, int(start_epoch))["sample_id"].dropna().tolist())
        end_ids = set(_method_epoch_mem(bundle, method_key, int(end_epoch))["sample_id"].dropna().tolist())
        common_sample_ids &= start_ids & end_ids
    anchor = anchor.loc[anchor["sample_id"].isin(common_sample_ids)].copy()
    rows = []
    for method_key in method_keys:
        start = _method_epoch_mem(bundle, method_key, int(start_epoch))[["sample_id", "memorization"]].rename(
            columns={"memorization": "start_memorization"}
        )
        end = _method_epoch_mem(bundle, method_key, int(end_epoch))[["sample_id", "memorization"]].rename(
            columns={"memorization": "end_memorization"}
        )
        frame = anchor.merge(start, on="sample_id", how="inner").merge(end, on="sample_id", how="inner").dropna()
        frame["start_memorization"] = _cap(frame["start_memorization"])
        frame["end_memorization"] = _cap(frame["end_memorization"])
        frame["memorization_rise"] = frame["end_memorization"] - frame["start_memorization"]
        frame["dataset_key"] = dataset_key
        frame["dataset"] = bundle["cfg"]["label"]
        frame["method_key"] = method_key
        frame["method"] = METHOD_LABELS[method_key]
        frame["start_epoch"] = int(start_epoch)
        frame["end_epoch"] = int(end_epoch)
        rows.append(
            frame[
                [
                    "dataset_key",
                    "dataset",
                    "method_key",
                    "method",
                    "sample_id",
                    "difficulty",
                    "pruning_rate_anchor",
                    "start_epoch",
                    "end_epoch",
                    "start_memorization",
                    "end_memorization",
                    "memorization_rise",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def plot_cifar10_pruning_rate_vs_late_memorization_rise(start_epoch: int = 87, end_epoch: int = 99) -> pd.DataFrame:
    table = build_cifar10_pruning_rate_vs_late_memorization_rise(start_epoch=start_epoch, end_epoch=end_epoch)
    save_table(table, f"cifar10_pruning_rate_vs_memorization_rise_{start_epoch}_{end_epoch}")
    method_keys = pruned_method_keys_for_dataset("cifar10")
    x = table["pruning_rate_anchor"].to_numpy(dtype=np.float64)
    y = table["memorization_rise"].to_numpy(dtype=np.float64)
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    x_pad = 0.04 * max(x_max - x_min, 1e-8)
    y_abs = float(np.nanquantile(np.abs(y), 0.995))
    y_lim = max(0.08, y_abs * 1.14)

    fig, axes = plt.subplots(1, len(method_keys), figsize=(4.9, 2.15), sharex=True, sharey=True, squeeze=False)
    for idx, method_key in enumerate(method_keys):
        ax = axes[0, idx]
        sub = table.loc[table["method_key"] == method_key].copy()
        x_sub = sub["pruning_rate_anchor"].to_numpy(dtype=np.float64)
        y_sub = sub["memorization_rise"].to_numpy(dtype=np.float64)
        color = METHOD_COLORS[method_key]
        ax.scatter(x_sub, y_sub, s=2.0, alpha=0.16, color=color, linewidths=0.0, rasterized=True)
        pearson = np.nan
        if len(sub) > 1 and np.nanstd(x_sub) > 0 and np.nanstd(y_sub) > 0:
            slope, intercept = np.polyfit(x_sub, y_sub, deg=1)
            x_line = np.linspace(float(np.nanmin(x_sub)), float(np.nanmax(x_sub)), 120)
            ax.plot(x_line, slope * x_line + intercept, color="#111827", linestyle="--", linewidth=1.0)
            pearson = float(np.corrcoef(x_sub, y_sub)[0, 1])
        ax.axhline(0.0, color="#334155", linewidth=0.75)
        ax.text(
            0.96,
            0.94,
            f"Pearson $r$ = {pearson:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CBD5E1", "linewidth": 0.6, "alpha": 0.9},
        )
        ax.set_title(METHOD_LABELS[method_key], fontsize=7.8, pad=4)
        ax.set_xlabel(f"Pruning Rate (fixed at epoch {start_epoch})")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(-y_lim, y_lim)
        ax.xaxis.label.set_size(7)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, axis="both", alpha=0.35)
    axes[0, 0].set_ylabel("Memorization Surge")
    axes[0, 0].yaxis.label.set_size(7)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.25, top=0.84, wspace=0.12)
    save_fig(fig, f"cifar10_pruning_rate_vs_memorization_rise_{start_epoch}_{end_epoch}")
    plt.show()
    plt.close(fig)

    for method_key in method_keys:
        fig, ax = plt.subplots(figsize=(2.65, 2.35))
        sub = table.loc[table["method_key"] == method_key].copy()
        x_sub = sub["pruning_rate_anchor"].to_numpy(dtype=np.float64)
        y_sub = sub["memorization_rise"].to_numpy(dtype=np.float64)
        color = METHOD_COLORS[method_key]
        ax.scatter(x_sub, y_sub, s=2.0, alpha=0.16, color=color, linewidths=0.0, rasterized=True)
        pearson = np.nan
        if len(sub) > 1 and np.nanstd(x_sub) > 0 and np.nanstd(y_sub) > 0:
            slope, intercept = np.polyfit(x_sub, y_sub, deg=1)
            x_line = np.linspace(float(np.nanmin(x_sub)), float(np.nanmax(x_sub)), 120)
            ax.plot(x_line, slope * x_line + intercept, color="#111827", linestyle="--", linewidth=1.0)
            pearson = float(np.corrcoef(x_sub, y_sub)[0, 1])
        ax.axhline(0.0, color="#334155", linewidth=0.75)
        ax.text(
            0.96,
            0.94,
            f"Pearson $r$ = {pearson:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CBD5E1", "linewidth": 0.6, "alpha": 0.9},
        )
        ax.set_title(METHOD_LABELS[method_key], fontsize=7.8, pad=4)
        ax.set_xlabel(f"Pruning Rate (fixed at epoch {start_epoch})")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(-y_lim, y_lim)
        ax.xaxis.label.set_size(7)
        ax.tick_params(axis="both", labelsize=6)
        ax.grid(True, axis="both", alpha=0.35)
        ax.set_ylabel("Memorization Surge")
        ax.yaxis.label.set_size(7)
        fig.subplots_adjust(left=0.22, right=0.96, bottom=0.24, top=0.88)
        save_fig(fig, f"cifar10_{method_key}_pruning_rate_vs_memorization_surge_{start_epoch}_{end_epoch}")
        plt.show()
        plt.close(fig)
    return table


def build_test_accuracy_trajectory(dataset_key: str) -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    rows = []
    for method_key in method_keys_for_dataset(dataset_key):
        curve = rpu.load_local_accuracy(bundle["cfg"]["methods"][method_key])[["epoch", "test_accuracy"]].copy()
        curve["dataset_key"] = dataset_key
        curve["dataset"] = bundle["cfg"]["label"]
        curve["method_key"] = method_key
        curve["method"] = METHOD_LABELS[method_key]
        rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def build_imagenet_test_accuracy_trajectory() -> pd.DataFrame:
    dataset_key = "imagenet"
    specs = [
        ("std", METHOD_LABELS["std"], ANALYSIS_ROOT / "imagenet" / "test_accuracy_standard.csv"),
        ("ib", METHOD_LABELS["ib"], ANALYSIS_ROOT / "imagenet" / "test_accuracy_infobatch.csv"),
        ("ib_prw", METHOD_LABELS["ib_prw"], ANALYSIS_ROOT / "imagenet" / "test_accuracy_wlib.csv"),
    ]
    rows = []
    for method_key, method_label, path in specs:
        frame = pd.read_csv(path)
        value_col = next((col for col in frame.columns if "test_acc" in col and not col.endswith("__MIN") and not col.endswith("__MAX")), None)
        if value_col is None:
            raise ValueError(f"No test_acc column found in {path}. Columns: {list(frame.columns)}")
        rows.append(
            pd.DataFrame(
                {
                    "epoch": frame["Step"].astype(int),
                    "test_accuracy": frame[value_col].astype(float),
                    "dataset_key": dataset_key,
                    "dataset": "ImageNet",
                    "method_key": method_key,
                    "method": method_label,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _smooth_accuracy(values: pd.Series, window: int = 5) -> pd.Series:
    return values.rolling(window=int(window), center=True, min_periods=1).mean()


def plot_test_accuracy_trajectory(dataset_key: str, *, smooth: bool | None = None) -> pd.DataFrame:
    if dataset_key == "imagenet":
        table = build_imagenet_test_accuracy_trajectory()
    else:
        table = build_test_accuracy_trajectory(dataset_key)
    save_table(table, f"{dataset_key}_test_accuracy_trajectory")
    if smooth is None:
        smooth = dataset_key == "cifar10"
    plot_table = table.copy()
    plot_table["plot_test_accuracy"] = plot_table["test_accuracy"]
    if smooth:
        plot_table["plot_test_accuracy"] = (
            plot_table.sort_values("epoch")
            .groupby("method_key", group_keys=False)["test_accuracy"]
            .transform(lambda values: _smooth_accuracy(values, window=5))
        )

    fig, ax = plt.subplots(figsize=(2.55, 2.35))
    method_keys = [key for key in ["std", "ib_prw", "ib"] if key in set(plot_table["method_key"])]
    y_min = float(plot_table["plot_test_accuracy"].min())
    y_max = float(plot_table["plot_test_accuracy"].max())
    y_span = max(y_max - y_min, 1e-6)
    for method_key in method_keys:
        sub = plot_table.loc[plot_table["method_key"] == method_key].sort_values("epoch")
        linestyles = IMAGENET_METHOD_LINESTYLES if dataset_key == "imagenet" else METHOD_LINESTYLES
        ax.plot(
            sub["epoch"],
            sub["plot_test_accuracy"],
            color=METHOD_COLORS[method_key],
            linestyle=linestyles.get(method_key, "-"),
            linewidth=1.05 if method_key == "ib" else 0.72 if method_key == "ib_prw" else 0.88,
            alpha=0.98 if method_key == "ib" else 0.70 if method_key == "ib_prw" else 0.88,
            zorder=4 if method_key == "ib" else 3 if method_key == "std" else 2,
            label=METHOD_LABELS[method_key],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(max(0.0, y_min - 0.08 * y_span), min(100.0, y_max + 0.10 * y_span))
    ax.grid(True, axis="y")
    if dataset_key == "imagenet":
        ax.legend(loc="lower right", fontsize=7.0, borderaxespad=0.25)
    # ax.legend(loc="lower right", fontsize=6.7, borderaxespad=0.25)
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.22, top=0.94)
    save_fig(fig, f"{dataset_key}_test_accuracy_trajectory")
    plt.show()
    plt.close(fig)
    return table


def plot_cifar10_epoch_violin_mean_tail(epochs: tuple[int, ...] = (99,), tail_threshold: float = 0.1) -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    method_keys = method_keys_for_dataset(dataset_key)
    rows = []
    score_frames = {}
    for epoch in epochs:
        for method_key in method_keys:
            mem = _method_epoch_mem(bundle, method_key, int(epoch))[["sample_id", "memorization"]].copy()
            mem["memorization"] = _cap(mem["memorization"])
            mem["dataset_key"] = dataset_key
            mem["dataset"] = bundle["cfg"]["label"]
            mem["method_key"] = method_key
            mem["method"] = METHOD_LABELS[method_key]
            mem["epoch"] = int(epoch)
            score_frames[(int(epoch), method_key)] = mem
            tail = mem.loc[mem["memorization"] >= float(tail_threshold), "memorization"]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": METHOD_LABELS[method_key],
                    "epoch": int(epoch),
                    "n": int(len(mem)),
                    "mean_memorization": float(mem["memorization"].mean()),
                    "tail_threshold": float(tail_threshold),
                    "tail_n": int(len(tail)),
                    "tail_share_pct": float(100.0 * len(tail) / max(len(mem), 1)),
                    "tail_mean_memorization": float(tail.mean()) if len(tail) else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    save_table(table, "cifar10_epoch_violin_mean_tail")

    fig_height = 2.55 if len(epochs) == 1 else 2.2 * len(epochs)
    fig, axes = plt.subplots(len(epochs), len(method_keys), figsize=(7.4, fig_height), sharey=True, squeeze=False)
    for row_idx, epoch in enumerate(epochs):
        for col_idx, method_key in enumerate(method_keys):
            ax = axes[row_idx, col_idx]
            scores = score_frames[(int(epoch), method_key)]["memorization"].to_numpy(dtype=np.float64)
            parts = ax.violinplot(
                scores,
                positions=[0.0],
                widths=0.82,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(lighten_color(METHOD_COLORS[method_key], 0.28))
                body.set_edgecolor(METHOD_COLORS[method_key])
                body.set_alpha(0.88)
                body.set_linewidth(0.75)
            summary = table.loc[(table["epoch"] == int(epoch)) & (table["method_key"] == method_key)].iloc[0]
            mean_val = float(summary["mean_memorization"])
            tail_val = float(summary["tail_mean_memorization"])
            line_specs = [
                (mean_val, "Mean", "#111827", 0.08),
                (tail_val, "Tail mean", METHOD_COLORS[method_key], 0.42),
            ]
            for y_value, label, color, y_offset in line_specs:
                if not np.isfinite(y_value):
                    continue
                ax.axhline(y_value, color=color, linestyle="--", linewidth=0.9, alpha=0.92)
                ax.text(
                    0.44,
                    y_value + y_offset,
                    f"{label}: {y_value:.3f}",
                    ha="left",
                    va="center",
                    fontsize=6.8,
                    color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
                )
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[method_key], fontsize=8.2, pad=4)
            if col_idx == 0:
                ax.set_ylabel(f"Epoch {epoch}\nMemorization")
            ax.set_xlim(-0.55, 1.15)
            ax.set_xticks([])
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, axis="y")
    bottom = 0.14 if len(epochs) == 1 else 0.08
    top = 0.84 if len(epochs) == 1 else 0.91
    fig.subplots_adjust(left=0.09, right=0.985, bottom=bottom, top=top, wspace=0.18, hspace=0.22)
    save_fig(fig, "cifar10_epoch_violin_mean_tail")
    plt.show()
    plt.close(fig)
    return table


def plot_memorization_exposure_metric() -> pd.DataFrame:
    traj = build_dataset_mean_trajectories()
    rows = []
    trapz = getattr(np, "trapezoid", np.trapz)
    for (dataset_key, method_key), sub in traj.groupby(["dataset_key", "method_key"], sort=False):
        sub = sub.sort_values("epoch")
        x = sub["epoch"].to_numpy(dtype=np.float64)
        x = (x - x.min()) / max(x.max() - x.min(), 1e-8)
        auc = float(trapz(sub["mean_memorization"].to_numpy(dtype=np.float64), x))
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": sub["dataset"].iloc[0],
                "method_key": method_key,
                "method": METHOD_LABELS[method_key],
                "cumulative_memorization_exposure": auc,
            }
        )
    table = pd.DataFrame(rows)
    std_auc = table.loc[table["method_key"] == "std", ["dataset_key", "cumulative_memorization_exposure"]].rename(columns={"cumulative_memorization_exposure": "standard_exposure"})
    table = table.merge(std_auc, on="dataset_key", how="left")
    table["reduction_vs_standard_pct"] = 100.0 * (table["standard_exposure"] - table["cumulative_memorization_exposure"]) / table["standard_exposure"]
    save_table(table, "cumulative_memorization_exposure")

    plot_combined_dataset_mean_trajectories(traj)
    plot_dataset_auc_reduction_companions(table)
    return table

def build_phase_memorization_growth() -> pd.DataFrame:
    trajectory = pd.concat([build_pruning_difficulty_trajectories(key) for key in MEM_DATASET_KEYS], ignore_index=True)
    rows = []
    for (dataset_key, method_key, difficulty), sub in trajectory.groupby(["dataset_key", "method_key", "difficulty"], observed=False):
        sub = sub.sort_values("epoch")
        epochs = sub["epoch"].to_numpy(dtype=np.float64)
        values = sub["mean_memorization"].to_numpy(dtype=np.float64)
        if len(epochs) < 2:
            continue
        boundaries = actual_phase_boundaries(epochs)
        for phase_idx, phase_name in enumerate(["Early", "Middle", "Late"]):
            start_epoch = float(boundaries[phase_idx])
            end_epoch = float(boundaries[phase_idx + 1])
            start_value = float(values[np.where(epochs == start_epoch)[0][0]])
            end_value = float(values[np.where(epochs == end_epoch)[0][0]])
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": sub["dataset"].iloc[0],
                    "method_key": method_key,
                    "method": METHOD_LABELS[method_key],
                    "difficulty": difficulty,
                    "phase": phase_name,
                    "phase_start_epoch": start_epoch,
                    "phase_end_epoch": end_epoch,
                    "start_mean_memorization": start_value,
                    "end_mean_memorization": end_value,
                    "mean_memorization_increase": end_value - start_value,
                }
            )
    return pd.DataFrame(rows)


def plot_late_phase_memorization_growth() -> pd.DataFrame:
    table = build_phase_memorization_growth()
    save_table(table, "phase_memorization_growth_by_pruning_difficulty")
    late = table.loc[table["phase"] == "Late"].copy()
    for dataset_key in MEM_DATASET_KEYS:
        fig, ax = plt.subplots(figsize=(2.5, 2.85))
        method_keys = method_keys_for_dataset(dataset_key)
        x, offsets, width = _group_positions(len(DIFFICULTY_ORDER_DESC), len(method_keys))
        for idx, method_key in enumerate(method_keys):
            sub = (
                late.loc[(late["dataset_key"] == dataset_key) & (late["method_key"] == method_key)]
                .set_index("difficulty")
                .reindex(DIFFICULTY_ORDER_DESC)
            )
            vals = sub["mean_memorization_increase"].to_numpy(dtype=np.float64)
            ax.bar(x + offsets[idx], vals, width=width, color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
        ax.axhline(0.0, color="#334155", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(DIFFICULTY_ORDER_DESC)
        ax.set_xlabel("Pruning-difficulty group")
        ax.set_ylabel("Late-phase memorization increase")
        ax.grid(True, axis="y")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(labels))
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.84)
        save_fig(fig, f"{dataset_key}_late_phase_memorization_growth_by_pruning_difficulty")
        plt.show()
        plt.close(fig)
    return table


def _mean_pruning_exposure(
    dataset_key: str,
    method_key: str,
    method_cfg: dict,
    *,
    start_epoch: int,
    end_epoch: int,
) -> pd.DataFrame:
    arrays = rpu.load_evolution_arrays(dataset_key, method_key, method_cfg)
    if "pruned_frac" not in arrays:
        return pd.DataFrame(columns=["sample_id", "mean_pruning_exposure"])
    mask = (arrays["epoch"] >= int(start_epoch)) & (arrays["epoch"] <= int(end_epoch))
    frame = pd.DataFrame(
        {
            "sample_id": arrays["sample_id"][mask].astype(np.int64),
            "pruning_rate": arrays["pruned_frac"][mask].astype(np.float64),
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["sample_id", "mean_pruning_exposure"])
    return (
        frame.groupby("sample_id", as_index=False)
        .agg(mean_pruning_exposure=("pruning_rate", "mean"))
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


def build_cifar10_first_phase_pruning_memorization(phase_end_epoch: int = 87) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    phase_end_epoch = int(phase_end_epoch)
    phase_start_epoch = int(
        min(epoch for epoch in bundle["mem_long"]["std"]["epoch"].unique() if int(epoch) <= phase_end_epoch)
    )
    anchor = _mean_pruning_exposure(
        dataset_key,
        "ib",
        bundle["cfg"]["methods"]["ib"],
        start_epoch=0,
        end_epoch=phase_end_epoch,
    )
    if anchor.empty:
        raise ValueError("No CIFAR-10 InfoBatch pruning exposure was available for the first phase.")
    lo = float(anchor["mean_pruning_exposure"].min())
    hi = float(anchor["mean_pruning_exposure"].max())
    if np.isclose(lo, hi):
        hi = lo + 1e-8
    edges = np.linspace(lo, hi, 4)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    anchor["first_phase_pruning_group"] = pd.cut(
        anchor["mean_pruning_exposure"],
        bins=edges,
        labels=["Hard / low-pruned", "Medium", "Easy / high-pruned"],
        include_lowest=True,
        ordered=True,
    )
    group_order = ["Hard / low-pruned", "Medium", "Easy / high-pruned"]
    anchor["first_phase_pruning_group"] = pd.Categorical(anchor["first_phase_pruning_group"], categories=group_order, ordered=True)
    anchor = anchor.rename(columns={"mean_pruning_exposure": "ib_anchor_mean_pruning_exposure"})

    exposure_by_method = {}
    for method_key in ["std", "ib", "ib_prw"]:
        if method_key == "std":
            exposure_by_method[method_key] = anchor[["sample_id"]].assign(mean_pruning_exposure=0.0)
            continue
        exposure_by_method[method_key] = _mean_pruning_exposure(
            dataset_key,
            method_key,
            bundle["cfg"]["methods"][method_key],
            start_epoch=0,
            end_epoch=phase_end_epoch,
        )

    trajectory_rows = []
    summary_rows = []
    for method_key in ["std", "ib", "ib_prw"]:
        mem_long = bundle["mem_long"][method_key]
        mem_long = mem_long.loc[mem_long["epoch"] <= phase_end_epoch, ["sample_id", "epoch", "memorization"]].copy()
        exposure = exposure_by_method[method_key]
        merged = (
            anchor.merge(mem_long, on="sample_id", how="inner")
            .merge(exposure, on="sample_id", how="left")
            .dropna(subset=["first_phase_pruning_group", "memorization"])
        )
        merged["memorization"] = _cap(merged["memorization"])
        grouped = (
            merged.groupby(["first_phase_pruning_group", "epoch"], observed=False)
            .agg(
                mean_memorization=("memorization", "mean"),
                n=("sample_id", "size"),
                mean_pruning_exposure=("mean_pruning_exposure", "mean"),
                ib_anchor_mean_pruning_exposure=("ib_anchor_mean_pruning_exposure", "mean"),
            )
            .reset_index()
        )
        grouped["dataset_key"] = dataset_key
        grouped["dataset"] = bundle["cfg"]["label"]
        grouped["method_key"] = method_key
        grouped["method"] = METHOD_LABELS[method_key]
        grouped["phase_start_epoch"] = phase_start_epoch
        grouped["phase_end_epoch"] = phase_end_epoch
        trajectory_rows.append(grouped)

        for group, sub in grouped.groupby("first_phase_pruning_group", observed=False):
            sub = sub.sort_values("epoch")
            epochs = sub["epoch"].to_numpy(dtype=np.float64)
            values = sub["mean_memorization"].to_numpy(dtype=np.float64)
            if len(epochs) < 2:
                continue
            start_value = float(np.interp(phase_start_epoch, epochs, values))
            end_value = float(np.interp(phase_end_epoch, epochs, values))
            summary_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": METHOD_LABELS[method_key],
                    "first_phase_pruning_group": group,
                    "phase_start_epoch": phase_start_epoch,
                    "phase_end_epoch": phase_end_epoch,
                    "start_mean_memorization": start_value,
                    "end_mean_memorization": end_value,
                    "mean_memorization_increase": end_value - start_value,
                    "mean_pruning_exposure": float(sub["mean_pruning_exposure"].iloc[-1]),
                    "ib_anchor_mean_pruning_exposure": float(sub["ib_anchor_mean_pruning_exposure"].iloc[-1]),
                    "n": int(sub["n"].iloc[-1]),
                }
            )

    trajectory = pd.concat(trajectory_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    save_table(trajectory, "cifar10_first_phase_pruning_memorization_trajectories")
    save_table(summary, "cifar10_first_phase_pruning_memorization_summary")
    return trajectory, summary


def plot_cifar10_first_phase_pruning_memorization(phase_end_epoch: int = 87) -> pd.DataFrame:
    trajectory, summary = build_cifar10_first_phase_pruning_memorization(phase_end_epoch)
    group_order = ["Hard / low-pruned", "Medium", "Easy / high-pruned"]
    method_keys = ["std", "ib", "ib_prw"]

    fig = plt.figure(figsize=(7.4, 4.45))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.50, wspace=0.30)
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    bar_ax = fig.add_subplot(gs[1, :])
    y_max = max(0.08, float(trajectory["mean_memorization"].max()) * 1.22)

    for ax, group in zip(axes, group_order):
        group_frame = trajectory.loc[trajectory["first_phase_pruning_group"] == group]
        for method_key in method_keys:
            sub = group_frame.loc[group_frame["method_key"] == method_key].sort_values("epoch")
            ax.plot(
                sub["epoch"],
                sub["mean_memorization"],
                color=METHOD_COLORS[method_key],
                marker="o",
                markersize=2.5,
                linewidth=1.25,
                label=METHOD_LABELS[method_key],
            )
        anchor_rate = float(group_frame["ib_anchor_mean_pruning_exposure"].mean())
        ax.set_title(f"{group}\nIB exposure {anchor_rate:.2f}", fontsize=8.0)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Epoch")
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Mean memorization")

    x, offsets, width = _group_positions(len(group_order), len(method_keys))
    y_min = min(0.0, float(summary["mean_memorization_increase"].min()) * 1.18)
    y_bar_max = max(0.05, float(summary["mean_memorization_increase"].max()) * 1.32)
    for idx, method_key in enumerate(method_keys):
        sub = (
            summary.loc[summary["method_key"] == method_key]
            .set_index("first_phase_pruning_group")
            .reindex(group_order)
        )
        vals = sub["mean_memorization_increase"].to_numpy(dtype=np.float64)
        bar_ax.bar(x + offsets[idx], vals, width=width, color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
    bar_ax.axhline(0.0, color="#334155", linewidth=0.8)
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(group_order)
    bar_ax.set_ylim(y_min, y_bar_max)
    bar_ax.set_ylabel("Memorization increase")
    bar_ax.set_xlabel("Cohorts from InfoBatch mean pruning exposure, epochs 0-87")
    bar_ax.grid(True, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.87)
    save_fig(fig, "cifar10_first_phase_pruning_memorization")
    plt.show()
    plt.close(fig)
    return summary


def _epoch_mem_column(bundle: dict, method_key: str, epoch: int) -> pd.DataFrame:
    mem = _method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]].copy()
    mem["memorization"] = _cap(mem["memorization"])
    return mem.rename(columns={"memorization": f"{method_key}_mem_{epoch}"})


def _epoch_evolution_columns(bundle: dict, method_key: str, epoch: int) -> pd.DataFrame:
    method_cfg = bundle["cfg"]["methods"][method_key]
    frame = rpu.evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
    cols = ["sample_id", "loss", "entropy", "pruning_rate", "status_transitions"]
    cols = [col for col in cols if col in frame.columns]
    return frame[cols].rename(columns={col: f"{method_key}_{col}_{epoch}" for col in cols if col != "sample_id"})


def build_cifar10_rescue_mechanism_sample_table(phase_end_epoch: int = 87) -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    phase_end_epoch = int(phase_end_epoch)
    phase_start_epoch = int(
        min(epoch for epoch in bundle["mem_long"]["std"]["epoch"].unique() if int(epoch) <= phase_end_epoch)
    )
    method_keys = ["std", "ib", "ib_prw"]
    table = pd.DataFrame({"sample_id": np.arange(bundle["cfg"]["n_samples"], dtype=np.int64)})
    for method_key in method_keys:
        for epoch in [phase_start_epoch, phase_end_epoch]:
            table = table.merge(_epoch_mem_column(bundle, method_key, epoch), on="sample_id", how="inner")
            table = table.merge(_epoch_evolution_columns(bundle, method_key, epoch), on="sample_id", how="inner")

    for method_key in method_keys:
        table[f"{method_key}_mem_growth"] = table[f"{method_key}_mem_{phase_end_epoch}"] - table[f"{method_key}_mem_{phase_start_epoch}"]
        table[f"{method_key}_entropy_drop"] = table[f"{method_key}_entropy_{phase_start_epoch}"] - table[f"{method_key}_entropy_{phase_end_epoch}"]
        table[f"{method_key}_loss_drop"] = table[f"{method_key}_loss_{phase_start_epoch}"] - table[f"{method_key}_loss_{phase_end_epoch}"]

    for method_key in ["ib", "ib_prw"]:
        table[f"{method_key}_rescue"] = table["std_mem_growth"] - table[f"{method_key}_mem_growth"]
        table[f"{method_key}_entropy_gap"] = table[f"{method_key}_entropy_{phase_end_epoch}"] - table[f"std_entropy_{phase_end_epoch}"]
        table[f"{method_key}_loss_gap"] = table[f"{method_key}_loss_{phase_end_epoch}"] - table[f"std_loss_{phase_end_epoch}"]

    table["mean_rescue"] = table[["ib_rescue", "ib_prw_rescue"]].mean(axis=1)
    rescue_threshold = float(table["mean_rescue"].quantile(0.90))
    table["rescue_cohort"] = np.where(table["mean_rescue"] >= rescue_threshold, "Rescued top 10%", "Other samples")
    table["phase_start_epoch"] = phase_start_epoch
    table["phase_end_epoch"] = phase_end_epoch

    exposure = _mean_pruning_exposure(
        dataset_key,
        "ib",
        bundle["cfg"]["methods"]["ib"],
        start_epoch=0,
        end_epoch=phase_end_epoch,
    ).rename(columns={"mean_pruning_exposure": "ib_first_phase_pruning_exposure"})
    table = table.merge(exposure, on="sample_id", how="left")

    save_table(table, "cifar10_rescue_mechanism_sample_table")
    return table


def build_cifar10_rescue_mechanism_summaries(phase_end_epoch: int = 87) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle = rpu.report_bundle("cifar10")
    sample_table = build_cifar10_rescue_mechanism_sample_table(phase_end_epoch)
    phase_start_epoch = int(sample_table["phase_start_epoch"].iloc[0])
    method_keys = ["std", "ib", "ib_prw"]
    rows = []
    for cohort, cohort_frame in sample_table.groupby("rescue_cohort", sort=False):
        for method_key in method_keys:
            rows.append(
                {
                    "cohort": cohort,
                    "method_key": method_key,
                    "method": METHOD_LABELS[method_key],
                    "n": int(len(cohort_frame)),
                    "mean_memorization_growth": float(cohort_frame[f"{method_key}_mem_growth"].mean()),
                    "mean_entropy_start": float(cohort_frame[f"{method_key}_entropy_{phase_start_epoch}"].mean()),
                    "mean_entropy_end": float(cohort_frame[f"{method_key}_entropy_{phase_end_epoch}"].mean()),
                    "mean_entropy_drop": float(cohort_frame[f"{method_key}_entropy_drop"].mean()),
                    "mean_loss_start": float(cohort_frame[f"{method_key}_loss_{phase_start_epoch}"].mean()),
                    "mean_loss_end": float(cohort_frame[f"{method_key}_loss_{phase_end_epoch}"].mean()),
                    "mean_loss_drop": float(cohort_frame[f"{method_key}_loss_drop"].mean()),
                    "mean_pruning_rate_end": float(cohort_frame[f"{method_key}_pruning_rate_{phase_end_epoch}"].mean()),
                    "mean_status_transitions_end": float(cohort_frame[f"{method_key}_status_transitions_{phase_end_epoch}"].mean()),
                }
            )
    summary = pd.DataFrame(rows)
    save_table(summary, "cifar10_rescue_mechanism_cohort_summary")

    trajectory_rows = []
    trajectory_epochs = [
        int(epoch)
        for epoch in bundle["mem_long"]["std"]["epoch"].drop_duplicates().sort_values().tolist()
        if int(epoch) <= int(phase_end_epoch)
    ]
    cohort_ids = {
        cohort: set(sample_table.loc[sample_table["rescue_cohort"] == cohort, "sample_id"].astype(int).tolist())
        for cohort in sample_table["rescue_cohort"].unique()
    }
    for method_key in method_keys:
        method_cfg = bundle["cfg"]["methods"][method_key]
        for epoch in trajectory_epochs:
            evo = rpu.evolution_epoch_frame("cifar10", method_key, method_cfg, epoch)
            for cohort, ids in cohort_ids.items():
                sub = evo.loc[evo["sample_id"].isin(ids)]
                trajectory_rows.append(
                    {
                        "cohort": cohort,
                        "method_key": method_key,
                        "method": METHOD_LABELS[method_key],
                        "epoch": epoch,
                        "mean_entropy": float(sub["entropy"].mean()),
                        "mean_loss": float(sub["loss"].mean()),
                        "mean_pruning_rate": float(sub["pruning_rate"].mean()) if "pruning_rate" in sub else np.nan,
                        "n": int(len(sub)),
                    }
                )
    trajectory = pd.DataFrame(trajectory_rows)
    save_table(trajectory, "cifar10_rescue_mechanism_evolution_trajectories")
    return sample_table, summary, trajectory


def plot_cifar10_rescue_mechanism(phase_end_epoch: int = 87) -> pd.DataFrame:
    sample_table, summary, trajectory = build_cifar10_rescue_mechanism_summaries(phase_end_epoch)
    phase_start_epoch = int(sample_table["phase_start_epoch"].iloc[0])
    phase_end_epoch = int(sample_table["phase_end_epoch"].iloc[0])
    pruned_methods = ["ib", "ib_prw"]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))
    scatter_axes = axes[0]
    for ax, method_key in zip(scatter_axes, pruned_methods):
        x = sample_table["std_mem_growth"].to_numpy(dtype=np.float64)
        y = sample_table[f"{method_key}_mem_growth"].to_numpy(dtype=np.float64)
        c = sample_table[f"{method_key}_pruning_rate_{phase_end_epoch}"].to_numpy(dtype=np.float64)
        sc = ax.scatter(x, y, c=c, s=3.0, alpha=0.24, linewidths=0.0, cmap="viridis", rasterized=True)
        lim = max(0.45, float(np.nanquantile(np.concatenate([x, y]), 0.995)) * 1.05)
        ax.plot([0, lim], [0, lim], color="#334155", linestyle="--", linewidth=0.9)
        ax.set_xlim(-0.02, lim)
        ax.set_ylim(-0.02, lim)
        ax.set_title(METHOD_LABELS[method_key], fontsize=8.2)
        ax.set_xlabel("Standard memorization growth")
        ax.set_ylabel(f"{METHOD_LABELS[method_key]} growth")
        ax.grid(True, axis="both", alpha=0.30)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.048, pad=0.02)
        cbar.set_label("Pruning rate", fontsize=6.8)
        cbar.ax.tick_params(labelsize=6.2)

    rescued = trajectory.loc[trajectory["cohort"] == "Rescued top 10%"]
    ax_entropy = axes[1, 0]
    ax_loss = axes[1, 1]
    for method_key in ["std", "ib", "ib_prw"]:
        sub = rescued.loc[rescued["method_key"] == method_key].sort_values("epoch")
        ax_entropy.plot(
            sub["epoch"],
            sub["mean_entropy"],
            marker="o",
            markersize=2.8,
            linewidth=1.25,
            color=METHOD_COLORS[method_key],
            label=METHOD_LABELS[method_key],
        )
        ax_loss.plot(
            sub["epoch"],
            sub["mean_loss"],
            marker="o",
            markersize=2.8,
            linewidth=1.25,
            color=METHOD_COLORS[method_key],
            label=METHOD_LABELS[method_key],
        )
    ax_entropy.set_title("Rescued samples keep more uncertainty", fontsize=8.2)
    ax_entropy.set_xlabel("Epoch")
    ax_entropy.set_ylabel("Mean entropy")
    ax_entropy.grid(True, axis="y")
    ax_loss.set_title("Rescued samples are not fit to zero loss", fontsize=8.2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Mean loss")
    ax_loss.grid(True, axis="y")

    handles, labels = ax_entropy.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3)
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.12, top=0.90, hspace=0.44, wspace=0.34)
    save_fig(fig, "cifar10_rescue_mechanism_samplewise")
    plt.show()
    plt.close(fig)
    return summary


def _instantaneous_pruned_by_epoch(dataset_key: str, method_key: str, method_cfg: dict, max_epoch: int) -> pd.DataFrame:
    arrays = rpu.load_evolution_arrays(dataset_key, method_key, method_cfg)
    if "pruned_frac" not in arrays:
        return pd.DataFrame(columns=["sample_id", "epoch", "instant_pruned"])
    mask = arrays["epoch"] <= int(max_epoch)
    frame = pd.DataFrame(
        {
            "sample_id": arrays["sample_id"][mask].astype(np.int64),
            "epoch": arrays["epoch"][mask].astype(np.int64),
            "pruned_frac": arrays["pruned_frac"][mask].astype(np.float64),
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["sample_id", "epoch", "instant_pruned"])
    frame = frame.sort_values(["sample_id", "epoch"]).reset_index(drop=True)
    frame["cumulative_pruned"] = np.rint(frame["pruned_frac"] * (frame["epoch"] + 1)).astype(np.int16)
    previous = frame.groupby("sample_id", sort=False)["cumulative_pruned"].shift(1).fillna(0).astype(np.int16)
    frame["instant_pruned"] = (frame["cumulative_pruned"] - previous).clip(lower=0, upper=1).astype(np.float64)
    return frame[["sample_id", "epoch", "instant_pruned"]]


def build_cifar10_selection_feedback_summary(phase_end_epoch: int = 87) -> pd.DataFrame:
    sample_table = build_cifar10_rescue_mechanism_sample_table(phase_end_epoch)
    bundle = rpu.report_bundle("cifar10")
    rescued_ids = set(sample_table.loc[sample_table["rescue_cohort"] == "Rescued top 10%", "sample_id"].astype(int).tolist())
    other_ids = set(sample_table.loc[sample_table["rescue_cohort"] == "Other samples", "sample_id"].astype(int).tolist())
    rows = []
    for method_key in ["ib", "ib_prw"]:
        method_cfg = bundle["cfg"]["methods"][method_key]
        arrays = rpu.load_evolution_arrays("cifar10", method_key, method_cfg)
        evo = pd.DataFrame(
            {
                "sample_id": arrays["sample_id"].astype(np.int64),
                "epoch": arrays["epoch"].astype(np.int64),
                "loss": arrays["loss"].astype(np.float64),
                "entropy": arrays["entropy"].astype(np.float64),
            }
        )
        evo = evo.loc[evo["epoch"] <= int(phase_end_epoch)]
        prune = _instantaneous_pruned_by_epoch("cifar10", method_key, method_cfg, phase_end_epoch)
        evo = evo.merge(prune, on=["sample_id", "epoch"], how="left")
        evo["instant_pruned"] = evo["instant_pruned"].fillna(0.0)
        for cohort, ids in [("Rescued top 10%", rescued_ids), ("Other samples", other_ids)]:
            sub = evo.loc[evo["sample_id"].isin(ids)]
            grouped = (
                sub.groupby("epoch", as_index=False)
                .agg(
                    mean_loss=("loss", "mean"),
                    mean_entropy=("entropy", "mean"),
                    instant_prune_rate=("instant_pruned", "mean"),
                    n=("sample_id", "size"),
                )
            )
            grouped["cohort"] = cohort
            grouped["method_key"] = method_key
            grouped["method"] = METHOD_LABELS[method_key]
            rows.append(grouped)
    table = pd.concat(rows, ignore_index=True)
    save_table(table, "cifar10_selection_feedback_summary")
    return table


def plot_cifar10_selection_feedback(phase_end_epoch: int = 87) -> pd.DataFrame:
    table = build_cifar10_selection_feedback_summary(phase_end_epoch)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85), sharey=False)
    for ax, method_key in zip(axes, ["ib", "ib_prw"]):
        method_table = table.loc[(table["method_key"] == method_key) & (table["cohort"] == "Rescued top 10%")].sort_values("epoch")
        color = METHOD_COLORS[method_key]
        ax2 = ax.twinx()
        ax.plot(
            method_table["epoch"],
            method_table["mean_entropy"],
            color=color,
            linewidth=1.35,
            label="Entropy",
        )
        ax.plot(
            method_table["epoch"],
            method_table["mean_loss"],
            color=lighten_color(color, 0.35),
            linewidth=1.2,
            linestyle="--",
            label="Loss",
        )
        ax2.plot(
            method_table["epoch"],
            100.0 * method_table["instant_prune_rate"],
            color="#111827",
            linewidth=1.1,
            label="Pruned this epoch",
        )
        ax.set_title(METHOD_LABELS[method_key], fontsize=8.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean loss / entropy")
        ax2.set_ylabel("Pruned this epoch (%)")
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_xlim(0, phase_end_epoch)
        ax2.set_ylim(0, max(4.0, float(100.0 * method_table["instant_prune_rate"].max()) * 1.22))
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=6.6)
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.25, top=0.87, wspace=0.38)
    save_fig(fig, "cifar10_selection_feedback_rescued_samples")
    plt.show()
    plt.close(fig)
    return table


def build_hard_pruning_band_late_growth() -> pd.DataFrame:
    rows = []
    for dataset_key in MEM_DATASET_KEYS:
        bundle = rpu.report_bundle(dataset_key)
        hard = build_hard_pruning_band_anchor(dataset_key, int(bundle["cfg"]["end_epoch"]))
        if hard.empty:
            continue
        for method_key in method_keys_for_dataset(dataset_key):
            method_rows = []
            mem_long = bundle["mem_long"][method_key]
            for epoch, chunk in mem_long.groupby("epoch", sort=True):
                merged = hard.merge(chunk[["sample_id", "memorization"]], on="sample_id", how="inner")
                merged["memorization"] = _cap(merged["memorization"])
                summary = (
                    merged.groupby("pruning_band", observed=False)
                    .agg(
                        mean_memorization=("memorization", "mean"),
                        n=("sample_id", "size"),
                        mean_pruning_rate=("pruning_rate_anchor", "mean"),
                        pruning_band_min=("pruning_band_min", "first"),
                        pruning_band_max=("pruning_band_max", "first"),
                    )
                    .reset_index()
                )
                summary["epoch"] = int(epoch)
                method_rows.append(summary)
            trajectory = pd.concat(method_rows, ignore_index=True)
            for pruning_band, sub in trajectory.groupby("pruning_band", observed=False):
                sub = sub.sort_values("epoch")
                epochs = sub["epoch"].to_numpy(dtype=np.float64)
                values = sub["mean_memorization"].to_numpy(dtype=np.float64)
                if len(epochs) < 2:
                    continue
                boundaries = actual_phase_boundaries(epochs)
                start_epoch = float(boundaries[2])
                end_epoch = float(boundaries[3])
                start_value = float(values[np.where(epochs == start_epoch)[0][0]])
                end_value = float(values[np.where(epochs == end_epoch)[0][0]])
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": bundle["cfg"]["label"],
                        "method_key": method_key,
                        "method": METHOD_LABELS[method_key],
                        "pruning_band": str(pruning_band),
                        "phase": "Late",
                        "phase_start_epoch": start_epoch,
                        "phase_end_epoch": end_epoch,
                        "start_mean_memorization": start_value,
                        "end_mean_memorization": end_value,
                        "mean_memorization_increase": end_value - start_value,
                        "n": int(sub["n"].iloc[-1]),
                        "mean_pruning_rate": float(sub["mean_pruning_rate"].iloc[-1]),
                        "pruning_band_min": float(sub["pruning_band_min"].iloc[-1]),
                        "pruning_band_max": float(sub["pruning_band_max"].iloc[-1]),
                    }
                )
    return pd.DataFrame(rows)


def plot_hard_pruning_band_late_growth() -> pd.DataFrame:
    table = build_hard_pruning_band_late_growth()
    save_table(table, "hard_sample_late_growth_by_pruning_band")
    y_min = min(0.0, float(table["mean_memorization_increase"].min()) * 1.20)
    y_max = max(0.06, float(table["mean_memorization_increase"].max()) * 1.34)
    for dataset_key in MEM_DATASET_KEYS:
        fig, ax = plt.subplots(figsize=(4.25, 2.55))
        dataset_table = table.loc[table["dataset_key"] == dataset_key].copy()
        band_meta = dataset_table[["pruning_band", "pruning_band_min"]].drop_duplicates().sort_values("pruning_band_min")
        bands = band_meta["pruning_band"].tolist()
        method_keys = method_keys_for_dataset(dataset_key)
        x_min = float(dataset_table["pruning_band_min"].min())
        x_max = float(dataset_table["pruning_band_max"].max())
        x_pad = 0.04 * max(x_max - x_min, 1e-8)
        for method_key in method_keys:
            sub = (
                dataset_table.loc[dataset_table["method_key"] == method_key]
                .set_index("pruning_band")
                .reindex(bands)
            )
            x = sub["mean_pruning_rate"].to_numpy(dtype=np.float64)
            y = sub["mean_memorization_increase"].to_numpy(dtype=np.float64)
            ax.plot(x, y, color=METHOD_COLORS[method_key], linewidth=0.9, alpha=0.55)
            ax.scatter(x, y, s=22, color=METHOD_COLORS[method_key], edgecolor="white", linewidth=0.55, zorder=3, label=METHOD_LABELS[method_key])
        ax.axhline(0.0, color="#334155", linewidth=0.8)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_xlabel("InfoBatch pruning-rate anchor, hard samples")
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Late-phase memorization increase")
        ax.grid(True, axis="y")
        ax.legend(loc="upper right", borderaxespad=0.35, fontsize=6.6)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.24, top=0.96)
        save_fig(fig, f"{dataset_key}_hard_sample_late_growth_by_pruning_band")
        plt.show()
        plt.close(fig)
    return table


def build_cifar10_hard_pruning_rate_vs_memorization_surge(start_epoch: int = 87, end_epoch: int = 99) -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    hard = build_hard_pruning_band_anchor(dataset_key, int(start_epoch), n_bins=10)
    rows = []
    for method_key in pruned_method_keys_for_dataset(dataset_key):
        start = _method_epoch_mem(bundle, method_key, int(start_epoch))[["sample_id", "memorization"]].rename(
            columns={"memorization": "start_memorization"}
        )
        end = _method_epoch_mem(bundle, method_key, int(end_epoch))[["sample_id", "memorization"]].rename(
            columns={"memorization": "end_memorization"}
        )
        merged = hard.merge(start, on="sample_id", how="inner").merge(end, on="sample_id", how="inner").dropna()
        merged["start_memorization"] = _cap(merged["start_memorization"])
        merged["end_memorization"] = _cap(merged["end_memorization"])
        merged["memorization_surge"] = merged["end_memorization"] - merged["start_memorization"]
        summary = (
            merged.groupby("pruning_band", observed=False)
            .agg(
                mean_memorization_surge=("memorization_surge", "mean"),
                start_mean_memorization=("start_memorization", "mean"),
                end_mean_memorization=("end_memorization", "mean"),
                n=("sample_id", "size"),
                mean_pruning_rate=("pruning_rate_anchor", "mean"),
                pruning_band_min=("pruning_band_min", "first"),
                pruning_band_max=("pruning_band_max", "first"),
            )
            .reset_index()
        )
        summary["dataset_key"] = dataset_key
        summary["dataset"] = bundle["cfg"]["label"]
        summary["method_key"] = method_key
        summary["method"] = METHOD_LABELS[method_key]
        summary["start_epoch"] = int(start_epoch)
        summary["end_epoch"] = int(end_epoch)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def plot_cifar10_hard_pruning_rate_vs_memorization_surge(start_epoch: int = 87, end_epoch: int = 99) -> pd.DataFrame:
    table = build_cifar10_hard_pruning_rate_vs_memorization_surge(start_epoch=start_epoch, end_epoch=end_epoch)
    save_table(table, f"cifar10_hard_pruning_rate_vs_memorization_surge_{start_epoch}_{end_epoch}")
    band_meta = table[["pruning_band", "pruning_band_min"]].drop_duplicates().sort_values("pruning_band_min")
    bands = band_meta["pruning_band"].tolist()
    x_min = float(table["pruning_band_min"].min())
    x_max = float(table["pruning_band_max"].max())
    x_pad = 0.04 * max(x_max - x_min, 1e-8)
    y_vals = table["mean_memorization_surge"].to_numpy(dtype=np.float64)
    y_min = min(0.0, float(np.nanmin(y_vals)) * 1.18)
    y_max = max(0.03, float(np.nanmax(y_vals)) * 1.22)

    fig, ax = plt.subplots(figsize=(3.05, 2.45))
    for method_key in pruned_method_keys_for_dataset("cifar10"):
        sub = table.loc[table["method_key"] == method_key].set_index("pruning_band").reindex(bands)
        x = sub["mean_pruning_rate"].to_numpy(dtype=np.float64)
        y = sub["mean_memorization_surge"].to_numpy(dtype=np.float64)
        ax.plot(x, y, color=METHOD_COLORS[method_key], linewidth=1.15, marker="o", markersize=3.0, label=METHOD_LABELS[method_key])
    ax.axhline(0.0, color="#334155", linewidth=0.8)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"Pruning Rate (epoch {start_epoch})")
    ax.set_ylabel("Memorization Surge\n(in the annealing phase)")
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", borderaxespad=0.35, fontsize=4.8)
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.24, top=0.95)
    save_fig(fig, f"cifar10_hard_pruning_rate_vs_memorization_surge_{start_epoch}_{end_epoch}")
    plt.show()
    plt.close(fig)
    return table


def build_cifar10_all_pruning_rate_vs_memorization_surge(start_epoch: int = 87, end_epoch: int = 99, n_bins: int = 10) -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    anchor = build_pruning_difficulty_anchor(dataset_key, int(start_epoch))[["sample_id", "pruning_rate_anchor"]].dropna().copy()
    lo = float(anchor["pruning_rate_anchor"].min())
    hi = float(anchor["pruning_rate_anchor"].max())
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    edges[-1] = np.nextafter(edges[-1], edges[-1] + 1.0)
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
    anchor["pruning_band"] = pd.cut(
        anchor["pruning_rate_anchor"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    anchor["pruning_band"] = pd.Categorical(anchor["pruning_band"], categories=labels, ordered=True)
    anchor["pruning_band_min"] = anchor["pruning_band"].map(dict(zip(labels, edges[:-1]))).astype(float)
    anchor["pruning_band_max"] = anchor["pruning_band"].map(dict(zip(labels, edges[1:]))).astype(float)

    rows = []
    for method_key in pruned_method_keys_for_dataset(dataset_key):
        start = _method_epoch_mem(bundle, method_key, int(start_epoch))[["sample_id", "memorization"]].rename(
            columns={"memorization": "start_memorization"}
        )
        end = _method_epoch_mem(bundle, method_key, int(end_epoch))[["sample_id", "memorization"]].rename(
            columns={"memorization": "end_memorization"}
        )
        merged = anchor.merge(start, on="sample_id", how="inner").merge(end, on="sample_id", how="inner").dropna()
        merged["start_memorization"] = _cap(merged["start_memorization"])
        merged["end_memorization"] = _cap(merged["end_memorization"])
        merged["memorization_surge"] = merged["end_memorization"] - merged["start_memorization"]
        summary = (
            merged.groupby("pruning_band", observed=False)
            .agg(
                mean_memorization_surge=("memorization_surge", "mean"),
                start_mean_memorization=("start_memorization", "mean"),
                end_mean_memorization=("end_memorization", "mean"),
                n=("sample_id", "size"),
                mean_pruning_rate=("pruning_rate_anchor", "mean"),
                pruning_band_min=("pruning_band_min", "first"),
                pruning_band_max=("pruning_band_max", "first"),
            )
            .reset_index()
        )
        summary["dataset_key"] = dataset_key
        summary["dataset"] = bundle["cfg"]["label"]
        summary["method_key"] = method_key
        summary["method"] = METHOD_LABELS[method_key]
        summary["start_epoch"] = int(start_epoch)
        summary["end_epoch"] = int(end_epoch)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def plot_cifar10_all_pruning_rate_vs_memorization_surge(start_epoch: int = 87, end_epoch: int = 99) -> pd.DataFrame:
    table = build_cifar10_all_pruning_rate_vs_memorization_surge(start_epoch=start_epoch, end_epoch=end_epoch)
    save_table(table, f"cifar10_all_pruning_rate_vs_memorization_surge_{start_epoch}_{end_epoch}")
    band_meta = table[["pruning_band", "pruning_band_min"]].drop_duplicates().sort_values("pruning_band_min")
    bands = band_meta["pruning_band"].tolist()
    x_min = float(table["pruning_band_min"].min())
    x_max = float(table["pruning_band_max"].max())
    x_pad = 0.04 * max(x_max - x_min, 1e-8)
    y_vals = table["mean_memorization_surge"].to_numpy(dtype=np.float64)
    y_min = min(0.0, float(np.nanmin(y_vals)) * 1.18)
    y_max = max(0.03, float(np.nanmax(y_vals)) * 1.22)

    fig, ax = plt.subplots(figsize=(3.05, 2.45))
    for method_key in pruned_method_keys_for_dataset("cifar10"):
        sub = table.loc[table["method_key"] == method_key].set_index("pruning_band").reindex(bands)
        x = sub["mean_pruning_rate"].to_numpy(dtype=np.float64)
        y = sub["mean_memorization_surge"].to_numpy(dtype=np.float64)
        ax.plot(x, y, color=METHOD_COLORS[method_key], linewidth=1.15, marker="o", markersize=3.0, label=METHOD_LABELS[method_key])
    ax.axhline(0.0, color="#334155", linewidth=0.8)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"Pruning Rate (epoch {start_epoch})")
    ax.set_ylabel("Memorization Surge\n(in the annealing phase)")
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", borderaxespad=0.35, fontsize=4.8)
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.24, top=0.95)
    save_fig(fig, f"cifar10_all_pruning_rate_vs_memorization_surge_{start_epoch}_{end_epoch}")
    plt.show()
    plt.close(fig)
    return table


def build_cifar10_pruning_rate_vs_loss(epoch: int = 10, pruning_epoch: int | None = None) -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    pruning_epoch = int(pruning_epoch if pruning_epoch is not None else bundle["cfg"]["end_epoch"])
    std_cfg = bundle["cfg"]["methods"]["std"]
    std_evo = rpu.evolution_epoch_frame(dataset_key, "std", std_cfg, epoch)
    loss_anchor = std_evo[["sample_id", "loss"]].rename(columns={"loss": "standard_loss"}).dropna().copy()
    standard_loss_threshold = float(loss_anchor["standard_loss"].mean())
    loss_anchor["standard_loss_difficulty"] = np.where(loss_anchor["standard_loss"] > standard_loss_threshold, "Hard", "Easy")
    rows = []
    for method_key in pruned_method_keys_for_dataset(dataset_key):
        loss_evo = _method_epoch_frame(bundle, method_key, epoch, rmia=False)[["sample_id", "loss"]]
        pruning_evo = _method_epoch_frame(bundle, method_key, pruning_epoch, rmia=False)
        pruning_col, pruning_label = _resolve_pruning_metric(bundle, method_key, pruning_evo)
        if pruning_col is None or pruning_col not in pruning_evo.columns:
            continue
        frame = (
            loss_anchor.merge(loss_evo[["sample_id", "loss"]], on="sample_id", how="inner")
            .merge(pruning_evo[["sample_id", pruning_col]], on="sample_id", how="inner")
            .rename(columns={"loss": "method_loss", pruning_col: "pruning_rate"})
            .dropna()
        )
        method_loss_threshold = float(frame["method_loss"].mean())
        frame["method_loss_difficulty"] = np.where(frame["method_loss"] > method_loss_threshold, "Hard", "Easy")
        lo = float(frame["pruning_rate"].min())
        hi = float(frame["pruning_rate"].max())
        if np.isclose(lo, hi):
            hi = lo + 1e-8
        edges = np.linspace(lo, hi, 4)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        labels = _range_labels(edges)
        frame["pruning_band"] = pd.cut(frame["pruning_rate"], bins=edges, labels=labels, include_lowest=True, ordered=True)
        frame["pruning_band"] = pd.Categorical(frame["pruning_band"], categories=labels, ordered=True)
        frame["pruning_band_min"] = frame["pruning_band"].map(dict(zip(labels, edges[:-1]))).astype(float)
        frame["pruning_band_max"] = frame["pruning_band"].map(dict(zip(labels, edges[1:]))).astype(float)
        frame["dataset_key"] = dataset_key
        frame["dataset"] = bundle["cfg"]["label"]
        frame["method_key"] = method_key
        frame["method"] = METHOD_LABELS[method_key]
        frame["loss_epoch"] = int(epoch)
        frame["pruning_epoch"] = int(pruning_epoch)
        frame["pruning_metric"] = pruning_label
        frame["standard_loss_threshold"] = standard_loss_threshold
        frame["method_loss_threshold"] = method_loss_threshold
        rows.append(frame)
    if not rows:
        raise ValueError(f"No pruning/loss rows available for CIFAR-10 loss epoch {epoch} and pruning epoch {pruning_epoch}.")
    return pd.concat(rows, ignore_index=True)


def plot_cifar10_pruning_rate_vs_loss(epoch: int = 10) -> pd.DataFrame:
    table = build_cifar10_pruning_rate_vs_loss(epoch)
    pruning_epoch = int(table["pruning_epoch"].iloc[0])
    method_keys = pruned_method_keys_for_dataset("cifar10")
    save_table(table, f"cifar10_pruning_rate_vs_loss_epoch_{epoch}")
    fig, axes = plt.subplots(1, len(method_keys), figsize=(3.1 * len(method_keys), 2.75), sharey=True, squeeze=False)
    axes = axes[0]
    stack_order = ["Easy", "Hard"]
    stack_colors = {"Easy": lighten_color(DIFFICULTY_COLORS["Easy"], 0.28), "Hard": lighten_color(DIFFICULTY_COLORS["Hard"], 0.18)}
    for ax, method_key in zip(axes, method_keys):
        sub = table.loc[table["method_key"] == method_key].copy()
        corr = sub[["standard_loss", "pruning_rate"]].corr(method="spearman").iloc[0, 1]
        band_order = (
            sub[["pruning_band", "pruning_band_min"]]
            .drop_duplicates()
            .sort_values("pruning_band_min")["pruning_band"]
            .astype(str)
            .tolist()
        )
        counts = (
            sub.groupby(["pruning_band", "standard_loss_difficulty"], observed=False)
            .size()
            .rename("n")
            .reset_index()
        )
        totals = counts.groupby("pruning_band", observed=False)["n"].transform("sum")
        counts["pct_within_band"] = 100.0 * counts["n"] / totals
        bottom = np.zeros(len(band_order), dtype=np.float64)
        x = np.arange(len(band_order))
        for difficulty in stack_order:
            vals = (
                counts.loc[counts["standard_loss_difficulty"] == difficulty]
                .set_index("pruning_band")
                .reindex(band_order)["pct_within_band"]
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            ax.bar(x, vals, bottom=bottom, width=0.62, color=stack_colors[difficulty], edgecolor="white", linewidth=0.55, label=difficulty)
            bottom += vals
        ax.text(
            0.0,
            1.04,
            f"{METHOD_LABELS[method_key]}   Spearman $\\rho$ = {corr:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="#475569",
            clip_on=False,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(band_order, fontsize=6.5)
        ax.set_xlabel(f"Pruning-rate range after epoch {pruning_epoch}")
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Samples in pruning band (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.82, wspace=0.18)
    save_fig(fig, f"cifar10_pruning_rate_vs_loss_epoch_{epoch}")
    plt.show()
    plt.close(fig)
    return table

def plot_pruning_anchor_vs_loss_paper(epoch: int = 10, dataset_key: str = "cifar10") -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    pruning_epoch = int(bundle["cfg"]["end_epoch"])
    method_keys = [method_key for method_key in ["std", "ib"] if method_key in bundle["cfg"]["methods"]]

    # --- pruning anchor (method-agnostic hardness) ---
    anchor = build_pruning_difficulty_anchor(dataset_key, pruning_epoch)
    anchor = anchor[["sample_id", "pruning_rate_anchor"]]

    df = anchor.copy()
    for method_key in method_keys:
        loss_col = f"{method_key}_loss"
        method_cfg = bundle["cfg"]["methods"][method_key]
        loss_frame = rpu.evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)[["sample_id", "loss"]].rename(columns={"loss": loss_col})
        df = df.merge(loss_frame, on="sample_id", how="inner")
    df = df.dropna()

    df["dataset_key"] = dataset_key
    df["dataset"] = bundle["cfg"]["label"]
    df["epoch"] = int(epoch)

    # --- plotting ---
    fig, axes = plt.subplots(1, len(method_keys), figsize=(2.7 * len(method_keys), 2.4), sharey=True, squeeze=False)
    axes = axes[0]
    specs = [(f"{method_key}_loss", METHOD_LABELS[method_key], METHOD_COLORS[method_key]) for method_key in method_keys]

    y_max = float(df["pruning_rate_anchor"].max()) * 1.05

    for ax, (loss_col, label, color) in zip(axes, specs):
        x = df[loss_col].to_numpy(dtype=np.float64)
        y = df["pruning_rate_anchor"].to_numpy(dtype=np.float64)

        # scatter
        ax.scatter(
            x,
            y,
            s=2.0,
            alpha=0.18,
            color=color,
            linewidths=0.0,
            rasterized=True,
        )

        # regression line
        if len(x) > 1:
            coeffs = np.polyfit(x, y, deg=1)
            x_line = np.linspace(0.0, float(np.quantile(x, 0.995)), 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, color="#111827", linewidth=1.0, linestyle="--")

            corr = np.corrcoef(x, y)[0, 1]
        else:
            corr = np.nan

        # small annotation (top-right)
        ax.text(
            0.98,
            0.96,
            f"$pearson-r$ = {corr:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#334155",
        )

        ax.set_xlabel(f"{label} loss (epoch {epoch})")
        ax.set_ylim(0.0, y_max)
        ax.grid(True, axis="both", alpha=0.45)

    axes[0].set_ylabel("Pruning-rate")

    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.26, top=0.92, wspace=0.22)

    save_fig(fig, f"{dataset_key}_pruning_anchor_vs_loss_paper_epoch_{epoch}")
    plt.show()
    plt.close(fig)

    return df

def plot_cifar10_pruning_rate_loss_scatter(epoch: int = 10) -> pd.DataFrame:
    table = build_cifar10_pruning_rate_vs_loss(epoch)
    save_table(table, f"cifar10_pruning_rate_loss_scatter_epoch_{epoch}")
    pruning_epoch = int(table["pruning_epoch"].iloc[0])
    method_keys = pruned_method_keys_for_dataset("cifar10")
    fig, axes = plt.subplots(2, len(method_keys), figsize=(3.2 * len(method_keys), 5.1), sharey=True, squeeze=False)
    y_max = max(0.02, float(table["pruning_rate"].max()) * 1.05)
    color_map = {"Easy": lighten_color(DIFFICULTY_COLORS["Easy"], 0.12), "Hard": lighten_color(DIFFICULTY_COLORS["Hard"], 0.08)}
    row_specs = [
        ("standard_loss", "standard_loss_difficulty", "standard_loss_threshold", "Standard loss"),
        ("method_loss", "method_loss_difficulty", "method_loss_threshold", "Method loss"),
    ]
    for col_idx, method_key in enumerate(method_keys):
        sub = table.loc[table["method_key"] == method_key].copy()
        band_edges = sub[["pruning_band_min", "pruning_band_max"]].drop_duplicates().sort_values("pruning_band_min").to_numpy(dtype=np.float64)
        for row_idx, (loss_col, difficulty_col, threshold_col, loss_label) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            corr = sub[[loss_col, "pruning_rate"]].corr(method="spearman").iloc[0, 1]
            threshold = float(sub[threshold_col].iloc[0])
            x_max = float(sub[loss_col].quantile(0.995))
            for difficulty in ["Easy", "Hard"]:
                points = sub.loc[sub[difficulty_col] == difficulty]
                ax.scatter(
                    points[loss_col],
                    points["pruning_rate"],
                    s=2.0,
                    alpha=0.18 if difficulty == "Easy" else 0.24,
                    color=color_map[difficulty],
                    linewidths=0.0,
                    rasterized=True,
                    label=difficulty,
                )
            for edge in np.unique(band_edges[:, 0])[1:]:
                ax.axhline(edge, color="#64748B", linewidth=0.65, linestyle=":", alpha=0.85)
            ax.axvline(threshold, color="#334155", linewidth=0.8, linestyle="--")
            ax.text(
                0.02,
                0.96,
                f"{METHOD_LABELS[method_key]}\n{loss_label}\nSpearman $\\rho$ = {corr:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                color="#334155",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.50},
            )
            ax.set_xlabel(f"{loss_label} after epoch {epoch}")
            ax.set_xlim(0.0, x_max)
            ax.set_ylim(0.0, y_max)
            ax.grid(True, axis="both", alpha=0.45)
    axes[0, 0].set_ylabel(f"Pruning rate\nafter epoch {pruning_epoch}")
    axes[1, 0].set_ylabel(f"Pruning rate\nafter epoch {pruning_epoch}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.90, wspace=0.18, hspace=0.30)
    save_fig(fig, f"cifar10_pruning_rate_loss_scatter_epoch_{epoch}")
    plt.show()
    plt.close(fig)
    return table


def build_privacy_exposure_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = rpu.section_build_rmia_table(RMIA_DATASET_KEYS, METHOD_KEYS)
    renamed = raw.rename(
        columns={
            "AUC ROC (%)": "Attack success AUC (%)",
            "AUC ROC std (%)": "Attack success AUC std (%)",
            "TPR @ 0% FPR (%)": "Detected @ 0% false positive rate (%)",
            "TPR @ 0% FPR std (%)": "Detected @ 0% false positive rate std (%)",
            "TPR @ 0.01% FPR (%)": "Detected @ 0.01% false positive rate (%)",
            "TPR @ 0.01% FPR std (%)": "Detected @ 0.01% false positive rate std (%)",
            "TPR @ 1% FPR (%)": "Detected @ 1% false positive rate (%)",
            "TPR @ 1% FPR std (%)": "Detected @ 1% false positive rate std (%)",
        }
    )
    renamed["method"] = renamed["method_key"].map(METHOD_LABELS).fillna(renamed["method"])
    pretty_rows = []
    specs = [
        ("Attack success AUC (%)", "Attack success AUC std (%)", "Attack success AUC"),
        ("Detected @ 0.01% false positive rate (%)", "Detected @ 0.01% false positive rate std (%)", "Detected @ 0.01% false positive rate"),
        ("Detected @ 1% false positive rate (%)", "Detected @ 1% false positive rate std (%)", "Detected @ 1% false positive rate"),
    ]
    for _, row in renamed.iterrows():
        out = {"Dataset": row["dataset"], "Method": METHOD_LABELS.get(row["method_key"], row["method"])}
        for mean_col, std_col, label in specs:
            out[label] = f"{float(row[mean_col]):.2f} +/- {float(row[std_col]):.2f}"
        pretty_rows.append(out)
    pretty = pd.DataFrame(pretty_rows)
    save_table(renamed, "membership_inference_table_raw")
    save_table(pretty, "membership_inference_table_paper", latex=True)
    return renamed, pretty


def build_entropy_memorization_summary(dataset_key: str = "cifar10", mem_threshold: float = 0.5) -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    epoch = int(bundle["cfg"]["end_epoch"])
    rows = []
    for method_key in METHOD_KEYS:
        frame = _method_epoch_frame(bundle, method_key, epoch, rmia=True)
        mem = _cap(frame["memorization"])
        norm_entropy = np.clip(frame["entropy"].to_numpy(dtype=np.float64) / rpu.theoretical_max_entropy(bundle), 0.0, 1.0)
        detection_pct = 100.0 * frame["detection_rate_0p01pct"].to_numpy(dtype=np.float64)
        groups = np.where(mem >= mem_threshold, f"High memorization\n>= {mem_threshold:.1f}", f"Low memorization\n< {mem_threshold:.1f}")
        for group in [f"Low memorization\n< {mem_threshold:.1f}", f"High memorization\n>= {mem_threshold:.1f}"]:
            mask = groups == group
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": METHOD_LABELS[method_key],
                    "memorization_group": group.replace("\n", " "),
                    "epoch": epoch,
                    "n": int(mask.sum()),
                    "sample_share_pct": float(100.0 * mask.mean()),
                    "mean_memorization": float(mem[mask].mean()) if mask.any() else np.nan,
                    "mean_normalized_entropy": float(norm_entropy[mask].mean()) if mask.any() else np.nan,
                    "mean_detected_pct": float(detection_pct[mask].mean()) if mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_entropy_memorization_summary(dataset_key: str = "cifar10", mem_threshold: float = 0.5) -> pd.DataFrame:
    table = build_entropy_memorization_summary(dataset_key, mem_threshold)
    save_table(table, f"{dataset_key}_entropy_by_memorization_group")
    groups = [f"Low memorization < {mem_threshold:.1f}", f"High memorization >= {mem_threshold:.1f}"]
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.6))
    x = np.arange(len(groups))
    width = 0.22
    specs = [
        ("mean_normalized_entropy", "Mean normalized entropy", "Predictive entropy"),
        ("mean_detected_pct", "Mean detected at 0.01% FPR (%)", "Membership detection"),
    ]
    for ax, (value_col, ylabel, panel_label) in zip(axes, specs):
        y_max = max(0.01, float(table[value_col].max()))
        for idx, method_key in enumerate(METHOD_KEYS):
            sub = table.loc[table["method_key"] == method_key].set_index("memorization_group").reindex(groups)
            vals = sub[value_col].to_numpy(dtype=np.float64)
            ax.bar(x + (idx - 1) * width, vals, width=width, color=METHOD_COLORS[method_key], label=METHOD_LABELS[method_key])
        ax.set_xticks(x)
        ax.set_xticklabels(["Low mem.", "High mem."])
        ax.set_ylabel(ylabel)
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#475569")
        ax.set_ylim(0, y_max * 1.28)
        ax.grid(True, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.82, wspace=0.32)
    save_fig(fig, f"{dataset_key}_entropy_by_memorization_group")
    plt.show()
    plt.close(fig)
    return table


def plot_membership_inference_3nts(mem_threshold: float = 0.5, entropy_threshold: float = 0.3) -> pd.DataFrame:
    rows = []
    for dataset_key in RMIA_DATASET_KEYS:
        bundle = rpu.report_bundle(dataset_key)
        epoch = int(bundle["cfg"]["end_epoch"])
        fig = plt.figure(figsize=(7.1, 2.18 * len(METHOD_KEYS) + 0.10))
        grid = fig.add_gridspec(len(METHOD_KEYS), 3, width_ratios=[2.20, 1.42, 0.055], hspace=0.34, wspace=0.30)
        vmax = 0.0
        frames = {}
        for method_key in METHOD_KEYS:
            frame = _method_epoch_frame(bundle, method_key, epoch, rmia=True)
            frames[method_key] = frame
            vmax = max(vmax, float(np.nanmax(100.0 * frame["detection_rate_0p01pct"].to_numpy(dtype=np.float64))))
        vmax = max(vmax, 1e-6)
        scatter = None
        for row_idx, method_key in enumerate(METHOD_KEYS):
            method_label = METHOD_LABELS[method_key]
            frame = frames[method_key]
            mem = _cap(frame["memorization"])
            norm_entropy = np.clip(frame["entropy"].to_numpy(dtype=np.float64) / rpu.theoretical_max_entropy(bundle), 0.0, 1.0)
            detection_pct = 100.0 * frame["detection_rate_0p01pct"].to_numpy(dtype=np.float64)
            stats = rpu.compute_rmia_norm_entropy_quadrants(
                mem,
                norm_entropy,
                detection_pct,
                mem_threshold=mem_threshold,
                norm_entropy_threshold=entropy_threshold,
            )
            ax = fig.add_subplot(grid[row_idx, 0])
            order = np.argsort(detection_pct, kind="stable")
            scatter = ax.scatter(mem[order], norm_entropy[order], c=detection_pct[order], cmap=rpu.SECTION_RMIA_CMAP, vmin=0.0, vmax=vmax, s=5.0, alpha=0.50, linewidths=0.0, rasterized=True)
            ax.axvline(mem_threshold, color="#475569", linestyle="--", linewidth=0.85)
            ax.axhline(entropy_threshold, color="#475569", linestyle="--", linewidth=0.85)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_ylabel(f"{method_label}\nNorm. entropy")
            if row_idx == len(METHOD_KEYS) - 1:
                ax.set_xlabel("Memorization score")
            else:
                ax.set_xticklabels([])
            ax.grid(True, alpha=0.26)

            axm = fig.add_subplot(grid[row_idx, 1])
            axm.set_xlim(-0.5, 1.5)
            axm.set_ylim(1.5, -0.5)
            axm.set_aspect("equal", adjustable="box")
            axm.set_xticks([0, 1])
            axm.set_xticklabels([f"< {mem_threshold:.1f}", f">= {mem_threshold:.1f}"])
            axm.set_yticks([0, 1])
            axm.set_yticklabels(["High ent.", "Low ent."])
            axm.tick_params(length=0)
            fill_max = max(float(np.nanmax(stats["fill_fraction"])), 1e-12)
            for y in range(2):
                for x_idx in range(2):
                    fill_strength = float(stats["fill_fraction"][y, x_idx]) / fill_max
                    fill = plt.cm.Blues(0.10 + 0.54 * fill_strength)
                    axm.add_patch(mpatches.Rectangle((x_idx - 0.5, y - 0.5), 1.0, 1.0, facecolor=fill, edgecolor="#94A3B8", linewidth=0.75))
                    n = int(stats["counts"][y, x_idx])
                    sample_share = float(stats["sample_share_pct"][y, x_idx])
                    mean_detected = float(stats["mean_detection_pct"][y, x_idx])
                    detection_share = float(stats["detection_share_pct"][y, x_idx])
                    normalized_detection_lift = float(stats["normalized_detection_lift"][y, x_idx])
                    axm.text(
                        x_idx,
                        y,
                        f"S={sample_share:.1f}%\nn={n:,}\nA={mean_detected:.2f}%\nD={detection_share:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7.6,
                        color="white" if fill_strength >= 0.58 else "#0F172A",
                        linespacing=1.13,
                    )
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": bundle["cfg"]["label"],
                            "method_key": method_key,
                            "method": method_label,
                            "entropy_bin": "high" if y == 0 else "low",
                            "memorization_bin": "high" if x_idx == 1 else "low",
                            "n": n,
                            "sample_share_pct": sample_share,
                            "average_detection_rate_pct": mean_detected,
                            "mean_detected_pct": mean_detected,
                            "detection_share_pct": detection_share,
                            "global_mean_detected_pct": float(stats["global_mean_detection_pct"]),
                            "normalized_detection_lift": normalized_detection_lift,
                        }
                    )
            for spine in axm.spines.values():
                spine.set_visible(False)
        cax = fig.add_subplot(grid[:, 2])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Detected at 0.01% false positive rate (%)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        fig.text(0.5, 0.01, "MIA = membership-inference attack. S = sample share; A = average detection rate in quadrant; D = share of total detected mass at 0.01% FPR.", ha="center", va="bottom", fontsize=8, color="#475569")
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.98)
        save_fig(fig, f"membership_inference_quadrants_{dataset_key}")
        plt.show()
        plt.close(fig)
    table = pd.DataFrame(rows)
    save_table(table, "membership_inference_quadrant_values")
    return table


def plot_membership_inference_quadrants(mem_threshold: float = 0.5, entropy_threshold: float = 0.3) -> pd.DataFrame:
    return plot_membership_inference_3nts(mem_threshold=mem_threshold, entropy_threshold=entropy_threshold)


def run_all(output_dir: str | Path = OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    summary = initialize(output_dir)
    raw_privacy, pretty_privacy = build_privacy_exposure_table()
    outputs = {
        "dataset_summary": summary,
        "privacy_table_raw": raw_privacy,
        "privacy_table_paper": pretty_privacy,
        "privacy_quadrants": plot_membership_inference_quadrants(),
        "cifar10_tail_histogram": plot_cifar10_tail_histogram_by_pruning_difficulty(),
        "pruning_difficulty_selection_summary": plot_pruning_difficulty_selection_summary(),
        "cifar10_pruning_difficulty_trajectories": plot_pruning_difficulty_trajectories(),
        "pruning_difficulty_trajectories_separate": plot_pruning_difficulty_trajectories_separate(),
        "cifar10_hard_sample_pruning_response": plot_hard_sample_pruning_response("cifar10"),
        "cifar10_hard_pruning_epoch_memorization": plot_cifar10_hard_pruning_epoch_memorization(),
        "dataset_mean_trajectories": plot_dataset_mean_trajectories(),
        "cumulative_memorization_exposure": plot_memorization_exposure_metric(),
        "phase_memorization_growth": plot_late_phase_memorization_growth(),
        "cifar10_pruning_rate_vs_memorization_rise_87_99": plot_cifar10_pruning_rate_vs_late_memorization_rise(),
        "hard_pruning_band_late_growth": plot_hard_pruning_band_late_growth(),
        "cifar10_hard_pruning_rate_vs_memorization_surge_87_99": plot_cifar10_hard_pruning_rate_vs_memorization_surge(),
        "cifar10_all_pruning_rate_vs_memorization_surge_87_99": plot_cifar10_all_pruning_rate_vs_memorization_surge(),
        "cifar10_pruning_anchor_vs_loss_paper_epoch_10": plot_pruning_anchor_vs_loss_paper(10),
        "cifar10_pruning_anchor_vs_loss_paper_epoch_50": plot_pruning_anchor_vs_loss_paper(50),
        "cifar10_hard_loss_phase_summary": plot_cifar10_hard_loss_phase_summary(),
        "entropy_memorization_summary": plot_entropy_memorization_summary("cifar10"),
    }
    return outputs


if __name__ == "__main__":
    run_all()
