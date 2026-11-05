from __future__ import annotations

import json
import os
from pathlib import Path

from prunepriv.paths import results_dir as _results_dir

ROOT = Path(os.environ.get("PRUNEPRIV_ANALYSIS_ROOT", _results_dir("analysis"))).resolve()
ANALYSIS_ROOT = ROOT
MPLCONFIGDIR = ANALYSIS_ROOT / "_cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from prunepriv.analysis import report_plot_utils as rpu


DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "outputs" / "paper_analysis_mechanism_v3"
PAPER_METHODS = ["std", "ib", "ib_prw"]
MAIN_DATASETS = ["cifar10", rpu.WRN_DATASET_KEY]
METHOD_COLORS = {key: rpu.METHOD_COLORS[key] for key in PAPER_METHODS}
TRAPZ = getattr(np, "trapezoid", np.trapz)


def threshold_slug(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


def paper_theme(font_size: int = 8) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": max(font_size - 1, 7),
            "ytick.labelsize": max(font_size - 1, 7),
            "legend.fontsize": max(font_size - 1, 7),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def initialize_paper_analysis(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    active_datasets: list[str] | None = None,
    rmia_datasets: list[str] | None = None,
    method_keys: list[str] | None = None,
    rmia_mem_threshold: float = 0.5,
    rmia_normalized_entropy_threshold: float = 0.3,
) -> pd.DataFrame:
    active_datasets = list(active_datasets or MAIN_DATASETS)
    rmia_datasets = list(rmia_datasets or MAIN_DATASETS)
    method_keys = list(method_keys or PAPER_METHODS)
    summary = rpu.initialize_report(
        {
            "active_datasets": active_datasets,
            "pruning_datasets": [key for key in active_datasets if key in MAIN_DATASETS],
            "rmia_datasets": rmia_datasets,
            "all_methods": method_keys,
            "pruned_methods": [key for key in method_keys if key != "std"],
            "save_figures": False,
            "save_vector": False,
            "output_dir": output_dir,
            "figure_dpi": 320,
            "font_size": 8,
            "rmia_mem_threshold": rmia_mem_threshold,
            "rmia_normalized_entropy_threshold": rmia_normalized_entropy_threshold,
        }
    )
    paper_theme()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return summary


def save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    out_dir = Path(output_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt.lower() == "png":
            kwargs["dpi"] = 320
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def save_table(
    table: pd.DataFrame,
    stem: str,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "tables",
    latex: bool = False,
) -> dict[str, Path]:
    out_dir = Path(output_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out_dir / f"{stem}.csv"}
    table.to_csv(paths["csv"], index=False)
    if latex:
        paths["tex"] = out_dir / f"{stem}.tex"
        paths["tex"].write_text(table.to_latex(index=False, escape=False), encoding="utf-8")
    return paths


def build_pruning_rank_anchor(
    bundle: dict,
    *,
    anchor_epoch: int,
    n_bins: int = 5,
    anchor_method_key: str = "ib",
) -> tuple[pd.DataFrame, str, str]:
    method_cfg = bundle["cfg"]["methods"][anchor_method_key]
    evo = rpu.evolution_epoch_frame(bundle["dataset_key"], anchor_method_key, method_cfg, anchor_epoch)
    pruning_col, pruning_label = rpu.resolve_pruning_metric(bundle, anchor_method_key, evo)
    if pruning_col is None:
        raise KeyError(f"No pruning metric found for {bundle['dataset_key']} / {anchor_method_key}")
    anchor = (
        evo[["sample_id", pruning_col]]
        .rename(columns={pruning_col: "anchor_pruning_rate"})
        .dropna(subset=["anchor_pruning_rate"])
        .copy()
    )
    anchor = anchor.sort_values(["anchor_pruning_rate", "sample_id"], ascending=[False, True]).reset_index(drop=True)
    n = len(anchor)
    ranks = np.arange(n, dtype=np.int64)
    anchor["bin_num"] = np.minimum((ranks * int(n_bins)) // max(n, 1) + 1, int(n_bins)).astype(int)
    anchor["anchor_pruning_percentile"] = 100.0 * (ranks + 0.5) / max(n, 1)
    anchor["anchor_method_key"] = anchor_method_key
    anchor["anchor_pruning_metric"] = pruning_col
    return anchor.sort_values("sample_id").reset_index(drop=True), pruning_col, pruning_label


def build_final_tail_sweep(
    *,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or ["cifar10"])
    method_keys = list(method_keys or PAPER_METHODS)
    thresholds = np.asarray(np.linspace(0.0, 1.0, 201) if thresholds is None else thresholds, dtype=np.float64)
    rows = []
    for dataset_key in dataset_keys:
        bundle = rpu.report_bundle(dataset_key)
        epoch = int(bundle["cfg"]["end_epoch"])
        active_methods = rpu.section_method_keys(bundle, method_keys)
        ccdf_by_method = {}
        for method_key in active_methods:
            scores = rpu.cap_memorization(
                rpu.get_method_epoch_mem(bundle, method_key, epoch)["memorization"].to_numpy(dtype=np.float64)
            )
            shares = np.array([100.0 * float(np.mean(scores >= threshold)) for threshold in thresholds])
            ccdf_by_method[method_key] = shares
            for threshold, share in zip(thresholds, shares):
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": bundle["cfg"]["label"],
                        "method_key": method_key,
                        "method": rpu.METHOD_LABELS[method_key],
                        "epoch": epoch,
                        "memorization_cutoff": float(threshold),
                        "share_above_cutoff_pct": float(share),
                        "standard_reduction_pp": np.nan,
                    }
                )
        if "std" in ccdf_by_method:
            std_share = ccdf_by_method["std"]
            for method_key in active_methods:
                if method_key == "std":
                    continue
                reduction = std_share - ccdf_by_method[method_key]
                for threshold, value in zip(thresholds, reduction):
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": bundle["cfg"]["label"],
                            "method_key": f"std-{method_key}",
                            "method": f"Standard - {rpu.METHOD_LABELS[method_key]}",
                            "epoch": epoch,
                            "memorization_cutoff": float(threshold),
                            "share_above_cutoff_pct": np.nan,
                            "standard_reduction_pp": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def plot_final_tail_sweep(
    *,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or ["cifar10"])
    method_keys = list(method_keys or PAPER_METHODS)
    table = build_final_tail_sweep(dataset_keys=dataset_keys, method_keys=method_keys)
    save_table(table, "final_memorization_tail_sweep", output_dir=output_dir)

    fig, axes = plt.subplots(1, len(dataset_keys), figsize=(3.55 * len(dataset_keys), 2.45), squeeze=False)
    for ax, dataset_key in zip(axes[0], dataset_keys):
        sub = table.loc[
            (table["dataset_key"] == dataset_key)
            & table["method_key"].isin([f"std-{key}" for key in method_keys if key != "std"])
        ]
        for method_key in [key for key in method_keys if key != "std"]:
            curve = sub.loc[sub["method_key"] == f"std-{method_key}"].sort_values("memorization_cutoff")
            if curve.empty:
                continue
            ax.plot(
                curve["memorization_cutoff"],
                curve["standard_reduction_pp"],
                color=METHOD_COLORS[method_key],
                linewidth=1.15,
                label=f"vs {rpu.METHOD_LABELS[method_key]}",
            )
        label = table.loc[table["dataset_key"] == dataset_key, "dataset"].iloc[0]
        ax.axhline(0.0, color="#334155", linewidth=0.8)
        ax.set_title(label, fontweight="normal")
        ax.set_xlabel("Memorization score cutoff t")
        ax.set_ylabel("Std - method sample share (p.p.)")
        ax.grid(True, axis="y")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.00), borderaxespad=0.0)
    fig.subplots_adjust(left=0.18 if len(dataset_keys) == 1 else 0.085, right=0.72 if len(dataset_keys) == 1 else 0.99, top=0.88, bottom=0.24, wspace=0.26)
    paths = save_figure(fig, "final_memorization_tail_sweep", output_dir=output_dir, subdir=subdir)
    plt.show()
    plt.close(fig)
    table.attrs["figure_paths"] = [str(path) for path in paths]
    return table


def build_standard_reference_response(
    *,
    dataset_key: str = "cifar10",
    method_keys: list[str] | None = None,
    n_bins: int = 20,
) -> pd.DataFrame:
    method_keys = list(method_keys or PAPER_METHODS)
    bundle = rpu.report_bundle(dataset_key)
    epoch = int(bundle["cfg"]["end_epoch"])
    standard = rpu.get_method_epoch_mem(bundle, "std", epoch)[["sample_id", "memorization"]].rename(
        columns={"memorization": "standard_memorization"}
    )
    standard["standard_memorization"] = rpu.cap_memorization(standard["standard_memorization"].to_numpy(dtype=np.float64))
    ranked = standard.sort_values(["standard_memorization", "sample_id"], ascending=[True, True]).reset_index(drop=True)
    n = len(ranked)
    ranks = np.arange(n, dtype=np.int64)
    ranked["standard_percentile_bin"] = np.minimum((ranks * int(n_bins)) // max(n, 1) + 1, int(n_bins)).astype(int)
    ranked["standard_percentile_mid"] = 100.0 * (ranked["standard_percentile_bin"] - 0.5) / int(n_bins)
    anchor = ranked[["sample_id", "standard_memorization", "standard_percentile_bin", "standard_percentile_mid"]]

    rows = []
    for method_key in rpu.section_method_keys(bundle, method_keys):
        mem = rpu.get_method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]].copy()
        mem["memorization"] = rpu.cap_memorization(mem["memorization"].to_numpy(dtype=np.float64))
        merged = anchor.merge(mem, on="sample_id", how="inner")
        if method_key != "std":
            frame = rpu.section_build_method_frame(bundle, epoch, method_key, include_rmia=False)
            pruning_col, pruning_label = rpu.resolve_pruning_metric(bundle, method_key, frame)
            if pruning_col is not None:
                merged = merged.merge(frame[["sample_id", pruning_col]], on="sample_id", how="left")
            else:
                pruning_label = "Pruning rate"
                merged["__missing_pruning__"] = np.nan
                pruning_col = "__missing_pruning__"
        else:
            pruning_label = "Pruning rate"
            pruning_col = None

        for bin_num, group in merged.groupby("standard_percentile_bin", sort=True):
            standard_mean = float(group["standard_memorization"].mean())
            method_mean = float(group["memorization"].mean())
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": rpu.METHOD_LABELS[method_key],
                    "epoch": epoch,
                    "standard_percentile_bin": int(bin_num),
                    "standard_percentile_mid": float(group["standard_percentile_mid"].iloc[0]),
                    "n": int(len(group)),
                    "standard_memorization_mean": standard_mean,
                    "method_memorization_mean": method_mean,
                    "delta_vs_standard": method_mean - standard_mean,
                    "pruning_rate_mean": np.nan if pruning_col is None else float(group[pruning_col].mean()),
                    "pruning_label": pruning_label,
                }
            )
    return pd.DataFrame(rows)


def plot_standard_reference_response(
    *,
    dataset_key: str = "cifar10",
    method_keys: list[str] | None = None,
    n_bins: int = 20,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> pd.DataFrame:
    method_keys = list(method_keys or PAPER_METHODS)
    table = build_standard_reference_response(dataset_key=dataset_key, method_keys=method_keys, n_bins=n_bins)
    save_table(table, f"{dataset_key}_standard_reference_percentile_response", output_dir=output_dir)
    dataset_label = table["dataset"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(5.65, 2.45), squeeze=False)
    ax_mem, ax_delta = axes[0]
    for method_key in method_keys:
        sub = table.loc[table["method_key"] == method_key].sort_values("standard_percentile_mid")
        ax_mem.plot(
            sub["standard_percentile_mid"],
            sub["method_memorization_mean"],
            color=METHOD_COLORS[method_key],
            linewidth=1.35,
            marker="o",
            markersize=2.4,
            label=rpu.METHOD_LABELS[method_key],
        )
    for method_key in [key for key in method_keys if key != "std"]:
        sub = table.loc[table["method_key"] == method_key].sort_values("standard_percentile_mid")
        ax_delta.plot(
            sub["standard_percentile_mid"],
            -sub["delta_vs_standard"],
            color=METHOD_COLORS[method_key],
            linewidth=1.35,
            marker="o",
            markersize=2.4,
            label=rpu.METHOD_LABELS[method_key],
        )

    ax_delta.axhline(0.0, color="#334155", linewidth=0.8)
    ax_mem.set_ylabel("Mean memorization")
    ax_delta.set_ylabel("Std - method")
    for ax in axes[0]:
        ax.set_xlabel("Standard memorization percentile")
        ax.set_xlim(0.0, 100.0)
        ax.grid(True, axis="y")
    ax_mem.set_title("Same samples")
    ax_delta.set_title("Reduction")
    handles, labels = ax_mem.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=len(handles))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.82, bottom=0.24, wspace=0.34)
    paths = save_figure(fig, f"{dataset_key}_standard_reference_percentile_response", output_dir=output_dir, subdir=subdir)
    plt.show()
    plt.close(fig)
    table.attrs["figure_paths"] = [str(path) for path in paths]
    return table


def build_standard_top_percentile_trajectory(
    *,
    dataset_key: str = "cifar10",
    method_keys: list[str] | None = None,
    top_fraction: float = 0.40,
) -> pd.DataFrame:
    method_keys = list(method_keys or PAPER_METHODS)
    bundle = rpu.report_bundle(dataset_key)
    epoch = int(bundle["cfg"]["end_epoch"])
    standard = rpu.get_method_epoch_mem(bundle, "std", epoch)[["sample_id", "memorization"]].copy()
    standard["memorization"] = rpu.cap_memorization(standard["memorization"].to_numpy(dtype=np.float64))
    n_keep = max(1, int(round(len(standard) * float(top_fraction))))
    selected = standard.sort_values(["memorization", "sample_id"], ascending=[False, True]).head(n_keep)
    selected_ids = selected[["sample_id"]].copy()
    rows = []
    for method_key in rpu.section_method_keys(bundle, method_keys):
        frame = bundle["mem_long"][method_key]
        for epoch_value, chunk in frame.groupby("epoch", sort=True):
            merged = selected_ids.merge(chunk[["sample_id", "memorization"]], on="sample_id", how="inner")
            scores = rpu.cap_memorization(merged["memorization"].to_numpy(dtype=np.float64))
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": rpu.METHOD_LABELS[method_key],
                    "epoch": int(epoch_value),
                    "mean_memorization": float(np.mean(scores)),
                    "n": int(len(scores)),
                    "top_fraction": float(top_fraction),
                    "cohort_rule": f"top {100.0 * float(top_fraction):.0f}% by Standard final memorization",
                }
            )
    return pd.DataFrame(rows)


def plot_standard_top_percentile_trajectory(
    *,
    dataset_key: str = "cifar10",
    method_keys: list[str] | None = None,
    top_fraction: float = 0.40,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> pd.DataFrame:
    method_keys = list(method_keys or PAPER_METHODS)
    table = build_standard_top_percentile_trajectory(
        dataset_key=dataset_key,
        method_keys=method_keys,
        top_fraction=top_fraction,
    )
    pct = int(round(100.0 * float(top_fraction)))
    save_table(table, f"{dataset_key}_standard_top_{pct}pct_trajectory", output_dir=output_dir)
    dataset_label = table["dataset"].iloc[0]

    fig, ax = plt.subplots(figsize=(3.75, 2.35))
    for method_key in method_keys:
        sub = table.loc[table["method_key"] == method_key].sort_values("epoch")
        ax.plot(
            sub["epoch"],
            sub["mean_memorization"],
            color=METHOD_COLORS[method_key],
            linewidth=1.15,
            marker="o",
            markersize=2.4,
            label=rpu.METHOD_LABELS[method_key],
        )
    ax.set_title(f"{dataset_label}: top {pct}% by Standard final memorization", fontweight="normal")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean memorization")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(left=0.16, right=0.78, top=0.86, bottom=0.23)
    paths = save_figure(fig, f"{dataset_key}_standard_top_{pct}pct_trajectory", output_dir=output_dir, subdir=subdir)
    plt.show()
    plt.close(fig)
    table.attrs["figure_paths"] = [str(path) for path in paths]
    return table


def format_rmia_table_for_paper(table: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("AUC ROC (%)", "AUC ROC std (%)", "AUC ROC"),
        ("TPR @ 0% FPR (%)", "TPR @ 0% FPR std (%)", "TPR @ 0% FPR"),
        ("TPR @ 0.01% FPR (%)", "TPR @ 0.01% FPR std (%)", "TPR @ 0.01% FPR"),
        ("TPR @ 1% FPR (%)", "TPR @ 1% FPR std (%)", "TPR @ 1% FPR"),
    ]
    rows = []
    for _, row in table.iterrows():
        formatted = {"Dataset": row["dataset"], "Method": row["method"]}
        for mean_col, std_col, label in specs:
            formatted[label] = f"{float(row[mean_col]):.2f} +/- {float(row[std_col]):.2f}"
        rows.append(formatted)
    return pd.DataFrame(rows)


def build_rmia_table_outputs(
    *,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_keys = list(dataset_keys or MAIN_DATASETS)
    method_keys = list(method_keys or PAPER_METHODS)
    raw = rpu.section_build_rmia_table(dataset_keys, method_keys)
    formatted = format_rmia_table_for_paper(raw)
    save_table(raw, "rmia_table_raw", output_dir=output_dir, latex=False)
    save_table(formatted, "rmia_table_paper", output_dir=output_dir, latex=True)
    return raw, formatted


def build_rmia_quadrant_checks(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    checks = rpu.build_report_rmia_quadrant_check_table()
    save_table(checks, "rmia_quadrant_threshold_check", output_dir=output_dir)
    return checks


def _draw_matrix_cell(
    ax: plt.Axes,
    x: int,
    y: int,
    *,
    n_cell: int,
    total_n: int,
    mean_detection_pct: float,
    detection_share_pct: float,
    facecolor,
) -> None:
    ax.add_patch(
        mpatches.Rectangle(
            (x - 0.5, y - 0.5),
            1.0,
            1.0,
            facecolor=facecolor,
            edgecolor="#94A3B8",
            linewidth=0.75,
        )
    )
    sample_share = 100.0 * n_cell / max(total_n, 1)
    ax.text(
        x,
        y,
        f"{sample_share:.1f}% samp.\nn={n_cell:,}\navg atk {mean_detection_pct:.2f}%\natk mass {detection_share_pct:.1f}%",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#0F172A",
        linespacing=1.16,
    )


def plot_rmia_quadrants(
    dataset_key: str,
    *,
    method_keys: list[str] | None = None,
    epoch: int | None = None,
    mem_threshold: float = 0.5,
    norm_entropy_threshold: float = 0.3,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> pd.DataFrame:
    bundle = rpu.report_bundle(dataset_key)
    method_keys = rpu.section_method_keys(bundle, method_keys or PAPER_METHODS)
    epoch = int(epoch if epoch is not None else bundle["cfg"]["end_epoch"])
    detection_limit = rpu.section_detection_vmax(bundle, method_keys)
    detection_limit = max(detection_limit, 1e-6)

    fig = plt.figure(figsize=(7.35, 2.10 * len(method_keys) + 0.35))
    grid = fig.add_gridspec(
        len(method_keys),
        3,
        width_ratios=[2.55, 1.18, 0.055],
        hspace=0.42,
        wspace=0.28,
    )
    scatter_axes = []
    rows = []

    for row_idx, method_key in enumerate(method_keys):
        method_label = rpu.METHOD_LABELS[method_key]
        frame = rpu.section_build_method_frame(bundle, epoch, method_key, include_rmia=True, require_rmia=True)
        mem = rpu.cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
        norm_entropy = frame["entropy"].to_numpy(dtype=np.float64) / rpu.theoretical_max_entropy(bundle)
        norm_entropy = np.clip(norm_entropy, 0.0, 1.0)
        detection_pct = 100.0 * frame["detection_rate_0p01pct"].to_numpy(dtype=np.float64)
        stats = rpu.compute_rmia_norm_entropy_quadrants(
            mem,
            norm_entropy,
            detection_pct,
            mem_threshold=mem_threshold,
            norm_entropy_threshold=norm_entropy_threshold,
        )

        ax = fig.add_subplot(grid[row_idx, 0])
        scatter_axes.append(ax)
        order = np.argsort(detection_pct, kind="stable")
        ax.scatter(
            mem[order],
            norm_entropy[order],
            c=detection_pct[order],
            cmap=rpu.SECTION_RMIA_CMAP,
            vmin=0.0,
            vmax=detection_limit,
            s=5.0,
            alpha=0.48,
            linewidths=0.0,
            rasterized=True,
        )
        ax.axvline(mem_threshold, color="#475569", linestyle="--", linewidth=0.85)
        ax.axhline(norm_entropy_threshold, color="#475569", linestyle="--", linewidth=0.85)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"{method_label}\nNorm. entropy")
        if row_idx == len(method_keys) - 1:
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
        if row_idx == 0:
            axm.set_title("Samples / RMIA share")
        fill_max = max(float(np.nanmax(stats["fill_fraction"])), 1e-12)
        for y in range(2):
            for x in range(2):
                _draw_matrix_cell(
                    axm,
                    x,
                    y,
                    n_cell=int(stats["counts"][y, x]),
                    total_n=int(stats["total_n"]),
                    mean_detection_pct=float(stats["mean_detection_pct"][y, x]),
                    detection_share_pct=float(stats["detection_share_pct"][y, x]),
                    facecolor=rpu.SECTION_MATRIX_FILL(0.12 + 0.82 * stats["fill_fraction"][y, x] / fill_max),
                )
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": bundle["cfg"]["label"],
                        "method_key": method_key,
                        "method": method_label,
                        "epoch": epoch,
                        "entropy_bin": "high" if y == 0 else "low",
                        "memorization_bin": "high" if x == 1 else "low",
                        "n": int(stats["counts"][y, x]),
                        "sample_share_pct": float(stats["sample_share_pct"][y, x]),
                        "mean_detection_pct": float(stats["mean_detection_pct"][y, x]),
                        "rmia_share_pct": float(stats["detection_share_pct"][y, x]),
                    }
                )

    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0.0, vmax=detection_limit),
        cmap=rpu.SECTION_RMIA_CMAP,
    )
    cax = fig.add_subplot(grid[:, 2])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("RMIA @ 0.01% FPR (%)")
    fig.suptitle(
        f"{bundle['cfg']['label']}: memorization, entropy, and RMIA quadrants",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=9.5,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        0.012,
        (
            f"Thresholds: mem={mem_threshold:.1f}, norm ent={norm_entropy_threshold:.1f}. "
            "avg atk = mean RMIA detection rate in cell; atk mass = sum cell detection rates / sum all detection rates."
        ),
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#475569",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.085)
    paths = save_figure(
        fig,
        f"rmia_quadrants_{dataset_key}_mem{threshold_slug(mem_threshold)}_ent{threshold_slug(norm_entropy_threshold)}",
        output_dir=output_dir,
        subdir=subdir,
    )
    plt.show()
    plt.close(fig)
    table = pd.DataFrame(rows)
    save_table(table, f"rmia_quadrants_{dataset_key}_matrix_values", output_dir=output_dir)
    table.attrs["figure_paths"] = [str(path) for path in paths]
    return table


def build_onset_table(
    *,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or MAIN_DATASETS)
    method_keys = list(method_keys or PAPER_METHODS)
    rows = []
    for dataset_key in dataset_keys:
        bundle = rpu.report_bundle(dataset_key)
        active_methods = rpu.section_method_keys(bundle, method_keys)
        common_epochs = None
        for method_key in active_methods:
            epochs = set(int(epoch) for epoch in bundle["mem_long"][method_key]["epoch"].unique())
            common_epochs = epochs if common_epochs is None else common_epochs & epochs
        if not common_epochs:
            continue
        for method_key in active_methods:
            frame = bundle["mem_long"][method_key]
            for epoch in sorted(common_epochs):
                scores = rpu.cap_memorization(
                    frame.loc[frame["epoch"] == epoch, "memorization"].to_numpy(dtype=np.float64)
                )
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": bundle["cfg"]["label"],
                        "method_key": method_key,
                        "method": rpu.METHOD_LABELS[method_key],
                        "epoch": int(epoch),
                        "threshold": float(threshold),
                        "percent_above_threshold": 100.0 * float(np.mean(scores >= threshold)),
                    }
                )
    return pd.DataFrame(rows)


def build_onset_crossing_table(
    onset: pd.DataFrame,
    *,
    crossing_percents: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0),
) -> pd.DataFrame:
    rows = []
    for (dataset_key, method_key), group in onset.groupby(["dataset_key", "method_key"], sort=False):
        group = group.sort_values("epoch")
        for crossing in crossing_percents:
            hit = group.loc[group["percent_above_threshold"] >= crossing]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": group["dataset"].iloc[0],
                    "method_key": method_key,
                    "method": group["method"].iloc[0],
                    "threshold": float(group["threshold"].iloc[0]),
                    "crossing_percent": float(crossing),
                    "first_epoch": np.nan if hit.empty else int(hit["epoch"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def plot_memorization_onset_exposure(
    *,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    threshold: float = 0.5,
    n_boot: int = 500,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> dict[str, pd.DataFrame]:
    dataset_keys = list(dataset_keys or MAIN_DATASETS)
    method_keys = list(method_keys or PAPER_METHODS)
    onset = build_onset_table(dataset_keys=dataset_keys, method_keys=method_keys, threshold=threshold)
    crossing = build_onset_crossing_table(onset)
    exposure, paired = rpu.compute_sample_level_temporal_exposure(
        dataset_keys=dataset_keys,
        method_keys=method_keys,
        thresholds=(threshold,),
        n_boot=n_boot,
    )
    save_table(onset, f"memorization_onset_threshold_{threshold_slug(threshold)}", output_dir=output_dir)
    save_table(crossing, f"memorization_onset_crossings_threshold_{threshold_slug(threshold)}", output_dir=output_dir)
    save_table(exposure, f"memorization_exposure_threshold_{threshold_slug(threshold)}", output_dir=output_dir)
    save_table(paired, f"memorization_exposure_paired_threshold_{threshold_slug(threshold)}", output_dir=output_dir)

    fig, axes = plt.subplots(
        2,
        len(dataset_keys),
        figsize=(3.55 * len(dataset_keys), 4.25),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.20, 0.90], "hspace": 0.46, "wspace": 0.26},
    )

    for col_idx, dataset_key in enumerate(dataset_keys):
        bundle = rpu.report_bundle(dataset_key)
        dataset_onset = onset.loc[onset["dataset_key"] == dataset_key]
        ax = axes[0, col_idx]
        for method_key in rpu.section_method_keys(bundle, method_keys):
            sub = dataset_onset.loc[dataset_onset["method_key"] == method_key].sort_values("epoch")
            ax.plot(
                sub["epoch"],
                sub["percent_above_threshold"],
                color=METHOD_COLORS[method_key],
                linewidth=1.9,
                marker="o",
                markersize=3.0,
                label=rpu.METHOD_LABELS[method_key],
            )
        ax.set_title(bundle["cfg"]["label"])
        ax.set_xlabel("Epoch")
        if col_idx == 0:
            ax.set_ylabel(f"Samples with mem. >= {threshold:.1f} (%)")
        ax.grid(True, axis="y")

        axb = axes[1, col_idx]
        dataset_exposure = exposure.loc[exposure["dataset_key"] == dataset_key]
        active_methods = rpu.section_method_keys(bundle, method_keys)
        values = []
        for method_key in active_methods:
            row = dataset_exposure.loc[dataset_exposure["method_key"] == method_key]
            values.append(float(row["mean_exposure"].iloc[0]) if not row.empty else np.nan)
        x = np.arange(len(active_methods), dtype=float)
        axb.bar(
            x,
            values,
            width=0.66,
            color=[METHOD_COLORS[method_key] for method_key in active_methods],
            alpha=0.93,
        )
        ymax = max(float(np.nanmax(values)), 1e-8)
        for xpos, value in zip(x, values):
            axb.text(xpos, value + 0.025 * ymax, f"{value:.4f}", ha="center", va="bottom", fontsize=6.8)
        axb.set_xticks(x)
        axb.set_xticklabels([rpu.METHOD_LABELS[method_key] for method_key in active_methods], rotation=22, ha="right")
        axb.set_ylim(0.0, ymax * 1.24)
        if col_idx == 0:
            axb.set_ylabel(f"Excess exposure >= {threshold:.1f}")
        axb.grid(True, axis="y")

    handles = [
        mpl.lines.Line2D([0], [0], color=METHOD_COLORS[method_key], marker="o", linewidth=1.9, label=rpu.METHOD_LABELS[method_key])
        for method_key in PAPER_METHODS
        if method_key in method_keys
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985), ncol=len(handles))
    fig.suptitle(
        f"Onset and exposure at memorization threshold {threshold:.1f}",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=9.5,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.83, bottom=0.16, hspace=0.55, wspace=0.26)
    paths = save_figure(
        fig,
        f"memorization_onset_exposure_threshold_{threshold_slug(threshold)}",
        output_dir=output_dir,
        subdir=subdir,
    )
    plt.show()
    plt.close(fig)
    return {"onset": onset, "crossing": crossing, "exposure": exposure, "paired": paired, "figure_paths": pd.DataFrame({"path": [str(p) for p in paths]})}


def build_difficulty_exposure_summary(
    *,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    thresholds: tuple[float, ...] = (0.0,),
    n_bins: int = 5,
    anchor_method_key: str = "ib",
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or MAIN_DATASETS)
    method_keys = list(method_keys or PAPER_METHODS)
    rows = []

    for dataset_key in dataset_keys:
        bundle = rpu.report_bundle(dataset_key)
        active_methods = rpu.section_method_keys(bundle, method_keys)
        anchor_epoch = int(bundle["cfg"]["end_epoch"])
        anchor, pruning_col, pruning_label = build_pruning_rank_anchor(
            bundle,
            anchor_epoch=anchor_epoch,
            n_bins=n_bins,
            anchor_method_key=anchor_method_key,
        )
        common_epochs = None
        for method_key in active_methods:
            epochs = set(int(epoch) for epoch in bundle["mem_long"][method_key]["epoch"].unique())
            common_epochs = epochs if common_epochs is None else common_epochs & epochs
        if not common_epochs:
            continue
        epoch_values = np.array(sorted(common_epochs), dtype=np.float64)
        epoch_span = max(float(epoch_values[-1] - epoch_values[0]), 1.0)

        for method_key in active_methods:
            frame = bundle["mem_long"][method_key]
            sample_ids = None
            epoch_scores = []
            for epoch in epoch_values.astype(int):
                sub = frame.loc[frame["epoch"] == epoch, ["sample_id", "memorization"]].sort_values("sample_id")
                current_ids = sub["sample_id"].to_numpy()
                if sample_ids is None:
                    sample_ids = current_ids
                elif not np.array_equal(sample_ids, current_ids):
                    raise ValueError(f"Sample-id order mismatch for {dataset_key} / {method_key}")
                epoch_scores.append(rpu.cap_memorization(sub["memorization"].to_numpy(dtype=np.float64)))
            score_matrix = np.vstack(epoch_scores)

            for threshold in thresholds:
                if threshold <= 0.0:
                    transformed = score_matrix
                    metric = "all_score_exposure"
                    metric_label = "All-score exposure"
                else:
                    transformed = np.maximum(score_matrix - threshold, 0.0) / max(1.0 - threshold, 1e-12)
                    metric = f"excess_exposure_ge_{threshold:.1f}"
                    metric_label = f"Excess exposure >= {threshold:.1f}"
                exposure = TRAPZ(transformed, epoch_values, axis=0) / epoch_span
                per_sample = pd.DataFrame({"sample_id": sample_ids, "exposure": exposure})
                merged = anchor[["sample_id", "bin_num", "anchor_pruning_rate"]].merge(per_sample, on="sample_id", how="inner")
                grouped = merged.groupby("bin_num", sort=True)
                for bin_num, group in grouped:
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": bundle["cfg"]["label"],
                            "method_key": method_key,
                            "method": rpu.METHOD_LABELS[method_key],
                            "metric": metric,
                            "metric_label": metric_label,
                            "threshold": float(threshold),
                            "bin_num": int(bin_num),
                            "n": int(len(group)),
                            "mean_exposure": float(group["exposure"].mean()),
                            "sem_exposure": float(group["exposure"].std(ddof=1) / np.sqrt(max(len(group), 1))),
                            "anchor_epoch": anchor_epoch,
                            "anchor_method_key": anchor_method_key,
                            "anchor_method": rpu.METHOD_LABELS[anchor_method_key],
                            "anchor_pruning_metric": pruning_col,
                            "anchor_pruning_label": pruning_label,
                            "anchor_bin_rule": f"{n_bins} equal-count pruning-rate rank bins",
                            "anchor_pruning_mean": float(group["anchor_pruning_rate"].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def plot_difficulty_exposure(
    *,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    thresholds: tuple[float, ...] = (0.0,),
    n_bins: int = 5,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> dict[str, pd.DataFrame]:
    dataset_keys = list(dataset_keys or MAIN_DATASETS)
    method_keys = list(method_keys or PAPER_METHODS)
    table = build_difficulty_exposure_summary(
        dataset_keys=dataset_keys,
        method_keys=method_keys,
        thresholds=thresholds,
        n_bins=n_bins,
    )
    save_table(table, "difficulty_bin_temporal_exposure", output_dir=output_dir)

    metric_order = list(dict.fromkeys(table["metric"].tolist()))
    fig, axes = plt.subplots(
        len(metric_order),
        len(dataset_keys),
        figsize=(3.55 * len(dataset_keys), 2.25 * len(metric_order) + 0.15),
        squeeze=False,
        sharex=True,
    )
    savings_rows = []

    for row_idx, metric in enumerate(metric_order):
        metric_sub = table.loc[table["metric"] == metric]
        metric_label = metric_sub["metric_label"].iloc[0]
        for col_idx, dataset_key in enumerate(dataset_keys):
            ax = axes[row_idx, col_idx]
            dataset_sub = metric_sub.loc[metric_sub["dataset_key"] == dataset_key]
            if dataset_sub.empty:
                ax.set_visible(False)
                continue
            ax.axvspan(n_bins - 0.5, n_bins + 0.5, color="#FEE2E2", alpha=0.42, linewidth=0)
            for method_key in rpu.section_method_keys(rpu.report_bundle(dataset_key), method_keys):
                sub = dataset_sub.loc[dataset_sub["method_key"] == method_key].sort_values("bin_num")
                ax.plot(
                    sub["bin_num"],
                    sub["mean_exposure"],
                    color=METHOD_COLORS[method_key],
                    linewidth=1.9,
                    marker="o",
                    markersize=3.7,
                    label=rpu.METHOD_LABELS[method_key],
                )

            pivot = dataset_sub.pivot_table(index="bin_num", columns="method_key", values="mean_exposure")
            if {"std", "ib", "ib_prw"}.issubset(pivot.columns):
                hard_bin = int(pivot.index.max())
                std_minus_ib = float(pivot.loc[hard_bin, "std"] - pivot.loc[hard_bin, "ib"])
                ib_minus_wl = float(pivot.loc[hard_bin, "ib"] - pivot.loc[hard_bin, "ib_prw"])
                std_minus_wl = float(pivot.loc[hard_bin, "std"] - pivot.loc[hard_bin, "ib_prw"])
                savings_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": dataset_sub["dataset"].iloc[0],
                        "metric": metric,
                        "metric_label": metric_label,
                        "hard_bin": hard_bin,
                        "standard_minus_infobatch": std_minus_ib,
                        "infobatch_minus_ib_wl": ib_minus_wl,
                        "standard_minus_ib_wl": std_minus_wl,
                    }
                )
                ax.text(
                    0.02,
                    0.96,
                    f"Hard bin savings\nStd-IB-WL {std_minus_wl:.4f}\nIB-IB-WL {ib_minus_wl:.4f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.3,
                    color="#475569",
                    bbox={"boxstyle": "square,pad=0.18", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
                )

            if row_idx == 0:
                ax.set_title(dataset_sub["dataset"].iloc[0])
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            if row_idx == len(metric_order) - 1:
                ax.set_xlabel("Easy/high pruning -> hard/low pruning")
            ax.set_xticks(range(1, n_bins + 1))
            ax.set_xticklabels(["1\nEasy", *[str(i) for i in range(2, n_bins)], f"{n_bins}\nHard"])
            ax.grid(True, axis="y")

    handles = [
        mpl.lines.Line2D([0], [0], color=METHOD_COLORS[method_key], marker="o", linewidth=1.9, label=rpu.METHOD_LABELS[method_key])
        for method_key in method_keys
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985), ncol=len(handles))
    fig.suptitle(
        "Memorization exposure by pruning difficulty",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=9.5,
        fontweight="semibold",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.83, bottom=0.14, hspace=0.42, wspace=0.22)
    paths = save_figure(fig, "difficulty_bin_memorization_exposure", output_dir=output_dir, subdir=subdir)
    plt.show()
    plt.close(fig)

    savings = pd.DataFrame(savings_rows)
    save_table(savings, "difficulty_bin_hard_savings", output_dir=output_dir)
    return {"summary": table, "hard_savings": savings, "figure_paths": pd.DataFrame({"path": [str(p) for p in paths]})}


def pruning_interval_tick_labels(edges: np.ndarray, n_bins: int) -> list[str]:
    labels = []
    for bin_num in range(1, n_bins + 1):
        low_idx = n_bins - bin_num
        high_idx = low_idx + 1
        low = float(edges[low_idx])
        high = float(edges[high_idx])
        if bin_num == 1:
            labels.append(f"{bin_num}\nEasy\n[{low:.2f},{high:.2f}]")
        elif bin_num == n_bins:
            labels.append(f"{bin_num}\nHard\n[{low:.2f},{high:.2f}]")
        else:
            labels.append(f"{bin_num}\n[{low:.2f},{high:.2f}]")
    return labels


def build_difficulty_mean_memorization_summary(
    *,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    n_bins: int = 5,
    anchor_method_key: str = "ib",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    dataset_keys = list(dataset_keys or MAIN_DATASETS)
    method_keys = list(method_keys or PAPER_METHODS)
    rows = []
    edges_by_dataset = {}
    for dataset_key in dataset_keys:
        bundle = rpu.report_bundle(dataset_key)
        epoch = int(bundle["cfg"]["end_epoch"])
        anchor, edges, pruning_col, pruning_label = rpu.build_infobatch_pruning_interval_anchor(
            bundle,
            anchor_epoch=epoch,
            n_bins=n_bins,
            anchor_method_key=anchor_method_key,
        )
        edges_by_dataset[dataset_key] = edges.copy()
        for method_key in rpu.section_method_keys(bundle, method_keys):
            mem = rpu.get_method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]].copy()
            mem["memorization"] = rpu.cap_memorization(mem["memorization"].to_numpy(dtype=np.float64))
            merged = anchor[["sample_id", "bin_num", "anchor_pruning_rate"]].merge(mem, on="sample_id", how="inner")
            for bin_num, group in merged.groupby("bin_num", sort=True):
                low_idx = n_bins - int(bin_num)
                high_idx = low_idx + 1
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": bundle["cfg"]["label"],
                        "method_key": method_key,
                        "method": rpu.METHOD_LABELS[method_key],
                        "epoch": epoch,
                        "bin_num": int(bin_num),
                        "n": int(len(group)),
                        "mean_memorization": float(group["memorization"].mean()),
                        "sem_memorization": float(group["memorization"].std(ddof=1) / np.sqrt(max(len(group), 1))),
                        "anchor_method_key": anchor_method_key,
                        "anchor_method": rpu.METHOD_LABELS[anchor_method_key],
                        "anchor_pruning_metric": pruning_col,
                        "anchor_pruning_label": pruning_label,
                        "pruning_interval_low": float(edges[low_idx]),
                        "pruning_interval_high": float(edges[high_idx]),
                    }
                )
    return pd.DataFrame(rows), edges_by_dataset


def plot_difficulty_mean_memorization(
    *,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
    n_bins: int = 5,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> dict[str, pd.DataFrame]:
    dataset_keys = list(dataset_keys or MAIN_DATASETS)
    method_keys = list(method_keys or PAPER_METHODS)
    table, edges_by_dataset = build_difficulty_mean_memorization_summary(
        dataset_keys=dataset_keys,
        method_keys=method_keys,
        n_bins=n_bins,
    )
    save_table(table, "difficulty_bin_mean_memorization", output_dir=output_dir)

    fig, axes = plt.subplots(1, len(dataset_keys), figsize=(3.85 * len(dataset_keys), 2.75), squeeze=False)
    for ax, dataset_key in zip(axes[0], dataset_keys):
        sub_all = table.loc[table["dataset_key"] == dataset_key]
        ax.axvspan(n_bins - 0.5, n_bins + 0.5, color="#FEE2E2", alpha=0.38, linewidth=0)
        for method_key in rpu.section_method_keys(rpu.report_bundle(dataset_key), method_keys):
            sub = sub_all.loc[sub_all["method_key"] == method_key].sort_values("bin_num")
            ax.plot(
                sub["bin_num"],
                sub["mean_memorization"],
                color=METHOD_COLORS[method_key],
                linewidth=1.15,
                marker="o",
                markersize=2.4,
                label=rpu.METHOD_LABELS[method_key],
            )
        ax.set_title(sub_all["dataset"].iloc[0], fontweight="normal")
        ax.set_xlabel("InfoBatch pruning-rate interval")
        ax.set_ylabel("Mean memorization")
        ax.set_xlim(0.75, n_bins + 0.95)
        ax.set_xticks(range(1, n_bins + 1))
        ax.set_xticklabels(pruning_interval_tick_labels(edges_by_dataset[dataset_key], n_bins), fontsize=6.2)
        ax.grid(True, axis="y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=len(handles))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.31, wspace=0.26)
    paths = save_figure(fig, "difficulty_bin_mean_memorization", output_dir=output_dir, subdir=subdir)
    plt.show()
    plt.close(fig)
    table.attrs["figure_paths"] = [str(path) for path in paths]
    return {"summary": table, "figure_paths": pd.DataFrame({"path": [str(p) for p in paths]})}


def _difficulty_interval_by_label(difficulty_meta: dict) -> dict[str, tuple[float, float]]:
    edges = np.asarray(difficulty_meta.get("edges", []), dtype=np.float64)
    if edges.size < 4:
        return {}
    return {
        "hard": (float(edges[0]), float(edges[1])),
        "medium": (float(edges[1]), float(edges[2])),
        "easy": (float(edges[2]), float(edges[3])),
    }


def build_cifar10_memorization_bin_composition(
    *,
    method_keys: list[str] | None = None,
    epoch: int | None = None,
    anchor_method_key: str = "ib",
) -> tuple[pd.DataFrame, dict]:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    epoch = int(epoch if epoch is not None else bundle["cfg"]["end_epoch"])
    method_keys = list(method_keys or PAPER_METHODS)
    active_methods = rpu.section_method_keys(bundle, method_keys)

    difficulty_frame, difficulty_meta = rpu.section_build_difficulty_frame(
        bundle,
        "pruning_rate_tertiles",
        pruning_epoch_by_dataset={dataset_key: epoch},
        pruning_method_key=anchor_method_key,
    )
    labels = list(difficulty_meta["labels"])
    interval_by_label = _difficulty_interval_by_label(difficulty_meta)

    rows = []
    for method_key in active_methods:
        mem = rpu.get_method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]].copy()
        mem["memorization"] = rpu.cap_memorization(mem["memorization"].to_numpy(dtype=np.float64))
        frame = rpu.merge_on_sample_id(
            mem,
            difficulty_frame[["sample_id", "difficulty_label", "difficulty_value"]],
            context=f"paper_cifar10_mem_bin_composition::{method_key}::epoch_{epoch}",
        )
        in_bin = (frame["memorization"] > rpu.MEMORIZED_THRESHOLD) & (frame["memorization"] <= 1.0)
        frame = frame.loc[in_bin].copy()
        frame["memorization_bin_id"] = rpu.memorization_bin_index(
            frame["memorization"].to_numpy(dtype=np.float64)
        )
        frame = frame.loc[
            (frame["memorization_bin_id"] >= 0) & (frame["memorization_bin_id"] < len(rpu.MEM_BIN_LABELS))
        ].copy()

        for bin_id, bin_label in enumerate(rpu.MEM_BIN_LABELS):
            bin_group = frame.loc[frame["memorization_bin_id"] == bin_id]
            total_in_bin = int(len(bin_group))
            mean_anchor_pruning = (
                float(bin_group["difficulty_value"].mean()) if total_in_bin else np.nan
            )
            sem_anchor_pruning = (
                float(bin_group["difficulty_value"].std(ddof=1) / np.sqrt(total_in_bin))
                if total_in_bin > 1
                else np.nan
            )
            for difficulty_label in labels:
                diff_group = bin_group.loc[bin_group["difficulty_label"] == difficulty_label]
                interval = interval_by_label.get(difficulty_label, (np.nan, np.nan))
                count = int(len(diff_group))
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": bundle["cfg"]["label"],
                        "method_key": method_key,
                        "method": rpu.METHOD_LABELS[method_key],
                        "epoch": epoch,
                        "memorization_bin_id": int(bin_id),
                        "memorization_bin": bin_label,
                        "difficulty_label": difficulty_label,
                        "difficulty_interval_low": float(interval[0]),
                        "difficulty_interval_high": float(interval[1]),
                        "count": count,
                        "n_in_method_bin": total_in_bin,
                        "pct_of_method_bin": 100.0 * count / total_in_bin if total_in_bin else np.nan,
                        "anchor_pruning_rate_mean": mean_anchor_pruning,
                        "anchor_pruning_rate_sem": sem_anchor_pruning,
                        "anchor_method_key": anchor_method_key,
                        "anchor_method": rpu.METHOD_LABELS[anchor_method_key],
                        "anchor_epoch": epoch,
                    }
                )
    table = pd.DataFrame(rows)
    return table, difficulty_meta


def plot_cifar10_memorization_bin_composition(
    *,
    method_keys: list[str] | None = None,
    epoch: int | None = None,
    anchor_method_key: str = "ib",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> pd.DataFrame:
    method_keys = list(method_keys or PAPER_METHODS)
    table, difficulty_meta = build_cifar10_memorization_bin_composition(
        method_keys=method_keys,
        epoch=epoch,
        anchor_method_key=anchor_method_key,
    )
    save_table(table, "cifar10_memorization_bin_composition", output_dir=output_dir)

    active_methods = [key for key in method_keys if key in table["method_key"].unique()]
    labels = list(difficulty_meta["labels"])
    interval_by_label = _difficulty_interval_by_label(difficulty_meta)
    difficulty_hatches = {"easy": "...", "medium": "xx", "hard": "///"}
    x = np.arange(len(rpu.MEM_BIN_LABELS), dtype=float)
    width = 0.78 / max(len(active_methods), 1)

    fig, ax = plt.subplots(figsize=(7.25, 3.18))
    ax2 = ax.twinx()
    for idx, method_key in enumerate(active_methods):
        method_table = table.loc[table["method_key"] == method_key]
        positions = x + (idx - (len(active_methods) - 1) / 2.0) * width
        bottoms = np.zeros(len(rpu.MEM_BIN_LABELS), dtype=np.float64)
        for difficulty_label in labels:
            counts = np.array(
                [
                    method_table.loc[
                        (method_table["memorization_bin_id"] == bin_id)
                        & (method_table["difficulty_label"] == difficulty_label),
                        "count",
                    ].sum()
                    for bin_id in range(len(rpu.MEM_BIN_LABELS))
                ],
                dtype=np.float64,
            )
            ax.bar(
                positions,
                counts,
                bottom=bottoms,
                width=width,
                color=rpu.section_shade_method_color(METHOD_COLORS[method_key], difficulty_label),
                edgecolor="#111827",
                linewidth=0.38,
                hatch=difficulty_hatches[difficulty_label],
                alpha=0.94,
            )
            bottoms += counts

        pruning_means = []
        for bin_id in range(len(rpu.MEM_BIN_LABELS)):
            row = method_table.loc[method_table["memorization_bin_id"] == bin_id]
            pruning_means.append(
                float(row["anchor_pruning_rate_mean"].iloc[0]) if not row.empty else np.nan
            )
        ax2.plot(
            positions,
            pruning_means,
            color=METHOD_COLORS[method_key],
            linewidth=1.05,
            marker="o",
            markersize=2.2,
            alpha=0.94,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(rpu.MEM_BIN_LABELS, rotation=38, ha="right")
    ax.set_xlabel("Final memorization score bin")
    ax.set_ylabel("Samples")
    ax.set_title("CIFAR-10 final epoch", loc="left", fontweight="normal", pad=4)
    ax.grid(True, axis="y", alpha=0.28)

    ax2.set_ylabel("InfoBatch pruning rate")
    finite_pruning = table["anchor_pruning_rate_mean"].to_numpy(dtype=np.float64)
    finite_pruning = finite_pruning[np.isfinite(finite_pruning)]
    top = max(float(np.nanmax(finite_pruning)) if finite_pruning.size else 0.1, 0.1)
    ax2.set_ylim(0.0, min(1.0, top * 1.18))
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#CBD5E1")

    method_handles = [
        mpatches.Patch(
            facecolor=METHOD_COLORS[method_key],
            edgecolor="#111827",
            label=rpu.METHOD_LABELS[method_key],
        )
        for method_key in active_methods
    ]
    difficulty_handles = []
    for difficulty_label in labels:
        low, high = interval_by_label.get(difficulty_label, (np.nan, np.nan))
        suffix = f" [{low:.2f},{high:.2f}]" if np.isfinite(low) and np.isfinite(high) else ""
        difficulty_handles.append(
            mpatches.Patch(
                facecolor="white",
                edgecolor="#111827",
                hatch=difficulty_hatches[difficulty_label],
                label=f"{difficulty_label.title()}{suffix}",
            )
        )
    pruning_handle = mpl.lines.Line2D(
        [0],
        [0],
        color="#111827",
        linewidth=1.05,
        marker="o",
        markersize=2.4,
        label="Mean pruning rate",
    )
    legend1 = ax.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(1.16, 1.00),
        title="Method",
        fontsize=7.0,
        title_fontsize=7.0,
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)
    legend2 = ax.legend(
        handles=difficulty_handles,
        loc="upper left",
        bbox_to_anchor=(1.16, 0.64),
        title="Pruning interval",
        fontsize=6.8,
        title_fontsize=7.0,
        borderaxespad=0.0,
    )
    ax.add_artist(legend2)
    ax2.legend(
        handles=[pruning_handle],
        loc="upper left",
        bbox_to_anchor=(1.16, 0.23),
        fontsize=7.0,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.085, right=0.68, top=0.92, bottom=0.25)
    paths = save_figure(
        fig,
        "cifar10_memorization_bin_composition",
        output_dir=output_dir,
        subdir=subdir,
    )
    plt.show()
    plt.close(fig)
    table.attrs["figure_paths"] = [str(path) for path in paths]
    return table


def plot_cifar10_difficulty_bar(
    *,
    method_keys: list[str] | None = None,
    n_bins: int = 3,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> pd.DataFrame:
    if int(n_bins) != 3:
        raise ValueError("The CIFAR-10 memorization-bin composition plot uses three pruning intervals.")
    return plot_cifar10_memorization_bin_composition(
        method_keys=method_keys,
        output_dir=output_dir,
        subdir=subdir,
    )


def build_hard_pruning_bin_trajectory(
    *,
    dataset_key: str = "cifar10",
    method_keys: list[str] | None = None,
    n_bins: int = 5,
    anchor_method_key: str = "ib",
) -> pd.DataFrame:
    method_keys = list(method_keys or PAPER_METHODS)
    bundle = rpu.report_bundle(dataset_key)
    anchor_epoch = int(bundle["cfg"]["end_epoch"])
    anchor, edges, pruning_col, pruning_label = rpu.build_infobatch_pruning_interval_anchor(
        bundle,
        anchor_epoch=anchor_epoch,
        n_bins=n_bins,
        anchor_method_key=anchor_method_key,
    )
    hard = anchor.loc[anchor["bin_num"] == n_bins, ["sample_id", "anchor_pruning_rate"]].copy()
    rows = []
    for method_key in rpu.section_method_keys(bundle, method_keys):
        frame = bundle["mem_long"][method_key]
        for epoch, chunk in frame.groupby("epoch", sort=True):
            merged = hard.merge(chunk[["sample_id", "memorization"]], on="sample_id", how="inner")
            scores = rpu.cap_memorization(merged["memorization"].to_numpy(dtype=np.float64))
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": rpu.METHOD_LABELS[method_key],
                    "epoch": int(epoch),
                    "mean_memorization": float(np.mean(scores)),
                    "n": int(len(scores)),
                    "anchor_method_key": anchor_method_key,
                    "anchor_method": rpu.METHOD_LABELS[anchor_method_key],
                    "anchor_epoch": anchor_epoch,
                    "anchor_pruning_metric": pruning_col,
                    "anchor_pruning_label": pruning_label,
                    "anchor_bin_rule": f"lowest-pruning interval out of {n_bins} equal-width pruning intervals",
                    "anchor_pruning_low": float(edges[0]),
                    "anchor_pruning_high": float(edges[1]),
                }
            )
    return pd.DataFrame(rows)


def plot_hard_pruning_bin_trajectory(
    *,
    dataset_key: str = "cifar10",
    method_keys: list[str] | None = None,
    n_bins: int = 5,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> pd.DataFrame:
    method_keys = list(method_keys or PAPER_METHODS)
    table = build_hard_pruning_bin_trajectory(dataset_key=dataset_key, method_keys=method_keys, n_bins=n_bins)
    save_table(table, f"{dataset_key}_hard_pruning_bin_trajectory", output_dir=output_dir)
    dataset_label = table["dataset"].iloc[0]
    interval_low = float(table["anchor_pruning_low"].iloc[0])
    interval_high = float(table["anchor_pruning_high"].iloc[0])

    fig, ax = plt.subplots(figsize=(3.9, 2.45))
    for method_key in method_keys:
        sub = table.loc[table["method_key"] == method_key].sort_values("epoch")
        ax.plot(
            sub["epoch"],
            sub["mean_memorization"],
            color=METHOD_COLORS[method_key],
            linewidth=1.15,
            marker="o",
            markersize=2.4,
            label=rpu.METHOD_LABELS[method_key],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean memorization")
    ax.text(
        0.03,
        0.97,
        f"{dataset_label} hard subset: pruning in [{interval_low:.2f}, {interval_high:.2f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color="#475569",
    )
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(left=0.16, right=0.78, top=0.94, bottom=0.22)
    paths = save_figure(fig, f"{dataset_key}_hard_pruning_bin_trajectory", output_dir=output_dir, subdir=subdir)
    plt.show()
    plt.close(fig)
    table.attrs["figure_paths"] = [str(path) for path in paths]
    return table


def plot_cifar10_pruning_movement(
    *,
    method_keys: tuple[str, ...] = ("ib", "ib_prw"),
    epoch: int = 99,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    subdir: str = "main",
) -> pd.DataFrame:
    dataset_key = "cifar10"
    bundle = rpu.report_bundle(dataset_key)
    fig, axes = plt.subplots(1, len(method_keys), figsize=(7.2, 3.15), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    rows = []

    for ax, method_key in zip(axes, method_keys):
        frame = rpu.section_build_method_frame(bundle, epoch, method_key, include_rmia=False)
        movement = rpu.movement_frame(bundle, epoch, method_key)[["sample_id", "movement", "delta_bins"]]
        frame = rpu.strict_sample_merge(
            frame,
            movement,
            context=f"paper_movement::{dataset_key}::{method_key}::epoch_{epoch}",
        )
        pruning_col, pruning_label = rpu.resolve_pruning_metric(bundle, method_key, frame)
        if pruning_col is None:
            raise KeyError(f"No pruning metric found for {dataset_key} / {method_key}")
        frame = frame.loc[frame[pruning_col].notna()].copy()
        x = rpu.cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
        y = frame[pruning_col].to_numpy(dtype=np.float64)

        draw_order = ["up_3p", "up_2", "up_1", "same", "down_1", "down_2", "down_3p"]
        alpha_by_prefix = {"up": 0.18, "same": 0.16, "down": 0.36}
        zorder_by_prefix = {"up": 1, "same": 2, "down": 4}
        for category in draw_order:
            style = rpu.MOVEMENT_STYLE[category]
            sub = frame.loc[frame["movement"] == category]
            if sub.empty:
                continue
            prefix = "down" if category.startswith("down") else ("up" if category.startswith("up") else "same")
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": rpu.METHOD_LABELS[method_key],
                    "movement": category,
                    "movement_label": rpu.movement_display_name(category),
                    "n": int(len(sub)),
                    "pct": 100.0 * len(sub) / max(len(frame), 1),
                }
            )
            ax.scatter(
                rpu.cap_memorization(sub["memorization"].to_numpy(dtype=np.float64)),
                sub[pruning_col].to_numpy(dtype=np.float64),
                s=8,
                alpha=alpha_by_prefix[prefix],
                color=style["color"],
                marker=style["marker"],
                edgecolors="none",
                rasterized=True,
                zorder=zorder_by_prefix[prefix],
            )

        if np.unique(np.round(x, 8)).size >= 2:
            coeff = np.polyfit(x, y, deg=1)
            x_line = np.linspace(max(0.0, float(np.nanmin(x))), 1.0, 160)
            y_line = coeff[0] * x_line + coeff[1]
            corr = float(np.corrcoef(x, y)[0, 1])
            ax.plot(x_line, y_line, color="#111827", linewidth=0.85, linestyle="--", zorder=5)
            ax.text(
                0.04,
                0.94,
                f"r = {corr:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.8,
                color="#111827",
                bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.88},
            )
        ax.set_title(rpu.METHOD_LABELS[method_key])
        ax.set_xlabel("Final memorization score")
        ax.grid(True, alpha=0.28)
        if ax is axes[0]:
            ax.set_ylabel(pruning_label)

    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(-0.02, 1.02)
    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="none",
            markerfacecolor=style["color"],
            markeredgecolor="none",
            markersize=5.5,
            label=rpu.movement_display_name(category),
        )
        for category, style in rpu.MOVEMENT_STYLE.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.52, 0.02), ncol=4)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.30, wspace=0.20)
    paths = save_figure(fig, "cifar10_pruning_vs_memorization_movement_bins", output_dir=output_dir, subdir=subdir)
    plt.show()
    plt.close(fig)
    counts = pd.DataFrame(rows)
    save_table(counts, "cifar10_movement_bin_counts", output_dir=output_dir)
    counts.attrs["figure_paths"] = [str(path) for path in paths]
    return counts


def write_manifest(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_dir = Path(output_dir)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
            }
            for path in files
        ]
    )
    save_table(manifest, "manifest", output_dir=output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(orient="records"), indent=2), encoding="utf-8")
    return manifest
