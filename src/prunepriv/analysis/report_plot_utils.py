from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

from prunepriv.paths import results_dir as _results_dir

ROOT = Path(os.environ.get("PRUNEPRIV_ANALYSIS_ROOT", _results_dir("analysis"))).resolve()
ANALYSIS_ROOT = ROOT
CACHE_ROOT = ANALYSIS_ROOT / "_cache"
FIGURE_ROOT = ANALYSIS_ROOT / "_figures"
MPLCONFIGDIR = CACHE_ROOT / "matplotlib"
IPYTHONDIR = CACHE_ROOT / "ipython"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
IPYTHONDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("IPYTHONDIR", str(IPYTHONDIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import Markdown, display
from matplotlib.colors import LinearSegmentedColormap

try:
    import wandb
except Exception:
    wandb = None

DATASET_ORDER = ["cifar10", "cifar100"]
ACTIVE_DATASETS = DATASET_ORDER[:]  # Set to a subset if you want a lighter run.
ACTIVE_METHODS = ["std", "ib", "ib_prw"]

MEMORIZED_THRESHOLD = 0.1
HARDNESS_EPOCH = 24
HARDNESS_REFERENCE_METHOD = "std"
ENTROPY_SPLIT = 1.0
RMIA_MATRIX_MEM_SPLIT = 0.5
RMIA_MATRIX_ENTROPY_MAX_FRACTION = 0.5
RMIA_MATRIX_DETECTED_RATE_THRESHOLD = 0.0
RMIA_MATRIX_THRESHOLDS = (0.5, 2.0)  # (memorization threshold, entropy threshold) for the 2 x 4 view.
DETECTION_RATE_BIN_EDGES = np.arange(0.0, 101.0, 10.0)
STD_IB_MATRIX_THRESHOLDS = (0.3, 0.7)
STD_IB_QUADRANT_THRESHOLDS = (0.5, 0.5)  # (x threshold, y threshold)
COMPARISON_PAIRS = [("std", "ib"), ("std", "ib_prw")]
QUADRANT_COMPARISON_PAIRS = [("std", "ib"), ("std", "ib_prw")]

PREFER_WANDB = False
SAVE_FIGURES = False
SAVE_VECTOR = False
VECTOR_FORMAT = "pdf"
FIG_DPI = 250
RANDOM_SEED = 3407

METHOD_ORDER = ["std", "ib", "ib_prw"]
METHOD_LABELS = {
    "std": "Standard",
    "ib": "InfoBatch",
    "ib_prw": "IB-WL",
}
METHOD_COLORS = {
    "std": "#2D7FF9",
    "ib": "#FF6B35",
    "ib_prw": "#4C9A2A",
}
METHOD_HATCHES = {
    "easy": "...",
    "hard": "///",
}
HARDNESS_COLORS = {
    "easy": "#4C78A8",
    "hard": "#E45756",
}
DIFFICULTY_BIN_COLORS = {
    1: "#2D7FF9",
    2: "#06A77D",
    3: "#F2C14E",
    4: "#F58518",
    5: "#D7263D",
}

MOVEMENT_STYLE = {
    "down_3p": {"color": "#0B3C5D", "hatch": "////", "marker": "s"},
    "down_2": {"color": "#2F80ED", "hatch": "////", "marker": "^"},
    "down_1": {"color": "#9FD6FF", "hatch": "////", "marker": "o"},
    "same": {"color": "#8E8E8E", "hatch": "", "marker": "D"},
    "up_1": {"color": "#FFD0A8", "hatch": "\\\\", "marker": "o"},
    "up_2": {"color": "#FF8C42", "hatch": "\\\\", "marker": "v"},
    "up_3p": {"color": "#C62828", "hatch": "\\\\", "marker": "P"},
}

DATASETS = {
    "cifar10": {
        "label": "CIFAR-10",
        "n_classes": 10,
        "n_samples": 50_000,
        "subset_path": ANALYSIS_ROOT / "cifar10" / "subsets_cifar_400.pt",
        "memorization_epochs": [10, 18, 25, 50, 57, 73, 75, 87, 88, 99],
        "end_epoch": 99,
        "bin_epochs": [99],
        "movement_epochs": [99],
        "matrix_epochs": [99],
        "comparison_epochs": [99],
        "entropy_hist_epochs": [99],
        "loss_hist_epochs": [99],
        "matched_accuracy_pairs": [],
        "methods": {
            "std": {
                "label": METHOD_LABELS["std"],
                "color": METHOD_COLORS["std"],
                "memorization_dir": ANALYSIS_ROOT / "cifar10" / "standard" / "memorization_scores",
                "memorization_glob": "loo_scores_epoch_*.pt",
                "seed_avg_path": ANALYSIS_ROOT / "cifar10" / "standard" / "seed_avg.pt",
                "evolution_log_path": ANALYSIS_ROOT / "cifar10" / "standard" / "cifar_evolution_log_standard.pt",
                "rmia_dir": ANALYSIS_ROOT / "cifar10" / "standard",
                "wandb_run": None,
                "wandb_train_keys": ["avg/train_accuracy"],
                "wandb_test_keys": ["avg/test_accuracy"],
                "local_train_keys": ["avg/train_accuracy"],
                "local_test_keys": ["avg/test_accuracy"],
            },
            "ib": {
                "label": METHOD_LABELS["ib"],
                "color": METHOD_COLORS["ib"],
                "memorization_dir": ANALYSIS_ROOT / "cifar10" / "IB" / "memorization_scores",
                "memorization_glob": "loo_scores_epoch_*.pt",
                "seed_avg_path": ANALYSIS_ROOT / "cifar10" / "IB" / "seed_avg.pt",
                "evolution_log_path": ANALYSIS_ROOT / "cifar10" / "IB" / "cifar_evolution_log_ib.pt",
                "rmia_dir": ANALYSIS_ROOT / "cifar10" / "IB",
                "wandb_run": None,
                "wandb_train_keys": ["avg/train_accuracy"],
                "wandb_test_keys": ["avg/test_accuracy"],
                "local_train_keys": ["avg/train_accuracy"],
                "local_test_keys": ["avg/test_accuracy"],
            },
            "ib_prw": {
                "label": METHOD_LABELS["ib_prw"],
                "color": METHOD_COLORS["ib_prw"],
                "memorization_dir": ANALYSIS_ROOT / "cifar10" / "IB-WL" / "memorization_scores",
                "memorization_glob": "loo_scores_epoch_*.pt",
                "seed_avg_path": ANALYSIS_ROOT / "cifar10" / "IB-WL" / "seed_avg.pt",
                "evolution_log_path": ANALYSIS_ROOT / "cifar10" / "IB-WL" / "cifar_evolution_log_ib_prw.pt",
                "rmia_dir": ANALYSIS_ROOT / "cifar10" / "IB-WL",
                "wandb_run": None,
                "wandb_train_keys": ["avg/train_accuracy"],
                "wandb_test_keys": ["avg/test_accuracy"],
                "local_train_keys": ["avg/train_accuracy"],
                "local_test_keys": ["avg/test_accuracy"],
            },
        },
    },
    "cifar100": {
        "label": "CIFAR-100",
        "n_classes": 100,
        "n_samples": 50_000,
        "subset_path": ANALYSIS_ROOT / "cifar100" / "subsets_cifar_400.pt",
        "memorization_epochs": [20, 28, 39, 66, 80, 88, 91, 99],
        "end_epoch": 99,
        "bin_epochs": [99],
        "movement_epochs": [99],
        "matrix_epochs": [99],
        "comparison_epochs": [99],
        "entropy_hist_epochs": [99],
        "loss_hist_epochs": [99],
        "matched_accuracy_pairs": [],
        "methods": {
            "std": {
                "label": METHOD_LABELS["std"],
                "color": METHOD_COLORS["std"],
                "memorization_dir": ANALYSIS_ROOT / "cifar100" / "standard" / "memorization_scores",
                "memorization_glob": "loo_scores_epoch_*.pt",
                "seed_avg_path": ANALYSIS_ROOT / "cifar100" / "standard" / "seed_avg.pt",
                "evolution_log_path": ANALYSIS_ROOT / "cifar100" / "standard" / "cifar_evolution_log_standard.pt",
                "rmia_dir": ANALYSIS_ROOT / "cifar100" / "standard",
                "wandb_run": None,
                "wandb_train_keys": ["avg/train_accuracy"],
                "wandb_test_keys": ["avg/test_accuracy"],
                "local_train_keys": ["avg/train_accuracy"],
                "local_test_keys": ["avg/test_accuracy"],
            },
            "ib": {
                "label": METHOD_LABELS["ib"],
                "color": METHOD_COLORS["ib"],
                "memorization_dir": ANALYSIS_ROOT / "cifar100" / "IB" / "memorization_scores",
                "memorization_glob": "loo_scores_epoch_*.pt",
                "seed_avg_path": ANALYSIS_ROOT / "cifar100" / "IB" / "seed_avg.pt",
                "evolution_log_path": ANALYSIS_ROOT / "cifar100" / "IB" / "cifar_evolution_log_ib.pt",
                "rmia_dir": ANALYSIS_ROOT / "cifar100" / "IB",
                "wandb_run": None,
                "wandb_train_keys": ["avg/train_accuracy"],
                "wandb_test_keys": ["avg/test_accuracy"],
                "local_train_keys": ["avg/train_accuracy"],
                "local_test_keys": ["avg/test_accuracy"],
            },
            "ib_prw": {
                "label": METHOD_LABELS["ib_prw"],
                "color": METHOD_COLORS["ib_prw"],
                "memorization_dir": ANALYSIS_ROOT / "cifar100" / "IB-WL" / "memorization_scores",
                "memorization_glob": "loo_scores_epoch_*.pt",
                "seed_avg_path": ANALYSIS_ROOT / "cifar100" / "IB-WL" / "seed_avg.pt",
                "evolution_log_path": ANALYSIS_ROOT / "cifar100" / "IB-WL" / "cifar_evolution_log_ib_prw.pt",
                "rmia_dir": ANALYSIS_ROOT / "cifar100" / "IB-WL",
                "wandb_run": None,
                "wandb_train_keys": ["avg/train_accuracy"],
                "wandb_test_keys": ["avg/test_accuracy"],
                "local_train_keys": ["avg/train_accuracy"],
                "local_test_keys": ["avg/test_accuracy"],
            },
        },
    },
}

sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": FIG_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.titlelocation": "left",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 11,
    }
)
np.random.seed(RANDOM_SEED)
DETECTION_CMAP = LinearSegmentedColormap.from_list("grey_to_red", ["#D3D8DE", "#D95858", "#8C1D1D"])
MATRIX_CMAP = LinearSegmentedColormap.from_list("matrix_fill", ["#FFFFFF", "#EDF2F7", "#D9E2EC"])


_LIGHTWEIGHT_CACHE = {}
_EVOLUTION_CACHE = {}
_RAW_EVO_SCHEMA_CACHE = {}
OUTPUT_ROOT = FIGURE_ROOT


def ordered_methods(cfg: dict, requested=None) -> list[str]:
    available = [key for key in METHOD_ORDER if key in cfg["methods"]]
    if requested is None:
        return available
    requested = list(requested)
    return [key for key in METHOD_ORDER if key in requested and key in cfg["methods"]]


def dataset_output_dir(dataset_key: str) -> Path:
    path = OUTPUT_ROOT / dataset_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def movement_display_name(key: str) -> str:
    return {
        "down_3p": "Down 3+",
        "down_2": "Down 2",
        "down_1": "Down 1",
        "same": "Same",
        "up_1": "Up 1",
        "up_2": "Up 2",
        "up_3p": "Up 3+",
    }[key]


def cap_memorization(scores):
    return np.clip(np.asarray(scores, dtype=np.float64), a_min=None, a_max=1.0)


def detection_vmax(values, floor: float = 0.0) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        return floor
    return max(float(np.nanmax(values)), floor)


def figure_label_from_stem(stem: str) -> str:
    label = stem
    replacements = [
        ("cifar10", "CIFAR-10"),
        ("cifar100", "CIFAR-100"),
        ("tinyimagenet", "TinyImageNet"),
        ("ib_prw", "IB-WL"),
        ("ib", "InfoBatch"),
        ("std", "Standard"),
        ("rmia", "RMIA"),
        ("0p01pct", "0.01%"),
        ("0p1pct", "0.1%"),
        ("pct", "%"),
    ]
    for src, dst in replacements:
        label = label.replace(src, dst)
    return label.replace("_", " | ")


def save_figure(fig: plt.Figure, stem: str, dataset_key: str) -> None:
    if not SAVE_FIGURES:
        return
    out_dir = dataset_output_dir(dataset_key)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    if SAVE_VECTOR:
        fig.savefig(out_dir / f"{stem}.{VECTOR_FORMAT}", bbox_inches="tight")


def finish_figure(
    fig: plt.Figure,
    stem: str,
    dataset_key: str,
    figure_label: str | None = None,
    tight_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.96),
) -> None:
    fig.text(
        0.01,
        0.995,
        figure_label or figure_label_from_stem(stem),
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="semibold",
        color="#334155",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=tight_rect)
    save_figure(fig, stem, dataset_key)
    plt.show()
    plt.close(fig)


def threshold_label_edges(thresholds: tuple[float, ...]) -> tuple[list[str], list[str]]:
    thresholds = tuple(float(t) for t in thresholds)
    x_labels = []
    lower = None
    for threshold in thresholds:
        if lower is None:
            x_labels.append(f"<{threshold:.1f}")
        else:
            x_labels.append(f"{lower:.1f}-{threshold:.1f}")
        lower = threshold
    x_labels.append(f">={thresholds[-1]:.1f}")
    y_labels = list(reversed(x_labels))
    return x_labels, y_labels


def add_detection_colorbar(
    fig: plt.Figure,
    scatter,
    ax: plt.Axes,
    *,
    method_label: str,
    vmax: float,
):
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-8), clip=True)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=DETECTION_CMAP)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"{method_label} RMIA detection rate @ 0.01% FPR (%)")
    if vmax > 0:
        cbar.set_ticks(np.linspace(0.0, vmax, 4))
    return cbar


def scatter_detection_layered(
    ax: plt.Axes,
    x,
    y,
    detection_pct,
    *,
    vmax: float,
    s: float = 12,
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    detection_pct = np.asarray(detection_pct, dtype=np.float64)
    order = np.argsort(detection_pct, kind="stable")
    x = x[order]
    y = y[order]
    detection_pct = detection_pct[order]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-8), clip=True)
    colors = DETECTION_CMAP(norm(detection_pct))
    alpha_scale = detection_pct / max(vmax, 1e-8)
    colors[:, 3] = 0.18 + 0.82 * np.power(np.clip(alpha_scale, 0.0, 1.0), 0.65)
    return ax.scatter(
        x,
        y,
        c=colors,
        s=s,
        edgecolors="none",
    )


def display_note(text: str) -> None:
    display(Markdown(text))


def load_torch(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern} under {path}, found {len(matches)}")
    return matches[0]


def pick_existing_key(mapping: dict, candidates: list[str]) -> str:
    for key in candidates:
        if key in mapping:
            return key
    raise KeyError(f"None of {candidates} found in mapping keys: {list(mapping.keys())[:20]}")


def raw_evolution_schema(method_cfg: dict, probe_size: int = 2048) -> dict:
    cache_key = (str(method_cfg["evolution_log_path"]), int(probe_size))
    if cache_key in _RAW_EVO_SCHEMA_CACHE:
        return _RAW_EVO_SCHEMA_CACHE[cache_key]

    payload = load_torch(method_cfg["evolution_log_path"])
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected evolution payload format in {method_cfg['evolution_log_path']}")

    probe = payload[: min(probe_size, len(payload))]
    frac_keys = sorted({key for record in probe for key in record.get("ib_status_frac", {}).keys()})
    has_correct = any("correct" in record for record in probe)
    record_keys = sorted({key for record in probe for key in record.keys()})
    pruning_field = None
    pruning_label = "Pruning rate"
    if "pruned_frac" in frac_keys:
        pruning_field = "frac__pruned_frac"
        pruning_label = "Pruning rate"
    elif "easy_pruned_frac" in frac_keys:
        pruning_field = "frac__easy_pruned_frac"
        pruning_label = "Easy-pruned fraction"

    schema = {
        "record_keys": record_keys,
        "frac_keys": frac_keys,
        "has_correct": has_correct,
        "n_records": int(len(payload)),
        "probe_size": int(len(probe)),
        "pruning_field": pruning_field,
        "pruning_label": pruning_label,
        "probe": {
            "epoch": np.fromiter((int(record["epoch"]) for record in probe), dtype=np.int16, count=len(probe)),
            "sample_id": np.fromiter((int(record["sample_index"]) for record in probe), dtype=np.int32, count=len(probe)),
            "loss": np.fromiter((float(record["loss"]) for record in probe), dtype=np.float32, count=len(probe)),
            "entropy": np.fromiter((float(record["uncertainty_entropy"]) for record in probe), dtype=np.float32, count=len(probe)),
        },
    }
    if has_correct:
        schema["probe"]["correct"] = np.fromiter(
            (float(record["correct"]) if "correct" in record else np.nan for record in probe),
            dtype=np.float32,
            count=len(probe),
        )
    for key in frac_keys:
        schema["probe"][f"frac__{key}"] = np.fromiter(
            (float(record.get("ib_status_frac", {}).get(key, np.nan)) for record in probe),
            dtype=np.float32,
            count=len(probe),
        )

    del payload
    _RAW_EVO_SCHEMA_CACHE[cache_key] = schema
    return schema


def subset_payload_to_dict(payload) -> tuple[dict[int, torch.Tensor], dict]:
    if isinstance(payload, dict) and "subsets" in payload:
        subsets = payload["subsets"]
        meta = payload.get("meta", {})
    elif isinstance(payload, dict):
        subsets = payload
        meta = {}
    else:
        raise ValueError(f"Unexpected subset payload type: {type(payload)}")
    subsets = {int(k): torch.as_tensor(v, dtype=torch.int64).cpu() for k, v in subsets.items()}
    return subsets, meta


def subset_coverage(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = load_torch(cfg["subset_path"])
    subsets, meta = subset_payload_to_dict(payload)
    n_samples = cfg["n_samples"]
    counts = np.zeros(n_samples, dtype=np.int32)
    for indices in subsets.values():
        counts[indices.numpy()] += 1
    n_subsets = len(subsets)
    frame = pd.DataFrame(
        {
            "sample_id": np.arange(n_samples, dtype=np.int64),
            "included_count": counts,
            "excluded_count": n_subsets - counts,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "dataset": cfg["label"],
                "n_samples": n_samples,
                "n_subsets": n_subsets,
                "mean_included": float(frame["included_count"].mean()),
                "mean_excluded": float(frame["excluded_count"].mean()),
                "min_included": int(frame["included_count"].min()),
                "max_included": int(frame["included_count"].max()),
                "min_excluded": int(frame["excluded_count"].min()),
                "max_excluded": int(frame["excluded_count"].max()),
                "subset_ratio": float(meta.get("subset_ratio", np.nan)),
            }
        ]
    )
    return frame, summary


def infer_epoch_from_path(path: Path) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    if not digits:
        raise ValueError(f"Could not infer epoch from {path}")
    return int(digits[-1])


def load_memorization_epoch(path: Path) -> pd.DataFrame:
    payload = load_torch(path)
    if not isinstance(payload, dict):
        scores = torch.as_tensor(payload).cpu().numpy().astype(np.float64)
        return pd.DataFrame(
            {
                "sample_id": np.arange(len(scores), dtype=np.int64),
                "memorization": scores,
                "epoch": infer_epoch_from_path(path),
            }
        )

    epoch = int(payload.get("epoch", infer_epoch_from_path(path)))
    frame = pd.DataFrame(
        {
            "sample_id": torch.as_tensor(payload["indices"]).cpu().numpy().astype(np.int64),
            "memorization": torch.as_tensor(payload["memorization"]).cpu().numpy().astype(np.float64),
            "epoch": epoch,
        }
    )
    if "labels" in payload:
        frame["mem_artifact_label"] = torch.as_tensor(payload["labels"]).cpu().numpy().astype(np.int64)
    optional_keys = {
        "per_sample_stderr": "memorization_stderr",
        "avg_correct_in": "avg_correct_in",
        "avg_correct_out": "avg_correct_out",
        "in_counts": "in_counts",
        "out_counts": "out_counts",
    }
    for payload_key, frame_key in optional_keys.items():
        if payload_key in payload:
            values = torch.as_tensor(payload[payload_key]).cpu().numpy()
            frame[frame_key] = values
    return frame.sort_values("sample_id").reset_index(drop=True)


def load_memorization_long(method_cfg: dict) -> pd.DataFrame:
    frames = [load_memorization_epoch(path) for path in sorted(method_cfg["memorization_dir"].glob(method_cfg["memorization_glob"]))]
    if not frames:
        raise FileNotFoundError(f"No memorization files found in {method_cfg['memorization_dir']}")
    return pd.concat(frames, ignore_index=True)


def load_local_accuracy(method_cfg: dict) -> pd.DataFrame:
    payload = load_torch(method_cfg["seed_avg_path"])
    train_key = pick_existing_key(payload, method_cfg["local_train_keys"])
    test_key = pick_existing_key(payload, method_cfg["local_test_keys"])
    train_acc = np.asarray(payload[train_key], dtype=np.float64)
    test_acc = np.asarray(payload[test_key], dtype=np.float64)
    frame = pd.DataFrame(
        {
            "epoch": np.arange(len(train_acc), dtype=np.int64),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "source": "local_seed_avg",
        }
    )
    for key in ("avg/train_loss", "avg/test_loss", "avg/active_ratio"):
        if key in payload:
            short = key.split("/")[-1]
            frame[short] = np.asarray(payload[key], dtype=np.float64)
    return frame


def load_wandb_accuracy(method_cfg: dict) -> pd.DataFrame | None:
    if wandb is None:
        return None
    try:
        api = wandb.Api(timeout=60)
        run = api.run(method_cfg["wandb_run"])
        requested_keys = list(dict.fromkeys(method_cfg["wandb_train_keys"] + method_cfg["wandb_test_keys"]))
        history = run.history(keys=requested_keys, pandas=True)
        if history.empty:
            return None
        history = history.reset_index(drop=True)
        train_key = next((key for key in method_cfg["wandb_train_keys"] if key in history.columns), None)
        test_key = next((key for key in method_cfg["wandb_test_keys"] if key in history.columns), None)
        if train_key is None or test_key is None:
            return None
        frame = pd.DataFrame(
            {
                "epoch": np.arange(len(history), dtype=np.int64),
                "train_accuracy": history[train_key].to_numpy(dtype=np.float64, copy=True),
                "test_accuracy": history[test_key].to_numpy(dtype=np.float64, copy=True),
                "source": "wandb",
            }
        )
        return frame.dropna(subset=["train_accuracy", "test_accuracy"]).reset_index(drop=True)
    except Exception as exc:
        print(f"W&B fallback for {method_cfg['wandb_run']}: {exc}")
        return None


def load_accuracy_curve(method_cfg: dict, prefer_wandb: bool = True) -> pd.DataFrame:
    if prefer_wandb:
        wandb_frame = load_wandb_accuracy(method_cfg)
        if wandb_frame is not None and not wandb_frame.empty:
            return wandb_frame
    return load_local_accuracy(method_cfg)


def resolve_rmia_artifact_dir(method_cfg: dict) -> Path:
    rmia_root = method_cfg["rmia_dir"]
    candidates = []

    for candidate in (
        rmia_root / "derived",
        rmia_root / "results",
        rmia_root / "RMIA" / "derived",
        rmia_root / "RMIA" / "results",
    ):
        if candidate.is_dir():
            candidates.append(candidate)

    candidates.extend(sorted(path for path in rmia_root.glob("*/derived") if path.is_dir()))
    candidates.extend(sorted(path for path in rmia_root.glob("*/results") if path.is_dir()))

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    required_patterns = (
        "rmia_overall_summary_*.json",
        "per_sample_privacy_profile_*.npz",
        "rmia_per_target_metrics_*.npz",
    )
    for candidate in unique_candidates:
        if all(any(candidate.glob(pattern)) for pattern in required_patterns):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve an RMIA artifact directory under {rmia_root}. "
        f"Tried: {[str(path) for path in unique_candidates]}"
    )


def load_rmia_artifacts(method_cfg: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    artifact_dir = resolve_rmia_artifact_dir(method_cfg)
    summary_path = find_one(artifact_dir, "rmia_overall_summary_*.json")
    profile_path = find_one(artifact_dir, "per_sample_privacy_profile_*.npz")
    per_target_path = find_one(artifact_dir, "rmia_per_target_metrics_*.npz")

    summary = json.loads(summary_path.read_text())
    profile_npz = np.load(profile_path, allow_pickle=True)
    per_target_npz = np.load(per_target_path, allow_pickle=True)

    profile = pd.DataFrame(
        {
            "sample_id": profile_npz["sample_index"].astype(np.int64),
            "rmia_mean_member_score": profile_npz["mean_member_score"].astype(np.float64),
            "rmia_std_member_score": profile_npz["std_member_score"].astype(np.float64),
            "detection_rate_1pct": profile_npz["hit_rate_at_1pct_fpr"].astype(np.float64),
            "detection_rate_0p1pct": profile_npz["hit_rate_at_0p1pct_fpr"].astype(np.float64),
            "detection_rate_0p01pct": profile_npz["hit_rate_at_0p01pct_fpr"].astype(np.float64),
        }
    ).sort_values("sample_id").reset_index(drop=True)
    if "class_label" in profile_npz.files:
        profile["rmia_artifact_label"] = profile_npz["class_label"].astype(np.int64)

    per_target = pd.DataFrame({key: per_target_npz[key] for key in per_target_npz.files})
    return summary, profile, per_target


def validate_rmia_alignment(mem_final: pd.DataFrame, rmia_profile: pd.DataFrame, dataset_key: str, method_key: str) -> dict:
    mem_ids = mem_final["sample_id"].to_numpy()
    rmia_ids = rmia_profile["sample_id"].to_numpy()
    sample_ids_match = np.array_equal(mem_ids, rmia_ids)
    allowed = bool(sample_ids_match)
    reason = "sample_id_ok" if allowed else "sample_id_mismatch"

    return {
        "method": method_key,
        "mem_rows": int(len(mem_final)),
        "rmia_rows": int(len(rmia_profile)),
        "sample_ids_match": bool(sample_ids_match),
        "artifact_labels_ignored": True,
        "rmia_join_allowed": bool(allowed),
        "reason": reason,
    }


def strict_sample_merge(left: pd.DataFrame, right: pd.DataFrame, context: str = "") -> pd.DataFrame:
    merged = left.merge(right, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(
            f"Sample merge mismatch in {context}: left={len(left)}, right={len(right)}, merged={len(merged)}"
        )
    return merged.sort_values("sample_id").reset_index(drop=True)


def load_lightweight_bundle(cfg: dict, dataset_key: str, method_keys=None, prefer_wandb: bool = True) -> dict:
    cache_key = (dataset_key, tuple(ordered_methods(cfg, method_keys)), prefer_wandb)
    if cache_key in _LIGHTWEIGHT_CACHE:
        return _LIGHTWEIGHT_CACHE[cache_key]

    method_keys = ordered_methods(cfg, method_keys)
    subset_counts, subset_summary = subset_coverage(cfg)

    mem_long = {}
    mem_final = {}
    accuracy_curves = {}
    rmia_summary = {}
    rmia_profile = {}
    rmia_per_target = {}
    alignment_rows = []

    for method_key in method_keys:
        method_cfg = cfg["methods"][method_key]
        mem_long[method_key] = load_memorization_long(method_cfg)
        end_epoch = cfg["end_epoch"]
        mem_final[method_key] = (
            mem_long[method_key]
            .query("epoch == @end_epoch")
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
        accuracy_curves[method_key] = load_accuracy_curve(method_cfg, prefer_wandb=prefer_wandb)
        rmia_summary[method_key], rmia_profile[method_key], rmia_per_target[method_key] = load_rmia_artifacts(method_cfg)
        alignment_rows.append(
            validate_rmia_alignment(mem_final[method_key], rmia_profile[method_key], dataset_key=dataset_key, method_key=method_key)
        )

    alignment_report = pd.DataFrame(alignment_rows)

    bundle = {
        "dataset_key": dataset_key,
        "cfg": cfg,
        "method_keys": method_keys,
        "subset_counts": subset_counts,
        "subset_summary": subset_summary,
        "mem_long": mem_long,
        "mem_final": mem_final,
        "accuracy_curves": accuracy_curves,
        "rmia_summary": rmia_summary,
        "rmia_profile": rmia_profile,
        "rmia_per_target": rmia_per_target,
        "alignment_report": alignment_report,
    }
    _LIGHTWEIGHT_CACHE[cache_key] = bundle
    return bundle


def rmia_join_allowed(bundle: dict, method_key: str) -> bool:
    row = bundle["alignment_report"].set_index("method").loc[method_key]
    return bool(row["rmia_join_allowed"])


def final_test_accuracy(bundle: dict, method_key: str) -> float:
    curve = bundle["accuracy_curves"][method_key]
    return float(curve["test_accuracy"].iloc[-1])


def bundle_detection_limit(bundle: dict, method_keys: list[str] | None = None) -> float:
    method_keys = method_keys or list(bundle["method_keys"])
    values = []
    for method_key in method_keys:
        if method_key not in bundle["rmia_profile"]:
            continue
        if not rmia_join_allowed(bundle, method_key):
            continue
        profile = bundle["rmia_profile"][method_key]
        values.append(100.0 * profile["detection_rate_0p01pct"].to_numpy(dtype=np.float64, copy=True))
    if not values:
        return detection_vmax(np.array([], dtype=np.float64))
    return detection_vmax(np.concatenate(values))


MEM_BIN_EDGES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
MEM_BIN_LABELS = [
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0",
]
MOVEMENT_LABELS = ["<0.1"] + MEM_BIN_LABELS


def evolution_cache_path(dataset_key: str, method_cfg: dict) -> Path:
    out_dir = CACHE_ROOT / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = method_cfg["evolution_log_path"]
    return out_dir / f"{log_path.stem}_{log_path.stat().st_mtime_ns}_flat_v1.npz"


def build_evolution_cache(log_path: Path, cache_path: Path) -> dict[str, np.ndarray]:
    payload = load_torch(log_path)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected evolution payload format in {log_path}")

    n_records = len(payload)
    probe = payload[: min(5000, n_records)]
    frac_keys = sorted({key for record in probe for key in record.get("ib_status_frac", {}).keys()})
    has_correct = any("correct" in record for record in probe)

    arrays = {
        "epoch": np.fromiter((int(record["epoch"]) for record in payload), dtype=np.int16, count=n_records),
        "sample_id": np.fromiter((int(record["sample_index"]) for record in payload), dtype=np.int32, count=n_records),
        "loss": np.fromiter((float(record["loss"]) for record in payload), dtype=np.float32, count=n_records),
        "entropy": np.fromiter((float(record["uncertainty_entropy"]) for record in payload), dtype=np.float32, count=n_records),
        "pruned_frac": np.fromiter(
            (float(record.get("ib_status_frac", {}).get("pruned_frac", np.nan)) for record in payload),
            dtype=np.float32,
            count=n_records,
        ),
        "status_transitions": np.fromiter(
            (float(record.get("ib_status_transitions", np.nan)) for record in payload),
            dtype=np.float32,
            count=n_records,
        ),
    }
    if has_correct:
        arrays["correct"] = np.fromiter(
            (float(record["correct"]) if "correct" in record else np.nan for record in payload),
            dtype=np.float32,
            count=n_records,
        )
    for key in frac_keys:
        arrays[f"frac__{key}"] = np.fromiter(
            (float(record.get("ib_status_frac", {}).get(key, np.nan)) for record in payload),
            dtype=np.float32,
            count=n_records,
        )

    np.savez_compressed(cache_path, **arrays)
    del payload
    return arrays


def load_evolution_arrays(dataset_key: str, method_key: str, method_cfg: dict) -> dict[str, np.ndarray]:
    cache_key = (dataset_key, method_key)
    if cache_key in _EVOLUTION_CACHE:
        return _EVOLUTION_CACHE[cache_key]

    cache_path = evolution_cache_path(dataset_key, method_cfg)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            arrays = {key: cached[key] for key in cached.files}
    else:
        arrays = build_evolution_cache(method_cfg["evolution_log_path"], cache_path)
    _EVOLUTION_CACHE[cache_key] = arrays
    return arrays


def evolution_epoch_frame(dataset_key: str, method_key: str, method_cfg: dict, epoch: int) -> pd.DataFrame:
    arrays = load_evolution_arrays(dataset_key, method_key, method_cfg)
    mask = arrays["epoch"] == int(epoch)
    default_pruning = np.full(int(mask.sum()), np.nan, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "sample_id": arrays["sample_id"][mask].astype(np.int64),
            "loss": arrays["loss"][mask].astype(np.float64),
            "entropy": arrays["entropy"][mask].astype(np.float64),
            "pruning_rate": arrays["pruned_frac"][mask].astype(np.float64) if "pruned_frac" in arrays else default_pruning,
            "status_transitions": arrays["status_transitions"][mask].astype(np.float64),
        }
    )
    for key, values in arrays.items():
        if key.startswith("frac__"):
            frame[key] = values[mask].astype(np.float64)
    if "correct" in arrays:
        frame["correct"] = arrays["correct"][mask].astype(np.float64)
    return frame.sort_values("sample_id").reset_index(drop=True)


def get_method_epoch_mem(bundle: dict, method_key: str, epoch: int) -> pd.DataFrame:
    frame = bundle["mem_long"][method_key]
    out = frame.query("epoch == @epoch").sort_values("sample_id").reset_index(drop=True)
    if out.empty:
        raise KeyError(f"No memorization data found for method={method_key}, epoch={epoch}")
    return out


def hardness_anchor_frame(bundle: dict, cfg: dict, reference_method: str | None = None) -> pd.DataFrame:
    reference_method = reference_method or HARDNESS_REFERENCE_METHOD
    method_cfg = cfg["methods"][reference_method]
    evo = evolution_epoch_frame(bundle["dataset_key"], reference_method, method_cfg, HARDNESS_EPOCH)
    loss_mean = float(evo["loss"].mean())
    out = evo[["sample_id", "loss"]].rename(columns={"loss": "hardness_anchor_loss"}).copy()
    out["hardness_reference_method"] = reference_method
    out["is_hard"] = out["hardness_anchor_loss"] > loss_mean
    out["is_easy"] = ~out["is_hard"]
    out["hardness_anchor_mean_loss"] = loss_mean
    return out.sort_values("sample_id").reset_index(drop=True)


def build_method_frame(
    bundle: dict,
    epoch: int,
    method_key: str,
    include_rmia: bool = True,
    require_rmia: bool = False,
) -> pd.DataFrame:
    cfg = bundle["cfg"]
    method_cfg = cfg["methods"][method_key]
    mem = get_method_epoch_mem(bundle, method_key, epoch)
    evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
    merged = strict_sample_merge(mem, evo, context=f"{bundle['dataset_key']}::{method_key}::epoch_{epoch}")
    merged = strict_sample_merge(merged, hardness_anchor_frame(bundle, cfg), context=f"{bundle['dataset_key']}::{method_key}::hardness")
    merged["method"] = method_key
    merged["method_label"] = method_cfg["label"]
    merged["is_memorized"] = merged["memorization"] > MEMORIZED_THRESHOLD

    if include_rmia:
        if rmia_join_allowed(bundle, method_key):
            merged = strict_sample_merge(
                merged,
                bundle["rmia_profile"][method_key],
                context=f"{bundle['dataset_key']}::{method_key}::rmia",
            )
        elif require_rmia:
            raise RuntimeError(
                f"RMIA join blocked for dataset={bundle['dataset_key']} method={method_key} because the sample_id ordering check failed."
            )
    return merged.sort_values("sample_id").reset_index(drop=True)


def validate_memorization_frame(mem_long: pd.DataFrame, n_samples: int) -> dict:
    epoch_ok = True
    epoch_count_ok = True
    for _, chunk in mem_long.groupby("epoch", sort=True):
        sample_ids = chunk["sample_id"].to_numpy()
        epoch_ok &= np.array_equal(np.sort(sample_ids), np.arange(n_samples, dtype=np.int64))
        epoch_count_ok &= len(chunk) == n_samples
    return {
        "mem_epoch_rows_ok": bool(epoch_count_ok),
        "mem_epoch_ids_ok": bool(epoch_ok),
    }


def validate_rmia_profile(profile: pd.DataFrame, n_samples: int) -> dict:
    sample_ids = profile["sample_id"].to_numpy()
    rates = profile[["detection_rate_1pct", "detection_rate_0p1pct", "detection_rate_0p01pct"]].to_numpy(dtype=np.float64)
    return {
        "rmia_rows_ok": bool(len(profile) == n_samples),
        "rmia_ids_ok": bool(np.array_equal(sample_ids, np.arange(n_samples, dtype=np.int64))),
        "rmia_rates_in_unit_interval": bool(np.nanmin(rates) >= 0.0 and np.nanmax(rates) <= 1.0),
    }


def validate_evolution_cache_against_raw(dataset_key: str, method_key: str, method_cfg: dict, probe_size: int = 2048) -> dict:
    arrays = load_evolution_arrays(dataset_key, method_key, method_cfg)
    schema = raw_evolution_schema(method_cfg, probe_size=probe_size)
    probe = schema["probe"]
    n_probe = schema["probe_size"]

    keys_to_compare = ["epoch", "sample_id", "loss", "entropy"]
    if schema["has_correct"] and "correct" in arrays:
        keys_to_compare.append("correct")
    for key in schema["frac_keys"]:
        cache_key = f"frac__{key}"
        if cache_key in arrays:
            keys_to_compare.append(cache_key)

    probe_ok = True
    for key in keys_to_compare:
        left = np.asarray(arrays[key][:n_probe])
        right = np.asarray(probe[key])
        if left.dtype.kind in {"f"} or right.dtype.kind in {"f"}:
            match = np.allclose(left, right, equal_nan=True, rtol=1e-6, atol=1e-6)
        else:
            match = np.array_equal(left, right)
        probe_ok &= bool(match)

    cache_frac_keys = sorted([key[len("frac__"):] for key in arrays.keys() if key.startswith("frac__")])
    return {
        "evo_rows_ok": bool(len(arrays["epoch"]) == schema["n_records"]),
        "evo_probe_matches_raw": bool(probe_ok),
        "evo_frac_keys_match_raw": bool(cache_frac_keys == schema["frac_keys"]),
        "raw_pruning_field": schema["pruning_field"] or "none",
        "raw_pruning_label": schema["pruning_label"],
    }


def build_integrity_report(bundle: dict, probe_size: int = 2048) -> pd.DataFrame:
    rows = []
    n_samples = bundle["cfg"]["n_samples"]
    for method_key in bundle["method_keys"]:
        method_cfg = bundle["cfg"]["methods"][method_key]
        mem_checks = validate_memorization_frame(bundle["mem_long"][method_key], n_samples)
        rmia_checks = validate_rmia_profile(bundle["rmia_profile"][method_key], n_samples)
        evo_checks = validate_evolution_cache_against_raw(bundle["dataset_key"], method_key, method_cfg, probe_size=probe_size)
        row = {
            "method": METHOD_LABELS[method_key],
            "sample_id_alignment_ok": bool(rmia_join_allowed(bundle, method_key)),
            **mem_checks,
            **rmia_checks,
            **evo_checks,
        }
        row["all_checked_ok"] = bool(
            row["sample_id_alignment_ok"]
            and row["mem_epoch_rows_ok"]
            and row["mem_epoch_ids_ok"]
            and row["rmia_rows_ok"]
            and row["rmia_ids_ok"]
            and row["rmia_rates_in_unit_interval"]
            and row["evo_rows_ok"]
            and row["evo_probe_matches_raw"]
            and row["evo_frac_keys_match_raw"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def movement_bin_ids(scores: np.ndarray) -> np.ndarray:
    return np.digitize(cap_memorization(scores), MEM_BIN_EDGES, right=True)


def movement_category(delta: int) -> str:
    if delta <= -3:
        return "down_3p"
    if delta == -2:
        return "down_2"
    if delta == -1:
        return "down_1"
    if delta == 0:
        return "same"
    if delta == 1:
        return "up_1"
    if delta == 2:
        return "up_2"
    return "up_3p"


def movement_frame(bundle: dict, epoch: int, method_key: str) -> pd.DataFrame:
    if method_key == "std":
        raise ValueError("Movement is defined relative to standard training, so method_key cannot be 'std'.")
    standard = get_method_epoch_mem(bundle, "std", epoch)[["sample_id", "memorization"]].rename(columns={"memorization": "memorization_std"})
    other = get_method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]].rename(columns={"memorization": "memorization_other"})
    merged = strict_sample_merge(standard, other, context=f"movement::{bundle['dataset_key']}::{method_key}::epoch_{epoch}")
    merged["std_bin"] = movement_bin_ids(merged["memorization_std"].to_numpy())
    merged["other_bin"] = movement_bin_ids(merged["memorization_other"].to_numpy())
    merged["delta_bins"] = merged["other_bin"] - merged["std_bin"]
    merged["movement"] = [movement_category(int(delta)) for delta in merged["delta_bins"].to_numpy()]
    return merged


def resolve_pruning_metric(bundle: dict, method_key: str, frame: pd.DataFrame) -> tuple[str | None, str]:
    method_cfg = bundle["cfg"]["methods"][method_key]
    schema = raw_evolution_schema(method_cfg)
    pruning_field = schema["pruning_field"]
    pruning_label = schema["pruning_label"]

    if pruning_field is None:
        return None, pruning_label
    if pruning_field in frame.columns and not frame[pruning_field].isna().all():
        return pruning_field, pruning_label
    if pruning_field == "frac__pruned_frac" and "pruning_rate" in frame.columns and not frame["pruning_rate"].isna().all():
        return "pruning_rate", pruning_label
    return None, pruning_label


def occupancy_labels_with_counts(bin_ids: np.ndarray, labels: list[str]) -> list[str]:
    counts = pd.Series(bin_ids).value_counts().sort_index()
    out = []
    for idx, label in enumerate(labels):
        out.append(f"{label}\n(n={int(counts.get(idx, 0)):,})")
    return out


def detection_rate_bin_labels() -> list[str]:
    edges = DETECTION_RATE_BIN_EDGES.astype(int)
    return [f"{edges[i]}-{edges[i + 1]}%" for i in range(len(edges) - 1)]


def detection_rate_bin_index(rates_pct: np.ndarray) -> np.ndarray:
    rates_pct = np.asarray(rates_pct, dtype=np.float64)
    return np.digitize(rates_pct, DETECTION_RATE_BIN_EDGES[1:], right=True).astype(np.int64)


def plot_subset_coverage(bundle: dict) -> None:
    dataset_key = bundle["dataset_key"]
    counts = bundle["subset_counts"]
    display(bundle["subset_summary"])

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(counts["included_count"], bins=40, color="#4C78A8", alpha=0.85)
    ax.set_xlabel("Times included across subsets")
    ax.set_ylabel("Samples")
    finish_figure(fig, f"{dataset_key}_subset_inclusion_hist", dataset_key)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(counts["excluded_count"], bins=40, color="#F58518", alpha=0.85)
    ax.set_xlabel("Times excluded across subsets")
    ax.set_ylabel("Samples")
    finish_figure(fig, f"{dataset_key}_subset_exclusion_hist", dataset_key)


def plot_memorization_epoch_distributions(bundle: dict) -> None:
    dataset_key = bundle["dataset_key"]
    for method_key in bundle["method_keys"]:
        method_cfg = bundle["cfg"]["methods"][method_key]
        frame = bundle["mem_long"][method_key].copy()
        frame["memorization_plot"] = cap_memorization(frame["memorization"])
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        sns.violinplot(
            data=frame,
            x="epoch",
            y="memorization_plot",
            color=method_cfg["color"],
            inner="quartile",
            cut=0,
            linewidth=0.8,
            ax=ax,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Memorization score")
        ax.set_ylim(min(-0.05, float(frame["memorization_plot"].min()) - 0.01), 1.02)
        finish_figure(
            fig,
            f"{dataset_key}_{method_key}_mem_violin_by_epoch",
            dataset_key,
            figure_label=f"{bundle['cfg']['label']} | {method_cfg['label']} | Memorization violin",
        )


def annotate_curve_end(ax, x, y, color: str, text: str, y_offset: float = 0.0) -> None:
    x_last = float(x.iloc[-1] if hasattr(x, "iloc") else x[-1])
    y_last = float(y.iloc[-1] if hasattr(y, "iloc") else y[-1])
    ax.scatter([x_last], [y_last], s=28, color=color, zorder=5, edgecolors="white", linewidths=0.8)
    ax.annotate(
        text,
        (x_last, y_last),
        xytext=(8, y_offset),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="semibold",
        va="center",
    )


def plot_accuracy_curves(bundle: dict) -> None:
    dataset_key = bundle["dataset_key"]
    for metric_name, ylabel in (("train_accuracy", "Train accuracy (%)"), ("test_accuracy", "Test accuracy (%)")):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for line_idx, method_key in enumerate(bundle["method_keys"]):
            method_cfg = bundle["cfg"]["methods"][method_key]
            curve = bundle["accuracy_curves"][method_key]
            ax.plot(
                curve["epoch"],
                curve[metric_name],
                linewidth=2.0,
                color=method_cfg["color"],
                alpha=0.95,
                label=f"{method_cfg['label']} ({curve['source'].iloc[0]})",
            )
            annotate_curve_end(
                ax,
                curve["epoch"],
                curve[metric_name],
                method_cfg["color"],
                f"{method_cfg['label']} {float(curve[metric_name].iloc[-1]):.1f}%",
                y_offset=(line_idx - 1) * 11,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.10)
        ax.legend()
        finish_figure(
            fig,
            f"{dataset_key}_{metric_name}_curves",
            dataset_key,
            figure_label=f"{bundle['cfg']['label']} | {ylabel}",
        )


def mean_memorization_by_epoch(mem_frame: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    grouped = mem_frame.groupby("epoch", sort=True)
    rows = []
    for epoch, chunk in grouped:
        scores = chunk["memorization"].to_numpy()
        if threshold is not None:
            scores = scores[scores > threshold]
        rows.append(
            {
                "epoch": int(epoch),
                "mean_memorization": float(np.mean(scores)) if len(scores) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_memorization_trajectories(bundle: dict) -> None:
    dataset_key = bundle["dataset_key"]
    for threshold, suffix, ylabel in (
        (None, "overall", "Mean memorization score"),
        (MEMORIZED_THRESHOLD, "tail", f"Mean memorization score (> {MEMORIZED_THRESHOLD})"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method_idx, method_key in enumerate(bundle["method_keys"]):
            method_cfg = bundle["cfg"]["methods"][method_key]
            summary = mean_memorization_by_epoch(bundle["mem_long"][method_key], threshold=threshold)
            final_acc = final_test_accuracy(bundle, method_key)
            ax.plot(
                summary["epoch"],
                summary["mean_memorization"],
                marker="o",
                linewidth=2.0,
                color=method_cfg["color"],
                label=f"{method_cfg['label']} ({final_acc:.2f}% test)",
            )
            y_offset = (-1, 0, 1)[method_idx] * 8 if len(bundle["method_keys"]) >= 3 else 6 * method_idx
            final_epoch = float(summary["epoch"].iloc[-1])
            final_mean = float(summary["mean_memorization"].iloc[-1])
            if not np.isnan(final_mean):
                ax.annotate(
                    f"{final_mean:.2f}",
                    (final_epoch, final_mean),
                    xytext=(8, y_offset),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=method_cfg["color"],
                    fontweight="semibold",
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        finish_figure(fig, f"{dataset_key}_memorization_trajectory_{suffix}", dataset_key)


def memorization_tail_group_specs() -> list[dict]:
    return [
        {"key": "all", "label": "All samples", "kind": "all", "color": "#111827", "linewidth": 2.8},
        {"key": "gt_0p1", "label": f"Score > {MEMORIZED_THRESHOLD}", "kind": "threshold", "threshold": MEMORIZED_THRESHOLD, "color": "#D7263D", "linewidth": 2.4},
        {"key": "top_90", "label": "Top 90%", "kind": "top_fraction", "fraction": 0.90, "color": "#2D7FF9", "linewidth": 1.9},
        {"key": "top_75", "label": "Top 75%", "kind": "top_fraction", "fraction": 0.75, "color": "#06A77D", "linewidth": 1.9},
        {"key": "top_50", "label": "Top 50%", "kind": "top_fraction", "fraction": 0.50, "color": "#F2C14E", "linewidth": 1.9},
        {"key": "top_25", "label": "Top 25%", "kind": "top_fraction", "fraction": 0.25, "color": "#F58518", "linewidth": 1.9},
        {"key": "top_10", "label": "Top 10%", "kind": "top_fraction", "fraction": 0.10, "color": "#7B2CBF", "linewidth": 1.9},
    ]


def build_memorization_tail_trajectory_summary(
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
) -> pd.DataFrame:
    specs = memorization_tail_group_specs()
    rows = []
    for dataset_key in DATASET_ORDER:
        if dataset_key not in bundles:
            continue
        bundle = bundles[dataset_key]
        active_methods = method_keys or list(bundle["method_keys"])
        active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
        for method_key in active_methods:
            method_cfg = bundle["cfg"]["methods"][method_key]
            frame = bundle["mem_long"][method_key]
            for epoch, chunk in frame.groupby("epoch", sort=True):
                scores = cap_memorization(chunk["memorization"].to_numpy(dtype=np.float64))
                scores = scores[np.isfinite(scores)]
                if len(scores) == 0:
                    continue
                sorted_scores = np.sort(scores)
                for order, spec in enumerate(specs):
                    if spec["kind"] == "all":
                        selected = scores
                    elif spec["kind"] == "threshold":
                        selected = scores[scores > float(spec["threshold"])]
                    elif spec["kind"] == "top_fraction":
                        n_select = max(1, int(np.ceil(len(sorted_scores) * float(spec["fraction"]))))
                        selected = sorted_scores[-n_select:]
                    else:
                        raise ValueError(f"Unknown memorization tail kind: {spec['kind']}")

                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": bundle["cfg"]["label"],
                            "method_key": method_key,
                            "method": method_cfg["label"],
                            "epoch": int(epoch),
                            "group_key": spec["key"],
                            "group": spec["label"],
                            "group_order": order,
                            "n": int(len(selected)),
                            "mean_memorization": float(np.mean(selected)) if len(selected) else np.nan,
                        }
                    )

    return pd.DataFrame(rows)


def build_memorization_tail_trajectory_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    sort_cols = ["dataset_key", "method_key", "group_order", "epoch"]
    ordered = summary_df.sort_values(sort_cols)
    first = (
        ordered.groupby(["dataset_key", "method_key", "group_key"], sort=False)
        .first()
        .reset_index()
        [["dataset_key", "method_key", "group_key", "epoch", "mean_memorization", "n"]]
        .rename(columns={"epoch": "first_epoch", "mean_memorization": "first_mean", "n": "first_n"})
    )
    final = (
        ordered.groupby(["dataset_key", "method_key", "group_key"], sort=False)
        .last()
        .reset_index()
        [
            [
                "dataset_key",
                "dataset",
                "method_key",
                "method",
                "group_key",
                "group",
                "group_order",
                "epoch",
                "mean_memorization",
                "n",
            ]
        ]
        .rename(columns={"epoch": "final_epoch", "mean_memorization": "final_mean", "n": "final_n"})
    )
    table = final.merge(first, on=["dataset_key", "method_key", "group_key"], how="left")
    table["change_from_first"] = table["final_mean"] - table["first_mean"]
    dataset_order = {dataset_key: idx for idx, dataset_key in enumerate(DATASET_ORDER)}
    method_order = {method_key: idx for idx, method_key in enumerate(METHOD_ORDER)}
    table["_dataset_order"] = table["dataset_key"].map(dataset_order).fillna(len(dataset_order)).astype(int)
    table["_method_order"] = table["method_key"].map(method_order).fillna(len(method_order)).astype(int)
    return (
        table.sort_values(["_dataset_order", "_method_order", "group_order"])
        .drop(columns=["_dataset_order", "_method_order"])
        .reset_index(drop=True)
    )


def plot_memorization_tail_trajectory_grid(
    summary_df: pd.DataFrame,
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
) -> None:
    if summary_df.empty:
        display_note("**Memorization tail trajectories skipped:** no summary rows were available.")
        return

    specs = memorization_tail_group_specs()
    spec_by_key = {spec["key"]: spec for spec in specs}
    dataset_keys = [dataset_key for dataset_key in DATASET_ORDER if dataset_key in summary_df["dataset_key"].unique()]
    active_method_keys = method_keys or METHOD_ORDER
    active_method_keys = [
        method_key
        for method_key in METHOD_ORDER
        if method_key in active_method_keys and method_key in summary_df["method_key"].unique()
    ]
    if not dataset_keys or not active_method_keys:
        display_note("**Memorization tail trajectories skipped:** no configured datasets or methods were present.")
        return

    fig, axes = plt.subplots(
        len(dataset_keys),
        len(active_method_keys),
        figsize=(5.4 * len(active_method_keys), 3.85 * len(dataset_keys)),
        sharey=True,
        squeeze=False,
    )

    for row_idx, dataset_key in enumerate(dataset_keys):
        bundle = bundles[dataset_key]
        for col_idx, method_key in enumerate(active_method_keys):
            ax = axes[row_idx, col_idx]
            sub = summary_df.loc[
                (summary_df["dataset_key"] == dataset_key) & (summary_df["method_key"] == method_key)
            ]
            if sub.empty:
                ax.axis("off")
                continue

            for spec in specs:
                group_sub = sub.loc[sub["group_key"] == spec["key"]].sort_values("epoch")
                if group_sub.empty:
                    continue
                ax.plot(
                    group_sub["epoch"],
                    group_sub["mean_memorization"],
                    color=spec["color"],
                    linewidth=spec["linewidth"],
                    marker="o" if spec["key"] in {"all", "gt_0p1", "top_10"} else None,
                    markersize=3.3,
                    label=spec["label"],
                )

            if row_idx == 0:
                ax.set_title(bundle["cfg"]["methods"][method_key]["label"], fontsize=12.5, pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{bundle['cfg']['label']}\nMean memorization")
            ax.set_xlabel("Epoch")
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, axis="y", alpha=0.25)

    legend_handles = [
        mpl.lines.Line2D([0], [0], color=spec["color"], linewidth=spec["linewidth"], label=spec["label"])
        for spec in specs
    ]
    fig.legend(
        legend_handles,
        [spec["label"] for spec in specs],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.987),
        ncol=4,
        frameon=False,
    )
    finish_figure(
        fig,
        "cross_dataset_memorization_tail_trajectory_grid",
        "cross_dataset",
        figure_label=(
            "Mean memorization trajectories for all samples, score-threshold tail, "
            "and top memorization slices"
        ),
        tight_rect=(0.0, 0.0, 1.0, 0.945),
    )


def _annotate_tail_line_ends(ax: plt.Axes, end_points: list[dict], *, x_last: float) -> None:
    if not end_points:
        return

    y_low, y_high = ax.get_ylim()
    y_span = max(y_high - y_low, 1e-6)
    min_gap = 0.052 * y_span
    lower = y_low + 0.055 * y_span
    upper = y_high - 0.055 * y_span

    ordered = sorted(end_points, key=lambda item: item["y"])
    adjusted = []
    for item in ordered:
        y = float(np.clip(item["y"], lower, upper))
        if adjusted and y - adjusted[-1]["label_y"] < min_gap:
            y = adjusted[-1]["label_y"] + min_gap
        adjusted.append({**item, "label_y": y})

    overflow = adjusted[-1]["label_y"] - upper if adjusted else 0.0
    if overflow > 0:
        for item in adjusted:
            item["label_y"] -= overflow
    if adjusted and adjusted[0]["label_y"] < lower:
        shift = lower - adjusted[0]["label_y"]
        for item in adjusted:
            item["label_y"] += shift

    x_left, x_right = ax.get_xlim()
    x_offset = max(0.35, 0.015 * max(x_right - x_left, 1.0))
    for item in adjusted:
        ax.annotate(
            f"{item['y']:.2f}",
            xy=(x_last, item["y"]),
            xytext=(x_last + x_offset, item["label_y"]),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=8.6,
            fontweight="semibold",
            color=item["color"],
            arrowprops={
                "arrowstyle": "-",
                "color": item["color"],
                "linewidth": 0.8,
                "alpha": 0.72,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            clip_on=False,
        )


def _annotate_grouped_point_values(
    ax: plt.Axes,
    points: list[dict],
    *,
    decimals: int = 2,
    x_offset_points: float = 0.0,
) -> None:
    if not points:
        return
    y_low, y_high = ax.get_ylim()
    y_span = max(y_high - y_low, 1e-6)
    min_gap = 0.045 * y_span
    lower = y_low + 0.05 * y_span
    upper = y_high - 0.05 * y_span

    x_low, x_high = ax.get_xlim()
    x_span = max(x_high - x_low, 1e-9)
    x_mid = 0.5 * (x_low + x_high)
    x_offset = 0.018 * x_span + 0.0025 * float(x_offset_points)

    for x_value in sorted({float(point["x"]) for point in points}):
        group = [point for point in points if np.isclose(float(point["x"]), x_value)]
        group = sorted(group, key=lambda point: float(point["y"]))
        adjusted = []
        for idx, point in enumerate(group):
            point_y = float(point["y"])
            if point_y + 0.03 * y_span <= upper:
                preferred_y = point_y + 0.03 * y_span
            else:
                preferred_y = point_y - 0.03 * y_span

            label_y = float(np.clip(preferred_y, lower, upper))
            if adjusted and label_y - adjusted[-1]["label_y"] < min_gap:
                label_y = adjusted[-1]["label_y"] + min_gap
            adjusted.append(
                {
                    "x": float(point["x"]),
                    "y": point_y,
                    "text": f"{float(point['y']):.{decimals}f}",
                    "color": point["color"],
                    "label_y": label_y,
                }
            )

        if adjusted:
            overflow = adjusted[-1]["label_y"] - upper
            if overflow > 0:
                for point in adjusted:
                    point["label_y"] -= overflow
            if adjusted[0]["label_y"] < lower:
                shift = lower - adjusted[0]["label_y"]
                for point in adjusted:
                    point["label_y"] += shift

        side = -1.0 if x_value > x_mid else 1.0
        if abs(x_value - x_mid) < 0.10 * x_span:
            side = 0.0

        for point in adjusted:
            bbox = dict(PRESENTATION_LINE_LABEL_BOX)
            bbox["edgecolor"] = point["color"]
            label_x = point["x"] + side * x_offset
            ax.annotate(
                point["text"],
                xy=(label_x, point["label_y"]),
                xytext=(label_x, point["label_y"]),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="semibold",
                color=point["color"],
                bbox=bbox,
                clip_on=True,
                zorder=20,
            )


def plot_memorization_tail_category_figures(
    summary_df: pd.DataFrame,
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
) -> None:
    if summary_df.empty:
        display_note("**Memorization tail trajectories skipped:** no summary rows were available.")
        return

    dataset_keys = [dataset_key for dataset_key in DATASET_ORDER if dataset_key in summary_df["dataset_key"].unique()]
    active_methods = method_keys or METHOD_ORDER
    active_methods = [
        method_key
        for method_key in METHOD_ORDER
        if method_key in active_methods and method_key in summary_df["method_key"].unique()
    ]
    if not dataset_keys or not active_methods:
        display_note("**Memorization tail trajectories skipped:** no configured datasets or methods were present.")
        return

    for spec in memorization_tail_group_specs():
        fig, axes = plt.subplots(
            1,
            len(dataset_keys),
            figsize=(5.35 * len(dataset_keys), 4.25),
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]

        for ax, dataset_key in zip(axes, dataset_keys):
            bundle = bundles[dataset_key]
            dataset_frame = summary_df.loc[
                (summary_df["dataset_key"] == dataset_key) & (summary_df["group_key"] == spec["key"])
            ]
            if dataset_frame.empty:
                ax.axis("off")
                continue

            end_points = []
            x_values = []
            for method_key in active_methods:
                method_frame = dataset_frame.loc[dataset_frame["method_key"] == method_key].sort_values("epoch")
                if method_frame.empty:
                    continue
                method_cfg = bundle["cfg"]["methods"][method_key]
                final_acc = final_test_accuracy(bundle, method_key)
                ax.plot(
                    method_frame["epoch"],
                    method_frame["mean_memorization"],
                    color=method_cfg["color"],
                    linewidth=2.35,
                    marker="o",
                    markersize=3.2,
                    label=f"{method_cfg['label']} ({final_acc:.2f}% acc)",
                )
                x_values.extend(method_frame["epoch"].tolist())
                final_row = method_frame.iloc[-1]
                end_points.append(
                    {
                        "method_key": method_key,
                        "y": float(final_row["mean_memorization"]),
                        "color": method_cfg["color"],
                    }
                )

            if x_values:
                x_min = float(np.nanmin(x_values))
                x_max = float(np.nanmax(x_values))
                x_pad = max(1.0, 0.09 * max(x_max - x_min, 1.0))
                ax.set_xlim(x_min, x_max + x_pad)
                _annotate_tail_line_ends(ax, end_points, x_last=x_max)

            ax.set_title(bundle["cfg"]["label"], fontsize=12.2, pad=8)
            ax.set_xlabel("Epoch")
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, axis="y", alpha=0.26)
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(0.995, 0.995),
                borderaxespad=0.15,
                borderpad=0.25,
                handlelength=1.45,
                handletextpad=0.45,
                labelspacing=0.24,
                frameon=True,
                framealpha=0.84,
                facecolor="white",
                edgecolor="#E5E7EB",
                fontsize=7.8,
            )

        axes[0].set_ylabel("Mean memorization score")
        finish_figure(
            fig,
            f"cross_dataset_memorization_tail_{spec['key']}_trajectory",
            "cross_dataset",
            figure_label=(
                f"Memorization trajectory | {spec['label']} | "
                "final labels show mean memorization"
            ),
            tight_rect=(0.0, 0.0, 1.0, 0.93),
        )


def build_memorization_tail_final_mean_table(
    summary_df: pd.DataFrame,
    bundles: dict,
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    ordered = summary_df.sort_values(["dataset_key", "method_key", "group_order", "epoch"])
    final = (
        ordered.groupby(["dataset_key", "dataset", "method_key", "method", "group_key", "group", "group_order"], sort=False)
        .last()
        .reset_index()
    )
    value_table = (
        final.pivot_table(
            index=["dataset_key", "dataset", "method_key", "method"],
            columns="group",
            values="mean_memorization",
            aggfunc="first",
        )
        .reset_index()
    )

    group_order = [spec["label"] for spec in memorization_tail_group_specs()]
    for group in group_order:
        if group not in value_table.columns:
            value_table[group] = np.nan

    value_table["Test acc (%)"] = value_table.apply(
        lambda row: final_test_accuracy(bundles[row["dataset_key"]], row["method_key"])
        if row["dataset_key"] in bundles and row["method_key"] in bundles[row["dataset_key"]]["method_keys"]
        else np.nan,
        axis=1,
    )
    dataset_order = {dataset_key: idx for idx, dataset_key in enumerate(DATASET_ORDER)}
    method_order = {method_key: idx for idx, method_key in enumerate(METHOD_ORDER)}
    value_table["_dataset_order"] = value_table["dataset_key"].map(dataset_order).fillna(len(dataset_order)).astype(int)
    value_table["_method_order"] = value_table["method_key"].map(method_order).fillna(len(method_order)).astype(int)
    ordered_cols = ["dataset", "method", "Test acc (%)", *group_order]
    return (
        value_table.sort_values(["_dataset_order", "_method_order"])
        .loc[:, ordered_cols]
        .reset_index(drop=True)
    )


def style_memorization_tail_final_mean_table(
    table: pd.DataFrame,
    *,
    tolerance: float = 0.05,
):
    if table.empty:
        display_note("**Memorization tail table skipped:** no summary rows were available.")
        return None

    value_cols = [spec["label"] for spec in memorization_tail_group_specs() if spec["label"] in table.columns]
    view = table.set_index(["dataset", "method"])
    style_grid = pd.DataFrame("", index=view.index, columns=view.columns)
    palette = {
        "low": "background-color: #EAF7EA;",
        "mid": "background-color: #F4F4F5;",
        "high": "background-color: #FCEBEC;",
    }

    for dataset_name, dataset_rows in table.groupby("dataset", sort=False):
        for col in value_cols:
            values = pd.to_numeric(dataset_rows[col], errors="coerce")
            valid = values.dropna()
            if valid.empty:
                continue
            spread = float(valid.max() - valid.min())
            index_keys = [(row["dataset"], row["method"]) for _, row in dataset_rows.iterrows()]
            if spread <= tolerance:
                for index_key in index_keys:
                    style_grid.loc[index_key, col] = palette["mid"]
                continue

            low = float(valid.min())
            high = float(valid.max())
            for _, row in dataset_rows.iterrows():
                index_key = (row["dataset"], row["method"])
                value = float(row[col])
                if np.isclose(value, low, atol=1e-12, rtol=0.0):
                    style_grid.loc[index_key, col] = palette["low"]
                elif np.isclose(value, high, atol=1e-12, rtol=0.0):
                    style_grid.loc[index_key, col] = palette["high"]
                else:
                    style_grid.loc[index_key, col] = palette["mid"]

    return (
        view.style
        .format({"Test acc (%)": "{:.2f}", **{col: "{:.3f}" for col in value_cols}})
        .apply(lambda _: style_grid, axis=None)
        .set_caption(
            "Final-epoch mean memorization by sample tail. Green is the lowest method value, red is the highest; "
            f"all methods are gray when the spread is <= {tolerance:.2f}."
        )
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("caption-side", "top"),
                        ("text-align", "left"),
                        ("font-weight", "600"),
                        ("font-size", "13px"),
                        ("color", "#334155"),
                        ("padding", "0 0 8px 0"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#F8FAFC"),
                        ("color", "#334155"),
                        ("font-weight", "600"),
                        ("border-bottom", "1px solid #CBD5E1"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-bottom", "1px solid #E5E7EB"),
                        ("padding", "6px 10px"),
                        ("text-align", "right"),
                    ],
                },
            ]
        )
    )


def final_memorization_tail_threshold_specs() -> list[dict]:
    return [
        {"label": "Top 10%", "quantile": 0.90, "color": "#7B2CBF"},
        {"label": "Top 25%", "quantile": 0.75, "color": "#D7263D"},
        {"label": "Top 50%", "quantile": 0.50, "color": "#F58518"},
        {"label": "Top 75%", "quantile": 0.25, "color": "#06A77D"},
        {"label": "Top 90%", "quantile": 0.10, "color": "#2D7FF9"},
    ]


def build_final_memorization_tail_threshold_table(
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    specs = final_memorization_tail_threshold_specs()
    for dataset_key in DATASET_ORDER:
        if dataset_key not in bundles:
            continue
        bundle = bundles[dataset_key]
        epoch = int(bundle["cfg"]["end_epoch"])
        active_methods = method_keys or list(bundle["method_keys"])
        active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
        for method_key in active_methods:
            mem = get_method_epoch_mem(bundle, method_key, epoch)
            scores = cap_memorization(mem["memorization"].to_numpy(dtype=np.float64))
            scores = scores[np.isfinite(scores)]
            if len(scores) == 0:
                continue
            for spec in specs:
                threshold = float(np.quantile(scores, float(spec["quantile"])))
                selected = scores[scores >= threshold]
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": bundle["cfg"]["label"],
                        "method_key": method_key,
                        "method": bundle["cfg"]["methods"][method_key]["label"],
                        "epoch": epoch,
                        "tail": spec["label"],
                        "quantile": float(spec["quantile"]),
                        "threshold": threshold,
                        "n_above_threshold": int(len(selected)),
                        "mean_above_threshold": float(np.mean(selected)) if len(selected) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _spread_threshold_labels(rows: list[dict], lower: float = 0.03, upper: float = 0.98, min_gap: float = 0.055) -> list[dict]:
    if not rows:
        return []
    adjusted = []
    for row in sorted(rows, key=lambda item: item["y"]):
        y = float(np.clip(row["y"], lower, upper))
        if adjusted and y - adjusted[-1]["label_y"] < min_gap:
            y = adjusted[-1]["label_y"] + min_gap
        adjusted.append({**row, "label_y": y})
    overflow = adjusted[-1]["label_y"] - upper
    if overflow > 0:
        for row in adjusted:
            row["label_y"] -= overflow
    if adjusted[0]["label_y"] < lower:
        shift = lower - adjusted[0]["label_y"]
        for row in adjusted:
            row["label_y"] += shift
    return adjusted


def plot_final_memorization_violin_thresholds(
    bundles: dict,
    *,
    threshold_table: pd.DataFrame | None = None,
    method_keys: list[str] | None = None,
) -> pd.DataFrame:
    threshold_table = (
        build_final_memorization_tail_threshold_table(bundles, method_keys=method_keys)
        if threshold_table is None
        else threshold_table
    )
    if threshold_table.empty:
        display_note("**Final memorization violin plot skipped:** no threshold rows were available.")
        return threshold_table

    specs = final_memorization_tail_threshold_specs()
    color_by_tail = {spec["label"]: spec["color"] for spec in specs}
    dataset_keys = [dataset_key for dataset_key in DATASET_ORDER if dataset_key in threshold_table["dataset_key"].unique()]
    active_methods = method_keys or METHOD_ORDER
    active_methods = [
        method_key
        for method_key in METHOD_ORDER
        if method_key in active_methods and method_key in threshold_table["method_key"].unique()
    ]

    fig, axes = plt.subplots(
        len(dataset_keys),
        len(active_methods),
        figsize=(4.55 * len(active_methods), 3.72 * len(dataset_keys)),
        sharey=True,
        squeeze=False,
    )

    for row_idx, dataset_key in enumerate(dataset_keys):
        bundle = bundles[dataset_key]
        epoch = int(bundle["cfg"]["end_epoch"])
        for col_idx, method_key in enumerate(active_methods):
            ax = axes[row_idx, col_idx]
            if method_key not in bundle["method_keys"]:
                ax.axis("off")
                continue
            method_cfg = bundle["cfg"]["methods"][method_key]
            mem = get_method_epoch_mem(bundle, method_key, epoch)
            scores = cap_memorization(mem["memorization"].to_numpy(dtype=np.float64))
            scores = scores[np.isfinite(scores)]
            if len(scores) == 0:
                ax.axis("off")
                continue

            violin = ax.violinplot(
                [scores],
                positions=[0.0],
                widths=0.62,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(method_cfg["color"])
                body.set_edgecolor(method_cfg["color"])
                body.set_alpha(0.24)
            violin["cmedians"].set_color(method_cfg["color"])
            violin["cmedians"].set_linewidth(1.7)

            ax.axhline(
                MEMORIZED_THRESHOLD,
                color="#111827",
                linestyle="--",
                linewidth=1.05,
                alpha=0.75,
            )
            ax.text(
                0.48,
                MEMORIZED_THRESHOLD,
                f"{MEMORIZED_THRESHOLD:.1f}",
                ha="left",
                va="center",
                fontsize=8.1,
                color="#111827",
            )

            label_rows = []
            sub_thresholds = threshold_table.loc[
                (threshold_table["dataset_key"] == dataset_key)
                & (threshold_table["method_key"] == method_key)
            ]
            for _, threshold_row in sub_thresholds.iterrows():
                y = float(threshold_row["threshold"])
                tail_label = str(threshold_row["tail"])
                color = color_by_tail.get(tail_label, "#475569")
                ax.hlines(y, -0.27, 0.27, color=color, linewidth=1.15, alpha=0.92)
                label_rows.append(
                    {
                        "y": y,
                        "color": color,
                        "text": (
                            f"{tail_label}: {y:.2f} "
                            f"({float(threshold_row['mean_above_threshold']):.2f})"
                        ),
                    }
                )

            for label in _spread_threshold_labels(label_rows):
                ax.plot([0.27, 0.40], [label["y"], label["label_y"]], color=label["color"], linewidth=0.75, alpha=0.75)
                ax.text(
                    0.43,
                    label["label_y"],
                    label["text"],
                    ha="left",
                    va="center",
                    fontsize=7.35,
                    color=label["color"],
                    fontweight="semibold",
                )

            ax.set_xlim(-0.62, 1.55)
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks([])
            if row_idx == 0:
                ax.set_title(method_cfg["label"], fontsize=12.0, pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{bundle['cfg']['label']}\nMemorization score")
            ax.grid(True, axis="y", alpha=0.22)

    finish_figure(
        fig,
        "cross_dataset_final_memorization_violin_tail_thresholds",
        "cross_dataset",
        figure_label=(
            "Final-epoch memorization distributions with tail thresholds; "
            "parentheses show mean memorization above each threshold"
        ),
        tight_rect=(0.0, 0.0, 1.0, 0.94),
    )
    return threshold_table


def plot_tiny_matched_accuracy_view(bundle: dict) -> None:
    if bundle["dataset_key"] != "tinyimagenet":
        return
    cfg = bundle["cfg"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for line_idx, method_key in enumerate(("std", "ib")):
        method_cfg = cfg["methods"][method_key]
        rows = []
        for pair in cfg["matched_accuracy_pairs"]:
            epoch = pair["standard"] if method_key == "std" else pair["ib"]
            mem_epoch = get_method_epoch_mem(bundle, method_key, epoch)
            acc_curve = bundle["accuracy_curves"][method_key]
            test_acc = float(acc_curve.loc[acc_curve["epoch"] == epoch, "test_accuracy"].iloc[0])
            rows.append({"epoch": epoch, "test_accuracy": test_acc, "mean_memorization": float(mem_epoch["memorization"].mean())})
        frame = pd.DataFrame(rows)
        ax.plot(
            frame["test_accuracy"],
            frame["mean_memorization"],
            marker="o",
            linewidth=2.0,
            color=method_cfg["color"],
            alpha=0.95,
            label=method_cfg["label"],
        )
        annotate_curve_end(
            ax,
            frame["test_accuracy"],
            frame["mean_memorization"],
            method_cfg["color"],
            f"{method_cfg['label']} {float(frame['mean_memorization'].iloc[-1]):.3f}",
            y_offset=-10 if line_idx == 0 else 10,
        )
    ax.set_xlabel("Average test accuracy (%)")
    ax.set_ylabel("Mean memorization score")
    ax.legend()
    finish_figure(
        fig,
        "tinyimagenet_accuracy_matched_memorization",
        "tinyimagenet",
        figure_label="TinyImageNet | Accuracy-matched memorization",
    )


def build_memorization_summary_table(bundle: dict, epoch: int) -> pd.DataFrame:
    rows = []
    hardness = hardness_anchor_frame(bundle, bundle["cfg"])
    for method_key in bundle["method_keys"]:
        frame = strict_sample_merge(
            get_method_epoch_mem(bundle, method_key, epoch),
            hardness,
            context=f"summary::{bundle['dataset_key']}::{method_key}::epoch_{epoch}",
        )
        memorized = frame["memorization"] > MEMORIZED_THRESHOLD
        mem_scores = frame["memorization"].to_numpy()
        mem_tail = frame.loc[memorized, "memorization"].to_numpy()
        easy_tail = frame.loc[memorized & frame["is_easy"], "sample_id"].size
        hard_tail = frame.loc[memorized & frame["is_hard"], "sample_id"].size
        rows.append(
            {
                "method": METHOD_LABELS[method_key],
                "epoch": epoch,
                "pct_memorized": 100.0 * memorized.mean(),
                "mean_mem_all": float(np.mean(mem_scores)),
                "median_mem_all": float(np.median(mem_scores)),
                "mean_mem_memorized": float(np.mean(mem_tail)) if len(mem_tail) else np.nan,
                "median_mem_memorized": float(np.median(mem_tail)) if len(mem_tail) else np.nan,
                "pct_easy_within_memorized": 100.0 * easy_tail / max(int(memorized.sum()), 1),
                "pct_hard_within_memorized": 100.0 * hard_tail / max(int(memorized.sum()), 1),
            }
        )
    return pd.DataFrame(rows)


def memorization_bin_index(scores: np.ndarray) -> np.ndarray:
    return np.digitize(cap_memorization(scores), MEM_BIN_EDGES, right=True) - 1


def plot_binned_counts_with_hardness(bundle: dict, epoch: int) -> None:
    dataset_key = bundle["dataset_key"]
    hardness = hardness_anchor_frame(bundle, bundle["cfg"])
    methods = bundle["method_keys"]
    x = np.arange(len(MEM_BIN_LABELS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for idx, method_key in enumerate(methods):
        frame = strict_sample_merge(
            get_method_epoch_mem(bundle, method_key, epoch),
            hardness,
            context=f"binned::{dataset_key}::{method_key}::epoch_{epoch}",
        )
        mask = (frame["memorization"] > MEMORIZED_THRESHOLD) & (frame["memorization"] <= 1.0)
        bins = memorization_bin_index(frame.loc[mask, "memorization"].to_numpy())
        easy = frame.loc[mask, "is_easy"].to_numpy()
        hard = frame.loc[mask, "is_hard"].to_numpy()
        easy_counts = np.zeros(len(MEM_BIN_LABELS), dtype=np.int64)
        hard_counts = np.zeros(len(MEM_BIN_LABELS), dtype=np.int64)
        for bin_id in range(len(MEM_BIN_LABELS)):
            easy_counts[bin_id] = int(np.sum((bins == bin_id) & easy))
            hard_counts[bin_id] = int(np.sum((bins == bin_id) & hard))
        positions = x + (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(
            positions,
            easy_counts,
            width=width,
            color=bundle["cfg"]["methods"][method_key]["color"],
            alpha=0.94,
            hatch=METHOD_HATCHES["easy"],
            edgecolor="black",
            linewidth=0.4,
            label=f"{METHOD_LABELS[method_key]} easy" if idx == 0 else None,
        )
        ax.bar(
            positions,
            hard_counts,
            bottom=easy_counts,
            width=width,
            color=bundle["cfg"]["methods"][method_key]["color"],
            alpha=0.94,
            hatch=METHOD_HATCHES["hard"],
            edgecolor="black",
            linewidth=0.4,
            label=f"{METHOD_LABELS[method_key]} hard" if idx == 0 else None,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(MEM_BIN_LABELS, rotation=45, ha="right")
    ax.set_xlabel("Memorization bin")
    ax.set_ylabel("Samples")

    method_handles = [
        mpatches.Patch(facecolor=bundle["cfg"]["methods"][method_key]["color"], edgecolor="black", label=METHOD_LABELS[method_key])
        for method_key in methods
    ]
    hardness_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=METHOD_HATCHES["easy"], label="Easy"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=METHOD_HATCHES["hard"], label="Hard"),
    ]
    legend1 = ax.legend(handles=method_handles, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(legend1)
    ax.legend(handles=hardness_handles, loc="upper right", bbox_to_anchor=(1.0, 0.78))
    finish_figure(
        fig,
        f"{dataset_key}_binned_hardness_epoch_{epoch}",
        dataset_key,
        figure_label=f"{bundle['cfg']['label']} | Memorization bins | Epoch {epoch}",
    )


def plot_movement_stacked(bundle: dict, epoch: int, method_key: str) -> None:
    dataset_key = bundle["dataset_key"]
    movement = movement_frame(bundle, epoch, method_key)
    categories = list(MOVEMENT_STYLE.keys())
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    x = np.arange(len(MOVEMENT_LABELS))
    bottoms = np.zeros(len(MOVEMENT_LABELS), dtype=np.float64)

    standard_bin = movement["std_bin"].to_numpy()
    counts = np.bincount(standard_bin, minlength=len(MOVEMENT_LABELS))
    for category in categories:
        mask = movement["movement"] == category
        cat_counts = np.bincount(standard_bin[mask], minlength=len(MOVEMENT_LABELS)).astype(np.float64)
        proportion = np.divide(cat_counts, counts, out=np.zeros_like(cat_counts), where=counts > 0)
        style = MOVEMENT_STYLE[category]
        ax.bar(
            x,
            proportion * 100.0,
            bottom=bottoms * 100.0,
            color=style["color"],
            edgecolor="black",
            linewidth=0.3,
            hatch=style["hatch"],
            alpha=0.94,
            label=movement_display_name(category),
        )
        bottoms += proportion

    ax.set_xticks(x)
    ax.set_xticklabels(occupancy_labels_with_counts(standard_bin, MOVEMENT_LABELS), rotation=0)
    ax.set_xlabel("Standard-training memorization bin")
    ax.set_ylabel("Share of samples in the bin (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    finish_figure(
        fig,
        f"{dataset_key}_{method_key}_movement_epoch_{epoch}",
        dataset_key,
        figure_label=f"{bundle['cfg']['label']} | Movement vs Standard | {METHOD_LABELS[method_key]} | Epoch {epoch}",
        tight_rect=(0.0, 0.0, 0.80, 0.96),
    )


def plot_pruning_vs_memorization(bundle: dict, epoch: int, method_key: str) -> None:
    dataset_key = bundle["dataset_key"]
    frame = build_method_frame(bundle, epoch=epoch, method_key=method_key, include_rmia=False)
    movement = movement_frame(bundle, epoch, method_key)[["sample_id", "movement"]]
    frame = strict_sample_merge(frame, movement, context=f"scatter::{dataset_key}::{method_key}::epoch_{epoch}")
    categories = frame["movement"].to_numpy()

    pruning_col, pruning_label = resolve_pruning_metric(bundle, method_key, frame)
    if pruning_col is None:
        display_note(
            f"**Pruning plot skipped for {METHOD_LABELS[method_key]} at epoch {epoch}:** "
            "no non-empty pruning field was found in the evolution log."
        )
        return

    valid = frame[pruning_col].notna().to_numpy()
    frame = frame.loc[valid].reset_index(drop=True)
    categories = frame["movement"].to_numpy()
    if frame.empty:
        display_note(
            f"**Pruning plot skipped for {METHOD_LABELS[method_key]} at epoch {epoch}:** "
            "the pruning field exists but is empty at this epoch."
        )
        return

    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    for category in MOVEMENT_STYLE:
        mask = categories == category
        if not np.any(mask):
            continue
        style = MOVEMENT_STYLE[category]
        ax.scatter(
            cap_memorization(frame.loc[mask, "memorization"]),
            frame.loc[mask, pruning_col],
            s=18,
            alpha=0.62,
            color=style["color"],
            label=movement_display_name(category),
            marker=style["marker"],
            edgecolors="white",
            linewidths=0.25,
        )
    x = cap_memorization(frame["memorization"].to_numpy())
    y = frame[pruning_col].to_numpy()
    coeff = np.polyfit(x, y, deg=1)
    x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
    y_line = coeff[0] * x_line + coeff[1]
    corr = np.corrcoef(x, y)[0, 1]
    ax.plot(x_line, y_line, color="black", linewidth=1.5, linestyle="--", label=f"r={corr:.3f}")
    ax.set_xlabel("Memorization score")
    ax.set_ylabel(pruning_label)
    ax.set_xlim(min(-0.05, float(np.nanmin(x)) - 0.01), 1.0)
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    finish_figure(
        fig,
        f"{dataset_key}_{method_key}_pruning_vs_mem_epoch_{epoch}",
        dataset_key,
        figure_label=f"{bundle['cfg']['label']} | {pruning_label} vs memorization | {METHOD_LABELS[method_key]} | Epoch {epoch}",
        tight_rect=(0.0, 0.0, 0.79, 0.96),
    )


def plot_detection_rate_binned_counts_with_hardness(bundle: dict, epoch: int) -> None:
    dataset_key = bundle["dataset_key"]
    valid_methods = [method_key for method_key in bundle["method_keys"] if rmia_join_allowed(bundle, method_key)]
    if not valid_methods:
        display_note("**RMIA detection-bin plot skipped:** no method passed the sample-index safety checks for this dataset.")
        return

    labels = detection_rate_bin_labels()
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    for idx, method_key in enumerate(valid_methods):
        frame = build_method_frame(bundle, epoch=epoch, method_key=method_key, include_rmia=True, require_rmia=True)
        rates_pct = 100.0 * frame["detection_rate_0p01pct"].to_numpy(dtype=np.float64, copy=True)
        mask = rates_pct > 0.0
        if not np.any(mask):
            continue
        bins = detection_rate_bin_index(rates_pct[mask])
        easy = frame.loc[mask, "is_easy"].to_numpy()
        hard = frame.loc[mask, "is_hard"].to_numpy()
        easy_counts = np.zeros(len(labels), dtype=np.int64)
        hard_counts = np.zeros(len(labels), dtype=np.int64)
        for bin_id in range(len(labels)):
            easy_counts[bin_id] = int(np.sum((bins == bin_id) & easy))
            hard_counts[bin_id] = int(np.sum((bins == bin_id) & hard))
        positions = x + (idx - (len(valid_methods) - 1) / 2.0) * width
        ax.bar(
            positions,
            easy_counts,
            width=width,
            color=bundle["cfg"]["methods"][method_key]["color"],
            alpha=0.94,
            hatch=METHOD_HATCHES["easy"],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            positions,
            hard_counts,
            bottom=easy_counts,
            width=width,
            color=bundle["cfg"]["methods"][method_key]["color"],
            alpha=0.94,
            hatch=METHOD_HATCHES["hard"],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("RMIA detection-rate bin @ 0.01% FPR")
    ax.set_ylabel("Samples with detection rate > 0")

    method_handles = [
        mpatches.Patch(facecolor=bundle["cfg"]["methods"][method_key]["color"], edgecolor="black", label=METHOD_LABELS[method_key])
        for method_key in valid_methods
    ]
    hardness_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=METHOD_HATCHES["easy"], label="Easy"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=METHOD_HATCHES["hard"], label="Hard"),
    ]
    legend1 = ax.legend(handles=method_handles, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(legend1)
    ax.legend(handles=hardness_handles, loc="upper right", bbox_to_anchor=(1.0, 0.78))
    finish_figure(
        fig,
        f"{dataset_key}_rmia_detection_binned_hardness_epoch_{epoch}",
        dataset_key,
        figure_label=f"{bundle['cfg']['label']} | RMIA detection-rate bins | Epoch {epoch}",
    )


def plot_pruning_vs_detection(bundle: dict, epoch: int, method_key: str) -> None:
    dataset_key = bundle["dataset_key"]
    if not rmia_join_allowed(bundle, method_key):
        display_note(
            f"**Pruning vs detection skipped for {METHOD_LABELS[method_key]} at epoch {epoch}:** "
            "the RMIA sample-index alignment checks did not pass."
        )
        return

    frame = build_method_frame(bundle, epoch=epoch, method_key=method_key, include_rmia=True, require_rmia=True)
    movement = movement_frame(bundle, epoch, method_key)[["sample_id", "movement"]]
    frame = strict_sample_merge(frame, movement, context=f"pruning_detection::{dataset_key}::{method_key}::epoch_{epoch}")

    pruning_col, pruning_label = resolve_pruning_metric(bundle, method_key, frame)
    if pruning_col is None:
        display_note(
            f"**Pruning vs detection skipped for {METHOD_LABELS[method_key]} at epoch {epoch}:** "
            "no non-empty pruning field was found in the evolution log."
        )
        return

    frame["detection_rate_pct"] = 100.0 * frame["detection_rate_0p01pct"]
    valid = frame[pruning_col].notna() & frame["detection_rate_pct"].gt(0.0)
    frame = frame.loc[valid].reset_index(drop=True)
    if frame.empty:
        display_note(
            f"**Pruning vs detection skipped for {METHOD_LABELS[method_key]} at epoch {epoch}:** "
            "no samples had both a non-zero RMIA detection rate and a valid pruning value."
        )
        return

    categories = frame["movement"].to_numpy()
    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    for category in MOVEMENT_STYLE:
        mask = categories == category
        if not np.any(mask):
            continue
        style = MOVEMENT_STYLE[category]
        ax.scatter(
            frame.loc[mask, "detection_rate_pct"],
            frame.loc[mask, pruning_col],
            s=18,
            alpha=0.62,
            color=style["color"],
            label=movement_display_name(category),
            marker=style["marker"],
            edgecolors="white",
            linewidths=0.25,
        )

    x = frame["detection_rate_pct"].to_numpy(dtype=np.float64, copy=True)
    y = frame[pruning_col].to_numpy(dtype=np.float64, copy=True)
    if len(np.unique(x)) >= 2:
        coeff = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        y_line = coeff[0] * x_line + coeff[1]
        corr = np.corrcoef(x, y)[0, 1]
        ax.plot(x_line, y_line, color="black", linewidth=1.5, linestyle="--", label=f"r={corr:.3f}")
    ax.set_xlabel("RMIA detection rate @ 0.01% FPR (%)")
    ax.set_ylabel(pruning_label)
    ax.set_xlim(left=0.0)
    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    finish_figure(
        fig,
        f"{dataset_key}_{method_key}_pruning_vs_detection_epoch_{epoch}",
        dataset_key,
        figure_label=f"{bundle['cfg']['label']} | {pruning_label} vs RMIA detection | {METHOD_LABELS[method_key]} | Epoch {epoch}",
        tight_rect=(0.0, 0.0, 0.79, 0.96),
    )


def build_rmia_summary_table(bundle: dict) -> pd.DataFrame:
    rows = []
    for method_key in bundle["method_keys"]:
        summary = bundle["rmia_summary"][method_key]
        rows.append(
            {
                "dataset_key": bundle["dataset_key"],
                "dataset": bundle["cfg"]["label"],
                "method_key": method_key,
                "method": METHOD_LABELS[method_key],
                "target_models": int(summary["num_target_models"]),
                "auc": float(summary["auc_mean"]),
                "auc_std": float(summary["auc_std"]),
                "attack_accuracy_pct": 100.0 * float(summary["accuracy_mean"]),
                "attack_accuracy_std_pct": 100.0 * float(summary["accuracy_std"]),
                "tpr_at_1pct_fpr_pct": 100.0 * float(summary["tpr_at_1pct_fpr_mean"]),
                "tpr_at_1pct_fpr_std_pct": 100.0 * float(summary["tpr_at_1pct_fpr_std"]),
                "tpr_at_0p1pct_fpr_pct": 100.0 * float(summary["tpr_at_0p1pct_fpr_mean"]),
                "tpr_at_0p1pct_fpr_std_pct": 100.0 * float(summary["tpr_at_0p1pct_fpr_std"]),
                "tpr_at_0p01pct_fpr_pct": 100.0 * float(summary["tpr_at_0p01pct_fpr_mean"]),
                "tpr_at_0p01pct_fpr_std_pct": 100.0 * float(summary["tpr_at_0p01pct_fpr_std"]),
                "tpr_at_0p001pct_fpr_pct": 100.0 * float(summary["tpr_at_0p001pct_fpr_mean"]),
                "tpr_at_0p001pct_fpr_std_pct": 100.0 * float(summary["tpr_at_0p001pct_fpr_std"]),
            }
        )
    return pd.DataFrame(rows)


def build_rmia_results_table(
    bundles: dict[str, dict],
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
) -> pd.DataFrame:
    dataset_keys = dataset_keys or DATASET_ORDER
    method_keys = method_keys or METHOD_ORDER
    frames = []
    for dataset_key in dataset_keys:
        if dataset_key not in bundles:
            continue
        bundle = bundles[dataset_key]
        frame = build_rmia_summary_table(bundle)
        frame = frame.loc[frame["method_key"].isin(method_keys)].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    table = pd.concat(frames, ignore_index=True)
    dataset_order = {dataset_key: idx for idx, dataset_key in enumerate(DATASET_ORDER)}
    method_order = {method_key: idx for idx, method_key in enumerate(METHOD_ORDER)}
    table["_dataset_order"] = table["dataset_key"].map(dataset_order).fillna(len(dataset_order)).astype(int)
    table["_method_order"] = table["method_key"].map(method_order).fillna(len(method_order)).astype(int)
    table = table.sort_values(["_dataset_order", "_method_order"]).drop(columns=["_dataset_order", "_method_order"])
    return table.reset_index(drop=True)


def style_rmia_results_table(table: pd.DataFrame):
    if table.empty:
        display_note("**RMIA table skipped:** no RMIA summaries were loaded.")
        return None

    def fmt_mean_std(row: pd.Series, mean_col: str, std_col: str, decimals: int) -> str:
        mean_value = row[mean_col]
        std_value = row[std_col]
        if pd.isna(mean_value):
            return ""
        if pd.isna(std_value):
            return f"{mean_value:.{decimals}f}"
        return f"{mean_value:.{decimals}f} +/- {std_value:.{decimals}f}"

    display_columns = {
        "target_models": "Targets",
        "auc_display": "AUC",
        "attack_accuracy_display": "Attack acc (%)",
        "tpr_at_1pct_fpr_display": "TPR @ 1% FPR (%)",
        "tpr_at_0p1pct_fpr_display": "TPR @ 0.1% FPR (%)",
        "tpr_at_0p01pct_fpr_display": "TPR @ 0.01% FPR (%)",
    }
    display_table = table.copy()
    display_table["auc_display"] = display_table.apply(lambda row: fmt_mean_std(row, "auc", "auc_std", 3), axis=1)
    display_table["attack_accuracy_display"] = display_table.apply(
        lambda row: fmt_mean_std(row, "attack_accuracy_pct", "attack_accuracy_std_pct", 2),
        axis=1,
    )
    display_table["tpr_at_1pct_fpr_display"] = display_table.apply(
        lambda row: fmt_mean_std(row, "tpr_at_1pct_fpr_pct", "tpr_at_1pct_fpr_std_pct", 2),
        axis=1,
    )
    display_table["tpr_at_0p1pct_fpr_display"] = display_table.apply(
        lambda row: fmt_mean_std(row, "tpr_at_0p1pct_fpr_pct", "tpr_at_0p1pct_fpr_std_pct", 2),
        axis=1,
    )
    display_table["tpr_at_0p01pct_fpr_display"] = display_table.apply(
        lambda row: fmt_mean_std(row, "tpr_at_0p01pct_fpr_pct", "tpr_at_0p01pct_fpr_std_pct", 2),
        axis=1,
    )
    metric_columns = [
        "auc",
        "attack_accuracy_pct",
        "tpr_at_1pct_fpr_pct",
        "tpr_at_0p1pct_fpr_pct",
        "tpr_at_0p01pct_fpr_pct",
    ]
    tolerances = {
        "auc": 0.005,
        "attack_accuracy_pct": 0.25,
        "tpr_at_1pct_fpr_pct": 0.25,
        "tpr_at_0p1pct_fpr_pct": 0.25,
        "tpr_at_0p01pct_fpr_pct": 0.25,
    }
    metric_to_display = {
        "auc": "AUC",
        "attack_accuracy_pct": "Attack acc (%)",
        "tpr_at_1pct_fpr_pct": "TPR @ 1% FPR (%)",
        "tpr_at_0p1pct_fpr_pct": "TPR @ 0.1% FPR (%)",
        "tpr_at_0p01pct_fpr_pct": "TPR @ 0.01% FPR (%)",
    }
    palette = {
        "low": "background-color: #EAF7EA;",
        "mid": "background-color: #F4F4F5;",
        "high": "background-color: #FCEBEC;",
        "plain": "",
    }

    view = (
        display_table[["dataset", "method", *display_columns.keys()]]
        .rename(columns=display_columns)
        .set_index(["dataset", "method"])
    )
    style_grid = pd.DataFrame("", index=view.index, columns=view.columns)

    for dataset_name, dataset_rows in table.groupby("dataset", sort=False):
        dataset_methods = dataset_rows["method"].tolist()
        for metric_col in metric_columns:
            shown_col = metric_to_display[metric_col]
            values = pd.to_numeric(dataset_rows[metric_col], errors="coerce")
            valid = values.dropna()
            if valid.empty:
                continue
            spread = float(valid.max() - valid.min())
            tol = tolerances[metric_col]
            if spread <= tol:
                for method_name in dataset_methods:
                    style_grid.loc[(dataset_name, method_name), shown_col] = palette["mid"]
                continue

            low = float(valid.min())
            high = float(valid.max())
            for _, row in dataset_rows.iterrows():
                value = float(row[metric_col])
                index_key = (row["dataset"], row["method"])
                if np.isclose(value, low, atol=1e-12, rtol=0.0):
                    style_grid.loc[index_key, shown_col] = palette["low"]
                elif np.isclose(value, high, atol=1e-12, rtol=0.0):
                    style_grid.loc[index_key, shown_col] = palette["high"]
                else:
                    style_grid.loc[index_key, shown_col] = palette["mid"]

    return (
        view.style
        .format(
            {
                "Targets": "{:.0f}",
            }
        )
        .apply(lambda _: style_grid, axis=None)
        .set_caption("RMIA audit results by dataset and target-training method. Cells show mean +/- std across targets; lower AUC/TPR means less membership leakage.")
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("caption-side", "top"),
                        ("text-align", "left"),
                        ("font-weight", "600"),
                        ("font-size", "13px"),
                        ("color", "#334155"),
                        ("padding", "0 0 8px 0"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#F8FAFC"),
                        ("color", "#334155"),
                        ("font-weight", "600"),
                        ("border-bottom", "1px solid #CBD5E1"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-bottom", "1px solid #E5E7EB"),
                        ("padding", "6px 10px"),
                        ("text-align", "right"),
                    ],
                },
            ]
        )
    )


def plot_detection_rate_violin(bundle: dict, epoch: int) -> None:
    valid_methods = [method_key for method_key in bundle["method_keys"] if rmia_join_allowed(bundle, method_key)]
    if not valid_methods:
        display_note("**RMIA join skipped:** no method passed the sample-index safety checks for this dataset.")
        return

    frames = []
    for method_key in valid_methods:
        frame = build_method_frame(bundle, epoch=epoch, method_key=method_key, include_rmia=True, require_rmia=True)
        frame["memorization_group"] = np.where(frame["memorization"] > MEMORIZED_THRESHOLD, "memorized", "not memorized")
        frame["method_label"] = METHOD_LABELS[method_key]
        frame["detection_rate_0p01pct_pct"] = 100.0 * frame["detection_rate_0p01pct"]
        frames.append(frame[["method_label", "memorization_group", "detection_rate_0p01pct_pct"]])
    plot_df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.boxplot(
        data=plot_df,
        x="memorization_group",
        y="detection_rate_0p01pct_pct",
        hue="method_label",
        palette=[METHOD_COLORS[key] for key in valid_methods],
        linewidth=0.8,
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Detection rate @ 0.01% FPR (%)")
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_rmia_detection_box_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=f"{bundle['cfg']['label']} | RMIA detection rate @ 0.01% FPR | Epoch {epoch}",
    )


def mem_entropy_display_bins(
    mem_scores: np.ndarray,
    entropy: np.ndarray,
    thresholds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    mem_threshold, entropy_threshold = thresholds
    mem_bins = np.where(cap_memorization(mem_scores) < mem_threshold, 0, 1)
    entropy_rows = np.where(np.asarray(entropy) >= entropy_threshold, 0, 1)
    return mem_bins.astype(np.int64), entropy_rows.astype(np.int64)


def rmia_mem_entropy_thresholds(
    bundle: dict,
    mem_split: float = RMIA_MATRIX_MEM_SPLIT,
    entropy_max_fraction: float = RMIA_MATRIX_ENTROPY_MAX_FRACTION,
) -> tuple[float, float, float]:
    mem_threshold = float(mem_split)
    max_entropy = float(np.log(float(bundle["cfg"]["n_classes"])))
    entropy_threshold = float(entropy_max_fraction) * max_entropy
    return mem_threshold, entropy_threshold, max_entropy


def mem_entropy_display_bins_2x4(
    mem_scores: np.ndarray,
    entropy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mem_scores = cap_memorization(mem_scores)
    mem_bins = np.digitize(mem_scores, [0.1, 0.4, 0.8], right=False)
    entropy_rows = np.where(np.asarray(entropy) >= RMIA_MATRIX_THRESHOLDS[1], 0, 1)
    return mem_bins.astype(np.int64), entropy_rows.astype(np.int64)


def rmia_mem_entropy_axis_limits(bundle: dict, max_entropy: float | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
    mem_mins = []
    entropy_mins = []
    entropy_maxs = []
    for epoch in bundle["cfg"]["matrix_epochs"]:
        for method_key in bundle["method_keys"]:
            if not rmia_join_allowed(bundle, method_key):
                continue
            frame = build_method_frame(bundle, epoch=epoch, method_key=method_key, include_rmia=True, require_rmia=True)
            mem_plot = cap_memorization(frame["memorization"].to_numpy())
            entropy = frame["entropy"].to_numpy(dtype=np.float64, copy=True)
            if len(mem_plot):
                mem_mins.append(float(np.nanmin(mem_plot)))
            if len(entropy):
                entropy_mins.append(float(np.nanmin(entropy)))
                entropy_maxs.append(float(np.nanmax(entropy)))

    x_lim = (0.0, 1.0)

    if max_entropy is not None and np.isfinite(max_entropy) and max_entropy > 0:
        y_lim = (0.0, 1.03 * float(max_entropy))
    elif entropy_mins and entropy_maxs:
        y_min_raw = min(entropy_mins)
        y_max_raw = max(entropy_maxs)
        y_span = max(y_max_raw - y_min_raw, 1e-6)
        y_pad = 0.03 * y_span
        y_lim = (max(0.0, y_min_raw - y_pad), y_max_raw + y_pad)
    else:
        y_lim = (0.0, 1.0)
    return x_lim, y_lim


def std_ib_display_bins(
    std_scores: np.ndarray,
    ib_scores: np.ndarray,
    thresholds: tuple[float, float] = STD_IB_MATRIX_THRESHOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    low, high = thresholds
    x_bins = np.digitize(cap_memorization(std_scores), [low, high], right=False)
    ib_scores = cap_memorization(ib_scores)
    y_rows = np.where(ib_scores >= high, 0, np.where(ib_scores >= low, 1, 2))
    return x_bins.astype(np.int64), y_rows.astype(np.int64)


def std_ib_quadrant_bins(
    std_scores: np.ndarray,
    ib_scores: np.ndarray,
    thresholds: tuple[float, float] = STD_IB_QUADRANT_THRESHOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    x_threshold, y_threshold = thresholds
    x_bins = np.where(cap_memorization(std_scores) < x_threshold, 0, 1)
    y_rows = np.where(cap_memorization(ib_scores) >= y_threshold, 0, 1)
    return x_bins.astype(np.int64), y_rows.astype(np.int64)


def _matrix_text(
    n_cell: int,
    n_total: int,
    n_detected: int,
    total_detected: int,
    include_detection: bool = True,
    extra: str | None = None,
    pct_decimals: int = 1,
) -> dict:
    pct_total = 100.0 * n_cell / max(n_total, 1)
    text = {
        "count": f"n = {n_cell:,}",
        "total": f"{pct_total:.{pct_decimals}f}% of total samples",
        "detected_cell": None,
        "detected_share": None,
        "extra": extra,
    }
    if include_detection:
        pct_cell_detected = 100.0 * n_detected / max(n_cell, 1)
        pct_share_detected = 100.0 * n_detected / max(total_detected, 1)
        text["detected_cell"] = f"Detected in cell: {pct_cell_detected:.{pct_decimals}f}%"
        text["detected_share"] = f"Of all detected: {pct_share_detected:.{pct_decimals}f}%"
    return text


def draw_matrix_cell(
    ax: plt.Axes,
    x: int,
    y: int,
    *,
    facecolor,
    n_cell: int,
    total_n: int,
    n_detected: int,
    total_detected: int,
    include_detection: bool = True,
    extra_lines: list[tuple[str, str]] | None = None,
    pct_decimals: int = 1,
) -> None:
    text = _matrix_text(
        n_cell,
        total_n,
        n_detected,
        total_detected,
        include_detection=include_detection,
        pct_decimals=pct_decimals,
    )
    ax.add_patch(
        mpatches.Rectangle(
            (x - 0.5, y - 0.5),
            1,
            1,
            facecolor=facecolor,
            edgecolor="#CBD5E1",
            linewidth=1.0,
        )
    )
    ax.text(x, y - 0.24, text["count"], ha="center", va="center", fontsize=10.4, fontweight="semibold", color="#0F172A")
    ax.text(x, y - 0.03, text["total"], ha="center", va="center", fontsize=8.5, color="#334155")

    next_y = 0.18
    if include_detection:
        ax.text(x, y + next_y, text["detected_cell"], ha="center", va="center", fontsize=8.1, color="#475569")
        next_y += 0.17
        ax.text(x, y + next_y, text["detected_share"], ha="center", va="center", fontsize=8.0, color="#64748B")
        next_y += 0.17

    if extra_lines:
        for line, _color in extra_lines:
            ax.text(x, y + next_y, line, ha="center", va="center", fontsize=7.9, color="#475569")
            next_y += 0.16


def plot_mem_entropy_matrix(bundle: dict, epoch: int, method_key: str) -> None:
    if not rmia_join_allowed(bundle, method_key):
        display_note(
            f"**RMIA join skipped for {METHOD_LABELS[method_key]} at epoch {epoch}:** "
            f"the local alignment checks did not pass."
        )
        return

    frame = build_method_frame(bundle, epoch=epoch, method_key=method_key, include_rmia=True, require_rmia=True)
    mem_plot = cap_memorization(frame["memorization"].to_numpy())
    entropy = frame["entropy"].to_numpy(dtype=np.float64, copy=True)
    mem_threshold, entropy_threshold, max_entropy = rmia_mem_entropy_thresholds(bundle)
    mem_bins, entropy_rows = mem_entropy_display_bins(
        mem_plot,
        entropy,
        thresholds=(mem_threshold, entropy_threshold),
    )
    detection_rates = frame["detection_rate_0p01pct"].to_numpy(dtype=np.float64, copy=True)
    detected = (detection_rates > RMIA_MATRIX_DETECTED_RATE_THRESHOLD).astype(np.int32)
    detection_pct = 100.0 * frame["detection_rate_0p01pct"].to_numpy()
    detection_limit = bundle_detection_limit(bundle)
    x_lim, y_lim = rmia_mem_entropy_axis_limits(bundle, max_entropy=max_entropy)

    fig = plt.figure(figsize=(13.4, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.24)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[0, 1])

    scatter = scatter_detection_layered(
        ax_scatter,
        mem_plot,
        entropy,
        detection_pct,
        vmax=detection_limit,
        s=12,
    )
    ax_scatter.axhline(entropy_threshold, color="#475569", linestyle="--", linewidth=1.0)
    ax_scatter.axvline(mem_threshold, color="#475569", linestyle="--", linewidth=1.0)
    ax_scatter.axhline(max_entropy, color="#64748B", linestyle=":", linewidth=1.0)
    ax_scatter.set_xlabel("Memorization score")
    ax_scatter.set_ylabel("Entropy (nats)")
    ax_scatter.set_xlim(*x_lim)
    ax_scatter.set_ylim(*y_lim)
    ax_scatter.set_title(
        (
            f"Split at memorization {mem_threshold:.2f} and "
            f"{RMIA_MATRIX_ENTROPY_MAX_FRACTION:.2f} x log({bundle['cfg']['n_classes']}) = {entropy_threshold:.2f}"
        ),
        fontsize=11.0,
        pad=8,
    )
    ax_scatter.text(
        0.985,
        max_entropy,
        "100%",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#64748B",
        transform=ax_scatter.get_yaxis_transform(),
    )
    add_detection_colorbar(
        fig,
        scatter,
        ax_scatter,
        method_label=METHOD_LABELS[method_key],
        vmax=detection_limit,
    )

    ax_matrix.set_xlim(-0.5, 1.5)
    ax_matrix.set_ylim(1.5, -0.5)
    ax_matrix.set_xticks(range(2))
    ax_matrix.set_xticklabels([f"Mem < {mem_threshold:.2f}", f"Mem >= {mem_threshold:.2f}"])
    ax_matrix.set_yticks(range(2))
    ax_matrix.set_yticklabels(
        [
            f"Entropy >= {entropy_threshold:.2f}\n(high)",
            f"Entropy < {entropy_threshold:.2f}\n(low)",
        ]
    )
    ax_matrix.set_title("Cell composition", fontsize=11.0, pad=8)
    ax_matrix.set_xlabel("")
    ax_matrix.set_ylabel("")
    ax_matrix.tick_params(length=0)
    ax_matrix.set_aspect("equal", adjustable="box")
    total_n = len(frame)
    total_detected = int(detected.sum())
    cell_pct = np.zeros((2, 2), dtype=np.float64)
    for r in range(2):
        for c in range(2):
            cell_pct[r, c] = np.mean((entropy_rows == r) & (mem_bins == c))
    matrix_cmap = MATRIX_CMAP

    for r in range(2):
        for c in range(2):
            mask = (entropy_rows == r) & (mem_bins == c)
            n_cell = int(mask.sum())
            n_detect = int(detected[mask].sum())
            facecolor = matrix_cmap(0.10 + 0.75 * cell_pct[r, c] / max(cell_pct.max(), 1e-8))
            draw_matrix_cell(
                ax_matrix,
                c,
                r,
                facecolor=facecolor,
                n_cell=n_cell,
                total_n=total_n,
                n_detected=n_detect,
                total_detected=total_detected,
                include_detection=True,
                pct_decimals=2,
            )

    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{method_key}_mem_entropy_matrix_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{bundle['cfg']['label']} | Memorization x entropy x RMIA | "
            f"{METHOD_LABELS[method_key]} | Epoch {epoch}"
        ),
    )


def plot_mem_entropy_matrix_2x4(bundle: dict, epoch: int, method_key: str) -> None:
    if not rmia_join_allowed(bundle, method_key):
        display_note(
            f"**RMIA join skipped for {METHOD_LABELS[method_key]} at epoch {epoch}:** "
            f"the local alignment checks did not pass."
        )
        return

    frame = build_method_frame(bundle, epoch=epoch, method_key=method_key, include_rmia=True, require_rmia=True)
    mem_plot = cap_memorization(frame["memorization"].to_numpy())
    mem_bins, entropy_rows = mem_entropy_display_bins_2x4(mem_plot, frame["entropy"].to_numpy())
    detected = (frame["detection_rate_0p01pct"].to_numpy() > 0).astype(np.int32)
    detection_pct = 100.0 * frame["detection_rate_0p01pct"].to_numpy()
    detection_limit = bundle_detection_limit(bundle)
    entropy_threshold = RMIA_MATRIX_THRESHOLDS[1]
    x_lim, y_lim = rmia_mem_entropy_axis_limits(bundle)

    fig = plt.figure(figsize=(12.4, 8.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 2.2], hspace=0.28)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[1, 0])

    scatter = scatter_detection_layered(
        ax_scatter,
        mem_plot,
        frame["entropy"],
        detection_pct,
        vmax=detection_limit,
        s=12,
    )
    ax_scatter.axhline(entropy_threshold, color="#475569", linestyle="--", linewidth=1.0)
    for xline in (0.1, 0.4, 0.8):
        ax_scatter.axvline(xline, color="#475569", linestyle="--", linewidth=1.0)
    ax_scatter.set_xlabel("Memorization score")
    ax_scatter.set_ylabel("Entropy")
    ax_scatter.set_xlim(*x_lim)
    ax_scatter.set_ylim(*y_lim)
    add_detection_colorbar(
        fig,
        scatter,
        ax_scatter,
        method_label=METHOD_LABELS[method_key],
        vmax=detection_limit,
    )

    ax_matrix.set_xlim(-0.5, 3.5)
    ax_matrix.set_ylim(1.5, -0.5)
    ax_matrix.set_xticks(range(4))
    ax_matrix.set_xticklabels(["<0.1", "0.1-0.4", "0.4-0.8", ">=0.8"])
    ax_matrix.set_yticks(range(2))
    ax_matrix.set_yticklabels([f"entropy >= {entropy_threshold:.1f}", f"entropy < {entropy_threshold:.1f}"])
    ax_matrix.set_xlabel("Memorization bins")
    ax_matrix.set_ylabel("Entropy bands")
    ax_matrix.tick_params(length=0)
    total_n = len(frame)
    total_detected = int(detected.sum())
    cell_pct = np.zeros((2, 4), dtype=np.float64)
    for r in range(2):
        for c in range(4):
            cell_pct[r, c] = np.mean((entropy_rows == r) & (mem_bins == c))

    for r in range(2):
        for c in range(4):
            mask = (entropy_rows == r) & (mem_bins == c)
            n_cell = int(mask.sum())
            n_detect = int(detected[mask].sum())
            facecolor = MATRIX_CMAP(0.10 + 0.75 * cell_pct[r, c] / max(cell_pct.max(), 1e-8))
            draw_matrix_cell(
                ax_matrix,
                c,
                r,
                facecolor=facecolor,
                n_cell=n_cell,
                total_n=total_n,
                n_detected=n_detect,
                total_detected=total_detected,
                include_detection=True,
                pct_decimals=3,
            )

    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{method_key}_mem_entropy_matrix_2x4_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{bundle['cfg']['label']} | Memorization × entropy × RMIA | "
            f"{METHOD_LABELS[method_key]} | Epoch {epoch} | 2x4 bins"
        ),
    )


def plot_entropy_histograms(bundle: dict, epoch: int) -> None:
    for method_key in bundle["method_keys"]:
        method_cfg = bundle["cfg"]["methods"][method_key]
        evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
        entropy = evo["entropy"].dropna().to_numpy()
        q1, med, q3 = np.quantile(entropy, [0.25, 0.5, 0.75])
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.hist(
            entropy,
            bins=70,
            color=method_cfg["color"],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.25,
        )
        ax.axvline(q1, color="#1F2937", linestyle="--", linewidth=1.2, label=f"Q1 = {q1:.2f}")
        ax.axvline(med, color="#111827", linestyle="-", linewidth=1.5, label=f"Median = {med:.2f}")
        ax.axvline(q3, color="#1F2937", linestyle=":", linewidth=1.5, label=f"Q3 = {q3:.2f}")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Samples")
        ax.legend(loc="upper right")
        finish_figure(
            fig,
            f"{bundle['dataset_key']}_{method_key}_entropy_hist_epoch_{epoch}",
            bundle["dataset_key"],
            figure_label=f"{bundle['cfg']['label']} | Entropy histogram | {METHOD_LABELS[method_key]} | Epoch {epoch}",
        )


def pairwise_mem_frame(bundle: dict, epoch: int, x_method: str, y_method: str, include_detection: bool = True) -> tuple[pd.DataFrame, bool]:
    x_name = METHOD_LABELS[x_method]
    y_name = METHOD_LABELS[y_method]
    x_mem = get_method_epoch_mem(bundle, x_method, epoch)[["sample_id", "memorization"]].rename(columns={"memorization": f"memorization_{x_method}"})
    y_mem = get_method_epoch_mem(bundle, y_method, epoch)[["sample_id", "memorization"]].rename(columns={"memorization": f"memorization_{y_method}"})
    merged = strict_sample_merge(x_mem, y_mem, context=f"{x_method}_vs_{y_method}::{bundle['dataset_key']}::epoch_{epoch}")

    use_detection = bool(include_detection and rmia_join_allowed(bundle, y_method))
    if use_detection:
        detection = build_method_frame(bundle, epoch=epoch, method_key=y_method, include_rmia=True, require_rmia=True)[["sample_id", "detection_rate_0p01pct"]]
        merged = strict_sample_merge(merged, detection, context=f"{x_method}_vs_{y_method}::{bundle['dataset_key']}::epoch_{epoch}::rmia")
    return merged, use_detection


def pairwise_mem_hardness_frame(bundle: dict, epoch: int, x_method: str, y_method: str) -> pd.DataFrame:
    x_mem = get_method_epoch_mem(bundle, x_method, epoch)[["sample_id", "memorization"]].rename(columns={"memorization": f"memorization_{x_method}"})
    y_mem = get_method_epoch_mem(bundle, y_method, epoch)[["sample_id", "memorization"]].rename(columns={"memorization": f"memorization_{y_method}"})
    merged = strict_sample_merge(x_mem, y_mem, context=f"{x_method}_vs_{y_method}::{bundle['dataset_key']}::epoch_{epoch}::hardness_base")
    merged = strict_sample_merge(
        merged,
        hardness_anchor_frame(bundle, bundle["cfg"])[["sample_id", "is_easy", "is_hard"]],
        context=f"{x_method}_vs_{y_method}::{bundle['dataset_key']}::epoch_{epoch}::hardness",
    )
    return merged


def plot_method_comparison(
    bundle: dict,
    epoch: int,
    x_method: str,
    y_method: str,
    thresholds: tuple[float, float] = STD_IB_MATRIX_THRESHOLDS,
) -> None:
    merged, include_detection = pairwise_mem_frame(bundle, epoch, x_method, y_method, include_detection=True)
    x_label = METHOD_LABELS[x_method]
    y_label = METHOD_LABELS[y_method]
    x_color = METHOD_COLORS[x_method]
    y_color = METHOD_COLORS[y_method]

    fig = plt.figure(figsize=(13.2, 10.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.05, 2.85], hspace=0.30)
    top = outer[0].subgridspec(1, 3, width_ratios=[0.32, 1.0, 0.32], wspace=0.0)
    ax_scatter = fig.add_subplot(top[0, 1])
    ax_matrix = fig.add_subplot(outer[1, 0])

    x_plot = cap_memorization(merged[f"memorization_{x_method}"].to_numpy())
    y_plot = cap_memorization(merged[f"memorization_{y_method}"].to_numpy())
    if include_detection:
        detection_pct = 100.0 * merged["detection_rate_0p01pct"].to_numpy()
        detection_limit = bundle_detection_limit(bundle)
        scatter = ax_scatter.scatter(
            x_plot,
            y_plot,
            c=detection_pct,
            cmap=DETECTION_CMAP,
            vmin=0.0,
            vmax=detection_limit,
            s=12,
            alpha=0.60,
            edgecolors="none",
        )
        add_detection_colorbar(
            fig,
            scatter,
            ax_scatter,
            method_label=y_label,
            vmax=detection_limit,
        )
    else:
        ax_scatter.scatter(x_plot, y_plot, s=12, alpha=0.45, color="#4D4D4D", edgecolors="none")
    lower = min(-0.05, float(np.nanmin([x_plot.min(), y_plot.min()])) - 0.01)
    upper = 1.0
    ax_scatter.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.3, color="black")
    for threshold in thresholds:
        ax_scatter.axvline(threshold, linestyle=":", linewidth=1.3, color="#374151")
        ax_scatter.axhline(threshold, linestyle=":", linewidth=1.3, color="#374151")
    ax_scatter.set_xlabel(f"{x_label} memorization score")
    ax_scatter.set_ylabel(f"{y_label} memorization score")
    ax_scatter.set_xlim(lower, 1.0)
    ax_scatter.set_ylim(lower, 1.0)

    x_bins, y_rows = std_ib_display_bins(x_plot, y_plot, thresholds=thresholds)
    detected = (merged["detection_rate_0p01pct"].to_numpy() > 0).astype(np.int32) if include_detection else np.zeros(len(merged), dtype=np.int32)
    total_detected = int(detected.sum())

    x_labels, y_labels = threshold_label_edges(thresholds)
    ax_matrix.set_xlim(-0.5, 2.5)
    ax_matrix.set_ylim(2.5, -0.5)
    ax_matrix.set_xticks(range(3))
    ax_matrix.set_xticklabels(x_labels)
    ax_matrix.set_yticks(range(3))
    ax_matrix.set_yticklabels(y_labels)
    ax_matrix.set_xlabel(f"{x_label} memorization bins")
    ax_matrix.set_ylabel(f"{y_label} memorization bins")
    ax_matrix.tick_params(length=0)

    total_n = len(merged)
    cell_pct = np.zeros((3, 3), dtype=np.float64)
    for r in range(3):
        for c in range(3):
            cell_pct[r, c] = np.mean((y_rows == r) & (x_bins == c))
    matrix_cmap = MATRIX_CMAP
    for r in range(3):
        for c in range(3):
            mask = (y_rows == r) & (x_bins == c)
            n_cell = int(mask.sum())
            y_better = 100.0 * np.mean(y_plot[mask] > x_plot[mask]) if n_cell else 0.0
            x_better = 100.0 * np.mean(x_plot[mask] > y_plot[mask]) if n_cell else 0.0
            n_detect = int(detected[mask].sum()) if include_detection else 0
            facecolor = matrix_cmap(0.10 + 0.75 * cell_pct[r, c] / max(cell_pct.max(), 1e-8))
            draw_matrix_cell(
                ax_matrix,
                c,
                r,
                facecolor=facecolor,
                n_cell=n_cell,
                total_n=total_n,
                n_detected=n_detect,
                total_detected=total_detected,
                include_detection=include_detection,
                extra_lines=[
                    (f"{y_label} > {x_label}: {y_better:.1f}%", y_color),
                    (f"{x_label} > {y_label}: {x_better:.1f}%", x_color),
                ],
            )

    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{x_method}_vs_{y_method}_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{bundle['cfg']['label']} | {x_label} vs {y_label} | "
            f"Epoch {epoch} | thresholds {thresholds[0]:.1f}, {thresholds[1]:.1f}"
        ),
    )


def plot_method_comparison_quadrants(
    bundle: dict,
    epoch: int,
    x_method: str,
    y_method: str,
    thresholds: tuple[float, float] = STD_IB_QUADRANT_THRESHOLDS,
) -> None:
    merged, include_detection = pairwise_mem_frame(bundle, epoch, x_method, y_method, include_detection=True)
    x_label = METHOD_LABELS[x_method]
    y_label = METHOD_LABELS[y_method]
    x_color = METHOD_COLORS[x_method]
    y_color = METHOD_COLORS[y_method]

    fig = plt.figure(figsize=(11.2, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.75], hspace=0.24)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[1, 0])

    x_plot = cap_memorization(merged[f"memorization_{x_method}"].to_numpy())
    y_plot = cap_memorization(merged[f"memorization_{y_method}"].to_numpy())
    if include_detection:
        detection_pct = 100.0 * merged["detection_rate_0p01pct"].to_numpy()
        detection_limit = bundle_detection_limit(bundle)
        scatter = ax_scatter.scatter(
            x_plot,
            y_plot,
            c=detection_pct,
            cmap=DETECTION_CMAP,
            vmin=0.0,
            vmax=detection_limit,
            s=12,
            alpha=0.60,
            edgecolors="none",
        )
        add_detection_colorbar(
            fig,
            scatter,
            ax_scatter,
            method_label=y_label,
            vmax=detection_limit,
        )
    else:
        ax_scatter.scatter(x_plot, y_plot, s=12, alpha=0.45, color="#4D4D4D", edgecolors="none")
    lower = min(-0.05, float(np.nanmin([x_plot.min(), y_plot.min()])) - 0.01)
    upper = 1.0
    ax_scatter.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.3, color="black")
    ax_scatter.axvline(thresholds[0], linestyle=":", linewidth=1.4, color="#374151")
    ax_scatter.axhline(thresholds[1], linestyle=":", linewidth=1.4, color="#374151")
    ax_scatter.set_xlabel(f"{x_label} memorization score")
    ax_scatter.set_ylabel(f"{y_label} memorization score")
    ax_scatter.set_xlim(lower, 1.0)
    ax_scatter.set_ylim(lower, 1.0)

    x_bins, y_rows = std_ib_quadrant_bins(x_plot, y_plot, thresholds=thresholds)
    detected = (merged["detection_rate_0p01pct"].to_numpy() > 0).astype(np.int32) if include_detection else np.zeros(len(merged), dtype=np.int32)
    total_detected = int(detected.sum())

    ax_matrix.set_xlim(-0.5, 1.5)
    ax_matrix.set_ylim(1.5, -0.5)
    ax_matrix.set_xticks(range(2))
    ax_matrix.set_xticklabels([f"<{thresholds[0]:.1f}", f">={thresholds[0]:.1f}"])
    ax_matrix.set_yticks(range(2))
    ax_matrix.set_yticklabels([f">={thresholds[1]:.1f}", f"<{thresholds[1]:.1f}"])
    ax_matrix.set_xlabel(f"{x_label} memorization bins")
    ax_matrix.set_ylabel(f"{y_label} memorization bins")
    ax_matrix.tick_params(length=0)

    total_n = len(merged)
    cell_pct = np.zeros((2, 2), dtype=np.float64)
    for r in range(2):
        for c in range(2):
            cell_pct[r, c] = np.mean((y_rows == r) & (x_bins == c))
    matrix_cmap = MATRIX_CMAP
    for r in range(2):
        for c in range(2):
            mask = (y_rows == r) & (x_bins == c)
            n_cell = int(mask.sum())
            y_better = 100.0 * np.mean(y_plot[mask] > x_plot[mask]) if n_cell else 0.0
            x_better = 100.0 * np.mean(x_plot[mask] > y_plot[mask]) if n_cell else 0.0
            n_detect = int(detected[mask].sum()) if include_detection else 0
            facecolor = matrix_cmap(0.10 + 0.75 * cell_pct[r, c] / max(cell_pct.max(), 1e-8))
            draw_matrix_cell(
                ax_matrix,
                c,
                r,
                facecolor=facecolor,
                n_cell=n_cell,
                total_n=total_n,
                n_detected=n_detect,
                total_detected=total_detected,
                include_detection=include_detection,
                extra_lines=[
                    (f"{y_label} > {x_label}: {y_better:.1f}%", y_color),
                    (f"{x_label} > {y_label}: {x_better:.1f}%", x_color),
                ],
            )

    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{x_method}_vs_{y_method}_quadrants_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{bundle['cfg']['label']} | {x_label} vs {y_label} quadrants | "
            f"Epoch {epoch} | x={thresholds[0]:.1f}, y={thresholds[1]:.1f}"
        ),
    )


def plot_standard_vs_infobatch(
    bundle: dict,
    epoch: int,
    thresholds: tuple[float, float] = STD_IB_MATRIX_THRESHOLDS,
) -> None:
    plot_method_comparison(bundle, epoch, x_method="std", y_method="ib", thresholds=thresholds)


def plot_standard_vs_infobatch_quadrants(
    bundle: dict,
    epoch: int,
    thresholds: tuple[float, float] = STD_IB_QUADRANT_THRESHOLDS,
) -> None:
    plot_method_comparison_quadrants(bundle, epoch, x_method="std", y_method="ib", thresholds=thresholds)


def plot_method_comparison_hardness(bundle: dict, epoch: int, x_method: str, y_method: str) -> None:
    merged = pairwise_mem_hardness_frame(bundle, epoch, x_method, y_method)
    x_label = METHOD_LABELS[x_method]
    y_label = METHOD_LABELS[y_method]
    x_plot = cap_memorization(merged[f"memorization_{x_method}"].to_numpy())
    y_plot = cap_memorization(merged[f"memorization_{y_method}"].to_numpy())

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for group_key, mask in (
        ("easy", merged["is_easy"].to_numpy()),
        ("hard", merged["is_hard"].to_numpy()),
    ):
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            continue
        ax.scatter(
            x_plot[mask],
            y_plot[mask],
            s=12,
            alpha=0.45,
            color=HARDNESS_COLORS[group_key],
            edgecolors="none",
            label=group_key.capitalize(),
        )

    lower = min(-0.05, float(np.nanmin([x_plot.min(), y_plot.min()])) - 0.01)
    upper = 1.0
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.3, color="black")
    ax.set_xlabel(f"{x_label} memorization score")
    ax.set_ylabel(f"{y_label} memorization score")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.legend(loc="upper left")
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{x_method}_vs_{y_method}_hardness_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=f"{bundle['cfg']['label']} | {x_label} vs {y_label} | Easy vs hard | Epoch {epoch}",
    )


def display_dataset_heading(bundle: dict, level: int = 3) -> None:
    display(Markdown(f"{'#' * level} {bundle['cfg']['label']}"))


def build_all_bundles(active_datasets=None, method_keys=None, prefer_wandb: bool = False) -> dict[str, dict]:
    dataset_keys = list(active_datasets or DATASET_ORDER)
    bundles = {}
    for dataset_key in dataset_keys:
        cfg = DATASETS[dataset_key]
        bundles[dataset_key] = load_lightweight_bundle(
            cfg,
            dataset_key,
            method_keys=method_keys,
            prefer_wandb=prefer_wandb,
        )
    return bundles


def common_mem_epochs(bundle: dict, method_keys: list[str]) -> list[int]:
    epoch_sets = []
    for method_key in method_keys:
        frame = bundle["mem_long"][method_key]
        epoch_sets.append(set(int(epoch) for epoch in frame["epoch"].unique()))
    common = sorted(set.intersection(*epoch_sets))
    if not common:
        raise ValueError(f"No common memorization epochs found across {method_keys} for {bundle['dataset_key']}")
    return common


def all_evolution_epochs(bundle: dict, method_key: str = "std") -> list[int]:
    method_cfg = bundle["cfg"]["methods"][method_key]
    arrays = load_evolution_arrays(bundle["dataset_key"], method_key, method_cfg)
    return sorted(int(epoch) for epoch in np.unique(arrays["epoch"]))


def common_evolution_epochs(bundle: dict, method_keys: list[str]) -> list[int]:
    epoch_sets = [set(all_evolution_epochs(bundle, method_key)) for method_key in method_keys]
    common = sorted(set.intersection(*epoch_sets))
    if not common:
        raise ValueError(f"No common evolution epochs found across {method_keys} for {bundle['dataset_key']}")
    return common


def build_std_loss_anchor(bundle: dict, anchor_epoch: int = 10, n_bins: int = 10) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    std_cfg = bundle["cfg"]["methods"]["std"]
    std_evo = evolution_epoch_frame(bundle["dataset_key"], "std", std_cfg, anchor_epoch)
    anchor = std_evo[["sample_id", "loss"]].rename(columns={"loss": "anchor_loss"}).dropna().copy()

    loss_min = float(anchor["anchor_loss"].min())
    loss_max = float(anchor["anchor_loss"].max())
    if np.isclose(loss_min, loss_max):
        loss_max = loss_min + 1e-8

    edges = np.linspace(loss_min, loss_max, n_bins + 1, dtype=np.float64)
    edges[-1] = np.nextafter(edges[-1], np.inf)

    anchor["bin_idx"] = pd.cut(
        anchor["anchor_loss"],
        bins=edges,
        labels=False,
        include_lowest=True,
    ).astype(int)
    anchor["bin_num"] = anchor["bin_idx"] + 1
    bin_labels = [f"{edges[i]:.3f}-{edges[i + 1]:.3f}" for i in range(n_bins)]
    return anchor.sort_values("sample_id").reset_index(drop=True), edges, bin_labels


def summarize_fixed_bins(
    bundle: dict,
    anchor_frame: pd.DataFrame,
    method_keys: list[str],
    epochs: list[int],
    *,
    source: str = "memorization",
    threshold: float = 0.1,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for method_key in method_keys:
        method_cfg = bundle["cfg"]["methods"][method_key]
        for epoch in epochs:
            if source in {"memorization", "memorized_fraction"}:
                frame = bundle["mem_long"][method_key]
                source_frame = (
                    frame.loc[frame["epoch"] == epoch, ["sample_id", "memorization"]]
                    .sort_values("sample_id")
                    .reset_index(drop=True)
                )
                if source_frame.empty:
                    continue
                source_frame = source_frame.rename(columns={"memorization": "value"})
            elif source == "loss":
                evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
                source_frame = evo[["sample_id", "loss"]].rename(columns={"loss": "value"})
            elif source == "pruning":
                evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
                pruning_col, _ = resolve_pruning_metric(bundle, method_key, evo)
                if pruning_col is None:
                    continue
                source_frame = evo[["sample_id", pruning_col]].rename(columns={pruning_col: "value"})
            else:
                raise ValueError(f"Unknown source={source}")

            merged = strict_sample_merge(
                anchor_frame[["sample_id", "bin_num"]],
                source_frame,
                context=f"{bundle['dataset_key']}::{method_key}::{source}::fixed_bins::epoch_{epoch}",
            ).dropna(subset=["value"]).copy()

            if source == "memorized_fraction":
                merged["metric"] = (merged["value"] > threshold).astype(float)
            else:
                merged["metric"] = merged["value"]

            summary = (
                merged.groupby("bin_num", sort=True)
                .agg(metric_mean=("metric", "mean"), n=("sample_id", "size"))
                .reindex(range(1, n_bins + 1))
                .reset_index()
            )
            summary["method_key"] = method_key
            summary["method"] = method_cfg["label"]
            summary["epoch"] = int(epoch)
            summary["source"] = source
            rows.append(summary)

    if not rows:
        raise ValueError(f"No summaries were built for source={source} on {bundle['dataset_key']}")
    return pd.concat(rows, ignore_index=True)


def summarize_fixed_bins_from_evolution(
    bundle: dict,
    anchor_frame: pd.DataFrame,
    method_keys: list[str],
    epochs: list[int],
    field: str,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for method_key in method_keys:
        method_cfg = bundle["cfg"]["methods"][method_key]
        for epoch in epochs:
            evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
            if field not in evo.columns or evo[field].isna().all():
                continue

            merged = strict_sample_merge(
                anchor_frame[["sample_id", "bin_num"]],
                evo[["sample_id", field]].rename(columns={field: "value"}),
                context=f"{bundle['dataset_key']}::{method_key}::{field}::fixed_bins::epoch_{epoch}",
            ).dropna(subset=["value"]).copy()

            summary = (
                merged.groupby("bin_num", sort=True)
                .agg(metric_mean=("value", "mean"), n=("sample_id", "size"))
                .reindex(range(1, n_bins + 1))
                .reset_index()
            )
            summary["method_key"] = method_key
            summary["method"] = method_cfg["label"]
            summary["epoch"] = int(epoch)
            summary["source"] = field
            rows.append(summary)

    if not rows:
        return pd.DataFrame(columns=["bin_num", "metric_mean", "n", "method_key", "method", "epoch", "source"])
    return pd.concat(rows, ignore_index=True)


def evolution_field_availability_report(bundle: dict, method_keys: list[str] | None = None) -> pd.DataFrame:
    method_keys = method_keys or list(bundle["method_keys"])
    rows = []
    for method_key in method_keys:
        method_cfg = bundle["cfg"]["methods"][method_key]
        schema = raw_evolution_schema(method_cfg)
        rows.append(
            {
                "dataset": bundle["cfg"]["label"],
                "method_key": method_key,
                "method": method_cfg["label"],
                "has_correct": bool(schema["has_correct"]),
                "pruning_field": schema["pruning_field"] or "none",
                "pruning_label": schema["pruning_label"],
                "record_keys": ", ".join(schema["record_keys"]),
                "frac_keys": ", ".join(schema["frac_keys"]),
            }
        )
    return pd.DataFrame(rows)


def plot_fixed_bin_trajectories(
    summary_df: pd.DataFrame,
    bundle: dict,
    anchor_frame: pd.DataFrame,
    bin_labels: list[str],
    focus_bins: list[int],
    *,
    ylabel: str,
    stem: str,
    figure_label: str,
    method_keys: list[str],
    ncols: int = 5,
) -> None:
    ncols = max(1, min(ncols, len(focus_bins)))
    nrows = int(np.ceil(len(focus_bins) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, bin_num in zip(axes, focus_bins):
        n_bin = int((anchor_frame["bin_num"] == bin_num).sum())
        sub = summary_df.loc[summary_df["bin_num"] == bin_num].copy()

        for method_key in method_keys:
            method_sub = sub.loc[sub["method_key"] == method_key].sort_values("epoch")
            if method_sub.empty:
                continue
            ax.plot(
                method_sub["epoch"],
                method_sub["metric_mean"],
                marker="o",
                markersize=7,
                linewidth=2.3,
                color=bundle["cfg"]["methods"][method_key]["color"],
                label=bundle["cfg"]["methods"][method_key]["label"],
            )

        ax.set_title(f"Bin {bin_num} | n={n_bin:,}\n{bin_labels[bin_num - 1]}", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes[len(focus_bins):]:
        ax.axis("off")

    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, loc="best")
    finish_figure(fig, stem, bundle["dataset_key"], figure_label=figure_label)


def plot_metric_histograms(bundle: dict, epoch: int, metric: str, method_keys: list[str] | None = None) -> None:
    if metric not in {"loss", "entropy"}:
        raise ValueError("metric must be 'loss' or 'entropy'")
    method_keys = method_keys or list(bundle["method_keys"])
    fig, axes = plt.subplots(1, len(method_keys), figsize=(5.2 * len(method_keys), 4.8), sharey=True)
    if len(method_keys) == 1:
        axes = [axes]

    for ax, method_key in zip(axes, method_keys):
        method_cfg = bundle["cfg"]["methods"][method_key]
        evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
        values = evo[metric].dropna().to_numpy(dtype=np.float64)
        if len(values) == 0:
            ax.text(0.5, 0.5, "No values", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.hist(
            values,
            bins=70,
            color=method_cfg["color"],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.25,
        )

        mean_value = float(np.mean(values))
        ax.axvline(
            mean_value,
            color="#111827",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {mean_value:.2f}",
        )
        ax.set_title(method_cfg["label"])
        ax.set_xlabel(metric.capitalize())
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper right")

    axes[0].set_ylabel("Samples")
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{metric}_histograms_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=f"{bundle['cfg']['label']} | {metric.capitalize()} histograms | Epoch {epoch}",
    )


def theoretical_max_entropy(bundle: dict) -> float:
    return float(np.log(float(bundle["cfg"]["n_classes"])))


def normalized_entropy_percent_formatter():
    return mpl.ticker.PercentFormatter(xmax=1.0, decimals=0)


def difficulty_bin_color(bin_num: int) -> str:
    return DIFFICULTY_BIN_COLORS.get(int(bin_num), "#475569")


def difficulty_bin_counts(summary_df: pd.DataFrame, n_bins: int) -> pd.Series:
    if summary_df.empty or "n" not in summary_df:
        return pd.Series(0, index=range(1, n_bins + 1), dtype=np.int64)
    counts = summary_df.groupby("bin_num", sort=True)["n"].max()
    return counts.reindex(range(1, n_bins + 1), fill_value=0).astype(np.int64)


def difficulty_bin_xticklabels(summary_df: pd.DataFrame, n_bins: int) -> list[str]:
    counts = difficulty_bin_counts(summary_df, n_bins)
    labels = []
    for bin_num in range(1, n_bins + 1):
        parts = [str(bin_num)]
        if bin_num == 1:
            parts.extend(["high prune", "easy"])
        elif bin_num == n_bins:
            parts.extend(["low prune", "hard"])
        parts.append(f"n={int(counts.loc[bin_num]):,}")
        labels.append("\n".join(parts))
    return labels


def infobatch_pruning_interval_labels(edges: np.ndarray) -> list[str]:
    edges = np.asarray(edges, dtype=np.float64)
    n_bins = len(edges) - 1
    display_edges = edges.copy()
    display_edges[-1] = np.nextafter(display_edges[-1], -np.inf)
    labels = []
    for bin_num in range(1, n_bins + 1):
        low_to_high_idx = n_bins - bin_num
        labels.append(f"{display_edges[low_to_high_idx]:.3f}-{display_edges[low_to_high_idx + 1]:.3f}")
    return labels


def infobatch_anchor_method_key(bundle: dict) -> str:
    for method_key in ("ib", "infobatch"):
        if method_key in bundle["method_keys"]:
            return method_key
    raise KeyError(f"No InfoBatch method is available for {bundle['dataset_key']}")


def build_infobatch_pruning_interval_anchor(
    bundle: dict,
    *,
    anchor_epoch: int,
    n_bins: int = 5,
    anchor_method_key: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, str, str]:
    anchor_method_key = anchor_method_key or infobatch_anchor_method_key(bundle)
    method_cfg = bundle["cfg"]["methods"][anchor_method_key]
    evo = evolution_epoch_frame(bundle["dataset_key"], anchor_method_key, method_cfg, anchor_epoch)
    pruning_col, pruning_label = resolve_pruning_metric(bundle, anchor_method_key, evo)
    if pruning_col is None:
        raise KeyError(
            f"No InfoBatch pruning metric found for {bundle['dataset_key']}::{anchor_method_key} "
            f"at epoch {anchor_epoch}"
        )

    anchor = (
        evo[["sample_id", pruning_col]]
        .rename(columns={pruning_col: "anchor_pruning_rate"})
        .dropna(subset=["anchor_pruning_rate"])
        .copy()
    )
    pruning_min = float(anchor["anchor_pruning_rate"].min())
    pruning_max = float(anchor["anchor_pruning_rate"].max())
    if np.isclose(pruning_min, pruning_max):
        pruning_max = pruning_min + 1e-8

    edges = np.linspace(pruning_min, pruning_max, n_bins + 1, dtype=np.float64)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    low_to_high = pd.cut(
        anchor["anchor_pruning_rate"],
        bins=edges,
        labels=False,
        include_lowest=True,
    ).astype(int)

    # Difficulty increases as InfoBatch pruning decreases:
    # bin 1 = highest pruning/easiest, bin n = lowest pruning/hardest.
    anchor["bin_num"] = n_bins - low_to_high
    anchor["anchor_method_key"] = anchor_method_key
    anchor["anchor_pruning_metric"] = pruning_col
    return (
        anchor.sort_values("sample_id").reset_index(drop=True),
        edges,
        pruning_col,
        pruning_label,
    )


def build_entropy_pruning_difficulty_bin_summary(
    bundle: dict,
    epoch: int,
    *,
    method_keys: list[str] | None = None,
    anchor_epoch: int | None = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    method_keys = method_keys or list(bundle["method_keys"])
    anchor_epoch = int(epoch if anchor_epoch is None else anchor_epoch)
    anchor_frame, edges, pruning_col, pruning_label = build_infobatch_pruning_interval_anchor(
        bundle,
        anchor_epoch=anchor_epoch,
        n_bins=n_bins,
    )
    max_entropy = theoretical_max_entropy(bundle)
    rows = []

    for method_key in method_keys:
        method_cfg = bundle["cfg"]["methods"][method_key]
        evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
        if "entropy" not in evo.columns or evo["entropy"].isna().all():
            continue

        merged = strict_sample_merge(
            anchor_frame[["sample_id", "bin_num", "anchor_pruning_rate"]],
            evo[["sample_id", "entropy"]],
            context=f"{bundle['dataset_key']}::{method_key}::entropy_story::epoch_{epoch}",
        ).dropna(subset=["entropy"])

        summary = (
            merged.groupby("bin_num", sort=True)
            .agg(
                entropy_mean=("entropy", "mean"),
                entropy_std=("entropy", "std"),
                n=("sample_id", "size"),
            )
            .reindex(range(1, n_bins + 1))
            .reset_index()
        )
        summary["entropy_sem"] = summary["entropy_std"] / np.sqrt(summary["n"].clip(lower=1))
        summary["entropy_mean_norm"] = summary["entropy_mean"] / max_entropy
        summary["entropy_std_norm"] = summary["entropy_std"] / max_entropy
        summary["entropy_sem_norm"] = summary["entropy_sem"] / max_entropy
        summary["dataset_key"] = bundle["dataset_key"]
        summary["dataset"] = bundle["cfg"]["label"]
        summary["n_classes"] = int(bundle["cfg"]["n_classes"])
        summary["max_entropy"] = max_entropy
        summary["method_key"] = method_key
        summary["method"] = method_cfg["label"]
        summary["epoch"] = int(epoch)
        summary["anchor_epoch"] = int(anchor_epoch)
        summary["anchor_method_key"] = str(anchor_frame["anchor_method_key"].iloc[0])
        summary["anchor_pruning_metric"] = pruning_col
        summary["anchor_pruning_label"] = pruning_label
        summary["anchor_pruning_min"] = float(edges[0])
        summary["anchor_pruning_max"] = float(edges[-1])
        rows.append(summary)

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset",
                "method_key",
                "method",
                "epoch",
                "bin_num",
                "entropy_mean",
                "entropy_std",
                "entropy_sem",
                "n",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _weighted_entropy_mean(frame: pd.DataFrame, value_col: str = "entropy_mean") -> tuple[float, int]:
    frame = frame.dropna(subset=[value_col, "n"])
    if frame.empty:
        return float("nan"), 0
    n = frame["n"].to_numpy(dtype=np.float64)
    values = frame[value_col].to_numpy(dtype=np.float64)
    return float(np.average(values, weights=n)), int(n.sum())


def _bin_span_label(bins: tuple[int, ...]) -> str:
    bins = tuple(int(b) for b in bins)
    if not bins:
        return ""
    if len(bins) == 1 or min(bins) == max(bins):
        return str(min(bins))
    return f"{min(bins)}-{max(bins)}"


def build_cross_dataset_entropy_story_tables(
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
    epoch_by_dataset: dict[str, int] | None = None,
    anchor_epoch: int | None = None,
    n_bins: int = 5,
    easy_bins: tuple[int, ...] = (1,),
    hard_bins: tuple[int, ...] = (5,),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decile_frames = []
    story_rows = []

    for dataset_key in DATASET_ORDER:
        if dataset_key not in bundles:
            continue
        bundle = bundles[dataset_key]
        active_methods = method_keys or list(bundle["method_keys"])
        active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
        epoch = int((epoch_by_dataset or {}).get(dataset_key, bundle["cfg"]["end_epoch"]))
        active_anchor_epoch = int(epoch if anchor_epoch is None else anchor_epoch)

        deciles = build_entropy_pruning_difficulty_bin_summary(
            bundle,
            epoch,
            method_keys=active_methods,
            anchor_epoch=active_anchor_epoch,
            n_bins=n_bins,
        )
        if deciles.empty:
            continue
        decile_frames.append(deciles)

        zone_means = {}
        for method_key in active_methods:
            method_deciles = deciles.loc[deciles["method_key"] == method_key]
            easy_mean, easy_n = _weighted_entropy_mean(method_deciles.loc[method_deciles["bin_num"].isin(easy_bins)])
            hard_mean, hard_n = _weighted_entropy_mean(method_deciles.loc[method_deciles["bin_num"].isin(hard_bins)])
            easy_norm, _ = _weighted_entropy_mean(method_deciles.loc[method_deciles["bin_num"].isin(easy_bins)], "entropy_mean_norm")
            hard_norm, _ = _weighted_entropy_mean(method_deciles.loc[method_deciles["bin_num"].isin(hard_bins)], "entropy_mean_norm")
            zone_means[method_key] = {
                "easy_entropy": easy_mean,
                "hard_entropy": hard_mean,
                "easy_entropy_norm": easy_norm,
                "hard_entropy_norm": hard_norm,
                "easy_n": easy_n,
                "hard_n": hard_n,
            }

        std_hard = zone_means.get("std", {}).get("hard_entropy", np.nan)
        std_hard_norm = zone_means.get("std", {}).get("hard_entropy_norm", np.nan)
        for method_key in active_methods:
            values = zone_means[method_key]
            story_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "n_classes": int(bundle["cfg"]["n_classes"]),
                    "max_entropy": float(deciles["max_entropy"].iloc[0]),
                    "method_key": method_key,
                    "method": bundle["cfg"]["methods"][method_key]["label"],
                    "epoch": epoch,
                    "anchor_epoch": active_anchor_epoch,
                    "anchor_method": bundle["cfg"]["methods"][deciles["anchor_method_key"].iloc[0]]["label"],
                    "anchor_pruning_metric": deciles["anchor_pruning_metric"].iloc[0],
                    "easy_bins": _bin_span_label(easy_bins),
                    "hard_bins": _bin_span_label(hard_bins),
                    "easy_entropy": values["easy_entropy"],
                    "hard_entropy": values["hard_entropy"],
                    "hard_minus_easy_entropy": values["hard_entropy"] - values["easy_entropy"],
                    "std_minus_method_hard_entropy": std_hard - values["hard_entropy"],
                    "easy_entropy_pct_max": 100.0 * values["easy_entropy_norm"],
                    "hard_entropy_pct_max": 100.0 * values["hard_entropy_norm"],
                    "hard_minus_easy_pct_points": 100.0 * (values["hard_entropy_norm"] - values["easy_entropy_norm"]),
                    "std_minus_method_hard_pct_points": 100.0 * (std_hard_norm - values["hard_entropy_norm"]),
                    "easy_n": values["easy_n"],
                    "hard_n": values["hard_n"],
                }
            )

    decile_df = pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    story_df = pd.DataFrame(story_rows)
    return decile_df, story_df


def plot_cross_dataset_entropy_difficulty_story(
    decile_df: pd.DataFrame,
    story_df: pd.DataFrame,
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
    hard_bins: tuple[int, ...] = (5,),
) -> None:
    if decile_df.empty or story_df.empty:
        display_note("**Entropy story plot skipped:** no entropy summary rows were available.")
        return

    dataset_keys = [dataset_key for dataset_key in DATASET_ORDER if dataset_key in decile_df["dataset_key"].unique()]
    n_datasets = len(dataset_keys)
    if n_datasets == 0:
        display_note("**Entropy story plot skipped:** no configured datasets were present.")
        return

    fig, axes = plt.subplots(1, n_datasets, figsize=(5.9 * n_datasets, 5.4), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, dataset_key in zip(axes, dataset_keys):
        bundle = bundles[dataset_key]
        active_methods = method_keys or list(bundle["method_keys"])
        active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
        dataset_deciles = decile_df.loc[decile_df["dataset_key"] == dataset_key]
        dataset_story = story_df.loc[story_df["dataset_key"] == dataset_key]
        epoch = int(dataset_deciles["epoch"].iloc[0])
        anchor_method = str(dataset_story["anchor_method"].iloc[0]) if "anchor_method" in dataset_story else "InfoBatch"
        anchor_epoch = int(dataset_deciles["anchor_epoch"].iloc[0])
        n_bins = int(dataset_deciles["bin_num"].max())
        n_classes = int(dataset_deciles["n_classes"].iloc[0])
        max_entropy = float(dataset_deciles["max_entropy"].iloc[0])

        ax.axvspan(
            min(hard_bins) - 0.5,
            max(hard_bins) + 0.5,
            color="#E45756",
            alpha=0.07,
            linewidth=0,
        )
        ax.axhline(max_entropy, color="#64748B", linestyle=":", linewidth=1.2)

        gap_lines = []
        for method_key in active_methods:
            method_deciles = dataset_deciles.loc[dataset_deciles["method_key"] == method_key].sort_values("bin_num")
            if method_deciles.empty:
                continue
            method_cfg = bundle["cfg"]["methods"][method_key]
            ax.plot(
                method_deciles["bin_num"],
                method_deciles["entropy_mean"],
                marker="o",
                markersize=5.5,
                linewidth=2.3,
                color=method_cfg["color"],
                label=method_cfg["label"],
            )

            story_row = dataset_story.loc[dataset_story["method_key"] == method_key]
            if not story_row.empty:
                gap = float(story_row["hard_minus_easy_entropy"].iloc[0])
                gap_lines.append(f"{method_cfg['label']}: {gap:+.3f}")

        ax.set_title(
            f"{bundle['cfg']['label']} | epoch {epoch}\nmax entropy = log({n_classes}) = {max_entropy:.2f}",
            fontsize=12.5,
            pad=8,
        )
        ax.set_xlabel("Difficulty bin fixed from InfoBatch pruning rate")
        xticks = list(range(1, n_bins + 1))
        ax.set_xticks(xticks)
        ax.set_xticklabels(difficulty_bin_xticklabels(dataset_deciles, n_bins))
        ax.set_ylim(0.0, max_entropy * 1.05)
        ax.set_ylabel("Mean entropy")
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(
            0.03,
            0.97,
            f"100% = log({n_classes})\nHard - easy entropy\n" + "\n".join(gap_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.88},
        )

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(len(legend_handles), 4),
            frameon=False,
        )

    finish_figure(
        fig,
        "cross_dataset_normalized_entropy_difficulty_story",
        "cross_dataset",
        figure_label=(
            "Entropy in raw nats with the theoretical maximum log(num classes) marked; "
            f"with difficulty fixed from {anchor_method} pruning-rate bins"
        ),
        tight_rect=(0.0, 0.0, 1.0, 0.90),
    )


def build_memorization_pruning_difficulty_bin_summary(
    bundle: dict,
    epoch: int,
    *,
    method_keys: list[str] | None = None,
    anchor_epoch: int | None = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    method_keys = method_keys or list(bundle["method_keys"])
    anchor_epoch = int(epoch if anchor_epoch is None else anchor_epoch)
    anchor_frame, edges, pruning_col, pruning_label = build_infobatch_pruning_interval_anchor(
        bundle,
        anchor_epoch=anchor_epoch,
        n_bins=n_bins,
    )
    rows = []

    for method_key in method_keys:
        method_cfg = bundle["cfg"]["methods"][method_key]
        mem = get_method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]].copy()
        mem["memorization"] = cap_memorization(mem["memorization"].to_numpy(dtype=np.float64))
        merged = strict_sample_merge(
            anchor_frame[["sample_id", "bin_num", "anchor_pruning_rate"]],
            mem,
            context=f"{bundle['dataset_key']}::{method_key}::memorization_pruning_story::epoch_{epoch}",
        ).dropna(subset=["memorization"])

        summary = (
            merged.groupby("bin_num", sort=True)
            .agg(
                memorization_mean=("memorization", "mean"),
                memorization_std=("memorization", "std"),
                n=("sample_id", "size"),
            )
            .reindex(range(1, n_bins + 1))
            .reset_index()
        )
        summary["memorization_sem"] = summary["memorization_std"] / np.sqrt(summary["n"].clip(lower=1))
        summary["dataset_key"] = bundle["dataset_key"]
        summary["dataset"] = bundle["cfg"]["label"]
        summary["method_key"] = method_key
        summary["method"] = method_cfg["label"]
        summary["epoch"] = int(epoch)
        summary["anchor_epoch"] = int(anchor_epoch)
        summary["anchor_method_key"] = str(anchor_frame["anchor_method_key"].iloc[0])
        summary["anchor_pruning_metric"] = pruning_col
        summary["anchor_pruning_label"] = pruning_label
        summary["anchor_pruning_min"] = float(edges[0])
        summary["anchor_pruning_max"] = float(edges[-1])
        rows.append(summary)

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset",
                "method_key",
                "method",
                "epoch",
                "bin_num",
                "memorization_mean",
                "memorization_std",
                "memorization_sem",
                "n",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def build_cross_dataset_memorization_story_tables(
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
    epoch_by_dataset: dict[str, int] | None = None,
    anchor_epoch: int | None = None,
    n_bins: int = 5,
    easy_bins: tuple[int, ...] = (1,),
    hard_bins: tuple[int, ...] = (5,),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_frames = []
    story_rows = []

    for dataset_key in DATASET_ORDER:
        if dataset_key not in bundles:
            continue
        bundle = bundles[dataset_key]
        active_methods = method_keys or list(bundle["method_keys"])
        active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
        epoch = int((epoch_by_dataset or {}).get(dataset_key, bundle["cfg"]["end_epoch"]))
        active_anchor_epoch = int(epoch if anchor_epoch is None else anchor_epoch)

        bins = build_memorization_pruning_difficulty_bin_summary(
            bundle,
            epoch,
            method_keys=active_methods,
            anchor_epoch=active_anchor_epoch,
            n_bins=n_bins,
        )
        if bins.empty:
            continue
        bin_frames.append(bins)

        zone_means = {}
        for method_key in active_methods:
            method_bins = bins.loc[bins["method_key"] == method_key]
            easy_mean, easy_n = _weighted_entropy_mean(method_bins.loc[method_bins["bin_num"].isin(easy_bins)], "memorization_mean")
            hard_mean, hard_n = _weighted_entropy_mean(method_bins.loc[method_bins["bin_num"].isin(hard_bins)], "memorization_mean")
            zone_means[method_key] = {
                "easy_memorization": easy_mean,
                "hard_memorization": hard_mean,
                "easy_n": easy_n,
                "hard_n": hard_n,
            }

        std_hard = zone_means.get("std", {}).get("hard_memorization", np.nan)
        for method_key in active_methods:
            values = zone_means[method_key]
            story_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": bundle["cfg"]["methods"][method_key]["label"],
                    "epoch": epoch,
                    "anchor_epoch": active_anchor_epoch,
                    "anchor_method": bundle["cfg"]["methods"][bins["anchor_method_key"].iloc[0]]["label"],
                    "anchor_pruning_metric": bins["anchor_pruning_metric"].iloc[0],
                    "easy_bins": _bin_span_label(easy_bins),
                    "hard_bins": _bin_span_label(hard_bins),
                    "easy_memorization": values["easy_memorization"],
                    "hard_memorization": values["hard_memorization"],
                    "hard_minus_easy_memorization": values["hard_memorization"] - values["easy_memorization"],
                    "std_minus_method_hard_memorization": std_hard - values["hard_memorization"],
                    "easy_n": values["easy_n"],
                    "hard_n": values["hard_n"],
                }
            )

    bin_df = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    story_df = pd.DataFrame(story_rows)
    return bin_df, story_df


def plot_cross_dataset_memorization_difficulty_story(
    bin_df: pd.DataFrame,
    story_df: pd.DataFrame,
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
    hard_bins: tuple[int, ...] = (5,),
) -> None:
    if bin_df.empty or story_df.empty:
        display_note("**Memorization story plot skipped:** no memorization summary rows were available.")
        return

    dataset_keys = [dataset_key for dataset_key in DATASET_ORDER if dataset_key in bin_df["dataset_key"].unique()]
    n_datasets = len(dataset_keys)
    if n_datasets == 0:
        display_note("**Memorization story plot skipped:** no configured datasets were present.")
        return

    fig, axes = plt.subplots(1, n_datasets, figsize=(5.9 * n_datasets, 5.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, dataset_key in zip(axes, dataset_keys):
        bundle = bundles[dataset_key]
        active_methods = method_keys or list(bundle["method_keys"])
        active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
        dataset_bins = bin_df.loc[bin_df["dataset_key"] == dataset_key]
        dataset_story = story_df.loc[story_df["dataset_key"] == dataset_key]
        epoch = int(dataset_bins["epoch"].iloc[0])
        anchor_method = str(dataset_story["anchor_method"].iloc[0]) if "anchor_method" in dataset_story else "InfoBatch"
        anchor_epoch = int(dataset_bins["anchor_epoch"].iloc[0])
        n_bins = int(dataset_bins["bin_num"].max())

        ax.axvspan(
            min(hard_bins) - 0.5,
            max(hard_bins) + 0.5,
            color="#E45756",
            alpha=0.07,
            linewidth=0,
        )

        gap_lines = []
        for method_key in active_methods:
            method_bins = dataset_bins.loc[dataset_bins["method_key"] == method_key].sort_values("bin_num")
            if method_bins.empty:
                continue
            method_cfg = bundle["cfg"]["methods"][method_key]
            ax.plot(
                method_bins["bin_num"],
                method_bins["memorization_mean"],
                marker="o",
                markersize=5.5,
                linewidth=2.3,
                color=method_cfg["color"],
                label=method_cfg["label"],
            )

            story_row = dataset_story.loc[dataset_story["method_key"] == method_key]
            if not story_row.empty:
                gap = float(story_row["hard_minus_easy_memorization"].iloc[0])
                gap_lines.append(f"{method_cfg['label']}: {gap:+.3f}")

        ax.set_title(
            f"{bundle['cfg']['label']} | epoch {epoch}\n{anchor_method} pruning anchor {anchor_epoch}",
            fontsize=12.5,
            pad=8,
        )
        ax.set_xlabel("Difficulty bin fixed from InfoBatch pruning rate")
        xticks = list(range(1, n_bins + 1))
        ax.set_xticks(xticks)
        ax.set_xticklabels(difficulty_bin_xticklabels(dataset_bins, n_bins))
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Mean memorization score")
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(
            0.03,
            0.97,
            "Hard - easy\nmemorization\n" + "\n".join(gap_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.88},
        )

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(len(legend_handles), 4),
            frameon=False,
        )

    finish_figure(
        fig,
        "cross_dataset_memorization_difficulty_story",
        "cross_dataset",
        figure_label=(
            "Memorization by InfoBatch-pruning difficulty bins, "
            "with y-axis fixed to [0, 1]"
        ),
        tight_rect=(0.0, 0.0, 1.0, 0.90),
    )


def build_loss_pruning_difficulty_bin_summary(
    bundle: dict,
    epoch: int,
    *,
    method_keys: list[str] | None = None,
    anchor_epoch: int | None = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    method_keys = method_keys or list(bundle["method_keys"])
    anchor_epoch = int(epoch if anchor_epoch is None else anchor_epoch)
    anchor_frame, edges, pruning_col, pruning_label = build_infobatch_pruning_interval_anchor(
        bundle,
        anchor_epoch=anchor_epoch,
        n_bins=n_bins,
    )
    random_guess_loss = theoretical_max_entropy(bundle)
    rows = []

    for method_key in method_keys:
        method_cfg = bundle["cfg"]["methods"][method_key]
        evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
        if "loss" not in evo.columns or evo["loss"].isna().all():
            continue

        merged = strict_sample_merge(
            anchor_frame[["sample_id", "bin_num", "anchor_pruning_rate"]],
            evo[["sample_id", "loss"]],
            context=f"{bundle['dataset_key']}::{method_key}::loss_pruning_story::epoch_{epoch}",
        ).dropna(subset=["loss"])

        summary = (
            merged.groupby("bin_num", sort=True)
            .agg(
                loss_mean=("loss", "mean"),
                loss_std=("loss", "std"),
                n=("sample_id", "size"),
            )
            .reindex(range(1, n_bins + 1))
            .reset_index()
        )
        summary["loss_sem"] = summary["loss_std"] / np.sqrt(summary["n"].clip(lower=1))
        summary["loss_mean_norm"] = summary["loss_mean"] / random_guess_loss
        summary["loss_std_norm"] = summary["loss_std"] / random_guess_loss
        summary["loss_sem_norm"] = summary["loss_sem"] / random_guess_loss
        summary["dataset_key"] = bundle["dataset_key"]
        summary["dataset"] = bundle["cfg"]["label"]
        summary["n_classes"] = int(bundle["cfg"]["n_classes"])
        summary["random_guess_loss"] = random_guess_loss
        summary["method_key"] = method_key
        summary["method"] = method_cfg["label"]
        summary["epoch"] = int(epoch)
        summary["anchor_epoch"] = int(anchor_epoch)
        summary["anchor_method_key"] = str(anchor_frame["anchor_method_key"].iloc[0])
        summary["anchor_pruning_metric"] = pruning_col
        summary["anchor_pruning_label"] = pruning_label
        summary["anchor_pruning_min"] = float(edges[0])
        summary["anchor_pruning_max"] = float(edges[-1])
        rows.append(summary)

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset",
                "method_key",
                "method",
                "epoch",
                "bin_num",
                "loss_mean",
                "loss_std",
                "loss_sem",
                "n",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def build_cross_dataset_loss_story_tables(
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
    epoch_by_dataset: dict[str, int] | None = None,
    anchor_epoch: int | None = None,
    n_bins: int = 5,
    easy_bins: tuple[int, ...] = (1,),
    hard_bins: tuple[int, ...] = (5,),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bin_frames = []
    story_rows = []

    for dataset_key in DATASET_ORDER:
        if dataset_key not in bundles:
            continue
        bundle = bundles[dataset_key]
        active_methods = method_keys or list(bundle["method_keys"])
        active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
        epoch = int((epoch_by_dataset or {}).get(dataset_key, bundle["cfg"]["end_epoch"]))
        active_anchor_epoch = int(epoch if anchor_epoch is None else anchor_epoch)

        bins = build_loss_pruning_difficulty_bin_summary(
            bundle,
            epoch,
            method_keys=active_methods,
            anchor_epoch=active_anchor_epoch,
            n_bins=n_bins,
        )
        if bins.empty:
            continue
        bin_frames.append(bins)

        zone_means = {}
        for method_key in active_methods:
            method_bins = bins.loc[bins["method_key"] == method_key]
            easy_mean, easy_n = _weighted_entropy_mean(method_bins.loc[method_bins["bin_num"].isin(easy_bins)], "loss_mean")
            hard_mean, hard_n = _weighted_entropy_mean(method_bins.loc[method_bins["bin_num"].isin(hard_bins)], "loss_mean")
            easy_norm, _ = _weighted_entropy_mean(method_bins.loc[method_bins["bin_num"].isin(easy_bins)], "loss_mean_norm")
            hard_norm, _ = _weighted_entropy_mean(method_bins.loc[method_bins["bin_num"].isin(hard_bins)], "loss_mean_norm")
            zone_means[method_key] = {
                "easy_loss": easy_mean,
                "hard_loss": hard_mean,
                "easy_loss_norm": easy_norm,
                "hard_loss_norm": hard_norm,
                "easy_n": easy_n,
                "hard_n": hard_n,
            }

        std_hard = zone_means.get("std", {}).get("hard_loss", np.nan)
        std_hard_norm = zone_means.get("std", {}).get("hard_loss_norm", np.nan)
        for method_key in active_methods:
            values = zone_means[method_key]
            story_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "n_classes": int(bundle["cfg"]["n_classes"]),
                    "random_guess_loss": float(bins["random_guess_loss"].iloc[0]),
                    "method_key": method_key,
                    "method": bundle["cfg"]["methods"][method_key]["label"],
                    "epoch": epoch,
                    "anchor_epoch": active_anchor_epoch,
                    "anchor_method": bundle["cfg"]["methods"][bins["anchor_method_key"].iloc[0]]["label"],
                    "anchor_pruning_metric": bins["anchor_pruning_metric"].iloc[0],
                    "easy_bins": _bin_span_label(easy_bins),
                    "hard_bins": _bin_span_label(hard_bins),
                    "easy_loss": values["easy_loss"],
                    "hard_loss": values["hard_loss"],
                    "hard_minus_easy_loss": values["hard_loss"] - values["easy_loss"],
                    "std_minus_method_hard_loss": std_hard - values["hard_loss"],
                    "easy_loss_pct_log_classes": 100.0 * values["easy_loss_norm"],
                    "hard_loss_pct_log_classes": 100.0 * values["hard_loss_norm"],
                    "hard_minus_easy_loss_pct_points": 100.0 * (values["hard_loss_norm"] - values["easy_loss_norm"]),
                    "std_minus_method_hard_loss_pct_points": 100.0 * (std_hard_norm - values["hard_loss_norm"]),
                    "easy_n": values["easy_n"],
                    "hard_n": values["hard_n"],
                }
            )

    bin_df = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    story_df = pd.DataFrame(story_rows)
    return bin_df, story_df


def plot_cross_dataset_loss_difficulty_story(
    bin_df: pd.DataFrame,
    story_df: pd.DataFrame,
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
    hard_bins: tuple[int, ...] = (5,),
) -> None:
    if bin_df.empty or story_df.empty:
        display_note("**Loss story plot skipped:** no loss summary rows were available.")
        return

    dataset_keys = [dataset_key for dataset_key in DATASET_ORDER if dataset_key in bin_df["dataset_key"].unique()]
    n_datasets = len(dataset_keys)
    if n_datasets == 0:
        display_note("**Loss story plot skipped:** no configured datasets were present.")
        return

    fig, axes = plt.subplots(1, n_datasets, figsize=(5.9 * n_datasets, 5.4), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, dataset_key in zip(axes, dataset_keys):
        bundle = bundles[dataset_key]
        active_methods = method_keys or list(bundle["method_keys"])
        active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
        dataset_bins = bin_df.loc[bin_df["dataset_key"] == dataset_key]
        dataset_story = story_df.loc[story_df["dataset_key"] == dataset_key]
        epoch = int(dataset_bins["epoch"].iloc[0])
        anchor_method = str(dataset_story["anchor_method"].iloc[0]) if "anchor_method" in dataset_story else "InfoBatch"
        anchor_epoch = int(dataset_bins["anchor_epoch"].iloc[0])
        n_bins = int(dataset_bins["bin_num"].max())
        n_classes = int(dataset_bins["n_classes"].iloc[0])
        random_guess_loss = float(dataset_bins["random_guess_loss"].iloc[0])
        y_top = max(1e-6, float(np.nanmax(dataset_bins["loss_mean"].to_numpy(dtype=np.float64)))) * 1.15

        ax.axvspan(
            min(hard_bins) - 0.5,
            max(hard_bins) + 0.5,
            color="#E45756",
            alpha=0.07,
            linewidth=0,
        )

        gap_lines = []
        for method_key in active_methods:
            method_bins = dataset_bins.loc[dataset_bins["method_key"] == method_key].sort_values("bin_num")
            if method_bins.empty:
                continue
            method_cfg = bundle["cfg"]["methods"][method_key]
            ax.plot(
                method_bins["bin_num"],
                method_bins["loss_mean"],
                marker="o",
                markersize=5.5,
                linewidth=2.3,
                color=method_cfg["color"],
                label=method_cfg["label"],
            )

            story_row = dataset_story.loc[dataset_story["method_key"] == method_key]
            if not story_row.empty:
                gap = float(story_row["hard_minus_easy_loss"].iloc[0])
                gap_lines.append(f"{method_cfg['label']}: {gap:+.4f}")

        ax.set_title(
            f"{bundle['cfg']['label']} | epoch {epoch}\nrandom-guess loss log({n_classes}) = {random_guess_loss:.2f}",
            fontsize=12.5,
            pad=8,
        )
        ax.set_xlabel("Difficulty bin fixed from InfoBatch pruning rate")
        xticks = list(range(1, n_bins + 1))
        ax.set_xticks(xticks)
        ax.set_xticklabels(difficulty_bin_xticklabels(dataset_bins, n_bins))
        ax.set_ylim(0.0, y_top)
        ax.set_ylabel("Mean loss")
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(
            0.03,
            0.97,
            "Hard - easy loss\n" + "\n".join(gap_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.88},
        )

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(len(legend_handles), 4),
            frameon=False,
        )

    finish_figure(
        fig,
        "cross_dataset_loss_difficulty_story",
        "cross_dataset",
        figure_label=(
            "Loss by InfoBatch-pruning difficulty bins in raw loss units"
        ),
        tight_rect=(0.0, 0.0, 1.0, 0.90),
    )


def build_integrated_difficulty_metric_table(
    entropy_bins: pd.DataFrame,
    memorization_bins: pd.DataFrame,
    loss_bins: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["dataset_key", "dataset", "method_key", "method", "epoch", "anchor_epoch", "bin_num"]
    entropy = entropy_bins[
        key_cols
        + [
            "n",
            "n_classes",
            "max_entropy",
            "anchor_pruning_metric",
            "entropy_mean",
            "entropy_mean_norm",
        ]
    ].rename(columns={"n": "n_entropy"})
    memorization = memorization_bins[
        key_cols + ["n", "memorization_mean"]
    ].rename(columns={"n": "n_memorization"})
    loss = loss_bins[
        key_cols + ["n", "random_guess_loss", "loss_mean", "loss_mean_norm"]
    ].rename(columns={"n": "n_loss"})

    merged = entropy.merge(memorization, on=key_cols, how="inner").merge(loss, on=key_cols, how="inner")
    merged["n"] = merged[["n_entropy", "n_memorization", "n_loss"]].min(axis=1).astype(np.int64)
    merged["entropy_pct_max"] = 100.0 * merged["entropy_mean_norm"]
    merged["loss_pct_log_classes"] = 100.0 * merged["loss_mean_norm"]
    dataset_order = {dataset_key: idx for idx, dataset_key in enumerate(DATASET_ORDER)}
    method_order = {method_key: idx for idx, method_key in enumerate(METHOD_ORDER)}
    merged["_dataset_order"] = merged["dataset_key"].map(dataset_order).fillna(len(dataset_order)).astype(int)
    merged["_method_order"] = merged["method_key"].map(method_order).fillna(len(method_order)).astype(int)
    return (
        merged.sort_values(["_dataset_order", "_method_order", "bin_num"])
        .drop(columns=["_dataset_order", "_method_order"])
        .reset_index(drop=True)
    )


def _spearman_or_nan(frame: pd.DataFrame, left: str, right: str) -> float:
    values = frame[[left, right]].dropna()
    if len(values) < 2:
        return float("nan")
    return float(values[left].corr(values[right], method="spearman"))


def build_integrated_difficulty_delta_table(
    integrated_df: pd.DataFrame,
    *,
    easy_bin: int = 1,
    hard_bin: int = 5,
) -> pd.DataFrame:
    rows = []
    for (dataset_key, method_key), frame in integrated_df.groupby(["dataset_key", "method_key"], sort=False):
        frame = frame.sort_values("bin_num")
        easy = frame.loc[frame["bin_num"] == easy_bin]
        hard = frame.loc[frame["bin_num"] == hard_bin]
        if easy.empty or hard.empty:
            continue
        easy = easy.iloc[0]
        hard = hard.iloc[0]
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": hard["dataset"],
                "method_key": method_key,
                "method": hard["method"],
                "easy_n": int(easy["n"]),
                "hard_n": int(hard["n"]),
                "easy_memorization": float(easy["memorization_mean"]),
                "hard_memorization": float(hard["memorization_mean"]),
                "delta_memorization": float(hard["memorization_mean"] - easy["memorization_mean"]),
                "easy_entropy_pct_max": float(easy["entropy_pct_max"]),
                "hard_entropy_pct_max": float(hard["entropy_pct_max"]),
                "easy_entropy": float(easy["entropy_mean"]),
                "hard_entropy": float(hard["entropy_mean"]),
                "delta_entropy": float(hard["entropy_mean"] - easy["entropy_mean"]),
                "delta_entropy_pct_points": float(hard["entropy_pct_max"] - easy["entropy_pct_max"]),
                "easy_loss": float(easy["loss_mean"]),
                "hard_loss": float(hard["loss_mean"]),
                "delta_loss": float(hard["loss_mean"] - easy["loss_mean"]),
                "easy_loss_pct_log_classes": float(easy["loss_pct_log_classes"]),
                "hard_loss_pct_log_classes": float(hard["loss_pct_log_classes"]),
                "delta_loss_pct_points": float(hard["loss_pct_log_classes"] - easy["loss_pct_log_classes"]),
                "rho_mem_entropy": _spearman_or_nan(frame, "memorization_mean", "entropy_mean_norm"),
                "rho_mem_loss": _spearman_or_nan(frame, "memorization_mean", "loss_mean_norm"),
                "rho_entropy_loss": _spearman_or_nan(frame, "entropy_mean_norm", "loss_mean_norm"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["all_three_higher_in_hard_bin"] = (
            (out["delta_memorization"] > 0)
            & (out["delta_entropy"] > 0)
            & (out["delta_loss"] > 0)
        )
    return out


def plot_integrated_difficulty_metric_grid(
    integrated_df: pd.DataFrame,
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
    hard_bins: tuple[int, ...] = (5,),
) -> None:
    if integrated_df.empty:
        display_note("**Integrated difficulty profile skipped:** no joined metric rows were available.")
        return

    dataset_keys = [dataset_key for dataset_key in DATASET_ORDER if dataset_key in integrated_df["dataset_key"].unique()]
    if not dataset_keys:
        display_note("**Integrated difficulty profile skipped:** no configured datasets were present.")
        return

    metrics = [
        ("memorization_mean", "Memorization", "Mean memorization", "memorization"),
        ("entropy_mean", "Entropy", "Mean entropy", "entropy"),
        ("loss_mean", "Loss", "Mean loss", "loss"),
    ]

    fig, axes = plt.subplots(
        len(dataset_keys),
        len(metrics),
        figsize=(5.6 * len(metrics), 4.25 * len(dataset_keys)),
        squeeze=False,
    )

    for row_idx, dataset_key in enumerate(dataset_keys):
        bundle = bundles[dataset_key]
        dataset_frame = integrated_df.loc[integrated_df["dataset_key"] == dataset_key]
        active_methods = method_keys or list(bundle["method_keys"])
        active_methods = [method_key for method_key in active_methods if method_key in dataset_frame["method_key"].unique()]
        n_bins = int(dataset_frame["bin_num"].max())
        xticks = list(range(1, n_bins + 1))

        for col_idx, (metric_col, metric_title, ylabel, metric_kind) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ax.axvspan(
                min(hard_bins) - 0.5,
                max(hard_bins) + 0.5,
                color="#E45756",
                alpha=0.07,
                linewidth=0,
            )

            if metric_kind == "entropy":
                max_entropy = float(dataset_frame["max_entropy"].iloc[0])
                n_classes = int(dataset_frame["n_classes"].iloc[0])
                ax.axhline(max_entropy, color="#64748B", linestyle=":", linewidth=1.0)
                ylim = (0.0, max_entropy * 1.10)
            elif metric_kind == "loss":
                ylim = (0.0, max(1e-6, float(np.nanmax(dataset_frame["loss_mean"].to_numpy(dtype=np.float64)))) * 1.20)
            else:
                ylim = (0.0, 1.06)

            ax.set_ylim(*ylim)
            annotation_points = []
            for method_key in active_methods:
                method_frame = dataset_frame.loc[dataset_frame["method_key"] == method_key].sort_values("bin_num")
                if method_frame.empty:
                    continue
                method_cfg = bundle["cfg"]["methods"][method_key]
                final_acc = final_test_accuracy(bundle, method_key)
                ax.plot(
                    method_frame["bin_num"],
                    method_frame[metric_col],
                    marker="o",
                    markersize=4.7,
                    linewidth=2.1,
                    color=method_cfg["color"],
                    label=f"{method_cfg['label']} ({final_acc:.2f}% acc)",
                )
                for _, point_row in method_frame.iterrows():
                    value = point_row[metric_col]
                    if pd.notna(value):
                        annotation_points.append(
                            {
                                "x": float(point_row["bin_num"]),
                                "y": float(value),
                                "color": method_cfg["color"],
                            }
                        )

            _annotate_grouped_point_values(ax, annotation_points, decimals=2)

            if row_idx == 0:
                ax.set_title(metric_title, fontsize=12.5, pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{bundle['cfg']['label']}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            ax.set_xticks(xticks)
            ax.set_xticklabels(difficulty_bin_xticklabels(dataset_frame, n_bins), fontsize=8)
            ax.set_xlabel("Difficulty bin" if row_idx == len(dataset_keys) - 1 else "")
            ax.grid(True, axis="y", alpha=0.25)
            if col_idx == 0:
                ax.legend(
                    loc="upper right",
                    bbox_to_anchor=(0.995, 0.995),
                    borderaxespad=0.15,
                    borderpad=0.25,
                    handlelength=1.45,
                    handletextpad=0.45,
                    labelspacing=0.24,
                    frameon=True,
                    framealpha=0.84,
                    facecolor="white",
                    edgecolor="#E5E7EB",
                    fontsize=7.8,
                )
            if metric_kind == "entropy":
                ax.text(
                    0.03,
                    0.96,
                    f"100% = log({n_classes})",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.2,
                    color="#475569",
                    bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.86},
                )

    finish_figure(
        fig,
        "cross_dataset_integrated_difficulty_profile",
        "cross_dataset",
        figure_label=(
            "Integrated difficulty profile: memorization, entropy, and loss over the same InfoBatch-pruning bins"
        ),
        tight_rect=(0.0, 0.0, 1.0, 0.955),
    )


def infobatch_pruning_anchor_table(anchor_frame: pd.DataFrame, bin_labels: list[str], n_bins: int) -> pd.DataFrame:
    table = (
        anchor_frame.groupby("bin_num", sort=True)
        .agg(
            n=("sample_id", "size"),
            mean_pruning_rate=("anchor_pruning_rate", "mean"),
            min_pruning_rate=("anchor_pruning_rate", "min"),
            max_pruning_rate=("anchor_pruning_rate", "max"),
        )
        .reindex(range(1, n_bins + 1))
        .reset_index()
    )
    table["pruning_interval"] = bin_labels
    table["difficulty"] = "middle"
    table.loc[table["bin_num"] == 1, "difficulty"] = "easy / highest-pruned"
    table.loc[table["bin_num"] == n_bins, "difficulty"] = "hard / lowest-pruned"
    return table[
        [
            "bin_num",
            "difficulty",
            "pruning_interval",
            "n",
            "mean_pruning_rate",
            "min_pruning_rate",
            "max_pruning_rate",
        ]
    ]


def plot_infobatch_pruning_entropy_trajectories(
    bundle: dict,
    *,
    anchor_epoch: int | None = None,
    method_keys: list[str] | None = None,
    n_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    active_methods = method_keys or list(bundle["method_keys"])
    active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
    if not active_methods:
        display_note(f"**Entropy trajectory skipped for {bundle['cfg']['label']}:** no requested methods are available.")
        return None

    anchor_epoch = int(bundle["cfg"]["end_epoch"] if anchor_epoch is None else anchor_epoch)
    anchor_frame, edges, pruning_col, pruning_label = build_infobatch_pruning_interval_anchor(
        bundle,
        anchor_epoch=anchor_epoch,
        n_bins=n_bins,
    )
    bin_labels = infobatch_pruning_interval_labels(edges)
    evo_epochs = common_evolution_epochs(bundle, active_methods)
    summary = summarize_fixed_bins_from_evolution(
        bundle,
        anchor_frame,
        active_methods,
        evo_epochs,
        field="entropy",
        n_bins=n_bins,
    )
    if summary.empty:
        display_note(f"**Entropy trajectory skipped for {bundle['cfg']['label']}:** no entropy field was found.")
        return None

    max_entropy = theoretical_max_entropy(bundle)
    summary = summary.copy()
    summary["dataset_key"] = bundle["dataset_key"]
    summary["dataset"] = bundle["cfg"]["label"]
    summary["metric_mean_norm"] = summary["metric_mean"] / max_entropy
    summary["metric_pct_max"] = 100.0 * summary["metric_mean_norm"]
    summary["anchor_epoch"] = anchor_epoch
    summary["anchor_pruning_metric"] = pruning_col
    summary["anchor_pruning_label"] = pruning_label
    summary["n_classes"] = int(bundle["cfg"]["n_classes"])
    summary["max_entropy"] = max_entropy

    anchor_table = infobatch_pruning_anchor_table(anchor_frame, bin_labels, n_bins)
    bin_count_lookup = anchor_table.set_index("bin_num")["n"].to_dict()

    fig, axes = plt.subplots(1, len(active_methods), figsize=(5.6 * len(active_methods), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method_key in zip(axes, active_methods):
        method_cfg = bundle["cfg"]["methods"][method_key]
        method_summary = summary.loc[summary["method_key"] == method_key]
        for bin_num in range(1, n_bins + 1):
            bin_summary = method_summary.loc[method_summary["bin_num"] == bin_num].sort_values("epoch")
            if bin_summary.empty:
                continue
            markevery = max(1, len(bin_summary) // 9)
            line_label = f"Bin {bin_num} (n={int(bin_count_lookup.get(bin_num, 0)):,})"
            if bin_num == 1:
                line_label += " easy"
            elif bin_num == n_bins:
                line_label += " hard"
            ax.plot(
                bin_summary["epoch"],
                bin_summary["metric_mean"],
                color=difficulty_bin_color(bin_num),
                linewidth=2.1 if bin_num not in (1, n_bins) else 2.6,
                marker="o",
                markersize=3.8,
                markevery=markevery,
                label=line_label,
            )

        ax.axhline(max_entropy, color="#64748B", linestyle=":", linewidth=1.0)
        ax.set_title(method_cfg["label"], fontsize=12.5, pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, max_entropy * 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(
            0.03,
            0.96,
            f"100% = log({bundle['cfg']['n_classes']})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="#475569",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.86},
        )

    axes[0].set_ylabel("Mean entropy")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8.7)
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_infobatch_pruning_bin_entropy_trajectories_anchor_{anchor_epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{bundle['cfg']['label']} | Entropy trajectories by InfoBatch-pruning difficulty "
            f"(anchor epoch {anchor_epoch}, max entropy log({bundle['cfg']['n_classes']})={max_entropy:.2f})"
        ),
        tight_rect=(0.0, 0.0, 1.0, 0.92),
    )
    return summary, anchor_table


def plot_entropy_trajectory_by_difficulty_bin_figures(
    summary_df: pd.DataFrame,
    bundles: dict,
    *,
    method_keys: list[str] | None = None,
    n_bins: int = 5,
) -> None:
    if summary_df.empty:
        display_note("**Entropy trajectory comparison skipped:** no entropy trajectory rows were available.")
        return

    dataset_keys = [dataset_key for dataset_key in DATASET_ORDER if dataset_key in summary_df["dataset_key"].unique()]
    active_methods = method_keys or METHOD_ORDER
    active_methods = [
        method_key
        for method_key in METHOD_ORDER
        if method_key in active_methods and method_key in summary_df["method_key"].unique()
    ]
    if not dataset_keys or not active_methods:
        display_note("**Entropy trajectory comparison skipped:** no configured datasets or methods were present.")
        return

    for bin_num in range(1, n_bins + 1):
        fig, axes = plt.subplots(
            1,
            len(dataset_keys),
            figsize=(5.35 * len(dataset_keys), 4.35),
            sharey=False,
            squeeze=False,
        )
        axes = axes[0]

        for ax, dataset_key in zip(axes, dataset_keys):
            bundle = bundles[dataset_key]
            dataset_frame = summary_df.loc[
                (summary_df["dataset_key"] == dataset_key) & (summary_df["bin_num"] == bin_num)
            ]
            if dataset_frame.empty:
                ax.axis("off")
                continue

            max_entropy = float(dataset_frame["max_entropy"].iloc[0])
            n_classes = int(dataset_frame["n_classes"].iloc[0])
            ax.axhline(max_entropy, color="#64748B", linestyle=":", linewidth=1.0)
            ax.set_ylim(0.0, max_entropy * 1.05)

            end_points = []
            x_values = []
            for method_key in active_methods:
                method_frame = dataset_frame.loc[dataset_frame["method_key"] == method_key].sort_values("epoch")
                if method_frame.empty:
                    continue
                method_cfg = bundle["cfg"]["methods"][method_key]
                final_acc = final_test_accuracy(bundle, method_key)
                markevery = max(1, len(method_frame) // 9)
                ax.plot(
                    method_frame["epoch"],
                    method_frame["metric_mean"],
                    color=method_cfg["color"],
                    linewidth=2.25,
                    marker="o",
                    markersize=3.4,
                    markevery=markevery,
                    label=f"{method_cfg['label']} ({final_acc:.2f}% acc)",
                )
                x_values.extend(method_frame["epoch"].tolist())
                final_row = method_frame.iloc[-1]
                end_points.append(
                    {
                        "method_key": method_key,
                        "y": float(final_row["metric_mean"]),
                        "color": method_cfg["color"],
                    }
                )

            if x_values:
                x_min = float(np.nanmin(x_values))
                x_max = float(np.nanmax(x_values))
                x_pad = max(1.0, 0.09 * max(x_max - x_min, 1.0))
                ax.set_xlim(x_min, x_max + x_pad)
                _annotate_tail_line_ends(ax, end_points, x_last=x_max)

            ax.set_title(
                f"{bundle['cfg']['label']}\n100% entropy = log({n_classes}) = {max_entropy:.2f}",
                fontsize=11.5,
                pad=8,
            )
            ax.set_xlabel("Epoch")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(0.995, 0.995),
                borderaxespad=0.15,
                borderpad=0.25,
                handlelength=1.45,
                handletextpad=0.45,
                labelspacing=0.24,
                frameon=True,
                framealpha=0.84,
                facecolor="white",
                edgecolor="#E5E7EB",
                fontsize=7.8,
            )

        axes[0].set_ylabel("Mean entropy")
        bin_label = f"Bin {bin_num}"
        if bin_num == 1:
            bin_label += " | easiest / highest-pruned"
        elif bin_num == n_bins:
            bin_label += " | hardest / lowest-pruned"
        finish_figure(
            fig,
            f"cross_dataset_entropy_trajectory_pruning_bin_{bin_num}",
            "cross_dataset",
            figure_label=(
                f"Entropy trajectories by method | {bin_label} | "
                "final labels show entropy"
            ),
            tight_rect=(0.0, 0.0, 1.0, 0.91),
        )


def plot_infobatch_pruning_loss_trajectories(
    bundle: dict,
    *,
    anchor_epoch: int | None = None,
    method_keys: list[str] | None = None,
    n_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    active_methods = method_keys or list(bundle["method_keys"])
    active_methods = [method_key for method_key in active_methods if method_key in bundle["method_keys"]]
    if not active_methods:
        display_note(f"**Loss trajectory skipped for {bundle['cfg']['label']}:** no requested methods are available.")
        return None

    anchor_epoch = int(bundle["cfg"]["end_epoch"] if anchor_epoch is None else anchor_epoch)
    anchor_frame, edges, pruning_col, pruning_label = build_infobatch_pruning_interval_anchor(
        bundle,
        anchor_epoch=anchor_epoch,
        n_bins=n_bins,
    )
    bin_labels = infobatch_pruning_interval_labels(edges)
    evo_epochs = common_evolution_epochs(bundle, active_methods)
    summary = summarize_fixed_bins_from_evolution(
        bundle,
        anchor_frame,
        active_methods,
        evo_epochs,
        field="loss",
        n_bins=n_bins,
    )
    if summary.empty:
        display_note(f"**Loss trajectory skipped for {bundle['cfg']['label']}:** no loss field was found.")
        return None

    random_guess_loss = theoretical_max_entropy(bundle)
    summary = summary.copy()
    summary["metric_mean_norm"] = summary["metric_mean"] / random_guess_loss
    summary["metric_pct_log_classes"] = 100.0 * summary["metric_mean_norm"]
    summary["anchor_epoch"] = anchor_epoch
    summary["anchor_pruning_metric"] = pruning_col
    summary["anchor_pruning_label"] = pruning_label
    summary["n_classes"] = int(bundle["cfg"]["n_classes"])
    summary["random_guess_loss"] = random_guess_loss

    anchor_table = infobatch_pruning_anchor_table(anchor_frame, bin_labels, n_bins)
    bin_count_lookup = anchor_table.set_index("bin_num")["n"].to_dict()
    y_top = max(1e-6, float(np.nanmax(summary["metric_mean"].to_numpy(dtype=np.float64)))) * 1.15

    fig, axes = plt.subplots(1, len(active_methods), figsize=(5.6 * len(active_methods), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method_key in zip(axes, active_methods):
        method_cfg = bundle["cfg"]["methods"][method_key]
        method_summary = summary.loc[summary["method_key"] == method_key]
        for bin_num in range(1, n_bins + 1):
            bin_summary = method_summary.loc[method_summary["bin_num"] == bin_num].sort_values("epoch")
            if bin_summary.empty:
                continue
            markevery = max(1, len(bin_summary) // 9)
            line_label = f"Bin {bin_num} (n={int(bin_count_lookup.get(bin_num, 0)):,})"
            if bin_num == 1:
                line_label += " easy"
            elif bin_num == n_bins:
                line_label += " hard"
            ax.plot(
                bin_summary["epoch"],
                bin_summary["metric_mean"],
                color=difficulty_bin_color(bin_num),
                linewidth=2.1 if bin_num not in (1, n_bins) else 2.6,
                marker="o",
                markersize=3.8,
                markevery=markevery,
                label=line_label,
            )

        ax.set_title(method_cfg["label"], fontsize=12.5, pad=8)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, y_top)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Mean loss")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8.7)
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_infobatch_pruning_bin_loss_trajectories_anchor_{anchor_epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{bundle['cfg']['label']} | Loss trajectories by InfoBatch-pruning difficulty "
            f"(anchor epoch {anchor_epoch}, raw loss units)"
        ),
        tight_rect=(0.0, 0.0, 1.0, 0.92),
    )
    return summary, anchor_table


def pruning_method_keys(bundle: dict) -> list[str]:
    return [method_key for method_key in bundle["method_keys"] if method_key != "std"]


def plot_sorted_pruning_curves(bundle: dict, epoch: int, method_keys: list[str] | None = None) -> None:
    method_keys = method_keys or pruning_method_keys(bundle)
    curve_frames = []
    for method_key in method_keys:
        method_cfg = bundle["cfg"]["methods"][method_key]
        evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
        pruning_col, pruning_label = resolve_pruning_metric(bundle, method_key, evo)
        if pruning_col is None:
            continue

        curve = (
            evo.loc[evo[pruning_col].notna(), ["sample_id", pruning_col]]
            .rename(columns={pruning_col: "pruning_value"})
            .sort_values("pruning_value", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        curve["sample_rank"] = np.arange(1, len(curve) + 1)
        curve["method_key"] = method_key
        curve["pruning_label"] = pruning_label
        curve_frames.append(curve)

    if not curve_frames:
        display_note(f"**Sorted pruning curves skipped for {bundle['cfg']['label']}:** no pruning metric was found.")
        return

    sorted_pruning_df = pd.concat(curve_frames, ignore_index=True)
    pruning_labels = sorted_pruning_df["pruning_label"].unique().tolist()
    y_label = pruning_labels[0] if len(pruning_labels) == 1 else "Pruning-related value"

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for method_key in method_keys:
        df_method = sorted_pruning_df.loc[sorted_pruning_df["method_key"] == method_key]
        if df_method.empty:
            continue
        ax.plot(
            df_method["sample_rank"],
            df_method["pruning_value"],
            color=bundle["cfg"]["methods"][method_key]["color"],
            linewidth=2.1,
            alpha=0.95,
            label=bundle["cfg"]["methods"][method_key]["label"],
        )

    ax.set_xlabel("Samples arranged by descending pruning value")
    ax.set_ylabel(y_label)
    ax.set_xlim(1, int(sorted_pruning_df["sample_rank"].max()))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_sorted_pruning_curves_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=f"{bundle['cfg']['label']} | Sorted pruning curves | Epoch {epoch}",
    )


def plot_mean_mem_by_shared_pruning_bins(
    bundle: dict,
    epoch: int,
    *,
    method_keys: list[str] | None = None,
    bin_width: float = 0.06,
    bin_start: float = 0.0,
    min_bin_end: float = 0.66,
) -> pd.DataFrame | None:
    method_keys = method_keys or pruning_method_keys(bundle)
    method_frames = {}
    all_pruning_vals = []

    for method_key in method_keys:
        method_cfg = bundle["cfg"]["methods"][method_key]
        mem = get_method_epoch_mem(bundle, method_key, epoch)
        evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
        pruning_col, _ = resolve_pruning_metric(bundle, method_key, evo)
        if pruning_col is None:
            continue

        merged = strict_sample_merge(
            mem[["sample_id", "memorization"]],
            evo[["sample_id", pruning_col]],
            context=f"{bundle['dataset_key']}::{method_key}::shared_pruning_bins::epoch_{epoch}",
        ).dropna(subset=["memorization", pruning_col]).copy()

        merged = merged.rename(columns={pruning_col: "pruning_value"})
        method_frames[method_key] = merged
        all_pruning_vals.append(merged["pruning_value"].to_numpy(dtype=np.float64))

    if not all_pruning_vals:
        display_note(f"**Shared pruning-bin analysis skipped for {bundle['cfg']['label']}:** no pruning data was found.")
        return None

    all_pruning_vals = np.concatenate(all_pruning_vals)
    max_val = float(np.max(all_pruning_vals))
    bin_end = max(min_bin_end, np.ceil(max_val / bin_width) * bin_width)
    shared_edges = np.arange(bin_start, bin_end + bin_width, bin_width, dtype=np.float64)
    shared_edges[-1] = np.nextafter(shared_edges[-1], np.inf)
    bin_labels = [f"{shared_edges[i]:.2f}-{shared_edges[i + 1]:.2f}" for i in range(len(shared_edges) - 1)]

    summaries = []
    for method_key in method_keys:
        if method_key not in method_frames:
            continue
        merged = method_frames[method_key].copy()
        merged["bin_idx"] = pd.cut(
            merged["pruning_value"],
            bins=shared_edges,
            labels=False,
            include_lowest=True,
        )
        summary = (
            merged.dropna(subset=["bin_idx"])
            .assign(bin_idx=lambda frame: frame["bin_idx"].astype(int))
            .groupby("bin_idx", sort=True)
            .agg(mean_memorization=("memorization", "mean"), n=("sample_id", "size"))
            .reset_index()
        )
        summary["bin_num"] = summary["bin_idx"] + 1
        summary["bin_label"] = [bin_labels[i] for i in summary["bin_idx"]]
        summary["method_key"] = method_key
        summary["method"] = bundle["cfg"]["methods"][method_key]["label"]
        summaries.append(summary)

    summary_df = pd.concat(summaries, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    for method_key in method_keys:
        method_summary = summary_df.loc[summary_df["method_key"] == method_key].sort_values("bin_num")
        if method_summary.empty:
            continue
        ax.plot(
            method_summary["bin_num"],
            method_summary["mean_memorization"],
            marker="o",
            markersize=8,
            linewidth=2.4,
            color=bundle["cfg"]["methods"][method_key]["color"],
            label=bundle["cfg"]["methods"][method_key]["label"],
        )

    ax.set_xlabel("Shared pruning-rate bin", fontsize=13)
    ax.set_ylabel("Mean memorization", fontsize=13)
    ax.set_xticks(range(1, len(bin_labels) + 1))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_mean_mem_by_shared_pruning_bins_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{bundle['cfg']['label']} | Mean memorization by shared {bin_width:.2f}-width pruning bins | Epoch {epoch}"
        ),
    )
    return summary_df


def plot_mean_mem_by_std_loss_bins(
    bundle: dict,
    *,
    anchor_epoch: int = 10,
    mem_epoch: int | None = None,
    method_keys: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    method_keys = method_keys or list(bundle["method_keys"])
    mem_epoch = int(mem_epoch if mem_epoch is not None else bundle["cfg"]["end_epoch"])
    anchor, _, bin_labels = build_std_loss_anchor(bundle, anchor_epoch=anchor_epoch, n_bins=n_bins)

    summary = summarize_fixed_bins(
        bundle,
        anchor,
        method_keys,
        epochs=[mem_epoch],
        source="memorization",
        n_bins=n_bins,
    )

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    for method_key in method_keys:
        method_summary = summary.loc[summary["method_key"] == method_key].sort_values("bin_num")
        if method_summary.empty:
            continue
        ax.plot(
            method_summary["bin_num"],
            method_summary["metric_mean"],
            marker="o",
            markersize=8,
            linewidth=2.4,
            color=bundle["cfg"]["methods"][method_key]["color"],
            label=bundle["cfg"]["methods"][method_key]["label"],
        )

    ax.set_xlabel(f"Standard-loss bin at epoch {anchor_epoch}", fontsize=13)
    ax.set_ylabel(f"Mean memorization at epoch {mem_epoch}", fontsize=13)
    ax.set_xticks(range(1, n_bins + 1))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_mean_mem_by_std_loss_bins_anchor_{anchor_epoch}_mem_{mem_epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{bundle['cfg']['label']} | Mean memorization by standard-loss bins "
            f"(anchor epoch {anchor_epoch}, mem epoch {mem_epoch})"
        ),
    )
    return summary


def plot_mean_pruning_by_std_loss_bins(
    bundle: dict,
    *,
    anchor_epoch: int = 10,
    prune_epoch: int | None = None,
    method_keys: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame | None:
    method_keys = method_keys or pruning_method_keys(bundle)
    prune_epoch = int(prune_epoch if prune_epoch is not None else bundle["cfg"]["end_epoch"])
    anchor, _, bin_labels = build_std_loss_anchor(bundle, anchor_epoch=anchor_epoch, n_bins=n_bins)

    try:
        summary = summarize_fixed_bins(
            bundle,
            anchor,
            method_keys,
            epochs=[prune_epoch],
            source="pruning",
            n_bins=n_bins,
        )
    except ValueError:
        display_note(f"**Standard-loss vs pruning analysis skipped for {bundle['cfg']['label']}:** no pruning data was found.")
        return None

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    for method_key in method_keys:
        method_summary = summary.loc[summary["method_key"] == method_key].sort_values("bin_num")
        if method_summary.empty:
            continue
        ax.plot(
            method_summary["bin_num"],
            method_summary["metric_mean"],
            marker="o",
            markersize=8,
            linewidth=2.4,
            color=bundle["cfg"]["methods"][method_key]["color"],
            label=bundle["cfg"]["methods"][method_key]["label"],
        )

    ax.set_xlabel(f"Standard-loss bin at epoch {anchor_epoch}", fontsize=13)
    ax.set_ylabel(f"Mean pruning at epoch {prune_epoch}", fontsize=13)
    ax.set_xticks(range(1, n_bins + 1))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_mean_pruning_by_std_loss_bins_anchor_{anchor_epoch}_epoch_{prune_epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{bundle['cfg']['label']} | Mean pruning by standard-loss bins "
            f"(anchor epoch {anchor_epoch}, prune epoch {prune_epoch})"
        ),
    )
    return summary


def weighted_focus_metric(summary_df: pd.DataFrame, method_key: str, focus_bins: list[int]) -> float:
    sub = summary_df.loc[(summary_df["method_key"] == method_key) & (summary_df["bin_num"].isin(focus_bins))].copy()
    sub = sub.dropna(subset=["metric_mean", "n"])
    if sub.empty or float(sub["n"].sum()) == 0.0:
        return float("nan")
    return float(np.average(sub["metric_mean"], weights=sub["n"]))


def build_cross_dataset_behavior_summary(
    bundles: dict[str, dict],
    *,
    anchor_epoch: int = 10,
    focus_bins: tuple[int, ...] = (8, 9, 10),
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for dataset_key in DATASET_ORDER:
        if dataset_key not in bundles:
            continue
        bundle = bundles[dataset_key]
        anchor, _, _ = build_std_loss_anchor(bundle, anchor_epoch=anchor_epoch, n_bins=n_bins)
        method_keys = list(bundle["method_keys"])
        prune_keys = pruning_method_keys(bundle)
        mem_epoch = int(bundle["cfg"]["end_epoch"])

        mem_summary = summarize_fixed_bins(
            bundle,
            anchor,
            method_keys,
            epochs=[mem_epoch],
            source="memorization",
            n_bins=n_bins,
        )
        frac_summary = summarize_fixed_bins(
            bundle,
            anchor,
            method_keys,
            epochs=[mem_epoch],
            source="memorized_fraction",
            threshold=MEMORIZED_THRESHOLD,
            n_bins=n_bins,
        )
        loss_summary = summarize_fixed_bins(
            bundle,
            anchor,
            method_keys,
            epochs=[mem_epoch],
            source="loss",
            n_bins=n_bins,
        )
        try:
            pruning_summary = summarize_fixed_bins(
                bundle,
                anchor,
                prune_keys,
                epochs=[mem_epoch],
                source="pruning",
                n_bins=n_bins,
            )
        except ValueError:
            pruning_summary = pd.DataFrame(columns=["method_key", "bin_num", "metric_mean", "n"])

        std_mem = weighted_focus_metric(mem_summary, "std", list(focus_bins))
        std_frac = weighted_focus_metric(frac_summary, "std", list(focus_bins))
        std_loss = weighted_focus_metric(loss_summary, "std", list(focus_bins))
        focus_n = int(anchor.loc[anchor["bin_num"].isin(focus_bins)].shape[0])

        for method_key in prune_keys:
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": bundle["cfg"]["label"],
                    "method_key": method_key,
                    "method": bundle["cfg"]["methods"][method_key]["label"],
                    "focus_bins": ",".join(str(bin_num) for bin_num in focus_bins),
                    "n_focus_samples": focus_n,
                    "std_mean_memorization_focus": std_mem,
                    "mean_memorization_focus": weighted_focus_metric(mem_summary, method_key, list(focus_bins)),
                    "delta_mean_memorization_focus": weighted_focus_metric(mem_summary, method_key, list(focus_bins)) - std_mem,
                    "std_fraction_memorized_focus": std_frac,
                    "fraction_memorized_focus": weighted_focus_metric(frac_summary, method_key, list(focus_bins)),
                    "delta_fraction_memorized_focus": weighted_focus_metric(frac_summary, method_key, list(focus_bins)) - std_frac,
                    "std_mean_loss_focus": std_loss,
                    "mean_loss_focus": weighted_focus_metric(loss_summary, method_key, list(focus_bins)),
                    "delta_mean_loss_focus": weighted_focus_metric(loss_summary, method_key, list(focus_bins)) - std_loss,
                    "mean_pruning_focus": weighted_focus_metric(pruning_summary, method_key, list(focus_bins)),
                }
            )

    return pd.DataFrame(rows)


def plot_cross_dataset_behavior_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        display_note("**Cross-dataset behavior summary skipped:** no summary rows were generated.")
        return

    dataset_keys = [dataset_key for dataset_key in DATASET_ORDER if dataset_key in summary_df["dataset_key"].unique()]
    dataset_labels = [DATASETS[dataset_key]["label"] for dataset_key in dataset_keys]
    method_keys = [method_key for method_key in ["ib", "ib_prw"] if method_key in summary_df["method_key"].unique()]
    x = np.arange(len(dataset_keys), dtype=np.float64)
    width = 0.35 if len(method_keys) == 2 else 0.55

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8))
    metric_specs = [
        ("delta_mean_memorization_focus", "Delta mean memorization vs std"),
        ("delta_fraction_memorized_focus", "Delta fraction memorized vs std"),
        ("mean_pruning_focus", "Mean pruning in hard bins"),
    ]

    for ax, (metric_col, title) in zip(axes, metric_specs):
        for idx, method_key in enumerate(method_keys):
            offsets = x + (idx - (len(method_keys) - 1) / 2.0) * width
            values = []
            for dataset_key in dataset_keys:
                row = summary_df.loc[
                    (summary_df["dataset_key"] == dataset_key) & (summary_df["method_key"] == method_key),
                    metric_col,
                ]
                values.append(float(row.iloc[0]) if not row.empty else np.nan)

            ax.bar(
                offsets,
                values,
                width=width,
                color=METHOD_COLORS[method_key],
                alpha=0.90,
                label=METHOD_LABELS[method_key],
            )

        ax.axhline(0.0, color="#475569", linewidth=1.0)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].legend(frameon=False, loc="best")
    finish_figure(
        fig,
        "cross_dataset_hard_bin_behavior_summary",
        "cross_dataset",
        figure_label="Cross-dataset hard-bin summary | bins 8-10 anchored by standard loss at epoch 10",
    )


from copy import deepcopy

WRN_DATASET_KEY = "cifar100_wrn28_2"
WRN_DATASET_LABEL = "CIFAR-100 / WRN-28-2"
WRN_DATASET_ROOT = ANALYSIS_ROOT / WRN_DATASET_KEY
WRN_SOURCE_DATASET_KEY = "cifar100"
WRN_MEMORIZATION_GLOB = "loo_scores_epoch_*.pt"

REPORT_DEFAULTS = {
    "active_datasets": ["cifar10", WRN_DATASET_KEY],
    "pruning_datasets": ["cifar10", WRN_DATASET_KEY],
    "rmia_datasets": ["cifar10", WRN_DATASET_KEY],
    "all_methods": ["std", "ib", "ib_prw"],
    "pruned_methods": ["ib", "ib_prw"],
    "memorization_epochs": {
        "cifar10": 99,
        WRN_DATASET_KEY: 99,
    },
    "easy_hard_epochs": {
        "cifar10": 10,
        WRN_DATASET_KEY: 20,
    },
    "easy_hard_reference_method": "std",
    "pruning_anchor_epochs": {
        "cifar10": 99,
        WRN_DATASET_KEY: 99,
    },
    "pruning_anchor_method": "ib",
    "tail_auc_fraction": 0.10,
    "exceedance_checkpoint_fractions": (0.45, 0.75, 1.0),
    "exceedance_threshold_min": 0.10,
    "exceedance_threshold_max": 1.00,
    "exceedance_threshold_points": 91,
    "exposure_thresholds": (0.0, 0.5, 0.7),
    "exposure_bootstrap_samples": 500,
    "onset_plot_thresholds": (0.3, 0.7),
    "bin_plot_difficulty_mode": "epoch_loss",
    "mean_memorization_filter": "all",
    "binned_trajectory_difficulty_mode": "pruning_rate_tertiles",
    "rmia_mem_threshold": 0.3,
    "rmia_normalized_entropy_threshold": 0.3,
    "save_figures": False,
    "save_vector": False,
    "vector_format": "svg",
    "figure_dpi": 140,
    "font_size": 10,
    "inline_format": "png",
    "output_dir": ANALYSIS_ROOT / "outputs" / "report",
    "difficulty_scatter_alpha": 0.50,
    "movement_scatter_alpha": 0.72,
}

REPORT_SETTINGS = {}
BUNDLES = {}


def _copy_mapping(mapping):
    return {key: deepcopy(value) for key, value in mapping.items()}


def _apply_report_defaults(overrides=None):
    settings = deepcopy(REPORT_DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if isinstance(settings.get(key), dict) and isinstance(value, dict):
                settings[key].update(deepcopy(value))
            else:
                settings[key] = deepcopy(value)
    return settings


def ensure_wrn_dataset_registered() -> None:
    if WRN_DATASET_KEY in DATASETS:
        return

    wrn_cfg = deepcopy(DATASETS[WRN_SOURCE_DATASET_KEY])
    wrn_cfg.update(
        {
            "label": WRN_DATASET_LABEL,
            "n_classes": 100,
            "n_samples": 50_000,
            "subset_path": DATASETS[WRN_SOURCE_DATASET_KEY]["subset_path"],
            "memorization_epochs": DATASETS[WRN_SOURCE_DATASET_KEY]["memorization_epochs"],
            "end_epoch": DATASETS[WRN_SOURCE_DATASET_KEY]["end_epoch"],
            "bin_epochs": DATASETS[WRN_SOURCE_DATASET_KEY]["bin_epochs"],
            "movement_epochs": DATASETS[WRN_SOURCE_DATASET_KEY]["movement_epochs"],
            "matrix_epochs": DATASETS[WRN_SOURCE_DATASET_KEY]["matrix_epochs"],
            "comparison_epochs": DATASETS[WRN_SOURCE_DATASET_KEY]["comparison_epochs"],
            "entropy_hist_epochs": DATASETS[WRN_SOURCE_DATASET_KEY]["entropy_hist_epochs"],
            "loss_hist_epochs": DATASETS[WRN_SOURCE_DATASET_KEY]["loss_hist_epochs"],
            "matched_accuracy_pairs": [],
        }
    )

    wrn_method_files = {
        "std": ("standard", "cifar_evolution_log_standard.pt"),
        "ib": ("IB", "cifar_evolution_log_ib.pt"),
        "ib_prw": ("IB-WL", "cifar_evolution_log_ib_prw.pt"),
    }
    for method_key, (method_dir, evolution_name) in wrn_method_files.items():
        method_root = WRN_DATASET_ROOT / method_dir
        method_cfg = wrn_cfg["methods"][method_key]
        method_cfg.update(
            {
                "memorization_dir": method_root / "memorization_scores",
                "memorization_glob": WRN_MEMORIZATION_GLOB,
                "seed_avg_path": method_root / "seed_avg.pt",
                "evolution_log_path": method_root / evolution_name,
                "rmia_dir": method_root,
                "wandb_run": None,
                "wandb_train_keys": ["avg/train_accuracy"],
                "wandb_test_keys": ["avg/test_accuracy"],
                "local_train_keys": ["avg/train_accuracy"],
                "local_test_keys": ["avg/test_accuracy"],
            }
        )

    DATASETS[WRN_DATASET_KEY] = wrn_cfg


def _apply_report_style(settings: dict) -> None:
    global SAVE_FIGURES, SAVE_VECTOR, VECTOR_FORMAT, FIG_DPI
    SAVE_FIGURES = bool(settings["save_figures"])
    SAVE_VECTOR = bool(settings["save_vector"])
    VECTOR_FORMAT = str(settings["vector_format"])
    FIG_DPI = int(settings["figure_dpi"])

    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats(str(settings.get("inline_format", "png")))
    except Exception:
        pass

    base_font = int(settings["font_size"])
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "font.size": base_font,
            "axes.titlesize": base_font,
            "axes.labelsize": base_font,
            "xtick.labelsize": max(base_font - 1, 8),
            "ytick.labelsize": max(base_font - 1, 8),
            "legend.fontsize": max(base_font - 2, 8),
        }
    )


def report_setting(key: str):
    return REPORT_SETTINGS[key]


def initialize_report(overrides=None) -> pd.DataFrame:
    global REPORT_SETTINGS, BUNDLES, OUTPUT_ROOT
    ensure_wrn_dataset_registered()
    REPORT_SETTINGS = _apply_report_defaults(overrides)
    _apply_report_style(REPORT_SETTINGS)
    output_dir = Path(REPORT_SETTINGS["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    OUTPUT_ROOT = output_dir
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    global ACTIVE_DATASETS, ACTIVE_METHODS, PREFER_WANDB
    ACTIVE_DATASETS = list(REPORT_SETTINGS["active_datasets"])
    ACTIVE_METHODS = list(REPORT_SETTINGS["all_methods"])
    PREFER_WANDB = False

    BUNDLES = build_all_bundles(
        active_datasets=ACTIVE_DATASETS,
        method_keys=ACTIVE_METHODS,
        prefer_wandb=PREFER_WANDB,
    )
    plt.close("all")
    return report_dataset_summary()


def report_dataset_summary() -> pd.DataFrame:
    if not BUNDLES:
        return pd.DataFrame(columns=["dataset", "dataset_key", "methods", "n_samples", "end_epoch", "rmia_root"])
    return pd.DataFrame(
        [
            {
                "dataset": bundle["cfg"]["label"],
                "dataset_key": dataset_key,
                "methods": ", ".join(bundle["cfg"]["methods"][method_key]["label"] for method_key in bundle["method_keys"]),
                "n_samples": bundle["cfg"]["n_samples"],
                "end_epoch": bundle["cfg"]["end_epoch"],
                "rmia_root": str(bundle["cfg"]["methods"]["std"]["rmia_dir"]),
            }
            for dataset_key, bundle in BUNDLES.items()
        ]
    )


def report_bundle(dataset_key: str) -> dict:
    if dataset_key not in BUNDLES:
        raise KeyError(f"Dataset {dataset_key!r} is not available. Initialize the report first.")
    return BUNDLES[dataset_key]


PRESENTATION_TOLERANCE = 0.05
PRESENTATION_LINE_LABEL_BOX = {
    "boxstyle": "square,pad=0.18",
    "facecolor": "white",
    "alpha": 0.94,
}


def presentation_dataset_keys(bundles: dict, frame: pd.DataFrame | None = None) -> list[str]:
    if frame is None or frame.empty or "dataset_key" not in frame.columns:
        return [dataset_key for dataset_key in DATASET_ORDER if dataset_key in bundles]
    return [dataset_key for dataset_key in DATASET_ORDER if dataset_key in bundles and dataset_key in set(frame["dataset_key"])]


def presentation_method_keys(bundle: dict, method_keys: list[str] | None = None) -> list[str]:
    requested = method_keys or METHOD_ORDER
    return [method_key for method_key in METHOD_ORDER if method_key in requested and method_key in bundle["method_keys"]]


def presentation_legend_location(ax: plt.Axes) -> str:
    x_low, x_high = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    x_cut = x_low + 0.64 * max(x_high - x_low, 1e-9)
    y_cut = y_low + 0.68 * max(y_high - y_low, 1e-9)
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=np.float64)
        y = np.asarray(line.get_ydata(), dtype=np.float64)
        if x.size == 0 or y.size == 0:
            continue
        mask = (x >= x_cut) & np.isfinite(y)
        if mask.any() and float(np.nanmax(y[mask])) >= y_cut:
            return "lower right"
    return "upper right"


def presentation_add_legend(ax: plt.Axes, *, fontsize: float = 7.8) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    loc = presentation_legend_location(ax)
    bbox = (0.995, 0.995) if loc == "upper right" else (0.995, 0.02)
    ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox,
        borderaxespad=0.15,
        borderpad=0.26,
        handlelength=1.45,
        handletextpad=0.45,
        labelspacing=0.24,
        frameon=True,
        framealpha=0.90,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=fontsize,
    )


def spread_label_positions(
    rows: list[dict],
    *,
    y_low: float,
    y_high: float,
    min_gap_frac: float = 0.060,
    pad_frac: float = 0.045,
) -> list[dict]:
    if not rows:
        return []
    y_span = max(y_high - y_low, 1e-9)
    lower = y_low + pad_frac * y_span
    upper = y_high - pad_frac * y_span
    min_gap = min_gap_frac * y_span

    adjusted = []
    for row in sorted(rows, key=lambda item: float(item["y"])):
        y = float(np.clip(float(row["y"]), lower, upper))
        if adjusted and y - adjusted[-1]["label_y"] < min_gap:
            y = adjusted[-1]["label_y"] + min_gap
        adjusted.append({**row, "label_y": y})

    overflow = adjusted[-1]["label_y"] - upper
    if overflow > 0:
        for row in adjusted:
            row["label_y"] -= overflow
    underflow = lower - adjusted[0]["label_y"]
    if underflow > 0:
        for row in adjusted:
            row["label_y"] += underflow
    return adjusted


def _estimate_text_box_pixels(text: str, fontsize: float) -> tuple[float, float]:
    width = max(30.0, 0.60 * fontsize * max(len(str(text)), 1) + 12.0)
    height = max(16.0, 1.85 * fontsize + 6.0)
    return width, height


def _rect_from_center(cx: float, cy: float, width: float, height: float) -> tuple[float, float, float, float]:
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def _rect_overlap_area(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    dx = min(left[2], right[2]) - max(left[0], right[0])
    dy = min(left[3], right[3]) - max(left[1], right[1])
    return max(dx, 0.0) * max(dy, 0.0)


def _rect_contains_point(rect: tuple[float, float, float, float], point: tuple[float, float], pad: float = 5.0) -> bool:
    return (
        rect[0] - pad <= point[0] <= rect[2] + pad
        and rect[1] - pad <= point[1] <= rect[3] + pad
    )


def _plan_offset_annotations(
    ax: plt.Axes,
    labels: list[dict],
    *,
    fontsize: float,
    candidate_offsets_points: list[tuple[float, float]],
    keep_inside_axes: bool,
    prefer_right_of_axes: bool = False,
) -> list[dict]:
    if not labels:
        return []

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    px_per_pt = fig.dpi / 72.0
    anchor_points = [
        tuple(ax.transData.transform((float(label["x"]), float(label["y"]))))
        for label in labels
    ]

    placements = []
    existing_rects: list[tuple[float, float, float, float]] = []
    for idx, label in enumerate(labels):
        anchor_px = anchor_points[idx]
        width_px, height_px = _estimate_text_box_pixels(label["text"], fontsize)
        best = None
        best_score = None
        for dx_pt, dy_pt in candidate_offsets_points:
            center_x = anchor_px[0] + dx_pt * px_per_pt
            center_y = anchor_px[1] + dy_pt * px_per_pt
            rect = _rect_from_center(center_x, center_y, width_px, height_px)
            score = 0.03 * (abs(dx_pt) + abs(dy_pt))

            if keep_inside_axes:
                if (
                    rect[0] < axes_bbox.x0 + 3.0
                    or rect[2] > axes_bbox.x1 - 3.0
                    or rect[1] < axes_bbox.y0 + 3.0
                    or rect[3] > axes_bbox.y1 - 3.0
                ):
                    score += 1e6
            else:
                if rect[1] < axes_bbox.y0 - 28.0 or rect[3] > axes_bbox.y1 + 28.0:
                    score += 4e5
                if prefer_right_of_axes and rect[0] < axes_bbox.x1 + 4.0:
                    score += 3e5

            if _rect_contains_point(rect, anchor_px, pad=7.0):
                score += 3e5
            for point_px in anchor_points:
                if _rect_contains_point(rect, point_px, pad=6.0):
                    score += 7e4
            for other_rect in existing_rects:
                overlap = _rect_overlap_area(rect, other_rect)
                if overlap > 0.0:
                    score += 1e6 + overlap

            if best_score is None or score < best_score:
                best = {"dx_pt": dx_pt, "dy_pt": dy_pt, "rect": rect}
                best_score = score

        if best is None:
            best = {
                "dx_pt": candidate_offsets_points[0][0],
                "dy_pt": candidate_offsets_points[0][1],
                "rect": _rect_from_center(anchor_px[0], anchor_px[1], width_px, height_px),
            }
        existing_rects.append(best["rect"])
        placements.append(best)

    return placements


def boxed_end_labels(
    ax: plt.Axes,
    end_points: list[dict],
    *,
    x_last: float,
    decimals: int = 2,
    min_gap_frac: float = 0.070,
) -> None:
    if not end_points:
        return
    y_low, y_high = ax.get_ylim()
    adjusted = spread_label_positions(
        end_points,
        y_low=y_low,
        y_high=y_high,
        min_gap_frac=min_gap_frac,
        pad_frac=0.055,
    )
    fig = ax.figure
    fig.canvas.draw()
    for item in adjusted:
        anchor_px = ax.transData.transform((x_last, float(item["y"])))
        label_px = ax.transData.transform((x_last, float(item["label_y"])))
        dy_points = float((label_px[1] - anchor_px[1]) * 72.0 / fig.dpi)
        bbox = dict(PRESENTATION_LINE_LABEL_BOX)
        bbox["edgecolor"] = item["color"]
        ax.annotate(
            f"{float(item['y']):.{decimals}f}",
            xy=(x_last, float(item["y"])),
            xytext=(24.0, dy_points),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="semibold",
            color=item["color"],
            bbox=bbox,
            arrowprops={
                "arrowstyle": "-",
                "color": item["color"],
                "linewidth": 0.75,
                "alpha": 0.75,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            clip_on=False,
            zorder=20,
        )


def boxed_point_labels(
    ax: plt.Axes,
    points: list[dict],
    *,
    decimals: int = 2,
    min_gap_frac: float = 0.055,
) -> None:
    if not points:
        return
    labels = [
        {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "text": f"{float(point['y']):.{decimals}f}",
            "color": point["color"],
        }
        for point in sorted(points, key=lambda item: (float(item["x"]), float(item["y"])))
    ]
    placements = _plan_offset_annotations(
        ax,
        labels,
        fontsize=7.0,
        candidate_offsets_points=[
            (12.0, 10.0),
            (12.0, -10.0),
            (-12.0, 10.0),
            (-12.0, -10.0),
            (18.0, 16.0),
            (18.0, -16.0),
            (-18.0, 16.0),
            (-18.0, -16.0),
        ],
        keep_inside_axes=True,
    )
    for label, placement in zip(labels, placements):
        bbox = dict(PRESENTATION_LINE_LABEL_BOX)
        bbox["edgecolor"] = label["color"]
        ax.annotate(
            label["text"],
            xy=(label["x"], label["y"]),
            xytext=(placement["dx_pt"], placement["dy_pt"]),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="semibold",
            color=label["color"],
            bbox=bbox,
            arrowprops={
                "arrowstyle": "-",
                "color": label["color"],
                "linewidth": 0.65,
                "alpha": 0.60,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            clip_on=False,
            zorder=20,
        )


def table_cell_styles_by_dataset(
    table: pd.DataFrame,
    value_cols: list[str],
    *,
    tolerance: float = PRESENTATION_TOLERANCE,
) -> pd.DataFrame:
    view = table.set_index("method")
    styles = pd.DataFrame("", index=view.index, columns=view.columns)
    palette = {
        "low": "background-color: #EAF7EA;",
        "mid": "background-color: #F4F4F5;",
        "high": "background-color: #FCEBEC;",
    }
    for col in value_cols:
        values = pd.to_numeric(table[col], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        spread = float(valid.max() - valid.min())
        if spread <= tolerance:
            for method in table["method"]:
                styles.loc[method, col] = palette["mid"]
            continue
        low = float(valid.min())
        high = float(valid.max())
        for _, row in table.iterrows():
            value = float(row[col])
            if np.isclose(value, low, atol=1e-12, rtol=0.0):
                styles.loc[row["method"], col] = palette["low"]
            elif np.isclose(value, high, atol=1e-12, rtol=0.0):
                styles.loc[row["method"], col] = palette["high"]
            else:
                styles.loc[row["method"], col] = palette["mid"]
    return styles


# Toggleable configuration for the new pre-presentation section.

from contextlib import contextmanager

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

# Section constants used by the helpers below; import-time rendering
# should follow the lightweight report defaults rather than forcing SVG/300 dpi.
SECTION_FONT_SIZE = REPORT_DEFAULTS["font_size"]
SECTION_DPI = REPORT_DEFAULTS["figure_dpi"]

SECTION_DATASET_ORDER = ["cifar10", WRN_DATASET_KEY]
SECTION_PRUNING_DATASET_ORDER = ["cifar10", WRN_DATASET_KEY]
SECTION_RMIA_DATASET_ORDER = ["cifar10", WRN_DATASET_KEY]
SECTION_ALL_METHODS = ["std", "ib", "ib_prw"]
SECTION_PRUNED_METHODS = ["ib", "ib_prw"]

SECTION_MEMORIZATION_EPOCHS = {
    "cifar10": 99,
    WRN_DATASET_KEY: 99,
}

# Used when difficulty_mode == "epoch_loss".
SECTION_EASY_HARD_EPOCHS = {
    "cifar10": 10,
    WRN_DATASET_KEY: 20,
}
SECTION_EASY_HARD_REFERENCE_METHOD = "std"

# Used when difficulty_mode == "pruning_rate_tertiles".
SECTION_PRUNING_ANCHOR_EPOCHS = {
    "cifar10": 99,
    WRN_DATASET_KEY: 99,
}
SECTION_PRUNING_ANCHOR_METHOD = "ib"

# Point 1: "epoch_loss" or "pruning_rate_tertiles".
SECTION_BIN_PLOT_DIFFICULTY_MODE = "epoch_loss"

# Point 2: default requested starter setting is pruning-rate bins.
SECTION_SCATTER_DIFFICULTY_MODE = "pruning_rate_tertiles"

# Points 4 and 6:
# - "all"
# - numeric threshold like 0.1 or 0.25
# - "top_10%", "top_25%", "top_50%", "top_75%"
SECTION_MEAN_MEMORIZATION_FILTER = "all"

# Point 6: easy / medium / hard trajectories need three bins,
# so the default is pruning-rate tertiles.
SECTION_BINNED_TRAJECTORY_DIFFICULTY_MODE = "pruning_rate_tertiles"

# Point 8.
SECTION_RMIA_MEM_THRESHOLD = 0.3
SECTION_RMIA_NORMALIZED_ENTROPY_THRESHOLD = 0.3


SECTION_DIFFICULTY_COLORS = {
    "easy": "#2D7FF9",
    "medium": "#F2C14E",
    "hard": "#D7263D",
}
SECTION_DIFFICULTY_HATCHES = {
    "easy": "",
    "medium": "xx",
    "hard": "///",
}
SECTION_MOVEMENT3_COLORS = {
    "down": "#0B3C5D",
    "same": "#8E8E8E",
    "up": "#F58518",
}
SECTION_MOVEMENT3_MARKERS = {
    "down": "v",
    "same": "o",
    "up": "^",
}
SECTION_RMIA_CMAP = LinearSegmentedColormap.from_list(
    "section_rmia_detection",
    ["#D7DCE2", "#F3A6A6", "#D95858", "#8C1D1D"],
)
SECTION_MATRIX_FILL = LinearSegmentedColormap.from_list(
    "section_rmia_matrix_fill",
    ["#FFFFFF", "#EDF2F7", "#D9E2EC"],
)


def save_figure(fig: plt.Figure, stem: str, dataset_key: str) -> None:
    if not SAVE_FIGURES:
        return
    out_dir = dataset_output_dir(dataset_key)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=FIG_DPI)
    if SAVE_VECTOR:
        fig.savefig(out_dir / f"{stem}.{VECTOR_FORMAT}", bbox_inches="tight")


@contextmanager
def temporary_dataset_order(dataset_keys):
    global DATASET_ORDER
    old_order = DATASET_ORDER[:]
    DATASET_ORDER = list(dataset_keys)
    try:
        yield
    finally:
        DATASET_ORDER = old_order


def section_dataset_label(bundle: dict) -> str:
    return {
        "cifar10": "CIFAR-10 / ResNet-20",
        WRN_DATASET_KEY: "CIFAR-100 / WRN-28-2",
        "tinyimagenet": "TinyImageNet / ResNet-50",
    }.get(bundle["dataset_key"], bundle["cfg"]["label"])


def section_method_keys(bundle: dict, requested=None) -> list[str]:
    requested = requested or SECTION_ALL_METHODS
    return [method_key for method_key in METHOD_ORDER if method_key in requested and method_key in bundle["method_keys"]]


def merge_on_sample_id(left: pd.DataFrame, right: pd.DataFrame, context: str = "") -> pd.DataFrame:
    merged = left.merge(right, on="sample_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError(f"Sample merge produced no rows in {context}")
    return merged.sort_values("sample_id").reset_index(drop=True)


def section_accuracy_label(bundle: dict, method_key: str) -> str:
    return f"{bundle['cfg']['methods'][method_key]['label']} ({final_test_accuracy(bundle, method_key):.2f}% test)"


def section_selector_info(filter_spec):
    if isinstance(filter_spec, dict) and "kind" in filter_spec and "label" in filter_spec and "stem" in filter_spec:
        return filter_spec
    if filter_spec is None:
        return {"kind": "all", "label": "All samples", "stem": "all"}
    if isinstance(filter_spec, (int, float)) and not isinstance(filter_spec, bool):
        value = float(filter_spec)
        return {
            "kind": "threshold",
            "threshold": value,
            "label": f"Memorization > {value:.2f}",
            "stem": f"gt_{str(value).replace('.', 'p').replace('-', 'neg_')}",
        }
    text = str(filter_spec).strip().lower().replace(" ", "")
    if text == "all":
        return {"kind": "all", "label": "All samples", "stem": "all"}
    if text.startswith("top"):
        digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
        if not digits:
            raise ValueError(f"Could not parse top-fraction filter: {filter_spec}")
        pct = float(digits)
        if pct <= 0.0 or pct > 100.0:
            raise ValueError(f"Top-fraction filter must be in (0, 100], got {pct}")
        return {
            "kind": "top_fraction",
            "fraction": pct / 100.0,
            "label": f"Top {pct:g}%",
            "stem": f"top_{str(int(pct) if pct.is_integer() else pct).replace('.', 'p')}",
        }
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"Unsupported memorization filter: {filter_spec}") from exc
    return {
        "kind": "threshold",
        "threshold": value,
        "label": f"Memorization > {value:.2f}",
        "stem": f"gt_{str(value).replace('.', 'p').replace('-', 'neg_')}",
    }


def section_select_scores(scores, filter_spec):
    selector = section_selector_info(filter_spec)
    arr = cap_memorization(np.asarray(scores, dtype=np.float64))
    arr = arr[np.isfinite(arr)]
    if selector["kind"] == "all":
        return arr
    if selector["kind"] == "threshold":
        return arr[arr > float(selector["threshold"])]
    if selector["kind"] == "top_fraction":
        if arr.size == 0:
            return arr
        n_select = max(1, int(np.ceil(arr.size * float(selector["fraction"]))))
        return np.sort(arr)[-n_select:]
    raise ValueError(f"Unknown selector kind: {selector['kind']}")


def section_info_box(ax, lines, *, loc="upper left", fontsize=8.2):
    lines = [str(line) for line in lines if str(line).strip()]
    if not lines:
        return
    positions = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="#334155",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "alpha": 0.92,
        },
        zorder=15,
    )


def section_shade_method_color(base_color: str, difficulty_label: str) -> tuple[float, float, float]:
    rgb = np.array(mpl.colors.to_rgb(base_color), dtype=np.float64)
    if difficulty_label == "easy":
        mixed = rgb * 0.70 + np.ones(3, dtype=np.float64) * 0.30
    elif difficulty_label == "medium":
        mixed = rgb * 0.88 + np.ones(3, dtype=np.float64) * 0.12
    else:
        mixed = np.clip(rgb * 0.70, 0.0, 1.0)
    return tuple(mixed.tolist())


def section_tail_auc_frames(
    *,
    dataset_keys=None,
    method_keys=None,
    tail_fraction: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_keys = list(dataset_keys or report_setting("active_datasets"))
    requested_methods = set(method_keys or report_setting("all_methods"))
    method_keys = [method_key for method_key in METHOD_ORDER if method_key in requested_methods]
    tail_fraction = float(report_setting("tail_auc_fraction") if tail_fraction is None else tail_fraction)
    trajectory_rows = []
    summary_rows = []

    for dataset_key in dataset_keys:
        bundle = report_bundle(dataset_key)
        active_methods = section_method_keys(bundle, method_keys)
        method_curves = {}
        for method_key in active_methods:
            curve_rows = []
            frame = bundle["mem_long"][method_key]
            for epoch in sorted(int(epoch) for epoch in frame["epoch"].unique()):
                sub = frame.loc[frame["epoch"] == epoch]
                scores = cap_memorization(sub["memorization"].to_numpy(dtype=np.float64))
                n_tail = max(1, int(np.ceil(tail_fraction * len(scores))))
                tail_scores = np.sort(scores)[-n_tail:]
                mean_mem = float(np.mean(tail_scores)) if len(tail_scores) else np.nan
                curve_rows.append({"epoch": epoch, "mean_memorization": mean_mem, "n": int(len(tail_scores))})
                trajectory_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": section_dataset_label(bundle),
                        "method_key": method_key,
                        "method": METHOD_LABELS[method_key],
                        "epoch": epoch,
                        "mean_memorization": mean_mem,
                        "n": int(len(scores)),
                        "tail_fraction": tail_fraction,
                    }
                )
            method_curves[method_key] = pd.DataFrame(curve_rows)

        std_curve = method_curves.get("std")
        std_final = (
            float(std_curve["mean_memorization"].iloc[-1])
            if std_curve is not None and not std_curve.empty
            else np.nan
        )
        onset_threshold = 0.5 * std_final if np.isfinite(std_final) else np.nan

        std_auc = np.nan
        method_summaries = {}
        for method_key, curve in method_curves.items():
            valid = curve.dropna(subset=["mean_memorization"])
            if len(valid) >= 2:
                epoch_span = float(valid["epoch"].iloc[-1] - valid["epoch"].iloc[0])
                auc = float(np.trapz(valid["mean_memorization"], valid["epoch"]) / max(epoch_span, 1e-8))
            elif len(valid) == 1:
                auc = float(valid["mean_memorization"].iloc[0])
            else:
                auc = np.nan
            if method_key == "std":
                std_auc = auc
            onset = valid.loc[valid["mean_memorization"] >= onset_threshold, "epoch"]
            method_summaries[method_key] = {
                "auc": auc,
                "final": float(valid["mean_memorization"].iloc[-1]) if len(valid) else np.nan,
                "peak": float(valid["mean_memorization"].max()) if len(valid) else np.nan,
                "onset_epoch": int(onset.iloc[0]) if len(onset) and np.isfinite(onset_threshold) else np.nan,
            }

        for method_key, stats in method_summaries.items():
            summary_rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": section_dataset_label(bundle),
                    "method_key": method_key,
                    "method": METHOD_LABELS[method_key],
                    "tail_fraction": tail_fraction,
                    "tail_n": int(method_curves[method_key]["n"].iloc[-1]) if method_key in method_curves else 0,
                    "auc": stats["auc"],
                    "auc_vs_standard": float(stats["auc"] / std_auc) if np.isfinite(std_auc) and std_auc > 0 else np.nan,
                    "final_mean": stats["final"],
                    "peak_mean": stats["peak"],
                    "onset_epoch_50pct_standard_final": stats["onset_epoch"],
                }
            )

    return pd.DataFrame(trajectory_rows), pd.DataFrame(summary_rows)


def section_display_tail_auc_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        display_note("**Tail memorization AUC table skipped:** no rows were available.")
        return
    view = summary[
        [
            "dataset",
            "method",
            "tail_n",
            "auc",
            "auc_vs_standard",
            "final_mean",
            "onset_epoch_50pct_standard_final",
        ]
    ].copy()
    view = view.rename(
        columns={
            "dataset": "Dataset",
            "method": "Method",
            "tail_n": "Tail n",
            "auc": "Tail AUC",
            "auc_vs_standard": "AUC / Standard",
            "final_mean": "Final tail mean",
            "onset_epoch_50pct_standard_final": "Onset epoch",
        }
    )
    display(
        view.style.format(
            {
                "Tail AUC": "{:.3f}",
                "AUC / Standard": "{:.2f}",
                "Final tail mean": "{:.3f}",
                "Onset epoch": "{:.0f}",
            },
            na_rep="--",
        )
        .hide(axis="index")
        .set_caption(
            "Current top-tail memorization: at each epoch, each method is evaluated on its own top-scoring samples. "
            "AUC is normalized by the observed epoch span."
        )
        .set_table_styles(
            [
                {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"), ("font-weight", "600"), ("font-size", "13px"), ("color", "#334155"), ("padding", "0 0 8px 0")]},
                {"selector": "th", "props": [("background-color", "#F8FAFC"), ("color", "#334155"), ("font-weight", "600"), ("border-bottom", "1px solid #CBD5E1"), ("padding", "6px 10px")]},
                {"selector": "td", "props": [("border-bottom", "1px solid #E5E7EB"), ("padding", "6px 10px"), ("text-align", "right")]},
            ]
        )
    )


def section_plot_tail_auc_summary(
    *,
    dataset_keys=None,
    method_keys=None,
    tail_fraction: float | None = None,
) -> pd.DataFrame:
    trajectory, summary = section_tail_auc_frames(
        dataset_keys=dataset_keys,
        method_keys=method_keys,
        tail_fraction=tail_fraction,
    )
    if trajectory.empty or summary.empty:
        display_note("**Tail memorization AUC skipped:** no trajectory rows were available.")
        return summary

    dataset_order = [key for key in report_setting("active_datasets") if key in trajectory["dataset_key"].unique()]
    requested_methods = set(method_keys or report_setting("all_methods"))
    method_order = [method_key for method_key in METHOD_ORDER if method_key in requested_methods]
    fig = plt.figure(figsize=(16.2, 7.0))
    gs = fig.add_gridspec(2, len(dataset_order), height_ratios=[2.15, 1.05], hspace=0.34, wspace=0.20)
    top_axes = [fig.add_subplot(gs[0, idx]) for idx in range(len(dataset_order))]
    ax_bar = fig.add_subplot(gs[1, :])

    handles = []
    labels = []
    for ax, dataset_key in zip(top_axes, dataset_order):
        sub = trajectory.loc[trajectory["dataset_key"] == dataset_key]
        bundle = report_bundle(dataset_key)
        for method_key in method_order:
            method_sub = sub.loc[sub["method_key"] == method_key].sort_values("epoch")
            if method_sub.empty:
                continue
            line, = ax.plot(
                method_sub["epoch"],
                method_sub["mean_memorization"],
                marker="o",
                markersize=4.0,
                linewidth=2.0,
                color=METHOD_COLORS[method_key],
                label=METHOD_LABELS[method_key],
            )
            if ax is top_axes[0]:
                handles.append(line)
                labels.append(METHOD_LABELS[method_key])
        ax.set_title(section_dataset_label(bundle), fontsize=10.0)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.03)
        ax.grid(True, axis="y", alpha=0.22)
    top_axes[0].set_ylabel("Mean memorization of current top-tail")
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.50, 0.965), ncol=len(handles), frameon=False, fontsize=8.6)

    x = np.arange(len(dataset_order), dtype=np.float64)
    width = min(0.22, 0.78 / max(len(method_order), 1))
    for idx, method_key in enumerate(method_order):
        vals = []
        for dataset_key in dataset_order:
            row = summary.loc[(summary["dataset_key"] == dataset_key) & (summary["method_key"] == method_key)]
            vals.append(float(row["auc_vs_standard"].iloc[0]) if not row.empty else np.nan)
        offsets = x + (idx - (len(method_order) - 1) / 2.0) * width
        bars = ax_bar.bar(
            offsets,
            vals,
            width=width,
            color=METHOD_COLORS[method_key],
            label=METHOD_LABELS[method_key],
            alpha=0.92,
        )
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.025,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.7,
                    color="#334155",
                )
    ax_bar.axhline(1.0, color="#475569", linewidth=1.0, linestyle="--")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([section_dataset_label(report_bundle(key)) for key in dataset_order])
    ax_bar.set_ylim(0.0, max(1.16, float(np.nanmax(summary["auc_vs_standard"].to_numpy(dtype=np.float64))) + 0.14))
    ax_bar.set_ylabel("Tail AUC / Standard")
    ax_bar.grid(True, axis="y", alpha=0.20)

    frac_label = f"{100.0 * float(summary['tail_fraction'].iloc[0]):.0f}%"
    section_footer(
        fig,
        [
            f"Tail: current method-native top {frac_label} at each epoch; no fixed sample cohort is assumed.",
            "AUC summarizes tail memorization onset and accumulation; lower values mean delayed or reduced tail memorization.",
        ],
        y=0.012,
    )
    finish_figure(
        fig,
        "cross_dataset_current_tail_auc",
        "cross_dataset",
        figure_label="Current top-tail memorization trajectories and normalized AUC",
        tight_rect=(0.0, 0.08, 1.0, 0.93),
    )
    section_display_tail_auc_table(summary)
    return summary


def section_plot_final_memorization_severity(
    *,
    dataset_keys=None,
    method_keys=None,
    thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or report_setting("active_datasets"))
    requested_methods = set(method_keys or report_setting("all_methods"))
    method_keys = [method_key for method_key in METHOD_ORDER if method_key in requested_methods]
    rows = []

    for dataset_key in dataset_keys:
        bundle = report_bundle(dataset_key)
        epoch = int(report_setting("memorization_epochs")[dataset_key])
        for method_key in section_method_keys(bundle, method_keys):
            frame = bundle["mem_long"][method_key]
            scores = cap_memorization(frame.loc[frame["epoch"] == epoch, "memorization"].to_numpy(dtype=np.float64))
            for threshold in thresholds:
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": section_dataset_label(bundle),
                        "method_key": method_key,
                        "method": METHOD_LABELS[method_key],
                        "epoch": epoch,
                        "threshold": float(threshold),
                        "percent_at_or_above": 100.0 * float(np.mean(scores >= float(threshold))),
                    }
                )

    table = pd.DataFrame(rows)
    if table.empty:
        display_note("**Final memorization severity skipped:** no rows were available.")
        return table

    dataset_order = [key for key in report_setting("active_datasets") if key in table["dataset_key"].unique()]
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(15.8, 4.6), sharey=False)
    axes = np.atleast_1d(axes)

    handles = []
    labels = []
    for ax, dataset_key in zip(axes, dataset_order):
        sub = table.loc[table["dataset_key"] == dataset_key]
        bundle = report_bundle(dataset_key)
        for method_key in method_keys:
            method_sub = sub.loc[sub["method_key"] == method_key].sort_values("threshold")
            if method_sub.empty:
                continue
            line, = ax.plot(
                method_sub["threshold"],
                method_sub["percent_at_or_above"],
                marker="o",
                markersize=4.2,
                linewidth=2.1,
                color=METHOD_COLORS[method_key],
                label=METHOD_LABELS[method_key],
            )
            if ax is axes[0]:
                handles.append(line)
                labels.append(METHOD_LABELS[method_key])
        ax.axvspan(0.5, 0.9, color="#F8FAFC", zorder=0)
        ax.set_title(section_dataset_label(bundle), fontsize=10.0)
        ax.set_xlabel("Memorization threshold")
        ax.set_xticks(list(thresholds)[::2])
        ax.grid(True, axis="y", alpha=0.22)
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylabel("Samples at or above threshold (%)")
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(handles), frameon=False, fontsize=8.8)

    section_footer(
        fig,
        [
            "Final-epoch threshold sweep; right side is the severe memorization tail.",
            "This avoids choosing a single cutoff and shows whether methods reduce mild, severe, or extreme memorization.",
        ],
        y=0.018,
    )
    finish_figure(
        fig,
        "cross_dataset_final_memorization_severity",
        "cross_dataset",
        figure_label="Final memorization severity curves",
        tight_rect=(0.0, 0.11, 1.0, 0.91),
    )
    return table


def section_plot_final_tail_profile(
    *,
    dataset_keys=None,
    method_keys=None,
    tail_percents: tuple[float, ...] = (1, 2, 5, 10, 15, 20, 25, 30, 40, 50),
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or report_setting("active_datasets"))
    requested_methods = set(method_keys or report_setting("all_methods"))
    method_keys = [method_key for method_key in METHOD_ORDER if method_key in requested_methods]
    rows = []

    for dataset_key in dataset_keys:
        bundle = report_bundle(dataset_key)
        epoch = int(report_setting("memorization_epochs")[dataset_key])
        for method_key in section_method_keys(bundle, method_keys):
            frame = bundle["mem_long"][method_key]
            scores = cap_memorization(frame.loc[frame["epoch"] == epoch, "memorization"].to_numpy(dtype=np.float64))
            sorted_scores = np.sort(scores)
            for percent in tail_percents:
                n_tail = max(1, int(np.ceil((float(percent) / 100.0) * len(sorted_scores))))
                tail_scores = sorted_scores[-n_tail:]
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": section_dataset_label(bundle),
                        "method_key": method_key,
                        "method": METHOD_LABELS[method_key],
                        "epoch": epoch,
                        "tail_percent": float(percent),
                        "tail_n": int(n_tail),
                        "tail_mean": float(np.mean(tail_scores)),
                        "tail_median": float(np.median(tail_scores)),
                    }
                )

    table = pd.DataFrame(rows)
    if table.empty:
        display_note("**Final tail profile skipped:** no rows were available.")
        return table

    dataset_order = [key for key in report_setting("active_datasets") if key in table["dataset_key"].unique()]
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(15.8, 4.7), sharey=True)
    axes = np.atleast_1d(axes)
    handles = []
    labels = []

    for ax, dataset_key in zip(axes, dataset_order):
        sub = table.loc[table["dataset_key"] == dataset_key]
        bundle = report_bundle(dataset_key)
        for method_key in method_keys:
            method_sub = sub.loc[sub["method_key"] == method_key].sort_values("tail_percent")
            if method_sub.empty:
                continue
            line, = ax.plot(
                method_sub["tail_percent"],
                method_sub["tail_mean"],
                marker="o",
                markersize=4.0,
                linewidth=2.1,
                color=METHOD_COLORS[method_key],
                label=METHOD_LABELS[method_key],
            )
            if ax is axes[0]:
                handles.append(line)
                labels.append(METHOD_LABELS[method_key])
        ax.axvspan(1, 10, color="#F8FAFC", zorder=0)
        ax.set_title(section_dataset_label(bundle), fontsize=10.0)
        ax.set_xlabel("Worst q% samples by final memorization")
        ax.set_xticks([1, 5, 10, 25, 50])
        ax.set_xlim(1, max(tail_percents))
        ax.grid(True, axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean memorization in worst q%")
    axes[0].set_ylim(0.0, 1.03)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(handles), frameon=False, fontsize=8.8)

    section_footer(
        fig,
        [
            "Final-epoch method-native tail profile: each method is evaluated on its own worst q% samples.",
            "The left shaded region is the extreme tail; widening q shows whether memorization is reduced or redistributed into broader tails.",
        ],
        y=0.018,
    )
    finish_figure(
        fig,
        "cross_dataset_final_tail_profile",
        "cross_dataset",
        figure_label="Final memorization tail profile",
        tight_rect=(0.0, 0.12, 1.0, 0.91),
    )
    return table


def section_plot_high_memorization_emergence(
    *,
    dataset_keys=None,
    method_keys=None,
    thresholds: tuple[float, ...] = (0.5, 0.8),
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or report_setting("active_datasets"))
    requested_methods = set(method_keys or report_setting("all_methods"))
    method_keys = [method_key for method_key in METHOD_ORDER if method_key in requested_methods]
    rows = []

    for dataset_key in dataset_keys:
        bundle = report_bundle(dataset_key)
        for method_key in section_method_keys(bundle, method_keys):
            frame = bundle["mem_long"][method_key]
            for epoch in sorted(int(epoch) for epoch in frame["epoch"].unique()):
                scores = cap_memorization(frame.loc[frame["epoch"] == epoch, "memorization"].to_numpy(dtype=np.float64))
                for threshold in thresholds:
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": section_dataset_label(bundle),
                            "method_key": method_key,
                            "method": METHOD_LABELS[method_key],
                            "epoch": epoch,
                            "threshold": float(threshold),
                            "percent_above_threshold": 100.0 * float(np.mean(scores >= float(threshold))),
                        }
                    )

    table = pd.DataFrame(rows)
    if table.empty:
        display_note("**High memorization emergence skipped:** no rows were available.")
        return table

    dataset_order = [key for key in report_setting("active_datasets") if key in table["dataset_key"].unique()]
    threshold_order = list(thresholds)
    fig, axes = plt.subplots(
        len(dataset_order),
        len(threshold_order),
        figsize=(12.8, 8.2),
        sharex=False,
        sharey="col",
    )
    axes = np.atleast_2d(axes)
    handles = []
    labels = []
    y_max_by_threshold = {}
    for threshold in threshold_order:
        sub = table.loc[np.isclose(table["threshold"], threshold), "percent_above_threshold"].to_numpy(dtype=np.float64)
        y_max_by_threshold[threshold] = max(1.0, float(np.nanmax(sub)) * 1.08)

    for row_idx, dataset_key in enumerate(dataset_order):
        bundle = report_bundle(dataset_key)
        dataset_sub = table.loc[table["dataset_key"] == dataset_key]
        for col_idx, threshold in enumerate(threshold_order):
            ax = axes[row_idx, col_idx]
            sub = dataset_sub.loc[np.isclose(dataset_sub["threshold"], threshold)]
            for method_key in method_keys:
                method_sub = sub.loc[sub["method_key"] == method_key].sort_values("epoch")
                if method_sub.empty:
                    continue
                line, = ax.plot(
                    method_sub["epoch"],
                    method_sub["percent_above_threshold"],
                    marker="o",
                    markersize=4.0,
                    linewidth=2.1,
                    color=METHOD_COLORS[method_key],
                    label=METHOD_LABELS[method_key],
                )
                if row_idx == 0 and col_idx == 0:
                    handles.append(line)
                    labels.append(METHOD_LABELS[method_key])
            if row_idx == 0:
                label = "Severe memorization" if np.isclose(threshold, 0.5) else "Extreme memorization"
                ax.set_title(f"{label}\nscore >= {threshold:.1f}", fontsize=10.0)
            ax.grid(True, axis="y", alpha=0.22)
            ax.set_ylim(0.0, y_max_by_threshold[threshold])
            ax.set_xlim(left=max(0.0, float(sub["epoch"].min()) - 1.5), right=float(sub["epoch"].max()) + 1.5)
            if row_idx == len(dataset_order) - 1:
                ax.set_xlabel("Epoch")
            if col_idx == 0:
                ax.set_ylabel("Samples above threshold (%)")
                ax.text(
                    -0.10,
                    0.50,
                    section_dataset_label(bundle),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10.0,
                    color="#334155",
                    fontweight="semibold",
                )

    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.50, 0.975), ncol=len(handles), frameon=False, fontsize=8.8)

    section_footer(
        fig,
        [
            "All-sample occupancy trajectories: each line shows the fraction of samples that have entered a high-memorization regime.",
            "Final saturation can hide differences; these panels show when severe and extreme memorization first appear and how quickly they spread.",
        ],
        y=0.015,
    )
    finish_figure(
        fig,
        "cross_dataset_high_memorization_emergence",
        "cross_dataset",
        figure_label="Emergence of severe and extreme memorization over training",
        tight_rect=(0.0, 0.08, 1.0, 0.94),
    )
    return table


MEM_STATE_EDGES = (0.3, 0.7)
MEM_STATE_LABELS = ("<0.3", "0.3-0.7", ">=0.7")
MEM_STATE_COLORS = ("#DCEAF7", "#8DB6D9", "#244A73")

EXCEEDANCE_CHECKPOINT_LABELS = ("Emergence window", "Late phase", "Final epoch")


def section_plot_memorization_state_flow(
    *,
    dataset_keys=None,
    method_keys=None,
    edges: tuple[float, float] = MEM_STATE_EDGES,
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or report_setting("active_datasets"))
    requested_methods = set(method_keys or report_setting("all_methods"))
    method_keys = [method_key for method_key in METHOD_ORDER if method_key in requested_methods]
    rows = []

    for dataset_key in dataset_keys:
        bundle = report_bundle(dataset_key)
        for method_key in section_method_keys(bundle, method_keys):
            frame = bundle["mem_long"][method_key]
            for epoch in sorted(int(epoch) for epoch in frame["epoch"].unique()):
                scores = cap_memorization(frame.loc[frame["epoch"] == epoch, "memorization"].to_numpy(dtype=np.float64))
                bins = np.digitize(scores, np.asarray(edges, dtype=np.float64), right=False)
                shares = [100.0 * float(np.mean(bins == idx)) for idx in range(len(MEM_STATE_LABELS))]
                for state_idx, (label, share) in enumerate(zip(MEM_STATE_LABELS, shares)):
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": section_dataset_label(bundle),
                            "method_key": method_key,
                            "method": METHOD_LABELS[method_key],
                            "epoch": epoch,
                            "state_idx": state_idx,
                            "state_label": label,
                            "share_pct": share,
                        }
                    )

    table = pd.DataFrame(rows)
    if table.empty:
        display_note("**Memorization state flow skipped:** no rows were available.")
        return table

    dataset_order = [key for key in report_setting("active_datasets") if key in table["dataset_key"].unique()]
    fig, axes = plt.subplots(
        len(dataset_order),
        len(method_keys),
        figsize=(13.8, 8.6),
        sharex=False,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row_idx, dataset_key in enumerate(dataset_order):
        bundle = report_bundle(dataset_key)
        dataset_sub = table.loc[table["dataset_key"] == dataset_key]
        for col_idx, method_key in enumerate(method_keys):
            ax = axes[row_idx, col_idx]
            method_sub = dataset_sub.loc[dataset_sub["method_key"] == method_key].sort_values(["epoch", "state_idx"])
            if method_sub.empty:
                ax.set_visible(False)
                continue
            epochs = np.array(sorted(method_sub["epoch"].unique()), dtype=np.float64)
            layers = []
            for state_idx in range(len(MEM_STATE_LABELS)):
                state_vals = (
                    method_sub.loc[method_sub["state_idx"] == state_idx]
                    .sort_values("epoch")["share_pct"]
                    .to_numpy(dtype=np.float64)
                )
                layers.append(state_vals)
            ax.stackplot(epochs, *layers, colors=MEM_STATE_COLORS, alpha=0.96, linewidth=0.0)
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[method_key], fontsize=10.0)
            ax.set_ylim(0.0, 100.0)
            ax.grid(True, axis="y", alpha=0.18)
            if row_idx == len(dataset_order) - 1:
                ax.set_xlabel("Epoch")
            if col_idx == 0:
                ax.set_ylabel("Samples (%)")
                ax.text(
                    -0.13,
                    0.50,
                    section_dataset_label(bundle),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10.0,
                    color="#334155",
                    fontweight="semibold",
                )

    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(MEM_STATE_COLORS, MEM_STATE_LABELS)
    ]
    fig.legend(
        legend_handles,
        MEM_STATE_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.975),
        ncol=len(MEM_STATE_LABELS),
        frameon=False,
        fontsize=8.8,
    )
    section_footer(
        fig,
        [
            "State flow over training: low (<0.3), intermediate (0.3-0.7), and high (>=0.7) memorization occupancy.",
            "A better method keeps mass out of the dark high-memorization band for longer; InfoBatch often shifts mass into the intermediate band, and IB-WL can further trim the late high band.",
        ],
        y=0.015,
    )
    finish_figure(
        fig,
        "cross_dataset_memorization_state_flow",
        "cross_dataset",
        figure_label="Memorization state flow over training",
        tight_rect=(0.0, 0.08, 1.0, 0.94),
    )
    return table


def section_plot_memorization_exceedance_checkpoints(
    *,
    dataset_keys=None,
    method_keys=None,
    checkpoint_fractions: tuple[float, ...] | None = None,
    threshold_min: float | None = None,
    threshold_max: float | None = None,
    threshold_points: int | None = None,
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or report_setting("active_datasets"))
    requested_methods = set(method_keys or report_setting("all_methods"))
    method_keys = [method_key for method_key in METHOD_ORDER if method_key in requested_methods]
    checkpoint_fractions = tuple(
        float(value)
        for value in (
            report_setting("exceedance_checkpoint_fractions")
            if checkpoint_fractions is None
            else checkpoint_fractions
        )
    )
    if not checkpoint_fractions:
        display_note("**Memorization exceedance curves skipped:** no checkpoint fractions were provided.")
        return pd.DataFrame()

    threshold_min = float(
        report_setting("exceedance_threshold_min") if threshold_min is None else threshold_min
    )
    threshold_max = float(
        report_setting("exceedance_threshold_max") if threshold_max is None else threshold_max
    )
    threshold_points = int(
        report_setting("exceedance_threshold_points") if threshold_points is None else threshold_points
    )
    thresholds = np.linspace(threshold_min, threshold_max, max(threshold_points, 2), dtype=np.float64)

    rows = []
    chosen_epochs_by_dataset = {}
    y_max_by_dataset = {}

    for dataset_key in dataset_keys:
        bundle = report_bundle(dataset_key)
        active_methods = section_method_keys(bundle, method_keys)
        if not active_methods:
            continue
        epoch_source = bundle["mem_long"][active_methods[0]]
        epochs = np.array(sorted(int(epoch) for epoch in epoch_source["epoch"].unique()), dtype=int)
        if epochs.size == 0:
            continue

        chosen_epochs = []
        for idx, fraction in enumerate(checkpoint_fractions):
            clipped = float(np.clip(fraction, 0.0, 1.0))
            epoch_idx = int(round((len(epochs) - 1) * clipped))
            epoch_idx = int(np.clip(epoch_idx, 0, len(epochs) - 1))
            chosen_epochs.append(int(epochs[epoch_idx]))
        if chosen_epochs:
            chosen_epochs[-1] = int(epochs[-1])
        chosen_epochs_by_dataset[dataset_key] = chosen_epochs

        dataset_y_max = 0.0
        for checkpoint_idx, epoch in enumerate(chosen_epochs):
            checkpoint_label = EXCEEDANCE_CHECKPOINT_LABELS[
                min(checkpoint_idx, len(EXCEEDANCE_CHECKPOINT_LABELS) - 1)
            ]
            for method_key in active_methods:
                frame = bundle["mem_long"][method_key]
                scores = cap_memorization(
                    frame.loc[frame["epoch"] == epoch, "memorization"].to_numpy(dtype=np.float64)
                )
                if scores.size == 0:
                    continue
                shares = np.array([100.0 * float(np.mean(scores >= threshold)) for threshold in thresholds])
                dataset_y_max = max(dataset_y_max, float(np.nanmax(shares)))
                for threshold, share_pct in zip(thresholds, shares):
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": section_dataset_label(bundle),
                            "checkpoint_idx": int(checkpoint_idx),
                            "checkpoint_label": checkpoint_label,
                            "epoch": int(epoch),
                            "method_key": method_key,
                            "method": METHOD_LABELS[method_key],
                            "threshold": float(threshold),
                            "percent_above_threshold": float(share_pct),
                        }
                    )

        y_cap = min(100.0, math.ceil(dataset_y_max / 5.0) * 5.0 + 5.0) if dataset_y_max > 0 else 5.0
        y_max_by_dataset[dataset_key] = y_cap

    table = pd.DataFrame(rows)
    if table.empty:
        display_note("**Memorization exceedance curves skipped:** no rows were available.")
        return table

    dataset_order = [key for key in report_setting("active_datasets") if key in table["dataset_key"].unique()]
    n_cols = len(checkpoint_fractions)
    fig, axes = plt.subplots(
        len(dataset_order),
        n_cols,
        figsize=(4.45 * n_cols, 2.95 * len(dataset_order) + 0.35),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for row_idx, dataset_key in enumerate(dataset_order):
        bundle = report_bundle(dataset_key)
        dataset_sub = table.loc[table["dataset_key"] == dataset_key]
        chosen_epochs = chosen_epochs_by_dataset.get(dataset_key, [])
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            checkpoint_label = EXCEEDANCE_CHECKPOINT_LABELS[min(col_idx, len(EXCEEDANCE_CHECKPOINT_LABELS) - 1)]
            epoch = chosen_epochs[col_idx] if col_idx < len(chosen_epochs) else None
            checkpoint_sub = dataset_sub.loc[dataset_sub["checkpoint_idx"] == col_idx]
            if checkpoint_sub.empty:
                ax.set_visible(False)
                continue

            for method_key in section_method_keys(bundle, method_keys):
                method_sub = checkpoint_sub.loc[checkpoint_sub["method_key"] == method_key].sort_values("threshold")
                if method_sub.empty:
                    continue
                ax.plot(
                    method_sub["threshold"],
                    method_sub["percent_above_threshold"],
                    color=bundle["cfg"]["methods"][method_key]["color"],
                    linewidth=2.15,
                    label=METHOD_LABELS[method_key],
                )

            if row_idx == 0:
                ax.set_title(checkpoint_label, fontsize=10.0)
            if col_idx == 0:
                ax.set_ylabel(f"{section_dataset_label(bundle)}\nSamples >= threshold (%)")
            if row_idx == len(dataset_order) - 1:
                ax.set_xlabel("Memorization threshold")
            ax.text(
                0.98,
                0.95,
                f"epoch {epoch}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.9,
                color="#475569",
            )
            ax.set_xlim(threshold_min, threshold_max)
            ax.set_ylim(0.0, y_max_by_dataset[dataset_key])
            ax.grid(True, alpha=0.20)

    legend_handles = [
        mpl.lines.Line2D([0], [0], color=METHOD_COLORS[method_key], linewidth=2.15, label=METHOD_LABELS[method_key])
        for method_key in method_keys
    ]
    fig.legend(
        legend_handles,
        [METHOD_LABELS[method_key] for method_key in method_keys],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.8,
    )
    section_footer(
        fig,
        [
            "At each checkpoint, the curve shows the share of samples with memorization >= threshold.",
            "Lower curves in the upper-threshold range mean delayed or reduced severe memorization. Low-threshold crossings reflect redistribution into intermediate memorization bins.",
            "Each dataset row uses its own y-scale so method separation stays legible.",
        ],
        y=0.015,
    )
    finish_figure(
        fig,
        "cross_dataset_memorization_exceedance_checkpoints",
        "cross_dataset",
        figure_label="Checkpointed memorization exceedance curves across thresholds",
        tight_rect=(0.0, 0.06, 1.0, 0.94),
    )
    return table


def exposure_metric_label(threshold: float) -> str:
    threshold = float(threshold)
    if threshold <= 0.0:
        return "All-score exposure"
    return f"Excess exposure >= {threshold:.1f}"


def exposure_metric_key(threshold: float) -> str:
    threshold = float(threshold)
    if threshold <= 0.0:
        return "memorization_exposure"
    return f"excess_exposure_ge_{threshold:.1f}"


def compute_sample_level_temporal_exposure(
    *,
    dataset_keys=None,
    method_keys=None,
    thresholds: tuple[float, ...] | None = None,
    n_boot: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_keys = list(dataset_keys or report_setting("active_datasets"))
    requested_methods = set(method_keys or report_setting("all_methods"))
    method_keys = [method_key for method_key in METHOD_ORDER if method_key in requested_methods]
    thresholds = tuple(
        float(value)
        for value in (report_setting("exposure_thresholds") if thresholds is None else thresholds)
    )
    n_boot = int(report_setting("exposure_bootstrap_samples") if n_boot is None else n_boot)
    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []
    paired_rows = []

    for dataset_key in dataset_keys:
        bundle = report_bundle(dataset_key)
        active_methods = section_method_keys(bundle, method_keys)
        if len(active_methods) < 2:
            continue

        common_epochs = None
        for method_key in active_methods:
            epochs = set(int(epoch) for epoch in bundle["mem_long"][method_key]["epoch"].unique())
            common_epochs = epochs if common_epochs is None else common_epochs & epochs
        if not common_epochs:
            continue
        epoch_values = np.array(sorted(common_epochs), dtype=np.float64)
        epoch_span = max(float(epoch_values[-1] - epoch_values[0]), 1.0)

        exposure_by_threshold = {threshold: {} for threshold in thresholds}
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
                    raise ValueError(
                        f"Sample-id order mismatch while computing exposure for {dataset_key} / {method_key}."
                    )
                epoch_scores.append(cap_memorization(sub["memorization"].to_numpy(dtype=np.float64)))
            score_matrix = np.vstack(epoch_scores)

            for threshold in thresholds:
                if threshold <= 0.0:
                    transformed = score_matrix
                else:
                    transformed = np.maximum(score_matrix - threshold, 0.0) / max(1.0 - threshold, 1e-12)
                exposure = np.trapz(transformed, epoch_values, axis=0) / epoch_span
                exposure_by_threshold[threshold][method_key] = pd.DataFrame(
                    {"sample_id": sample_ids, "exposure": exposure}
                )
                summary_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": section_dataset_label(bundle),
                        "metric": exposure_metric_key(threshold),
                        "metric_label": exposure_metric_label(threshold),
                        "threshold": threshold,
                        "method_key": method_key,
                        "method": METHOD_LABELS[method_key],
                        "mean_exposure": float(np.mean(exposure)),
                        "median_exposure": float(np.median(exposure)),
                        "q90_exposure": float(np.quantile(exposure, 0.90)),
                        "q95_exposure": float(np.quantile(exposure, 0.95)),
                    }
                )

        for threshold in thresholds:
            metric = exposure_metric_key(threshold)
            merged = None
            for method_key in active_methods:
                exposure_frame = exposure_by_threshold[threshold][method_key].rename(
                    columns={"exposure": f"exposure_{method_key}"}
                )
                merged = exposure_frame if merged is None else merged.merge(exposure_frame, on="sample_id", how="inner")
            if merged is None or merged.empty:
                continue

            for left_key, right_key in [("std", "ib"), ("ib", "ib_prw"), ("std", "ib_prw")]:
                if left_key not in active_methods or right_key not in active_methods:
                    continue
                diff = (
                    merged[f"exposure_{left_key}"].to_numpy(dtype=np.float64)
                    - merged[f"exposure_{right_key}"].to_numpy(dtype=np.float64)
                )
                point = float(np.mean(diff))
                if n_boot > 0 and diff.size:
                    boot_values = np.empty(n_boot, dtype=np.float64)
                    for boot_idx in range(n_boot):
                        idx = rng.integers(0, diff.size, size=diff.size)
                        boot_values[boot_idx] = float(np.mean(diff[idx]))
                    ci_low, ci_high = np.percentile(boot_values, [2.5, 97.5])
                else:
                    ci_low = np.nan
                    ci_high = np.nan
                paired_rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset": section_dataset_label(bundle),
                        "metric": metric,
                        "metric_label": exposure_metric_label(threshold),
                        "threshold": threshold,
                        "comparison": f"{METHOD_LABELS[left_key]} - {METHOD_LABELS[right_key]}",
                        "comparison_key": f"{left_key}-{right_key}",
                        "mean_diff": point,
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "pct_positive": 100.0 * float(np.mean(diff > 0.0)),
                        "pct_negative": 100.0 * float(np.mean(diff < 0.0)),
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def section_plot_memorization_exposure_evidence(
    *,
    dataset_keys=None,
    method_keys=None,
    thresholds: tuple[float, ...] | None = None,
    n_boot: int | None = None,
) -> dict[str, pd.DataFrame]:
    summary, paired = compute_sample_level_temporal_exposure(
        dataset_keys=dataset_keys,
        method_keys=method_keys,
        thresholds=thresholds,
        n_boot=n_boot,
    )
    if summary.empty:
        display_note("**Memorization exposure evidence skipped:** no exposure rows were available.")
        return {"summary": summary, "paired": paired}

    dataset_order = [key for key in report_setting("active_datasets") if key in summary["dataset_key"].unique()]
    metric_order = list(dict.fromkeys(summary["metric"].tolist()))
    active_methods = [method_key for method_key in METHOD_ORDER if method_key in summary["method_key"].unique()]

    fig, axes = plt.subplots(
        len(metric_order),
        len(dataset_order),
        figsize=(4.45 * len(dataset_order), 2.72 * len(metric_order) + 0.25),
        squeeze=False,
    )
    for row_idx, metric in enumerate(metric_order):
        metric_sub = summary.loc[summary["metric"] == metric]
        metric_label = metric_sub["metric_label"].iloc[0]
        for col_idx, dataset_key in enumerate(dataset_order):
            ax = axes[row_idx, col_idx]
            dataset_sub = metric_sub.loc[metric_sub["dataset_key"] == dataset_key]
            if dataset_sub.empty:
                ax.set_visible(False)
                continue
            values = []
            for method_key in active_methods:
                row = dataset_sub.loc[dataset_sub["method_key"] == method_key]
                values.append(float(row["mean_exposure"].iloc[0]) if not row.empty else np.nan)
            x = np.arange(len(active_methods), dtype=np.float64)
            ax.bar(
                x,
                values,
                color=[METHOD_COLORS[method_key] for method_key in active_methods],
                width=0.66,
                alpha=0.90,
            )
            y_max = max(float(np.nanmax(values)), 1e-8)
            for x_value, y_value in zip(x, values):
                if np.isfinite(y_value):
                    ax.text(
                        x_value,
                        y_value + 0.025 * y_max,
                        f"{y_value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        color="#334155",
                    )
            if row_idx == 0:
                ax.set_title(section_dataset_label(report_bundle(dataset_key)), fontsize=10.0)
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_LABELS[method_key] for method_key in active_methods], rotation=20, ha="right")
            ax.set_ylim(0.0, y_max * 1.26)
            ax.grid(True, axis="y", alpha=0.20)

            if not paired.empty and {"std", "ib", "ib_prw"}.issubset(set(active_methods)):
                diff_std_ib = paired.loc[
                    (paired["dataset_key"] == dataset_key)
                    & (paired["metric"] == metric)
                    & (paired["comparison_key"] == "std-ib")
                ]
                diff_ib_wl = paired.loc[
                    (paired["dataset_key"] == dataset_key)
                    & (paired["metric"] == metric)
                    & (paired["comparison_key"] == "ib-ib_prw")
                ]
                if not diff_std_ib.empty and not diff_ib_wl.empty:
                    row_a = diff_std_ib.iloc[0]
                    row_b = diff_ib_wl.iloc[0]
                    ax.text(
                        0.02,
                        0.94,
                        (
                            f"S-I {row_a['mean_diff']:.3g} "
                            f"[{row_a['ci95_low']:.3g}, {row_a['ci95_high']:.3g}]\n"
                            f"I-W {row_b['mean_diff']:.3g} "
                            f"[{row_b['ci95_low']:.3g}, {row_b['ci95_high']:.3g}]"
                        ),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6.8,
                        color="#475569",
                        bbox={
                            "boxstyle": "round,pad=0.18",
                            "fc": "white",
                            "ec": "#CBD5E1",
                            "lw": 0.60,
                            "alpha": 0.92,
                        },
                    )

    section_footer(
        fig,
        [
            "Exposure is the per-sample area under the memorization trajectory through the final epoch, averaged over samples.",
            "Excess exposure >= t integrates only the part of each trajectory above threshold t; brackets show paired bootstrap 95% CIs for method differences.",
        ],
        y=0.012,
    )
    finish_figure(
        fig,
        "cross_dataset_memorization_exposure_evidence",
        "cross_dataset",
        figure_label="Cumulative memorization exposure by the end of training",
        tight_rect=(0.0, 0.08, 1.0, 0.96),
    )
    return {"summary": summary, "paired": paired}


def section_plot_memorization_onset_trajectories(
    *,
    dataset_keys=None,
    method_keys=None,
    thresholds: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    dataset_keys = list(dataset_keys or report_setting("active_datasets"))
    requested_methods = set(method_keys or report_setting("all_methods"))
    method_keys = [method_key for method_key in METHOD_ORDER if method_key in requested_methods]
    thresholds = tuple(
        float(value)
        for value in (report_setting("onset_plot_thresholds") if thresholds is None else thresholds)
    )
    rows = []

    for dataset_key in dataset_keys:
        bundle = report_bundle(dataset_key)
        common_epochs = None
        for method_key in section_method_keys(bundle, method_keys):
            epochs = set(int(epoch) for epoch in bundle["mem_long"][method_key]["epoch"].unique())
            common_epochs = epochs if common_epochs is None else common_epochs & epochs
        if not common_epochs:
            continue
        for method_key in section_method_keys(bundle, method_keys):
            frame = bundle["mem_long"][method_key]
            for epoch in sorted(common_epochs):
                scores = cap_memorization(frame.loc[frame["epoch"] == epoch, "memorization"].to_numpy(dtype=np.float64))
                for threshold in thresholds:
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": section_dataset_label(bundle),
                            "method_key": method_key,
                            "method": METHOD_LABELS[method_key],
                            "epoch": int(epoch),
                            "threshold": threshold,
                            "percent_above_threshold": 100.0 * float(np.mean(scores >= threshold)),
                        }
                    )

    table = pd.DataFrame(rows)
    if table.empty:
        display_note("**Memorization onset trajectories skipped:** no rows were available.")
        return table

    dataset_order = [key for key in report_setting("active_datasets") if key in table["dataset_key"].unique()]
    fig, axes = plt.subplots(
        len(dataset_order),
        len(thresholds),
        figsize=(5.50 * len(thresholds), 2.62 * len(dataset_order) + 0.35),
        squeeze=False,
        sharex=False,
    )
    for row_idx, dataset_key in enumerate(dataset_order):
        bundle = report_bundle(dataset_key)
        dataset_sub = table.loc[table["dataset_key"] == dataset_key]
        for col_idx, threshold in enumerate(thresholds):
            ax = axes[row_idx, col_idx]
            threshold_sub = dataset_sub.loc[np.isclose(dataset_sub["threshold"], threshold)]
            for method_key in section_method_keys(bundle, method_keys):
                method_sub = threshold_sub.loc[threshold_sub["method_key"] == method_key].sort_values("epoch")
                ax.plot(
                    method_sub["epoch"],
                    method_sub["percent_above_threshold"],
                    color=METHOD_COLORS[method_key],
                    linewidth=2.0,
                    marker="o",
                    markersize=3.0,
                    label=METHOD_LABELS[method_key],
                )
            if row_idx == 0:
                ax.set_title(f"Share with memorization >= {threshold:.1f}", fontsize=10.0)
            if col_idx == 0:
                ax.set_ylabel(f"{section_dataset_label(bundle)}\nSamples (%)")
            if row_idx == len(dataset_order) - 1:
                ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.22)

    handles = [
        mpl.lines.Line2D([0], [0], color=METHOD_COLORS[method_key], linewidth=2.0, marker="o", label=METHOD_LABELS[method_key])
        for method_key in method_keys
    ]
    fig.legend(handles, [handle.get_label() for handle in handles], loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=len(handles), frameon=False)
    section_footer(
        fig,
        [
            "Onset view: Standard occupies intermediate and severe memorization thresholds earlier; InfoBatch and IB-WL lag behind during the growth phase.",
        ],
        y=0.012,
    )
    finish_figure(
        fig,
        "cross_dataset_memorization_onset_trajectories",
        "cross_dataset",
        figure_label="Onset of intermediate and severe memorization",
        tight_rect=(0.0, 0.07, 1.0, 0.94),
    )
    return table


def section_difficulty_labels(mode: str) -> list[str]:
    if mode == "epoch_loss":
        return ["easy", "hard"]
    if mode == "pruning_rate_tertiles":
        return ["easy", "medium", "hard"]
    raise ValueError(f"Unknown difficulty mode: {mode}")


def section_difficulty_summary_lines(meta: dict) -> list[str]:
    if meta["mode"] == "epoch_loss":
        return [
            (
                "Difficulty: easy / hard from "
                f"{METHOD_LABELS.get(meta['reference_method'], meta['reference_method'])} loss @ epoch {meta['anchor_epoch']}"
            )
        ]
    return [
        (
            "Difficulty: easy / medium / hard from "
            f"{METHOD_LABELS.get(meta['reference_method'], meta['reference_method'])} "
            f"{meta['pruning_label'].lower()} @ epoch {meta['anchor_epoch']}"
        ),
        "High pruning = easy",
    ]


def section_build_difficulty_frame(
    bundle: dict,
    mode: str,
    *,
    difficulty_epoch_by_dataset=None,
    pruning_epoch_by_dataset=None,
    epoch_reference_method="std",
    pruning_method_key="ib",
):
    dataset_key = bundle["dataset_key"]

    if mode == "epoch_loss":
        anchor_epoch = int((difficulty_epoch_by_dataset or {}).get(dataset_key, bundle["cfg"]["end_epoch"]))
        method_cfg = bundle["cfg"]["methods"][epoch_reference_method]
        evo = evolution_epoch_frame(dataset_key, epoch_reference_method, method_cfg, anchor_epoch)
        anchor = evo[["sample_id", "loss"]].dropna().rename(columns={"loss": "difficulty_value"}).copy()
        split_value = float(anchor["difficulty_value"].mean())
        anchor["difficulty_label"] = np.where(anchor["difficulty_value"] <= split_value, "easy", "hard")
        anchor["difficulty_order"] = anchor["difficulty_label"].map({"easy": 0, "hard": 1}).astype(int)
        meta = {
            "mode": mode,
            "labels": ["easy", "hard"],
            "anchor_epoch": anchor_epoch,
            "reference_method": epoch_reference_method,
            "split_value": split_value,
        }
        return anchor.sort_values("sample_id").reset_index(drop=True), meta

    if mode == "pruning_rate_tertiles":
        anchor_epoch = int((pruning_epoch_by_dataset or {}).get(dataset_key, bundle["cfg"]["end_epoch"]))
        method_cfg = bundle["cfg"]["methods"][pruning_method_key]
        evo = evolution_epoch_frame(dataset_key, pruning_method_key, method_cfg, anchor_epoch)
        pruning_col, pruning_label = resolve_pruning_metric(bundle, pruning_method_key, evo)
        if pruning_col is None:
            raise KeyError(
                f"No pruning metric was found for {dataset_key} / {pruning_method_key} at epoch {anchor_epoch}"
            )
        anchor = (
            evo[["sample_id", pruning_col]]
            .dropna()
            .rename(columns={pruning_col: "difficulty_value"})
            .copy()
        )
        low = float(anchor["difficulty_value"].min())
        high = float(anchor["difficulty_value"].max())
        if np.isclose(low, high):
            high = low + 1e-8
        edges = np.linspace(low, high, 4, dtype=np.float64)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        low_to_high = pd.cut(
            anchor["difficulty_value"],
            bins=edges,
            labels=False,
            include_lowest=True,
        ).astype(int)
        label_map = {2: "easy", 1: "medium", 0: "hard"}
        anchor["difficulty_label"] = [label_map[int(idx)] for idx in low_to_high.to_numpy()]
        anchor["difficulty_order"] = anchor["difficulty_label"].map({"easy": 0, "medium": 1, "hard": 2}).astype(int)
        meta = {
            "mode": mode,
            "labels": ["easy", "medium", "hard"],
            "anchor_epoch": anchor_epoch,
            "reference_method": pruning_method_key,
            "pruning_label": pruning_label,
            "edges": edges,
        }
        return anchor.sort_values("sample_id").reset_index(drop=True), meta

    raise ValueError(f"Unknown difficulty mode: {mode}")


def section_build_method_frame(
    bundle: dict,
    epoch: int,
    method_key: str,
    *,
    difficulty_frame=None,
    include_rmia=False,
    require_rmia=False,
) -> pd.DataFrame:
    method_cfg = bundle["cfg"]["methods"][method_key]
    mem = get_method_epoch_mem(bundle, method_key, epoch)
    evo = evolution_epoch_frame(bundle["dataset_key"], method_key, method_cfg, epoch)
    merged = strict_sample_merge(
        mem,
        evo,
        context=f"{bundle['dataset_key']}::{method_key}::section_frame::epoch_{epoch}",
    )
    if difficulty_frame is not None:
        merged = merge_on_sample_id(
            merged,
            difficulty_frame[["sample_id", "difficulty_label", "difficulty_order"]],
            context=f"{bundle['dataset_key']}::{method_key}::difficulty",
        )
    if include_rmia:
        if rmia_join_allowed(bundle, method_key):
            merged = strict_sample_merge(
                merged,
                bundle["rmia_profile"][method_key],
                context=f"{bundle['dataset_key']}::{method_key}::rmia",
            )
        elif require_rmia:
            raise RuntimeError(
                f"RMIA join is not allowed for {bundle['dataset_key']} / {method_key}"
            )
    merged["method_key"] = method_key
    merged["method"] = method_cfg["label"]
    return merged


def section_movement_three_way_frame(bundle: dict, epoch: int, method_key: str) -> pd.DataFrame:
    move = movement_frame(bundle, epoch, method_key)[["sample_id", "movement", "delta_bins"]].copy()
    move["movement3"] = np.where(
        move["delta_bins"] < 0,
        "down",
        np.where(move["delta_bins"] > 0, "up", "same"),
    )
    return move


def section_build_filtered_mean_mem_summary(mem_frame: pd.DataFrame, filter_spec) -> pd.DataFrame:
    rows = []
    for epoch, chunk in mem_frame.groupby("epoch", sort=True):
        selected = section_select_scores(chunk["memorization"].to_numpy(dtype=np.float64), filter_spec)
        rows.append(
            {
                "epoch": int(epoch),
                "mean_memorization": float(np.mean(selected)) if selected.size else np.nan,
                "n_selected": int(selected.size),
                "n_total": int(len(chunk)),
            }
        )
    return pd.DataFrame(rows)


def section_build_filtered_mean_mem_by_difficulty(
    bundle: dict,
    difficulty_frame: pd.DataFrame,
    filter_spec,
    *,
    method_keys=None,
):
    rows = []
    active_methods = section_method_keys(bundle, method_keys)
    for method_key in active_methods:
        method_mem = bundle["mem_long"][method_key]
        for epoch, chunk in method_mem.groupby("epoch", sort=True):
            merged = merge_on_sample_id(
                chunk[["sample_id", "memorization"]],
                difficulty_frame[["sample_id", "difficulty_label"]],
                context=f"{bundle['dataset_key']}::{method_key}::difficulty_mean::epoch_{epoch}",
            )
            for difficulty_label in sorted(difficulty_frame["difficulty_label"].unique()):
                scores = merged.loc[merged["difficulty_label"] == difficulty_label, "memorization"].to_numpy(dtype=np.float64)
                selected = section_select_scores(scores, filter_spec)
                rows.append(
                    {
                        "method_key": method_key,
                        "method": bundle["cfg"]["methods"][method_key]["label"],
                        "epoch": int(epoch),
                        "difficulty_label": difficulty_label,
                        "mean_memorization": float(np.mean(selected)) if selected.size else np.nan,
                        "n_selected": int(selected.size),
                        "n_anchor": int((difficulty_frame["difficulty_label"] == difficulty_label).sum()),
                    }
                )
    return pd.DataFrame(rows)


def section_detection_vmax(bundle: dict, method_keys=None) -> float:
    vals = []
    for method_key in section_method_keys(bundle, method_keys):
        if not rmia_join_allowed(bundle, method_key):
            continue
        profile = bundle["rmia_profile"][method_key]
        if "detection_rate_0p01pct" not in profile.columns:
            continue
        arr = 100.0 * profile["detection_rate_0p01pct"].to_numpy(dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return 1.0
    flat = np.concatenate(vals)
    return max(float(np.nanpercentile(flat, 99.5)), 1e-6)


def section_plot_memorization_bin_composition(
    bundle: dict,
    *,
    epoch: int,
    difficulty_mode: str,
    difficulty_epoch_by_dataset=None,
    pruning_epoch_by_dataset=None,
    method_keys=None,
) -> None:
    difficulty_frame, difficulty_meta = section_build_difficulty_frame(
        bundle,
        difficulty_mode,
        difficulty_epoch_by_dataset=difficulty_epoch_by_dataset,
        pruning_epoch_by_dataset=pruning_epoch_by_dataset,
        epoch_reference_method=SECTION_EASY_HARD_REFERENCE_METHOD,
        pruning_method_key=SECTION_PRUNING_ANCHOR_METHOD,
    )

    active_methods = section_method_keys(bundle, method_keys)
    x = np.arange(len(MEM_BIN_LABELS))
    width = 0.80 / max(len(active_methods), 1)
    fig, ax = plt.subplots(figsize=(11.4, 4.9))

    method_lines = []
    for idx, method_key in enumerate(active_methods):
        method_cfg = bundle["cfg"]["methods"][method_key]
        frame = merge_on_sample_id(
            get_method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]],
            difficulty_frame[["sample_id", "difficulty_label"]],
            context=f"{bundle['dataset_key']}::{method_key}::bin_plot::epoch_{epoch}",
        )
        frame["memorization"] = cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
        mask = (frame["memorization"] > MEMORIZED_THRESHOLD) & (frame["memorization"] <= 1.0)
        bins = memorization_bin_index(frame.loc[mask, "memorization"].to_numpy(dtype=np.float64))
        difficulty = frame.loc[mask, "difficulty_label"].to_numpy()
        positions = x + (idx - (len(active_methods) - 1) / 2.0) * width
        bottoms = np.zeros(len(MEM_BIN_LABELS), dtype=np.float64)

        for difficulty_label in difficulty_meta["labels"]:
            counts = np.array(
                [
                    np.sum((bins == bin_id) & (difficulty == difficulty_label))
                    for bin_id in range(len(MEM_BIN_LABELS))
                ],
                dtype=np.float64,
            )
            ax.bar(
                positions,
                counts,
                bottom=bottoms,
                width=width,
                color=section_shade_method_color(method_cfg["color"], difficulty_label),
                edgecolor="#111827",
                linewidth=0.45,
                hatch=SECTION_DIFFICULTY_HATCHES[difficulty_label],
                alpha=0.94,
            )
            bottoms += counts

        method_lines.append(
            f"{method_cfg['label']}: n={int(mask.sum()):,}, acc={final_test_accuracy(bundle, method_key):.2f}%"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MEM_BIN_LABELS, rotation=45, ha="right")
    ax.set_xlabel(f"Memorization bin (> {MEMORIZED_THRESHOLD})")
    ax.set_ylabel("Samples")
    ax.grid(True, axis="y", alpha=0.22)

    method_handles = [
        mpatches.Patch(
            facecolor=bundle["cfg"]["methods"][method_key]["color"],
            edgecolor="#111827",
            label=section_accuracy_label(bundle, method_key),
        )
        for method_key in active_methods
    ]
    difficulty_handles = [
        mpatches.Patch(
            facecolor=SECTION_DIFFICULTY_COLORS[difficulty_label],
            edgecolor="#111827",
            hatch=SECTION_DIFFICULTY_HATCHES[difficulty_label],
            label=difficulty_label.title(),
        )
        for difficulty_label in difficulty_meta["labels"]
    ]
    legend1 = ax.legend(handles=method_handles, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(legend1)
    ax.legend(handles=difficulty_handles, loc="upper right", bbox_to_anchor=(1.0, 0.72))

    section_info_box(
        ax,
        [
            f"Samples: {bundle['cfg']['n_samples']:,}",
            f"Epoch: {epoch}",
            *section_difficulty_summary_lines(difficulty_meta),
            *method_lines,
        ],
        loc="upper left",
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_section_mem_bins_{difficulty_mode}_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{section_dataset_label(bundle)} | Memorization bins with "
            f"{difficulty_mode.replace('_', ' ')} composition | Epoch {epoch}"
        ),
    )


def section_plot_pruning_vs_memorization_by_difficulty(
    bundle: dict,
    *,
    epoch: int,
    method_key: str,
    difficulty_mode: str,
    difficulty_epoch_by_dataset=None,
    pruning_epoch_by_dataset=None,
) -> None:
    difficulty_frame, difficulty_meta = section_build_difficulty_frame(
        bundle,
        difficulty_mode,
        difficulty_epoch_by_dataset=difficulty_epoch_by_dataset,
        pruning_epoch_by_dataset=pruning_epoch_by_dataset,
        epoch_reference_method=SECTION_EASY_HARD_REFERENCE_METHOD,
        pruning_method_key=SECTION_PRUNING_ANCHOR_METHOD,
    )
    frame = section_build_method_frame(
        bundle,
        epoch,
        method_key,
        difficulty_frame=difficulty_frame,
        include_rmia=False,
    )
    pruning_col, pruning_label = resolve_pruning_metric(bundle, method_key, frame)
    if pruning_col is None:
        display_note(
            f"**Pruning scatter skipped:** no pruning metric was found for "
            f"{section_dataset_label(bundle)} / {METHOD_LABELS[method_key]} / epoch {epoch}."
        )
        return
    frame = frame.loc[frame[pruning_col].notna()].copy()
    if frame.empty:
        display_note(
            f"**Pruning scatter skipped:** the pruning metric exists but is empty for "
            f"{section_dataset_label(bundle)} / {METHOD_LABELS[method_key]} / epoch {epoch}."
        )
        return

    fig, ax = plt.subplots(figsize=(6.9, 5.0))
    for difficulty_label in difficulty_meta["labels"]:
        sub = frame.loc[frame["difficulty_label"] == difficulty_label]
        if sub.empty:
            continue
        ax.scatter(
            cap_memorization(sub["memorization"].to_numpy(dtype=np.float64)),
            sub[pruning_col].to_numpy(dtype=np.float64),
            s=18,
            alpha=0.68,
            color=SECTION_DIFFICULTY_COLORS[difficulty_label],
            label=difficulty_label.title(),
            edgecolors="white",
            linewidths=0.25,
        )

    x = cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
    y = frame[pruning_col].to_numpy(dtype=np.float64)
    if np.unique(np.round(x, 8)).size >= 2:
        coeff = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        y_line = coeff[0] * x_line + coeff[1]
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.plot(x_line, y_line, color="#111827", linewidth=1.5, linestyle="--", label=f"r = {corr:.3f}")

    ax.set_xlabel("Memorization score")
    ax.set_ylabel(pruning_label)
    ax.set_xlim(max(0.0, float(np.nanmin(x)) - 0.01), 1.0)
    ax.grid(True, alpha=0.22)
    presentation_add_legend(ax, fontsize=7.6)
    section_info_box(
        ax,
        [
            f"Samples: {len(frame):,}",
            f"Accuracy: {final_test_accuracy(bundle, method_key):.2f}%",
            *section_difficulty_summary_lines(difficulty_meta),
        ],
        loc="upper left",
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{method_key}_section_pruning_vs_mem_{difficulty_mode}_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{section_dataset_label(bundle)} | {METHOD_LABELS[method_key]} | "
            f"{pruning_label} vs memorization | colored by {difficulty_mode.replace('_', ' ')} | epoch {epoch}"
        ),
    )


def section_plot_pruning_vs_memorization_by_movement(bundle: dict, *, epoch: int, method_key: str) -> None:
    frame = section_build_method_frame(bundle, epoch, method_key, include_rmia=False)
    move = section_movement_three_way_frame(bundle, epoch, method_key)
    frame = merge_on_sample_id(
        frame,
        move[["sample_id", "movement3"]],
        context=f"{bundle['dataset_key']}::{method_key}::movement_scatter::epoch_{epoch}",
    )
    pruning_col, pruning_label = resolve_pruning_metric(bundle, method_key, frame)
    if pruning_col is None:
        display_note(
            f"**Movement scatter skipped:** no pruning metric was found for "
            f"{section_dataset_label(bundle)} / {METHOD_LABELS[method_key]} / epoch {epoch}."
        )
        return
    frame = frame.loc[frame[pruning_col].notna()].copy()
    if frame.empty:
        display_note(
            f"**Movement scatter skipped:** the pruning metric exists but is empty for "
            f"{section_dataset_label(bundle)} / {METHOD_LABELS[method_key]} / epoch {epoch}."
        )
        return

    fig, ax = plt.subplots(figsize=(6.9, 5.0))
    for movement_label in ["down", "same", "up"]:
        sub = frame.loc[frame["movement3"] == movement_label]
        if sub.empty:
            continue
        ax.scatter(
            cap_memorization(sub["memorization"].to_numpy(dtype=np.float64)),
            sub[pruning_col].to_numpy(dtype=np.float64),
            s=18,
            alpha=0.68,
            color=SECTION_MOVEMENT3_COLORS[movement_label],
            label=f"{movement_label.title()} (n={len(sub):,})",
            marker=SECTION_MOVEMENT3_MARKERS[movement_label],
            edgecolors="white",
            linewidths=0.25,
        )

    x = cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
    y = frame[pruning_col].to_numpy(dtype=np.float64)
    if np.unique(np.round(x, 8)).size >= 2:
        coeff = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        y_line = coeff[0] * x_line + coeff[1]
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.plot(x_line, y_line, color="#111827", linewidth=1.5, linestyle="--", label=f"r = {corr:.3f}")

    ax.set_xlabel("Memorization score")
    ax.set_ylabel(pruning_label)
    ax.set_xlim(max(0.0, float(np.nanmin(x)) - 0.01), 1.0)
    ax.grid(True, alpha=0.22)
    presentation_add_legend(ax, fontsize=7.6)
    section_info_box(
        ax,
        [
            f"Samples: {len(frame):,}",
            f"Accuracy: {final_test_accuracy(bundle, method_key):.2f}%",
            "Movement is defined relative to Standard training",
        ],
        loc="upper left",
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{method_key}_section_pruning_vs_mem_movement_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{section_dataset_label(bundle)} | {METHOD_LABELS[method_key]} | "
            f"{pruning_label} vs memorization | colored by up / same / down | epoch {epoch}"
        ),
    )


def section_plot_mean_memorization_trajectory(bundle: dict, *, filter_spec, method_keys=None) -> None:
    selector = section_selector_info(filter_spec)
    active_methods = section_method_keys(bundle, method_keys)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    end_points = []
    x_last = None

    for method_key in active_methods:
        summary = section_build_filtered_mean_mem_summary(bundle["mem_long"][method_key], selector)
        summary = summary.dropna(subset=["mean_memorization"]).copy()
        if summary.empty:
            continue
        method_cfg = bundle["cfg"]["methods"][method_key]
        ax.plot(
            summary["epoch"],
            summary["mean_memorization"],
            marker="o",
            linewidth=2.2,
            markersize=4.8,
            color=method_cfg["color"],
            label=section_accuracy_label(bundle, method_key),
        )
        x_last = float(summary["epoch"].iloc[-1])
        end_points.append(
            {
                "y": float(summary["mean_memorization"].iloc[-1]),
                "color": method_cfg["color"],
            }
        )

    if x_last is not None and end_points:
        boxed_end_labels(ax, end_points, x_last=x_last, decimals=3, min_gap_frac=0.060)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean memorization score")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.22)
    presentation_add_legend(ax, fontsize=7.5)
    section_info_box(
        ax,
        [
            f"Samples: {bundle['cfg']['n_samples']:,}",
            selector["label"],
        ],
        loc="upper left",
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_section_mean_mem_trajectory_{selector['stem']}",
        bundle["dataset_key"],
        figure_label=f"{section_dataset_label(bundle)} | Mean memorization trajectory | {selector['label']}",
    )


def section_plot_binned_mean_memorization_trajectory(
    bundle: dict,
    *,
    filter_spec,
    difficulty_mode: str,
    difficulty_epoch_by_dataset=None,
    pruning_epoch_by_dataset=None,
    method_keys=None,
) -> None:
    difficulty_frame, difficulty_meta = section_build_difficulty_frame(
        bundle,
        difficulty_mode,
        difficulty_epoch_by_dataset=difficulty_epoch_by_dataset,
        pruning_epoch_by_dataset=pruning_epoch_by_dataset,
        epoch_reference_method=SECTION_EASY_HARD_REFERENCE_METHOD,
        pruning_method_key=SECTION_PRUNING_ANCHOR_METHOD,
    )
    if difficulty_meta["labels"] != ["easy", "medium", "hard"]:
        display_note(
            f"**Binned memorization trajectories skipped for {section_dataset_label(bundle)}:** "
            "this plot needs easy / medium / hard bins, so use `pruning_rate_tertiles`."
        )
        return

    selector = section_selector_info(filter_spec)
    summary = section_build_filtered_mean_mem_by_difficulty(
        bundle,
        difficulty_frame,
        selector,
        method_keys=method_keys,
    )
    active_methods = section_method_keys(bundle, method_keys)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), sharey=True)

    for ax, difficulty_label in zip(axes, ["easy", "medium", "hard"]):
        sub = summary.loc[summary["difficulty_label"] == difficulty_label].copy()
        end_points = []
        x_last = None
        for method_key in active_methods:
            method_sub = sub.loc[sub["method_key"] == method_key].dropna(subset=["mean_memorization"]).sort_values("epoch")
            if method_sub.empty:
                continue
            method_cfg = bundle["cfg"]["methods"][method_key]
            ax.plot(
                method_sub["epoch"],
                method_sub["mean_memorization"],
                marker="o",
                linewidth=2.1,
                markersize=4.6,
                color=method_cfg["color"],
                label=section_accuracy_label(bundle, method_key),
            )
            x_last = float(method_sub["epoch"].iloc[-1])
            end_points.append(
                {
                    "y": float(method_sub["mean_memorization"].iloc[-1]),
                    "color": method_cfg["color"],
                }
            )
        if x_last is not None and end_points:
            boxed_end_labels(ax, end_points, x_last=x_last, decimals=3, min_gap_frac=0.070)
        n_anchor = int((difficulty_frame["difficulty_label"] == difficulty_label).sum())
        ax.set_title(f"{difficulty_label.title()} | anchor n={n_anchor:,}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.22)
        section_info_box(ax, [selector["label"]], loc="lower right", fontsize=7.8)

    axes[0].set_ylabel("Mean memorization score")
    presentation_add_legend(axes[0], fontsize=7.2)
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_section_binned_mean_mem_trajectory_{selector['stem']}",
        bundle["dataset_key"],
        figure_label=(
            f"{section_dataset_label(bundle)} | Mean memorization trajectory by easy / medium / hard bins | "
            f"{selector['label']}"
        ),
    )


def section_plot_integrated_difficulty_profile(dataset_keys=None, method_keys=None):
    dataset_keys = dataset_keys or SECTION_DATASET_ORDER
    method_keys = method_keys or SECTION_ALL_METHODS
    with temporary_dataset_order(dataset_keys):
        entropy_bins, _ = build_cross_dataset_entropy_story_tables(
            BUNDLES,
            method_keys=method_keys,
            anchor_epoch=None,
            n_bins=5,
            easy_bins=(1,),
            hard_bins=(5,),
        )
        memorization_bins, _ = build_cross_dataset_memorization_story_tables(
            BUNDLES,
            method_keys=method_keys,
            anchor_epoch=None,
            n_bins=5,
            easy_bins=(1,),
            hard_bins=(5,),
        )
        loss_bins, _ = build_cross_dataset_loss_story_tables(
            BUNDLES,
            method_keys=method_keys,
            anchor_epoch=None,
            n_bins=5,
            easy_bins=(1,),
            hard_bins=(5,),
        )
        integrated = build_integrated_difficulty_metric_table(
            entropy_bins,
            memorization_bins,
            loss_bins,
        )
        plot_integrated_difficulty_metric_grid(
            integrated,
            BUNDLES,
            method_keys=method_keys,
            hard_bins=(5,),
        )


def section_build_rmia_table(dataset_keys=None, method_keys=None) -> pd.DataFrame:
    dataset_keys = dataset_keys or SECTION_RMIA_DATASET_ORDER
    method_keys = method_keys or SECTION_ALL_METHODS
    rows = []
    for dataset_key in dataset_keys:
        bundle = BUNDLES[dataset_key]
        for method_key in section_method_keys(bundle, method_keys):
            summary = bundle["rmia_summary"][method_key]
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "dataset": section_dataset_label(bundle),
                    "method_key": method_key,
                    "method": bundle["cfg"]["methods"][method_key]["label"],
                    "AUC ROC (%)": 100.0 * float(summary["auc_mean"]),
                    "AUC ROC std (%)": 100.0 * float(summary["auc_std"]),
                    "TPR @ 0% FPR (%)": 100.0 * float(summary["tpr_at_0pct_fpr_mean"]),
                    "TPR @ 0% FPR std (%)": 100.0 * float(summary["tpr_at_0pct_fpr_std"]),
                    "TPR @ 0.01% FPR (%)": 100.0 * float(summary["tpr_at_0p01pct_fpr_mean"]),
                    "TPR @ 0.01% FPR std (%)": 100.0 * float(summary["tpr_at_0p01pct_fpr_std"]),
                    "TPR @ 1% FPR (%)": 100.0 * float(summary["tpr_at_1pct_fpr_mean"]),
                    "TPR @ 1% FPR std (%)": 100.0 * float(summary["tpr_at_1pct_fpr_std"]),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    dataset_order = {dataset_key: idx for idx, dataset_key in enumerate(dataset_keys)}
    method_order = {method_key: idx for idx, method_key in enumerate(METHOD_ORDER)}
    table["_dataset_order"] = table["dataset_key"].map(dataset_order).astype(int)
    table["_method_order"] = table["method_key"].map(method_order).astype(int)
    return (
        table.sort_values(["_dataset_order", "_method_order"])
        .drop(columns=["_dataset_order", "_method_order"])
        .reset_index(drop=True)
    )


def section_display_rmia_table(table: pd.DataFrame) -> None:
    if table.empty:
        display_note("**RMIA table skipped:** no RMIA summaries were available.")
        return

    metric_cols = [
        "AUC ROC (%)",
        "TPR @ 0% FPR (%)",
        "TPR @ 0.01% FPR (%)",
        "TPR @ 1% FPR (%)",
    ]

    for dataset_key in SECTION_RMIA_DATASET_ORDER:
        raw = table.loc[table["dataset_key"] == dataset_key].copy()
        if raw.empty:
            continue
        dataset_label = raw["dataset"].iloc[0]
        display(Markdown(f"#### {dataset_label}"))

        display_rows = []
        for _, row in raw.iterrows():
            display_rows.append(
                {
                    "method": row["method"],
                    "AUC ROC (%)": f"{row['AUC ROC (%)']:.2f} +/- {row['AUC ROC std (%)']:.2f}",
                    "TPR @ 0% FPR (%)": f"{row['TPR @ 0% FPR (%)']:.2f} +/- {row['TPR @ 0% FPR std (%)']:.2f}",
                    "TPR @ 0.01% FPR (%)": f"{row['TPR @ 0.01% FPR (%)']:.2f} +/- {row['TPR @ 0.01% FPR std (%)']:.2f}",
                    "TPR @ 1% FPR (%)": f"{row['TPR @ 1% FPR (%)']:.2f} +/- {row['TPR @ 1% FPR std (%)']:.2f}",
                }
            )
        view = pd.DataFrame(display_rows).set_index("method")

        styles = pd.DataFrame("", index=view.index, columns=view.columns)
        palette = {
            "low": "background-color: #EAF7EA;",
            "mid": "background-color: #F4F4F5;",
            "high": "background-color: #FCEBEC;",
        }
        numeric = raw[["method", *metric_cols]].set_index("method")
        for col in metric_cols:
            values = numeric[col].astype(float)
            low = float(values.min())
            high = float(values.max())
            for method_name, value in values.items():
                if np.isclose(value, low, atol=1e-12, rtol=0.0):
                    styles.loc[method_name, col] = palette["low"]
                elif np.isclose(value, high, atol=1e-12, rtol=0.0):
                    styles.loc[method_name, col] = palette["high"]
                else:
                    styles.loc[method_name, col] = palette["mid"]

        display(
            view.style
            .apply(lambda _: styles, axis=None)
            .set_caption("RMIA results. Cells show mean +/- std across targets; lower is better.")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("caption-side", "top"),
                            ("text-align", "left"),
                            ("font-weight", "700"),
                            ("font-size", "13px"),
                            ("color", "#1F2937"),
                            ("padding", "0 0 8px 0"),
                        ],
                    },
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F8FAFC"),
                            ("color", "#111827"),
                            ("font-weight", "700"),
                            ("border-bottom", "1px solid #CBD5E1"),
                            ("padding", "6px 10px"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border-bottom", "1px solid #E5E7EB"),
                            ("padding", "7px 11px"),
                            ("text-align", "right"),
                            ("font-size", "12px"),
                        ],
                    },
                ]
            )
        )


def section_draw_rmia_matrix_cell(
    ax,
    x,
    y,
    *,
    n_cell,
    total_n,
    mean_detection_pct,
    detection_share_pct,
    facecolor,
):
    ax.add_patch(
        Rectangle(
            (x - 0.5, y - 0.5),
            1.0,
            1.0,
            facecolor=facecolor,
            edgecolor="#6B7280",
            linewidth=0.9,
        )
    )
    pct_samples = 100.0 * n_cell / max(total_n, 1)
    ax.text(
        x,
        y,
        (
            f"n = {n_cell:,}\n"
            f"{pct_samples:.2f}% samples\n"
            f"{mean_detection_pct:.2f}% mean det.\n"
            f"{detection_share_pct:.2f}% RMIA share"
        ),
        ha="center",
        va="center",
        fontsize=7.6,
        color="#111827",
        linespacing=1.25,
    )


def compute_rmia_norm_entropy_quadrants(
    mem: np.ndarray,
    norm_entropy: np.ndarray,
    detection_pct: np.ndarray,
    *,
    mem_threshold: float,
    norm_entropy_threshold: float,
) -> dict[str, np.ndarray | float | int]:
    mem = np.asarray(mem, dtype=np.float64)
    norm_entropy = np.asarray(norm_entropy, dtype=np.float64)
    detection_pct = np.asarray(detection_pct, dtype=np.float64)
    if mem.shape != norm_entropy.shape or mem.shape != detection_pct.shape:
        raise ValueError("mem, norm_entropy, and detection_pct must have identical shapes")

    mem_bin = np.where(mem < float(mem_threshold), 0, 1).astype(np.int64)
    ent_row = np.where(norm_entropy >= float(norm_entropy_threshold), 0, 1).astype(np.int64)
    total_n = int(mem.size)
    total_detection_mass = float(np.nansum(detection_pct))

    counts = np.zeros((2, 2), dtype=np.int64)
    detection_sums = np.zeros((2, 2), dtype=np.float64)
    for row_idx in range(2):
        for col_idx in range(2):
            mask = (ent_row == row_idx) & (mem_bin == col_idx)
            counts[row_idx, col_idx] = int(mask.sum())
            detection_sums[row_idx, col_idx] = float(np.nansum(detection_pct[mask]))

    if int(counts.sum()) != total_n:
        raise AssertionError("RMIA quadrant counts do not cover every sample exactly once")

    sample_share_pct = 100.0 * counts.astype(np.float64) / max(total_n, 1)
    mean_detection_pct = detection_sums / np.maximum(counts, 1)
    global_mean_detection_pct = total_detection_mass / max(total_n, 1)
    if global_mean_detection_pct > 0.0:
        normalized_detection_lift = mean_detection_pct / global_mean_detection_pct
    else:
        normalized_detection_lift = np.zeros((2, 2), dtype=np.float64)
    if total_detection_mass > 0.0:
        detection_share_pct = 100.0 * detection_sums / total_detection_mass
        fill_fraction = detection_sums / total_detection_mass
    else:
        detection_share_pct = np.zeros((2, 2), dtype=np.float64)
        fill_fraction = counts.astype(np.float64) / max(total_n, 1)

    return {
        "mem_bin": mem_bin,
        "ent_row": ent_row,
        "counts": counts,
        "sample_share_pct": sample_share_pct,
        "detection_sums": detection_sums,
        "mean_detection_pct": mean_detection_pct,
        "global_mean_detection_pct": global_mean_detection_pct,
        "normalized_detection_lift": normalized_detection_lift,
        "detection_share_pct": detection_share_pct,
        "fill_fraction": fill_fraction,
        "total_n": total_n,
        "total_detection_mass": total_detection_mass,
    }


SECTION_RMIA_SCORE_COL_CANDIDATES = [
    "rmia_mean_member_score",
    "mean_member_score",
    "rmia_score",
    "rmia_member_score",
    "member_score",
]


def section_pick_rmia_score_col(frame: pd.DataFrame) -> str:
    for col in SECTION_RMIA_SCORE_COL_CANDIDATES:
        if col in frame.columns:
            return col
    raise KeyError(
        "Could not find a per-sample RMIA score column. "
        f"Tried: {SECTION_RMIA_SCORE_COL_CANDIDATES}"
    )


def section_plot_rmia_score_violins(dataset_keys=None, method_keys=None) -> None:
    dataset_keys = dataset_keys or SECTION_RMIA_DATASET_ORDER
    method_keys = method_keys or SECTION_ALL_METHODS
    fig, axes = plt.subplots(1, len(dataset_keys), figsize=(6.2 * len(dataset_keys), 5.1), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, dataset_key in zip(axes, dataset_keys):
        bundle = BUNDLES[dataset_key]
        active_methods = section_method_keys(bundle, method_keys)
        positions = np.arange(1, len(active_methods) + 1)
        valid_positions = []
        valid_data = []
        valid_methods = []

        for position, method_key in zip(positions, active_methods):
            if not rmia_join_allowed(bundle, method_key):
                continue
            frame = section_build_method_frame(
                bundle,
                epoch=int(bundle["cfg"]["end_epoch"]),
                method_key=method_key,
                include_rmia=True,
                require_rmia=True,
            )
            score_col = section_pick_rmia_score_col(frame)
            values = frame[score_col].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            values = values[(values >= 0.0) & (values <= 1.0)]
            if values.size == 0:
                continue
            valid_positions.append(position)
            valid_data.append(values)
            valid_methods.append(method_key)

        if not valid_data:
            ax.text(0.5, 0.5, "No RMIA scores", ha="center", va="center", transform=ax.transAxes)
            continue

        parts = ax.violinplot(
            valid_data,
            positions=valid_positions,
            widths=0.78,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, method_key in zip(parts["bodies"], valid_methods):
            body.set_facecolor(bundle["cfg"]["methods"][method_key]["color"])
            body.set_edgecolor("#2F3542")
            body.set_linewidth(0.75)
            body.set_alpha(0.72)

        mean_labels = []
        for position, values, method_key in zip(valid_positions, valid_data, valid_methods):
            q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
            mean_value = float(np.mean(values))
            ax.vlines(position, q1, q3, color="#111827", linewidth=3.0, alpha=0.9, zorder=5)
            ax.scatter([position], [med], s=34, color="white", edgecolor="#111827", linewidth=0.75, zorder=6)
            ax.scatter([position], [mean_value], marker="D", s=24, color="#111827", edgecolor="white", linewidth=0.45, zorder=7)
            mean_labels.append(
                {
                    "x": float(position),
                    "y": mean_value,
                    "text": f"mean {mean_value:.3f}",
                    "color": "#111827",
                }
            )
            ax.text(
                position,
                0.018,
                f"n={len(values):,}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color="#334155",
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "fc": "white",
                    "ec": "#CBD5E1",
                    "lw": 0.60,
                    "alpha": 0.94,
                },
            )

        for label, placement in zip(
            mean_labels,
            _plan_offset_annotations(
                ax,
                mean_labels,
                fontsize=8.0,
                candidate_offsets_points=[
                    (0.0, 18.0),
                    (12.0, 18.0),
                    (-12.0, 18.0),
                    (0.0, 26.0),
                    (16.0, 12.0),
                    (-16.0, 12.0),
                ],
                keep_inside_axes=True,
            ),
        ):
            ax.annotate(
                label["text"],
                xy=(label["x"], label["y"]),
                xytext=(placement["dx_pt"], placement["dy_pt"]),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.0,
                color=label["color"],
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "#CBD5E1",
                    "lw": 0.65,
                    "alpha": 0.94,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#64748B",
                    "linewidth": 0.65,
                    "alpha": 0.55,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=8,
                clip_on=False,
            )

        ax.set_xticks(valid_positions)
        ax.set_xticklabels([bundle["cfg"]["methods"][method_key]["label"] for method_key in valid_methods], rotation=0)
        ax.set_title(section_dataset_label(bundle), fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.20)

    axes[0].set_ylabel("Per-sample RMIA score")
    finish_figure(
        fig,
        "section_rmia_score_violins",
        "cross_dataset",
        figure_label="Per-sample RMIA score violins for CIFAR-10 / ResNet-20 and CIFAR-100 / WRN-28-2",
    )


def section_short_method_accuracy(bundle: dict, method_key: str) -> str:
    return f"{bundle['cfg']['methods'][method_key]['label']} {final_test_accuracy(bundle, method_key):.2f}%"


def section_footer(fig: plt.Figure, lines, *, x: float = 0.01, y: float = 0.012, fontsize: float = 7.9) -> None:
    lines = [str(line).strip() for line in lines if str(line).strip()]
    if not lines:
        return
    fig.text(
        x,
        y,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=fontsize,
        color="#475569",
    )


def section_outside_legend(ax: plt.Axes, *, loc: str = "upper left", anchor=(1.02, 1.0), fontsize: float = 7.7, ncol: int = 1):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=fontsize,
        ncol=ncol,
    )


def section_bar_footer_lines(bundle: dict, *, epoch: int, difficulty_meta: dict, method_keys) -> list[str]:
    difficulty_line = section_difficulty_summary_lines(difficulty_meta)[0]
    return [f"n={bundle['cfg']['n_samples']:,} | epoch {epoch} | {difficulty_line}"]


def section_scatter_footer_lines(bundle: dict, *, epoch: int, method_key: str, extra_lines=None) -> list[str]:
    lines = [f"n={bundle['cfg']['n_samples']:,} | epoch {epoch} | acc {final_test_accuracy(bundle, method_key):.2f}%"]
    if extra_lines:
        lines[0] = lines[0] + " | " + " | ".join(str(line).rstrip(".") for line in extra_lines)
    return lines


def section_plot_memorization_bin_composition(
    bundle: dict,
    *,
    epoch: int,
    difficulty_mode: str,
    difficulty_epoch_by_dataset=None,
    pruning_epoch_by_dataset=None,
    method_keys=None,
) -> None:
    difficulty_frame, difficulty_meta = section_build_difficulty_frame(
        bundle,
        difficulty_mode,
        difficulty_epoch_by_dataset=difficulty_epoch_by_dataset,
        pruning_epoch_by_dataset=pruning_epoch_by_dataset,
        epoch_reference_method=SECTION_EASY_HARD_REFERENCE_METHOD,
        pruning_method_key=SECTION_PRUNING_ANCHOR_METHOD,
    )
    active_methods = section_method_keys(bundle, method_keys)
    x = np.arange(len(MEM_BIN_LABELS))
    width = 0.80 / max(len(active_methods), 1)
    fig, ax = plt.subplots(figsize=(13.2, 5.0))

    for idx, method_key in enumerate(active_methods):
        method_cfg = bundle["cfg"]["methods"][method_key]
        frame = merge_on_sample_id(
            get_method_epoch_mem(bundle, method_key, epoch)[["sample_id", "memorization"]],
            difficulty_frame[["sample_id", "difficulty_label"]],
            context=f"{bundle['dataset_key']}::{method_key}::bin_plot::epoch_{epoch}",
        )
        frame["memorization"] = cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
        mask = (frame["memorization"] > MEMORIZED_THRESHOLD) & (frame["memorization"] <= 1.0)
        bins = memorization_bin_index(frame.loc[mask, "memorization"].to_numpy(dtype=np.float64))
        difficulty = frame.loc[mask, "difficulty_label"].to_numpy()
        positions = x + (idx - (len(active_methods) - 1) / 2.0) * width
        bottoms = np.zeros(len(MEM_BIN_LABELS), dtype=np.float64)

        for difficulty_label in difficulty_meta["labels"]:
            counts = np.array(
                [
                    np.sum((bins == bin_id) & (difficulty == difficulty_label))
                    for bin_id in range(len(MEM_BIN_LABELS))
                ],
                dtype=np.float64,
            )
            ax.bar(
                positions,
                counts,
                bottom=bottoms,
                width=width,
                color=method_cfg["color"],
                edgecolor="#111827",
                linewidth=0.45,
                hatch=SECTION_DIFFICULTY_HATCHES[difficulty_label],
                alpha=0.94,
            )
            bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels(MEM_BIN_LABELS, rotation=45, ha="right")
    ax.set_xlabel(f"Memorization bin (> {MEMORIZED_THRESHOLD})")
    ax.set_ylabel("Samples")
    ax.grid(True, axis="y", alpha=0.22)

    method_handles = [
        mpatches.Patch(
            facecolor=bundle["cfg"]["methods"][method_key]["color"],
            edgecolor="#111827",
            label=section_accuracy_label(bundle, method_key),
        )
        for method_key in active_methods
    ]
    difficulty_handles = [
        mpatches.Patch(
            facecolor="white",
            edgecolor="#111827",
            hatch=SECTION_DIFFICULTY_HATCHES[difficulty_label],
            label=difficulty_label.title(),
        )
        for difficulty_label in difficulty_meta["labels"]
    ]
    legend1 = ax.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.5,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=difficulty_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.58),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.5,
    )

    section_footer(
        fig,
        section_bar_footer_lines(
            bundle,
            epoch=epoch,
            difficulty_meta=difficulty_meta,
            method_keys=active_methods,
        ),
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_section_mem_bins_{difficulty_mode}_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{section_dataset_label(bundle)} | Memorization bins with "
            f"{difficulty_mode.replace('_', ' ')} composition | Epoch {epoch}"
        ),
        tight_rect=(0.0, 0.09, 0.80, 0.96),
    )


def section_plot_pruning_vs_memorization_by_difficulty(
    bundle: dict,
    *,
    epoch: int,
    method_key: str,
    difficulty_mode: str,
    difficulty_epoch_by_dataset=None,
    pruning_epoch_by_dataset=None,
) -> None:
    difficulty_frame, difficulty_meta = section_build_difficulty_frame(
        bundle,
        difficulty_mode,
        difficulty_epoch_by_dataset=difficulty_epoch_by_dataset,
        pruning_epoch_by_dataset=pruning_epoch_by_dataset,
        epoch_reference_method=SECTION_EASY_HARD_REFERENCE_METHOD,
        pruning_method_key=SECTION_PRUNING_ANCHOR_METHOD,
    )
    frame = section_build_method_frame(
        bundle,
        epoch,
        method_key,
        difficulty_frame=difficulty_frame,
        include_rmia=False,
    )
    pruning_col, pruning_label = resolve_pruning_metric(bundle, method_key, frame)
    if pruning_col is None:
        display_note(
            f"**Pruning scatter skipped:** no pruning metric was found for "
            f"{section_dataset_label(bundle)} / {METHOD_LABELS[method_key]} / epoch {epoch}."
        )
        return
    frame = frame.loc[frame[pruning_col].notna()].copy()
    if frame.empty:
        display_note(
            f"**Pruning scatter skipped:** the pruning metric exists but is empty for "
            f"{section_dataset_label(bundle)} / {METHOD_LABELS[method_key]} / epoch {epoch}."
        )
        return

    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    for difficulty_label in difficulty_meta["labels"]:
        sub = frame.loc[frame["difficulty_label"] == difficulty_label]
        if sub.empty:
            continue
        ax.scatter(
            cap_memorization(sub["memorization"].to_numpy(dtype=np.float64)),
            sub[pruning_col].to_numpy(dtype=np.float64),
            s=18,
            alpha=float(report_setting("difficulty_scatter_alpha")),
            color=SECTION_DIFFICULTY_COLORS[difficulty_label],
            label=f"{difficulty_label.title()} (n={len(sub):,})",
            edgecolors="white",
            linewidths=0.25,
        )

    x = cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
    y = frame[pruning_col].to_numpy(dtype=np.float64)
    if np.unique(np.round(x, 8)).size >= 2:
        coeff = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        y_line = coeff[0] * x_line + coeff[1]
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.plot(x_line, y_line, color="#111827", linewidth=1.45, linestyle="--", label=f"r = {corr:.3f}")

    ax.set_xlabel("Memorization score")
    ax.set_ylabel(pruning_label)
    ax.set_xlim(max(0.0, float(np.nanmin(x)) - 0.01), 1.0)
    ax.grid(True, alpha=0.22)
    section_outside_legend(ax, fontsize=7.6)
    section_footer(
        fig,
        section_scatter_footer_lines(
            bundle,
            epoch=epoch,
            method_key=method_key,
            extra_lines=section_difficulty_summary_lines(difficulty_meta),
        ),
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{method_key}_section_pruning_vs_mem_{difficulty_mode}_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{section_dataset_label(bundle)} | {METHOD_LABELS[method_key]} | "
            f"{pruning_label} vs memorization | colored by {difficulty_mode.replace('_', ' ')} | epoch {epoch}"
        ),
        tight_rect=(0.0, 0.10, 0.82, 0.96),
    )


def section_plot_pruning_vs_memorization_by_movement(bundle: dict, *, epoch: int, method_key: str) -> None:
    frame = section_build_method_frame(bundle, epoch, method_key, include_rmia=False)
    move = section_movement_three_way_frame(bundle, epoch, method_key)
    frame = merge_on_sample_id(
        frame,
        move[["sample_id", "movement3"]],
        context=f"{bundle['dataset_key']}::{method_key}::movement_scatter::epoch_{epoch}",
    )
    pruning_col, pruning_label = resolve_pruning_metric(bundle, method_key, frame)
    if pruning_col is None:
        display_note(
            f"**Movement scatter skipped:** no pruning metric was found for "
            f"{section_dataset_label(bundle)} / {METHOD_LABELS[method_key]} / epoch {epoch}."
        )
        return
    frame = frame.loc[frame[pruning_col].notna()].copy()
    if frame.empty:
        display_note(
            f"**Movement scatter skipped:** the pruning metric exists but is empty for "
            f"{section_dataset_label(bundle)} / {METHOD_LABELS[method_key]} / epoch {epoch}."
        )
        return

    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    for movement_label in ["down", "same", "up"]:
        sub = frame.loc[frame["movement3"] == movement_label]
        if sub.empty:
            continue
        ax.scatter(
            cap_memorization(sub["memorization"].to_numpy(dtype=np.float64)),
            sub[pruning_col].to_numpy(dtype=np.float64),
            s=18,
            alpha=float(report_setting("movement_scatter_alpha")),
            color=SECTION_MOVEMENT3_COLORS[movement_label],
            label=f"{movement_label.title()} (n={len(sub):,})",
            marker=SECTION_MOVEMENT3_MARKERS[movement_label],
            edgecolors="white",
            linewidths=0.25,
        )

    x = cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
    y = frame[pruning_col].to_numpy(dtype=np.float64)
    if np.unique(np.round(x, 8)).size >= 2:
        coeff = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
        y_line = coeff[0] * x_line + coeff[1]
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.plot(x_line, y_line, color="#111827", linewidth=1.45, linestyle="--", label=f"r = {corr:.3f}")

    ax.set_xlabel("Memorization score")
    ax.set_ylabel(pruning_label)
    ax.set_xlim(max(0.0, float(np.nanmin(x)) - 0.01), 1.0)
    ax.grid(True, alpha=0.22)
    section_outside_legend(ax, fontsize=7.6)
    section_footer(
        fig,
        section_scatter_footer_lines(
            bundle,
            epoch=epoch,
            method_key=method_key,
            extra_lines=["Movement is defined relative to Standard training"],
        ),
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{method_key}_section_pruning_vs_mem_movement_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{section_dataset_label(bundle)} | {METHOD_LABELS[method_key]} | "
            f"{pruning_label} vs memorization | colored by up / same / down | epoch {epoch}"
        ),
        tight_rect=(0.0, 0.10, 0.82, 0.96),
    )


def section_plot_mean_memorization_trajectory(bundle: dict, *, filter_spec, method_keys=None) -> None:
    selector = section_selector_info(filter_spec)
    active_methods = section_method_keys(bundle, method_keys)
    fig, ax = plt.subplots(figsize=(12.0, 4.9))
    end_points = []
    x_last = None

    for method_key in active_methods:
        summary = section_build_filtered_mean_mem_summary(bundle["mem_long"][method_key], selector)
        summary = summary.dropna(subset=["mean_memorization"]).copy()
        if summary.empty:
            continue
        method_cfg = bundle["cfg"]["methods"][method_key]
        ax.plot(
            summary["epoch"],
            summary["mean_memorization"],
            marker="o",
            linewidth=2.2,
            markersize=4.8,
            color=method_cfg["color"],
            label=section_accuracy_label(bundle, method_key),
        )
        x_last = float(summary["epoch"].iloc[-1])
        end_points.append(
            {
                "y": float(summary["mean_memorization"].iloc[-1]),
                "color": method_cfg["color"],
            }
        )

    if x_last is not None and end_points:
        boxed_end_labels(ax, end_points, x_last=x_last, decimals=3, min_gap_frac=0.060)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean memorization score")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.22)
    section_outside_legend(ax, fontsize=7.4)
    section_footer(
        fig,
        [f"n={bundle['cfg']['n_samples']:,} | {selector['label']}"],
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_section_mean_mem_trajectory_{selector['stem']}",
        bundle["dataset_key"],
        figure_label=f"{section_dataset_label(bundle)} | Mean memorization trajectory | {selector['label']}",
        tight_rect=(0.0, 0.09, 0.82, 0.96),
    )


def section_plot_binned_mean_memorization_trajectory(
    bundle: dict,
    *,
    filter_spec,
    difficulty_mode: str,
    difficulty_epoch_by_dataset=None,
    pruning_epoch_by_dataset=None,
    method_keys=None,
) -> None:
    difficulty_frame, difficulty_meta = section_build_difficulty_frame(
        bundle,
        difficulty_mode,
        difficulty_epoch_by_dataset=difficulty_epoch_by_dataset,
        pruning_epoch_by_dataset=pruning_epoch_by_dataset,
        epoch_reference_method=SECTION_EASY_HARD_REFERENCE_METHOD,
        pruning_method_key=SECTION_PRUNING_ANCHOR_METHOD,
    )
    if difficulty_meta["labels"] != ["easy", "medium", "hard"]:
        display_note(
            f"**Binned memorization trajectories skipped for {section_dataset_label(bundle)}:** "
            "this plot needs easy / medium / hard bins, so use `pruning_rate_tertiles`."
        )
        return

    selector = section_selector_info(filter_spec)
    summary = section_build_filtered_mean_mem_by_difficulty(
        bundle,
        difficulty_frame,
        selector,
        method_keys=method_keys,
    )
    active_methods = section_method_keys(bundle, method_keys)
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 4.9), sharey=True)

    for ax, difficulty_label in zip(axes, ["easy", "medium", "hard"]):
        sub = summary.loc[summary["difficulty_label"] == difficulty_label].copy()
        end_points = []
        x_last = None
        for method_key in active_methods:
            method_sub = sub.loc[sub["method_key"] == method_key].dropna(subset=["mean_memorization"]).sort_values("epoch")
            if method_sub.empty:
                continue
            method_cfg = bundle["cfg"]["methods"][method_key]
            ax.plot(
                method_sub["epoch"],
                method_sub["mean_memorization"],
                marker="o",
                linewidth=2.1,
                markersize=4.6,
                color=method_cfg["color"],
                label=section_accuracy_label(bundle, method_key),
            )
            x_last = float(method_sub["epoch"].iloc[-1])
            end_points.append({"y": float(method_sub["mean_memorization"].iloc[-1]), "color": method_cfg["color"]})
        if x_last is not None and end_points:
            boxed_end_labels(ax, end_points, x_last=x_last, decimals=3, min_gap_frac=0.070)
        ax.set_title(difficulty_label.title(), fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.22)

    axes[0].set_ylabel("Mean memorization score")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(len(handles), 3),
            frameon=False,
            fontsize=7.6,
        )
    section_footer(
        fig,
        [
            f"{selector['label']} | {section_difficulty_summary_lines(difficulty_meta)[0]}",
        ],
        y=0.018,
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_section_binned_mean_mem_trajectory_{selector['stem']}",
        bundle["dataset_key"],
        figure_label=(
            f"{section_dataset_label(bundle)} | Mean memorization trajectory by easy / medium / hard bins | "
            f"{selector['label']}"
        ),
        tight_rect=(0.0, 0.10, 1.0, 0.92),
    )


def section_display_rmia_table(table: pd.DataFrame) -> None:
    if table.empty:
        display_note("**RMIA table skipped:** no RMIA summaries were available.")
        return

    metric_cols = [
        "AUC ROC (%)",
        "TPR @ 0% FPR (%)",
        "TPR @ 0.01% FPR (%)",
        "TPR @ 1% FPR (%)",
    ]

    for dataset_key in report_setting("rmia_datasets"):
        raw = table.loc[table["dataset_key"] == dataset_key].copy()
        if raw.empty:
            continue
        dataset_label = raw["dataset"].iloc[0]
        display(Markdown(f"#### {dataset_label}"))

        display_rows = []
        for _, row in raw.iterrows():
            display_rows.append(
                {
                    "method": row["method"],
                    "AUC ROC (%)": f"{row['AUC ROC (%)']:.2f} +/- {row['AUC ROC std (%)']:.2f}",
                    "TPR @ 0% FPR (%)": f"{row['TPR @ 0% FPR (%)']:.2f} +/- {row['TPR @ 0% FPR std (%)']:.2f}",
                    "TPR @ 0.01% FPR (%)": f"{row['TPR @ 0.01% FPR (%)']:.2f} +/- {row['TPR @ 0.01% FPR std (%)']:.2f}",
                    "TPR @ 1% FPR (%)": f"{row['TPR @ 1% FPR (%)']:.2f} +/- {row['TPR @ 1% FPR std (%)']:.2f}",
                }
            )
        view = pd.DataFrame(display_rows).set_index("method")
        styles = pd.DataFrame("", index=view.index, columns=view.columns)
        palette = {
            "low": "background-color: #EEF6FF;",
            "mid": "background-color: #F8FAFC;",
            "high": "background-color: #EAF7EA;",
        }
        numeric = raw[["method", *metric_cols]].set_index("method")
        for col in metric_cols:
            values = numeric[col].astype(float)
            low = float(values.min())
            high = float(values.max())
            for method_name, value in values.items():
                if np.isclose(value, high, atol=1e-12, rtol=0.0):
                    styles.loc[method_name, col] = palette["high"]
                elif np.isclose(value, low, atol=1e-12, rtol=0.0):
                    styles.loc[method_name, col] = palette["low"]
                else:
                    styles.loc[method_name, col] = palette["mid"]

        display(
            view.style
            .apply(lambda _: styles, axis=None)
            .set_caption("RMIA results. Cells show mean +/- std across targets; higher is better.")
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [
                            ("caption-side", "top"),
                            ("text-align", "left"),
                            ("font-weight", "700"),
                            ("font-size", "13px"),
                            ("color", "#1F2937"),
                            ("padding", "0 0 8px 0"),
                        ],
                    },
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#F8FAFC"),
                            ("color", "#111827"),
                            ("font-weight", "700"),
                            ("border-bottom", "1px solid #CBD5E1"),
                            ("padding", "6px 10px"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border-bottom", "1px solid #E5E7EB"),
                            ("padding", "7px 11px"),
                            ("text-align", "right"),
                            ("font-size", "12px"),
                        ],
                    },
                ]
            )
        )


def section_plot_rmia_norm_entropy_matrix(
    bundle: dict,
    *,
    epoch: int,
    method_key: str,
    mem_threshold: float,
    norm_entropy_threshold: float,
) -> None:
    if not rmia_join_allowed(bundle, method_key):
        display_note(
            f"**RMIA quadrant plot skipped:** sample-id alignment did not pass for "
            f"{section_dataset_label(bundle)} / {METHOD_LABELS[method_key]}."
        )
        return

    frame = section_build_method_frame(
        bundle,
        epoch,
        method_key,
        include_rmia=True,
        require_rmia=True,
    )
    mem = cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
    norm_entropy = frame["entropy"].to_numpy(dtype=np.float64) / theoretical_max_entropy(bundle)
    norm_entropy = np.clip(norm_entropy, 0.0, 1.0)
    detection_pct = 100.0 * frame["detection_rate_0p01pct"].to_numpy(dtype=np.float64)
    quadrant_stats = compute_rmia_norm_entropy_quadrants(
        mem,
        norm_entropy,
        detection_pct,
        mem_threshold=mem_threshold,
        norm_entropy_threshold=norm_entropy_threshold,
    )

    order = np.argsort(detection_pct, kind="stable")
    detection_limit = section_detection_vmax(bundle)

    fig = plt.figure(figsize=(14.2, 5.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.30, 0.07, 0.18, 1.45], wspace=0.10)
    ax_scatter = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[0, 3])

    scatter = ax_scatter.scatter(
        mem[order],
        norm_entropy[order],
        c=detection_pct[order],
        cmap=SECTION_RMIA_CMAP,
        vmin=0.0,
        vmax=detection_limit,
        s=10,
        alpha=0.50,
        edgecolors="none",
    )
    ax_scatter.axvline(mem_threshold, color="#475569", linestyle="--", linewidth=1.0)
    ax_scatter.axhline(norm_entropy_threshold, color="#475569", linestyle="--", linewidth=1.0)
    ax_scatter.set_xlim(0.0, 1.0)
    ax_scatter.set_ylim(0.0, 1.0)
    ax_scatter.set_xlabel("Memorization score")
    ax_scatter.set_ylabel("Normalized entropy / log(#classes)")
    ax_scatter.grid(True, alpha=0.18)

    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("RMIA @ 0.01% FPR (%)", fontsize=8.6, labelpad=4)
    cbar.ax.tick_params(labelsize=8.0)

    ax_matrix.set_xlim(-0.5, 1.5)
    ax_matrix.set_ylim(1.5, -0.5)
    ax_matrix.set_xticks([0, 1])
    ax_matrix.set_xticklabels([f"Mem < {mem_threshold:.2f}", f"Mem >= {mem_threshold:.2f}"])
    ax_matrix.set_yticks([0, 1])
    ax_matrix.set_yticklabels(["High ent.", "Low ent."])
    ax_matrix.yaxis.tick_right()
    ax_matrix.tick_params(axis="y", length=0, labelright=True, labelleft=False, pad=8)
    ax_matrix.tick_params(axis="x", length=0)
    ax_matrix.set_aspect("equal", adjustable="box")
    ax_matrix.set_title("2 x 2 sample + RMIA share", fontsize=10, pad=10)

    for row_idx in range(2):
        for col_idx in range(2):
            n_cell = int(quadrant_stats["counts"][row_idx, col_idx])
            facecolor = SECTION_MATRIX_FILL(
                0.10
                + 0.78
                * quadrant_stats["fill_fraction"][row_idx, col_idx]
                / max(float(np.max(quadrant_stats["fill_fraction"])), 1e-8)
            )
            section_draw_rmia_matrix_cell(
                ax_matrix,
                col_idx,
                row_idx,
                n_cell=n_cell,
                total_n=int(quadrant_stats["total_n"]),
                mean_detection_pct=float(quadrant_stats["mean_detection_pct"][row_idx, col_idx]),
                detection_share_pct=float(quadrant_stats["detection_share_pct"][row_idx, col_idx]),
                facecolor=facecolor,
            )

    section_footer(
        fig,
        [
            f"n={int(quadrant_stats['total_n']):,} | acc {final_test_accuracy(bundle, method_key):.2f}% | mem {mem_threshold:.2f} | norm ent {norm_entropy_threshold:.2f}",
            "mean det. = cell sum(detection rates) / cell sample count",
            "RMIA share = cell sum(detection rates) / total sum(detection rates)",
        ],
    )
    finish_figure(
        fig,
        f"{bundle['dataset_key']}_{method_key}_section_rmia_norm_entropy_matrix_epoch_{epoch}",
        bundle["dataset_key"],
        figure_label=(
            f"{section_dataset_label(bundle)} | {METHOD_LABELS[method_key]} | "
            f"Normalized entropy x memorization x RMIA | epoch {epoch}"
        ),
        tight_rect=(0.0, 0.09, 1.0, 0.97),
    )


def plot_report_memorization_bins() -> None:
    for dataset_key in report_setting("active_datasets"):
        bundle = report_bundle(dataset_key)
        section_plot_memorization_bin_composition(
            bundle,
            epoch=int(report_setting("memorization_epochs")[dataset_key]),
            difficulty_mode=report_setting("bin_plot_difficulty_mode"),
            difficulty_epoch_by_dataset=report_setting("easy_hard_epochs"),
            pruning_epoch_by_dataset=report_setting("pruning_anchor_epochs"),
            method_keys=report_setting("all_methods"),
        )


def plot_report_pruning_vs_memorization_difficulty() -> None:
    for dataset_key in report_setting("pruning_datasets"):
        bundle = report_bundle(dataset_key)
        epoch = int(report_setting("memorization_epochs")[dataset_key])
        for method_key in report_setting("pruned_methods"):
            section_plot_pruning_vs_memorization_by_difficulty(
                bundle,
                epoch=epoch,
                method_key=method_key,
                difficulty_mode="epoch_loss",
                difficulty_epoch_by_dataset=report_setting("easy_hard_epochs"),
                pruning_epoch_by_dataset=report_setting("pruning_anchor_epochs"),
            )


def plot_report_tail_memorization_auc() -> pd.DataFrame:
    return section_plot_tail_auc_summary(
        dataset_keys=report_setting("active_datasets"),
        method_keys=report_setting("all_methods"),
        tail_fraction=float(report_setting("tail_auc_fraction")),
    )


def plot_report_final_tail_profile() -> pd.DataFrame:
    return section_plot_final_tail_profile(
        dataset_keys=report_setting("active_datasets"),
        method_keys=report_setting("all_methods"),
    )


def plot_report_high_memorization_emergence() -> pd.DataFrame:
    return section_plot_high_memorization_emergence(
        dataset_keys=report_setting("active_datasets"),
        method_keys=report_setting("all_methods"),
    )


def plot_report_memorization_state_flow() -> pd.DataFrame:
    return section_plot_memorization_state_flow(
        dataset_keys=report_setting("active_datasets"),
        method_keys=report_setting("all_methods"),
    )


def plot_report_memorization_exceedance_checkpoints() -> pd.DataFrame:
    return section_plot_memorization_exceedance_checkpoints(
        dataset_keys=report_setting("active_datasets"),
        method_keys=report_setting("all_methods"),
    )


def plot_report_memorization_exposure_evidence() -> dict[str, pd.DataFrame]:
    evidence = section_plot_memorization_exposure_evidence(
        dataset_keys=report_setting("active_datasets"),
        method_keys=report_setting("all_methods"),
    )
    evidence["onset"] = section_plot_memorization_onset_trajectories(
        dataset_keys=report_setting("active_datasets"),
        method_keys=report_setting("all_methods"),
    )
    return evidence


def plot_report_pruning_vs_memorization_movement() -> None:
    for dataset_key in report_setting("pruning_datasets"):
        bundle = report_bundle(dataset_key)
        epoch = int(report_setting("memorization_epochs")[dataset_key])
        for method_key in report_setting("pruned_methods"):
            section_plot_pruning_vs_memorization_by_movement(bundle, epoch=epoch, method_key=method_key)


def plot_report_mean_memorization_trajectories() -> None:
    for dataset_key in report_setting("active_datasets"):
        section_plot_mean_memorization_trajectory(
            report_bundle(dataset_key),
            filter_spec=report_setting("mean_memorization_filter"),
            method_keys=report_setting("all_methods"),
        )


def plot_report_integrated_difficulty_profile() -> None:
    section_plot_integrated_difficulty_profile(
        dataset_keys=report_setting("active_datasets"),
        method_keys=report_setting("all_methods"),
    )


def plot_report_binned_memorization_trajectories() -> None:
    for dataset_key in report_setting("active_datasets"):
        section_plot_binned_mean_memorization_trajectory(
            report_bundle(dataset_key),
            filter_spec=report_setting("mean_memorization_filter"),
            difficulty_mode=report_setting("binned_trajectory_difficulty_mode"),
            difficulty_epoch_by_dataset=report_setting("easy_hard_epochs"),
            pruning_epoch_by_dataset=report_setting("pruning_anchor_epochs"),
            method_keys=report_setting("all_methods"),
        )


def display_report_rmia_table() -> pd.DataFrame:
    table = section_build_rmia_table(
        dataset_keys=report_setting("rmia_datasets"),
        method_keys=report_setting("all_methods"),
    )
    section_display_rmia_table(table)
    return table


def plot_report_rmia_quadrants() -> None:
    for dataset_key in report_setting("rmia_datasets"):
        bundle = report_bundle(dataset_key)
        epoch = int(report_setting("memorization_epochs")[dataset_key])
        for method_key in report_setting("all_methods"):
            section_plot_rmia_norm_entropy_matrix(
                bundle,
                epoch=epoch,
                method_key=method_key,
                mem_threshold=float(report_setting("rmia_mem_threshold")),
                norm_entropy_threshold=float(report_setting("rmia_normalized_entropy_threshold")),
            )


def build_report_rmia_quadrant_check_table() -> pd.DataFrame:
    rows = []
    mem_threshold = float(report_setting("rmia_mem_threshold"))
    norm_entropy_threshold = float(report_setting("rmia_normalized_entropy_threshold"))
    for dataset_key in report_setting("rmia_datasets"):
        bundle = report_bundle(dataset_key)
        epoch = int(report_setting("memorization_epochs")[dataset_key])
        for method_key in report_setting("all_methods"):
            if not rmia_join_allowed(bundle, method_key):
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "method_key": method_key,
                        "epoch": epoch,
                        "status": "skipped: rmia alignment failed",
                    }
                )
                continue

            frame = section_build_method_frame(
                bundle,
                epoch,
                method_key,
                include_rmia=True,
                require_rmia=True,
            )
            mem = cap_memorization(frame["memorization"].to_numpy(dtype=np.float64))
            norm_entropy = frame["entropy"].to_numpy(dtype=np.float64) / theoretical_max_entropy(bundle)
            norm_entropy = np.clip(norm_entropy, 0.0, 1.0)
            detection_pct = 100.0 * frame["detection_rate_0p01pct"].to_numpy(dtype=np.float64)
            stats = compute_rmia_norm_entropy_quadrants(
                mem,
                norm_entropy,
                detection_pct,
                mem_threshold=mem_threshold,
                norm_entropy_threshold=norm_entropy_threshold,
            )
            counts = stats["counts"]
            detection_share_pct = stats["detection_share_pct"]
            total_detection_mass = float(stats["total_detection_mass"])
            share_sum = float(np.sum(detection_share_pct))
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "method_key": method_key,
                    "epoch": epoch,
                    "status": "ok",
                    "mem_threshold": mem_threshold,
                    "norm_entropy_threshold": norm_entropy_threshold,
                    "n_total": int(stats["total_n"]),
                    "n_high_ent_low_mem": int(counts[0, 0]),
                    "n_high_ent_high_mem": int(counts[0, 1]),
                    "n_low_ent_low_mem": int(counts[1, 0]),
                    "n_low_ent_high_mem": int(counts[1, 1]),
                    "counts_sum": int(np.sum(counts)),
                    "counts_match_total": bool(int(np.sum(counts)) == int(stats["total_n"])),
                    "rmia_share_sum_pct": share_sum,
                    "rmia_share_sum_ok": bool(total_detection_mass <= 0.0 or np.isclose(share_sum, 100.0, atol=1e-8)),
                }
            )
    return pd.DataFrame(rows)


def plot_report_rmia_violins() -> None:
    section_plot_rmia_score_violins(
        dataset_keys=report_setting("rmia_datasets"),
        method_keys=report_setting("all_methods"),
    )


REPORT_PLOT_KINDS = [
    "mem_bins",
    "state_flow",
    "exceedance_curves",
    "exposure_evidence",
    "high_mem_emergence",
    "tail_auc",
    "final_tail_profile",
    "pruning_mem_difficulty",
    "pruning_mem_movement",
    "mean_trajectory",
    "integrated_profile",
    "binned_trajectory",
    "rmia_table",
    "rmia_quadrants",
    "rmia_violins",
]


FULL_REPORT_SECTIONS = [
    ("## 1. Memorization Bin Plots With Difficulty Composition", "plot_report_memorization_bins"),
    ("## 2. Memorization Exposure And Onset Evidence", "plot_report_memorization_exposure_evidence"),
    ("## 3. Pruning Rate Versus Memorization, Colored By Loss Difficulty", "plot_report_pruning_vs_memorization_difficulty"),
    ("## 4. Pruning Rate Versus Memorization, Colored By Up / Same / Down", "plot_report_pruning_vs_memorization_movement"),
    ("## 5. Mean Memorization Trajectories", "plot_report_mean_memorization_trajectories"),
    ("## 6. Integrated Difficulty Profile", "plot_report_integrated_difficulty_profile"),
    ("## 7. Mean Memorization Trajectories By Easy / Medium / Hard Bins", "plot_report_binned_memorization_trajectories"),
    ("## 8. RMIA Results Table", "display_report_rmia_table"),
    ("## 9. Entropy, Memorization, And RMIA Quadrants", "plot_report_rmia_quadrants"),
    ("## 10. RMIA Score Violin Plots", "plot_report_rmia_violins"),
]


def run_report_selection(
    plot_kind: str,
    *,
    dataset_keys=None,
    method_key: str | None = None,
    method_keys=None,
    filter_spec=None,
    bin_plot_difficulty_mode: str | None = None,
    binned_trajectory_difficulty_mode: str | None = None,
):
    if plot_kind not in REPORT_PLOT_KINDS:
        raise ValueError(f"Unknown plot_kind={plot_kind!r}. Choose from {REPORT_PLOT_KINDS}.")

    dataset_keys = list(dataset_keys or report_setting("active_datasets"))
    method_keys = list(method_keys or report_setting("all_methods"))
    filter_spec = report_setting("mean_memorization_filter") if filter_spec is None else filter_spec
    bin_plot_difficulty_mode = (
        report_setting("bin_plot_difficulty_mode")
        if bin_plot_difficulty_mode is None
        else bin_plot_difficulty_mode
    )
    binned_trajectory_difficulty_mode = (
        report_setting("binned_trajectory_difficulty_mode")
        if binned_trajectory_difficulty_mode is None
        else binned_trajectory_difficulty_mode
    )

    if plot_kind == "mem_bins":
        for dataset_key in dataset_keys:
            section_plot_memorization_bin_composition(
                report_bundle(dataset_key),
                epoch=int(report_setting("memorization_epochs")[dataset_key]),
                difficulty_mode=bin_plot_difficulty_mode,
                difficulty_epoch_by_dataset=report_setting("easy_hard_epochs"),
                pruning_epoch_by_dataset=report_setting("pruning_anchor_epochs"),
                method_keys=method_keys,
            )
        return None

    if plot_kind == "pruning_mem_difficulty":
        chosen_method = method_key or report_setting("pruned_methods")[0]
        for dataset_key in dataset_keys:
            section_plot_pruning_vs_memorization_by_difficulty(
                report_bundle(dataset_key),
                epoch=int(report_setting("memorization_epochs")[dataset_key]),
                method_key=chosen_method,
                difficulty_mode="epoch_loss",
                difficulty_epoch_by_dataset=report_setting("easy_hard_epochs"),
                pruning_epoch_by_dataset=report_setting("pruning_anchor_epochs"),
            )
        return None

    if plot_kind == "tail_auc":
        return section_plot_tail_auc_summary(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
            tail_fraction=float(report_setting("tail_auc_fraction")),
        )

    if plot_kind == "final_tail_profile":
        return section_plot_final_tail_profile(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
        )

    if plot_kind == "high_mem_emergence":
        return section_plot_high_memorization_emergence(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
        )

    if plot_kind == "state_flow":
        return section_plot_memorization_state_flow(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
        )

    if plot_kind == "exceedance_curves":
        return section_plot_memorization_exceedance_checkpoints(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
        )

    if plot_kind == "exposure_evidence":
        evidence = section_plot_memorization_exposure_evidence(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
        )
        evidence["onset"] = section_plot_memorization_onset_trajectories(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
        )
        return evidence

    if plot_kind == "pruning_mem_movement":
        chosen_method = method_key or report_setting("pruned_methods")[0]
        for dataset_key in dataset_keys:
            section_plot_pruning_vs_memorization_by_movement(
                report_bundle(dataset_key),
                epoch=int(report_setting("memorization_epochs")[dataset_key]),
                method_key=chosen_method,
            )
        return None

    if plot_kind == "mean_trajectory":
        for dataset_key in dataset_keys:
            section_plot_mean_memorization_trajectory(
                report_bundle(dataset_key),
                filter_spec=filter_spec,
                method_keys=method_keys,
            )
        return None

    if plot_kind == "integrated_profile":
        section_plot_integrated_difficulty_profile(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
        )
        return None

    if plot_kind == "binned_trajectory":
        for dataset_key in dataset_keys:
            section_plot_binned_mean_memorization_trajectory(
                report_bundle(dataset_key),
                filter_spec=filter_spec,
                difficulty_mode=binned_trajectory_difficulty_mode,
                difficulty_epoch_by_dataset=report_setting("easy_hard_epochs"),
                pruning_epoch_by_dataset=report_setting("pruning_anchor_epochs"),
                method_keys=method_keys,
            )
        return None

    if plot_kind == "rmia_table":
        table = section_build_rmia_table(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
        )
        section_display_rmia_table(table)
        return table

    if plot_kind == "rmia_quadrants":
        chosen_method = method_key or method_keys[0]
        for dataset_key in dataset_keys:
            section_plot_rmia_norm_entropy_matrix(
                report_bundle(dataset_key),
                epoch=int(report_setting("memorization_epochs")[dataset_key]),
                method_key=chosen_method,
                mem_threshold=float(report_setting("rmia_mem_threshold")),
                norm_entropy_threshold=float(report_setting("rmia_normalized_entropy_threshold")),
            )
        return None

    if plot_kind == "rmia_violins":
        section_plot_rmia_score_violins(
            dataset_keys=dataset_keys,
            method_keys=method_keys,
        )
        return None


def run_full_report():
    table = None
    for heading, fn_name in FULL_REPORT_SECTIONS:
        display(Markdown(heading))
        result = globals()[fn_name]()
        if fn_name == "display_report_rmia_table":
            table = result
    return table
