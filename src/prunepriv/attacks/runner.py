"""End-to-end RMIA/LiRA pipeline: train shadow + target banks, export logits, run the attack.

Dataset and model come from the registry; training uses ``training/engine.py`` and the attack runs
``attacks/rmia/main.py``.

Threat model & attack toggles:
  * defense-unaware : shadow_method == "standard" (auditor uses standard shadows)
  * defense-aware   : shadow_method == target_method (method-matched shadows)
  * online / offline: `online=True` -> audit.offline False; else True
  * RMIA / LiRA     : attack="rmia" -> sm_taylor_softmax_relative; "lira" -> loss_logit_rescaled

Bank layout read by the attack:
  <bank>/experiment-<i>_<N>/{last_epoch.pt, best_epoch.pt, keep.npy, hparams.json, logits/<ep>_<aug>.npy}
  <bank>/{x_train,y_train,x_test,y_test}.npy
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .. import paths
from ..configs import build_config, get_dataset_spec, build_model
from ..configs.methods.build import canonical_method
from ..training.engine import CifarLoader, run_single_training, set_seed

# --- constants ---
SHADOW_PKEEP = 0.5
SHADOW_KEEP_SEED = 0
TARGET_KEEP_SEED = 1
BASE_SEED = 42
EXPORT_BATCH_SIZE = 512
INFERENCE_EPOCH = 100
AUGMENTATION_COUNTS = (2, 18)

_SIGNAL = {"rmia": "sm_taylor_softmax_relative", "lira": "loss_logit_rescaled"}

# per-method seed offsets so different methods' banks use distinct seeds
_SEED_BONUS = {
    "standard": 0, "infobatch": 1_000, "ib_prw": 1_500, "ib_3seg_prw_norm": 1_750,
    "ib_noanneal": 3_000, "ib_earlystop_cooled": 3_250, "ib_earlystop_uncooled": 3_500,
    "wlib_loss": 4_000, "wlib_shuffle": 4_250, "wlib_hardconst": 4_500,
}


def method_seed_bonus(method: str) -> int:
    return _SEED_BONUS.get(canonical_method(method), 0)


# ---------------------------------------------------------------- keep matrices
def generate_keep_matrix(num_models: int, dataset_size: int, pkeep: float, seed: int) -> np.ndarray:
    """Rank-based membership: each sample kept in exactly floor(pkeep*num_models) models."""
    rng = np.random.RandomState(seed)
    random_values = rng.uniform(0.0, 1.0, size=(num_models, dataset_size))
    order = random_values.argsort(axis=0)
    return np.asarray(order < int(pkeep * num_models), dtype=bool)


def _balanced_keep(num_models: int, dataset_size: int, seed: int) -> np.ndarray:
    km = generate_keep_matrix(num_models, dataset_size, SHADOW_PKEEP, seed)
    counts = km.sum(axis=0)
    exp = num_models // 2
    if counts.min() != exp or counts.max() != exp:
        raise RuntimeError(f"balanced keep must include each sample in exactly {exp} models "
                           f"(got min={counts.min()}, max={counts.max()}).")
    return km


# ---------------------------------------------------------------- dataset root files
def ensure_dataset_root_files(exp_dir: Path, spec, data_root: str):
    exp_dir.mkdir(parents=True, exist_ok=True)
    req = [exp_dir / f"{n}.npy" for n in ("x_train", "y_train", "x_test", "y_test")]
    if all(p.exists() for p in req):
        return
    if spec.loader_backend == "image_pt":
        # 224px image datasets (e.g. waterbirds): use ImageLoader (NCHW uint8, ImageNet norm, 256->224
        # centre-crop). Labels drive the RMIA signal; x_* is only read by the (unused) gradient path,
        # so it is saved as the centre-cropped uint8 NHWC image purely so the four files exist.
        from ..data.image_loader import ImageLoader
        crop = int(spec.input_size)
        for split, train in (("train", True), ("test", False)):
            ld = ImageLoader(path=data_root, dataset_name=spec.name, train=train,
                             batch_size=EXPORT_BATCH_SIZE, shuffle=False,
                             mean=spec.mean, std=spec.std, crop_size=crop)
            np.save(exp_dir / f"y_{split}.npy", ld.labels.cpu().numpy().astype(np.int64))
            r = (ld.images.shape[-1] - crop) // 2
            x = ld.images[:, :, r:r + crop, r:r + crop].permute(0, 2, 3, 1).cpu().numpy()
            np.save(exp_dir / f"x_{split}.npy", x)
        return
    tr = CifarLoader(path=data_root, dataset_name=spec.name, train=True,
                     batch_size=EXPORT_BATCH_SIZE, shuffle=False)
    te = CifarLoader(path=data_root, dataset_name=spec.name, train=False,
                     batch_size=EXPORT_BATCH_SIZE, shuffle=False)
    np.save(exp_dir / "x_train.npy", tr.images.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32) * 2.0 - 1.0)
    np.save(exp_dir / "y_train.npy", tr.labels.cpu().numpy().astype(np.int64))
    np.save(exp_dir / "x_test.npy", te.images.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32) * 2.0 - 1.0)
    np.save(exp_dir / "y_test.npy", te.labels.cpu().numpy().astype(np.int64))


# ---------------------------------------------------------------- training one model
def _train_one_model(exp_dir: Path, run_name: str, dataset: str, model: str, method: str,
                     keep_mask: np.ndarray, seed_offset: int, epochs, force: bool,
                     use_compile=None, train_overrides=None):
    run_dir = exp_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if not force and (run_dir / "last_epoch.pt").exists() and (run_dir / "keep.npy").exists():
        return  # resume: already trained
    cfg = build_config(dataset, model, method)
    for k, v in (train_overrides or {}).items():   # recipe overrides (e.g. pretrained, learning_rate)
        setattr(cfg, k, v)
    if epochs is not None:
        cfg.num_epochs = int(epochs)
    if use_compile is not None:
        cfg.use_compile = bool(use_compile)
    cfg.save_last_checkpoint = True
    cfg.save_best_checkpoint = True
    spec = get_dataset_spec(dataset)
    ensure_dataset_root_files(exp_dir, spec, cfg.data_root_dir)

    subset = np.flatnonzero(keep_mask)
    if canonical_method(method) == "infomax":
        # InfoMax statically prunes within the membership pool: train on coreset ∩ keep-pool.
        coreset = np.load(cfg.infomax_coreset_path)
        subset = np.intersect1d(subset, coreset)
    run_seed = BASE_SEED + seed_offset
    set_seed(run_seed)
    subset_dev = torch.as_tensor(subset, dtype=torch.long, device=cfg.device)
    run_single_training(cfg, seed=run_seed, subset_indices=subset_dev,
                        unit_name=run_dir.name, output_dir=run_dir)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    np.save(run_dir / "keep.npy", np.asarray(keep_mask, dtype=bool))
    (run_dir / "hparams.json").write_text(json.dumps({
        "dataset": dataset, "model": cfg.model_name, "training_method": canonical_method(method),
        "subset_size": int(subset.size), "epochs": int(cfg.num_epochs), "seed": run_seed,
    }, indent=2), encoding="utf-8")
    (run_dir / "logits").mkdir(exist_ok=True)


def _experiment_dirs(exp_dir: Path) -> list[Path]:
    ds = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("experiment-")]
    return sorted(ds, key=lambda p: int(p.name.replace("experiment-", "").split("_")[0]))


# ---------------------------------------------------------------- logit export
def _query_specs(n_aug: int):
    shift = {2: 0, 18: 1, 50: 2}[n_aug]
    specs = [(reflect, shift, dx, dy)
             for reflect in (False, True)
             for dx in range(2 * shift + 1) for dy in range(2 * shift + 1)]
    assert len(specs) == n_aug, (len(specs), n_aug)
    return specs


def _apply_spec(images: torch.Tensor, spec) -> torch.Tensor:
    reflect, shift, dx, dy = spec
    t = images.flip(-1) if reflect else images
    if shift == 0:
        return t
    padded = F.pad(t, (shift, shift, shift, shift), mode="reflect")
    h, w = images.shape[-2], images.shape[-1]
    return padded[:, :, dx:dx + h, dy:dy + w]


def export_logits(exp_dir: Path, dataset: str, model: str):
    spec = get_dataset_spec(dataset)
    cfg = build_config(dataset, model, "standard")  # data_root + model token
    ensure_dataset_root_files(exp_dir, spec, cfg.data_root_dir)
    if spec.loader_backend == "image_pt":
        # 224px: ImageLoader gives NCHW uint8 blob; prep() does 256->224 centre-crop + ImageNet norm.
        from ..data.image_loader import ImageLoader
        loader = ImageLoader(path=cfg.data_root_dir, dataset_name=spec.name, train=True,
                             batch_size=EXPORT_BATCH_SIZE, shuffle=False,
                             mean=spec.mean, std=spec.std, crop_size=int(spec.input_size))
        raw = loader.images

        def prep(s, e):
            return loader._eval_batch(raw[s:e])
    else:
        loader = CifarLoader(path=cfg.data_root_dir, dataset_name=spec.name, train=True,
                             batch_size=EXPORT_BATCH_SIZE, shuffle=False, noise_type="clean")
        raw = loader.images

        def prep(s, e):
            return loader.normalize(raw[s:e])
    N, C = spec.train_size, spec.num_classes
    for run_dir in _experiment_dirs(exp_dir):
        ckpt = run_dir / "last_epoch.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"missing checkpoint {ckpt}")
        net = build_model(cfg.model_name, num_classes=C).to(cfg.device)
        net.load_state_dict(_load_state_dict(ckpt))
        net.train()  # BN in train mode during export (matches original)
        (run_dir / "logits").mkdir(exist_ok=True)
        for n_aug in AUGMENTATION_COUNTS:
            out_path = run_dir / "logits" / f"{INFERENCE_EPOCH:010d}_{n_aug:04d}.npy"
            if out_path.exists():
                continue
            logits = torch.empty((N, 1, n_aug, C), dtype=torch.float32)
            with torch.no_grad():
                for a, sp in enumerate(_query_specs(n_aug)):
                    for s in range(0, N, EXPORT_BATCH_SIZE):
                        e = min(s + EXPORT_BATCH_SIZE, N)
                        batch = _apply_spec(prep(s, e), sp)
                        logits[s:e, 0, a, :] = net(batch).detach().cpu().float()
            np.save(out_path, logits.numpy())
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_state_dict(path: Path) -> dict:
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location="cpu")
    if sd and all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd


# ---------------------------------------------------------------- attack config + run
_ATTACK_DIR = Path(__file__).resolve().parent / "rmia"  # holds main.py + signals.py + plot.py


def write_attack_config(cfg_path: Path, results_dir: Path, report_name: str, target_dir: Path,
                        reference_dir: Path, dataset_size: int, num_ref: int, attack: str,
                        online: bool, temperature: float = 2.0, population_cap: int = -1):
    # Build a dict and yaml.dump it (robust vs string templating for the variable audit block).
    audit = {
        "privacy_game": "privacy_loss_model",
        "report_log": report_name,
        "offline": (not online),          # main.py: False=online, True=offline, or a float
        "offline_a": 0.3,
        "num_ref_models": int(num_ref),
        "augmentation": "augmented",
        "nb_augmentation": 18,
        "top_k": -1,
        "signal": _SIGNAL[attack],
        "target_idx": "all",
        "subset": "all",
        "allout": False,
        "population_cap": int(population_cap),   # -1 = off; >0 caps the OUT population (large datasets)
    }
    if attack == "rmia":
        audit.update({"temperature": float(temperature), "gamma": 2.0, "proportiontocut": 0.2,
                      "taylor_m": 0.6, "taylor_n": 4})
    else:  # lira
        audit["fix_variance"] = False
    cfg = {
        "run": {"random_seed": 12345, "log_dir": str(results_dir), "time_log": True},
        "audit": audit,
        "data": {"target_dir": str(target_dir), "reference_dir": str(reference_dir),
                 "dataset_size": int(dataset_size), "epoch": INFERENCE_EPOCH},
    }
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def run_attack(cfg_path: Path, log_path: Path):
    import os
    env = dict(os.environ, MPLBACKEND="Agg")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "main.py", f"--cf={cfg_path}"],
                                cwd=str(_ATTACK_DIR), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in proc.stdout:
            log.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, "main.py")


# ---------------------------------------------------------------- top-level orchestrator
def run_attack_pipeline(dataset: str, model: str | None = None, target_method: str = "standard",
                        shadow_method: str = "standard", attack: str = "rmia", online: bool = True,
                        num_shadows: int = 64, num_targets: int = 1, epochs: int | None = None,
                        force: bool = False, use_compile=None, pretrained: bool = False,
                        lr: float | None = None, temperature: float = 2.0,
                        population_cap: int = -1) -> dict:
    spec = get_dataset_spec(dataset)
    model = model or spec.default_model
    target_method = canonical_method(target_method)
    shadow_method = canonical_method(shadow_method)
    defense_aware = shadow_method == target_method and target_method != "standard"

    # optional recipe override (e.g. pretrained fine-tune): applied to ALL bank models (shadows +
    # targets), and tagged with a suffix so PT banks/reports never collide with the scratch banks.
    train_overrides = {}
    if pretrained:
        train_overrides["pretrained"] = True
    if lr is not None:
        train_overrides["learning_rate"] = float(lr)
    rec = "_pt" if pretrained else ""

    def bank(role_method, role, n):
        return paths.banks_dir(f"{dataset}_{model}{rec}_{role_method}_{role}_{n}")

    shadow_dir = bank(shadow_method, "shadows", num_shadows)
    target_dir = bank(target_method, "targets", num_targets)

    # ---- shadow bank ----
    shadow_keep = _balanced_keep(num_shadows, spec.train_size, SHADOW_KEEP_SEED)
    np.save(paths.ensure(shadow_dir) / "shadow_keep_matrix.npy", shadow_keep)
    for i in range(num_shadows):
        _train_one_model(shadow_dir, f"experiment-{i}_{num_shadows}", dataset, model, shadow_method,
                         shadow_keep[i], seed_offset=i + method_seed_bonus(shadow_method),
                         epochs=epochs, force=force, use_compile=use_compile,
                         train_overrides=train_overrides)
    export_logits(shadow_dir, dataset, model)

    # ---- target bank ----
    if num_targets == 1:
        target_keep = generate_keep_matrix(num_shadows, spec.train_size, SHADOW_PKEEP, TARGET_KEEP_SEED)[0:1]
    else:
        target_keep = _balanced_keep(num_targets, spec.train_size, TARGET_KEEP_SEED)
    np.save(paths.ensure(target_dir) / "target_keep_matrix.npy", target_keep)
    for i in range(num_targets):
        _train_one_model(target_dir, f"experiment-{i}_{num_targets}", dataset, model, target_method,
                         target_keep[i], seed_offset=10_000 + i + method_seed_bonus(target_method),
                         epochs=epochs, force=force, use_compile=use_compile,
                         train_overrides=train_overrides)
    export_logits(target_dir, dataset, model)

    # ---- attack ----
    aware_tag = "aware" if defense_aware else "unaware"
    tag = f"{dataset}_{model}{rec}_{target_method}_vs_{shadow_method}_{attack}_{'online' if online else 'offline'}_{aware_tag}"
    results_dir = paths.ensure(paths.results_dir("attacks", tag))
    report_name = f"report_{attack}_{'online' if online else 'offline'}"
    cfg_path = results_dir / "attack_config.yaml"
    write_attack_config(cfg_path, results_dir, report_name, target_dir, shadow_dir,
                        spec.train_size, num_shadows, attack, online, temperature=temperature,
                        population_cap=population_cap)
    run_attack(cfg_path, results_dir / "attack.log")
    return {"tag": tag, "results_dir": str(results_dir),
            "report_dir": str(results_dir / report_name),
            "shadow_dir": str(shadow_dir), "target_dir": str(target_dir),
            "defense_aware": defense_aware, "attack": attack, "online": online}
