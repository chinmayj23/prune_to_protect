"""
ImageNet training harness with FFCV + DDP.

Supports three training methods, all keyed on the `index` field in the indexed
beton (no reliance on loader ordering):

  - standard:     plain DDP training over allowed_train_indices.
  - infobatch:    per-epoch, select a subset of "allowed_train_indices" based
                  on per-sample loss scores. Kept samples reweighted by
                  1/ib_ratio so the expected gradient is unbiased.
  - ib_prw_norm:  phase 1 = infobatch with the same rules, while counting per
                  sample how often each gets pruned. At switch_epoch, freeze
                  prw_weights = clamp(prune_counts / phase1_epochs, epsilon),
                  normalize to mean 1. Phase 2 trains on the full allowed set
                  with a per-sample reweighted loss using prw_weights.

LOO / membership subset: `loo.include_mask` is a boolean numpy file over all
1,281,167 training samples. Only samples where mask[i]=True are allowed to
appear in training. This is what RMIA shadow/target runs consume.
"""
from __future__ import annotations

import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from ffcv.loader import OrderOption
from torch import optim
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from utils.dataset import build_analysis_loader, build_test_loader, build_train_loader
from utils import schedulers as schedulers_mod


NUM_TRAIN_SAMPLES = 1281167
SUPPORTED_METHODS = {"standard", "infobatch", "ib_prw_norm"}


# --- fastargs sections ---
Section("dataset", "Dataset config").params(
    dataset_name=Param(str, default="ImageNet"),
    num_classes=Param(int, default=1000),
    label_smoothing=Param(float, default=0.1),
    batch_size=Param(int, default=256),
    num_workers=Param(int, default=8),
    data_root=Param(str, required=True),
    train_beton=Param(str, default="train_500_0.50_90_indexed.beton"),
    val_beton=Param(str, default="val_500_0.50_90_indexed.beton"),
)
Section("model_params", "Model config").params(
    model_name=Param(str, default="resnet50"),
    pretrained=Param(bool, default=False),
)
Section("optimizer", "Optimizer config").params(
    optim_type=Param(str, default="SGD"),
    lr=Param(float, default=0.25),
    lr_min=Param(float, default=0.0),
    momentum=Param(float, default=0.9),
    weight_decay=Param(float, default=5e-5),
    scheduler_type=Param(str, default="TriangularSchedule"),
    warmup_steps=Param(int, default=10),
)
Section("experiment_params", "Experiment config").params(
    expt_name=Param(str, default="run"),
    wandb_project=Param(str, default=""),
    epochs=Param(int, default=100),
    training_precision=Param(str, default="bfloat16"),
    seed=Param(int, default=42),
    base_dir=Param(str, required=True),
)
Section("data_pruning", "Data pruning config").params(
    method=Param(str, default="standard"),
    ib_ratio=Param(float, default=0.5),
    ib_delta=Param(float, default=0.875),
    prw_epsilon=Param(float, default=0.05),
)
Section("loo", "LOO / membership subset").params(
    enabled=Param(bool, default=False),
    include_mask_path=Param(str, default=""),
    run_tag=Param(str, default=""),
    save_epochs=Param(str, default=""),
)
Section("dist_params", "Distributed config").params(
    port=Param(int, default=12355),
    address=Param(str, default="localhost"),
    gpus_per_run=Param(int, default=4),
)


# --- helpers ---

def _resolve_method(m: str) -> str:
    m = str(m).lower()
    if m not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported data_pruning.method={m!r}. Supported: {sorted(SUPPORTED_METHODS)}")
    return m


def _parse_int_list(s: str):
    s = s.strip()
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _load_include_mask(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".pt"):
        arr = torch.load(path, map_location="cpu")
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
    else:
        raise ValueError(f"Unsupported include_mask format: {path}")
    arr = np.asarray(arr).astype(bool)
    if arr.ndim != 1 or arr.shape[0] != NUM_TRAIN_SAMPLES:
        raise ValueError(f"include_mask must be shape [{NUM_TRAIN_SAMPLES}], got {arr.shape}")
    return torch.from_numpy(arr)


class Harness:
    def __init__(self, gpu_id: int, expt_dir: str, model: nn.Module) -> None:
        self.config = get_current_config()
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.expt_dir = expt_dir

        self.method = _resolve_method(self.config["data_pruning.method"])
        self.ib_ratio = float(self.config["data_pruning.ib_ratio"])
        self.ib_delta = float(self.config["data_pruning.ib_delta"])
        self.prw_epsilon = float(self.config["data_pruning.prw_epsilon"])
        self.epochs = int(self.config["experiment_params.epochs"])
        self.switch_epoch = int(self.epochs * self.ib_delta)

        self.precision, self.use_amp = self._get_dtype_amp()

        # LOO / membership
        self.loo_enabled = _as_bool(self.config["loo.enabled"])
        if self.loo_enabled:
            mask = _load_include_mask(self.config["loo.include_mask_path"])
            self.allowed_train_indices = torch.from_numpy(np.flatnonzero(mask.numpy())).long().to(self.device)
            if self.gpu_id == 0:
                print(f"[LOO] allowed samples: {self.allowed_train_indices.numel()} / {NUM_TRAIN_SAMPLES}")
        else:
            self.allowed_train_indices = torch.arange(NUM_TRAIN_SAMPLES, dtype=torch.long, device=self.device)

        self.save_epochs = set(_parse_int_list(self.config["loo.save_epochs"]))

        # Extras tracking (enables per-sample loss + prune-rate logging on full-data runs).
        # Must be set before _get_criterion so we know whether to force reduction="none".
        self.track_extras = not self.loo_enabled

        # criterion
        self.criterion = self._get_criterion()

        # model / DDP
        model = model.to(self.device)
        self.model = DDP(model, device_ids=[gpu_id])

        # optimizer / scheduler
        self._create_optimizer()

        # pruning state (global over num_train; reads/writes only into allowed)
        if self.method != "standard":
            self.scores = torch.full((NUM_TRAIN_SAMPLES,), 3.0, device=self.device, dtype=torch.float32)
            # Per-epoch local score-update buffers. Each rank writes its shard's
            # per-sample losses into these during the epoch (no collectives);
            # one all_reduce at epoch end merges ranks into self.scores.
            # Safe because FFCV distributed=True partitions active indices
            # disjointly across ranks, so each index is written by exactly one
            # rank per epoch.
            self._epoch_loss_buf = torch.zeros(NUM_TRAIN_SAMPLES, device=self.device, dtype=torch.float32)
            self._epoch_touch_buf = torch.zeros(NUM_TRAIN_SAMPLES, device=self.device, dtype=torch.float32)
            # Persistent per-sample weight buffer (mirrors CIFAR reference's
            # self.weights, line 494). Reset in place each epoch in place of
            # re-allocating a fresh 5MB tensor.
            self._epoch_weights = torch.ones(NUM_TRAIN_SAMPLES, device=self.device, dtype=torch.float32)
        else:
            self.scores = None
            self._epoch_loss_buf = None
            self._epoch_touch_buf = None
            self._epoch_weights = None

        # PRW state
        self.prw_prune_counts = None
        self.prw_weights = None
        if self.method == "ib_prw_norm":
            self.prw_prune_counts = torch.zeros(NUM_TRAIN_SAMPLES, device=self.device, dtype=torch.float32)

        # Extras tracking for full-dataset runs only (disabled for LOO/RMIA runs).
        # Per-sample prune rate = sample_prune_counts / num_prune_epochs_so_far,
        # snapshotted at end of phase 1 and end of training. Updated rank-0-only
        # inside _select_active_infobatch (once per epoch, off the hot path).
        self.num_prune_epochs = 0
        self._last_epoch_extras: dict = {}
        if self.track_extras:
            self.sample_prune_counts = torch.zeros(NUM_TRAIN_SAMPLES, device=self.device, dtype=torch.float32)
            snaps = {self.epochs - 1}
            if self.method in {"infobatch", "ib_prw_norm"} and self.switch_epoch > 0:
                snaps.add(self.switch_epoch - 1)
            self.extras_snapshot_epochs = snaps
        else:
            self.sample_prune_counts = None
            self.extras_snapshot_epochs = set()

        # test loader (fixed across the run)
        self.test_loader = self._build_test_loader()

        # Train loader cache. We reuse a single Loader across epochs when the
        # active set is static (standard, PRW phase 2) so FFCV's internal
        # per-iter shuffle counter advances and each epoch sees a different
        # batch order. For dynamic methods (infobatch / PRW phase 1) we must
        # rebuild per epoch because `indices` is fixed at Loader construction.
        self._cached_train_loader = None
        self._cached_loader_size = None

    @param("experiment_params.training_precision")
    def _get_dtype_amp(self, training_precision):
        return {
            "bfloat16": (torch.bfloat16, True),
            "float16": (torch.float16, True),
            "float32": (torch.float32, False),
        }.get(training_precision, (torch.float32, False))

    @param("dataset.label_smoothing")
    def _get_criterion(self, label_smoothing: float):
        kwargs = {"reduction": "none"} if self.method != "standard" else {}
        if label_smoothing > 0.0:
            kwargs["label_smoothing"] = label_smoothing
        return nn.CrossEntropyLoss(**kwargs)

    @param("optimizer.lr")
    @param("optimizer.momentum")
    @param("optimizer.weight_decay")
    @param("optimizer.scheduler_type")
    @param("optimizer.optim_type")
    def _create_optimizer(self, lr, momentum, weight_decay, scheduler_type, optim_type):
        if optim_type.lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = getattr(schedulers_mod, scheduler_type)(optimizer=self.optimizer)
        if self.gpu_id == 0:
            print(f"[opt] {self.optimizer.__class__.__name__} | scheduler={scheduler_type}")

    # ---- loaders ----

    @param("dataset.batch_size")
    @param("dataset.num_workers")
    @param("dataset.data_root")
    @param("dataset.train_beton")
    @param("experiment_params.seed")
    def _build_train_loader_full(self, batch_size, num_workers, data_root, train_beton, seed,
                                 active_indices: torch.Tensor | None):
        idx_list = active_indices.cpu().tolist() if active_indices is not None else None
        return build_train_loader(
            data_root=data_root,
            beton_name=train_beton,
            batch_size=batch_size,
            num_workers=num_workers,
            device=self.device,
            seed=seed,
            distributed=True,
            indices=idx_list,
            order=OrderOption.RANDOM,
        )

    @param("dataset.batch_size")
    @param("dataset.num_workers")
    @param("dataset.data_root")
    @param("dataset.val_beton")
    @param("experiment_params.seed")
    def _build_test_loader(self, batch_size, num_workers, data_root, val_beton, seed):
        return build_test_loader(
            data_root=data_root,
            beton_name=val_beton,
            batch_size=batch_size,
            num_workers=num_workers,
            device=self.device,
            seed=seed,
            distributed=True,
        )

    # ---- InfoBatch selection ----

    def _select_active_infobatch(self, use_pruning: bool) -> torch.Tensor:
        """Return active_indices (long tensor on device). All ranks produce the SAME tensor."""
        allowed = self.allowed_train_indices

        if not use_pruning:
            return allowed

        # On rank 0, compute the subset; then broadcast.
        if self.gpu_id == 0:
            allowed_scores = self.scores[allowed]
            thresh = allowed_scores.mean()
            well_learned = allowed_scores < thresh
            kept_hard = allowed[~well_learned]
            well_pool = allowed[well_learned]

            num_select = int(self.ib_ratio * len(well_pool))
            if num_select > 0:
                perm = torch.randperm(len(well_pool), device=self.device)
                selected = well_pool[perm[:num_select]]
                active = torch.cat([kept_hard, selected], dim=0)
                pruned = well_pool[perm[num_select:]]
            else:
                active = kept_hard
                pruned = well_pool

            # Record pruned counts for PRW during phase 1
            if self.method == "ib_prw_norm" and self.prw_prune_counts is not None:
                self.prw_prune_counts[pruned] += 1.0

            # Extras: track per-sample pruning frequency across all pruning methods
            # (rank 0 only; only rank 0 reads/saves it later).
            if self.track_extras and self.sample_prune_counts is not None:
                self.sample_prune_counts[pruned] += 1.0

            active = active[torch.randperm(len(active), device=self.device)]
        else:
            active = torch.empty(0, dtype=torch.long, device=self.device)

        # Broadcast length, then the active-index tensor. prw_prune_counts
        # stays rank-0-only during phase 1 (it is written only on rank 0 above);
        # the final prw_weights are broadcast once at switch_epoch via
        # _freeze_prw_weights, which is the only point any rank actually needs
        # the counts-derived state.
        active_len = torch.tensor([active.numel()], device=self.device, dtype=torch.int64)
        dist.broadcast(active_len, src=0)
        if self.gpu_id != 0:
            active = torch.empty(int(active_len.item()), dtype=torch.long, device=self.device)
        dist.broadcast(active, src=0)

        # Round to multiple of (world_size * batch_size) so DDP sharding is clean.
        world_size = dist.get_world_size()
        bs = int(self.config["dataset.batch_size"])
        step = world_size * bs
        usable = (active.numel() // step) * step
        if usable == 0:
            raise ValueError(f"Active set {active.numel()} too small for world_size={world_size}, batch_size={bs}")
        return active[:usable]

    # ---- PRW: freeze weights at end of phase 1 ----

    def _freeze_prw_weights(self):
        assert self.method == "ib_prw_norm"
        if self.prw_weights is not None:
            return
        # Phase 1 runs for exactly `switch_epoch` epochs (0..switch_epoch-1)
        # under the `epoch < switch_epoch` convention.
        #
        # prw_prune_counts is maintained only on rank 0 during phase 1 (see
        # _select_active_infobatch). Compute the final weights on rank 0 and
        # broadcast once, so every rank gets a ready prw_weights tensor.
        phase1_len = max(1, self.switch_epoch)
        if self.gpu_id == 0:
            rates = self.prw_prune_counts / float(phase1_len)
            w = torch.clamp(rates, min=self.prw_epsilon)
            # Normalize over allowed samples only; rows outside allowed are never read.
            allowed = self.allowed_train_indices
            mean_allowed = w[allowed].mean().clamp(min=1e-8)
            w = w / mean_allowed
        else:
            w = torch.empty(NUM_TRAIN_SAMPLES, device=self.device, dtype=torch.float32)
        dist.broadcast(w, src=0)
        self.prw_weights = w
        if self.gpu_id == 0:
            aw = self.prw_weights[self.allowed_train_indices]
            print(f"[PRW] frozen weights: mean={aw.mean():.4f} min={aw.min():.4f} max={aw.max():.4f}")

    # ---- loss score update for InfoBatch ----

    def _record_local_scores(self, losses: torch.Tensor, indices: torch.Tensor):
        """Per-batch: record this rank's losses into the local epoch buffer.

        No collectives here. `_finalize_scores_epoch` does a single all_reduce
        at end of epoch to merge ranks into self.scores.
        """
        assert losses.ndim == 1 and indices.ndim == 1 and losses.shape == indices.shape
        i = indices.detach().long()
        self._epoch_loss_buf.index_copy_(0, i, losses.detach().float())
        self._epoch_touch_buf.index_fill_(0, i, 1.0)

    def _finalize_scores_epoch(self):
        """End of epoch: all_reduce local buffers and merge into self.scores."""
        dist.all_reduce(self._epoch_loss_buf, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._epoch_touch_buf, op=dist.ReduceOp.SUM)
        touched = self._epoch_touch_buf > 0
        self.scores[touched] = self._epoch_loss_buf[touched]
        self._epoch_loss_buf.zero_()
        self._epoch_touch_buf.zero_()

    # ---- train / test ----

    def train_one_epoch(self, epoch: int):
        self.model.train()
        self.scheduler.step()  # apply LR for this epoch before training

        # Decide phase and active set.
        # Convention: switch_epoch = int(epochs * ib_delta) is the FIRST
        # no-pruning epoch. So for epochs=100, ib_delta=0.875, switch_epoch=87:
        # epochs 0..86 prune (87 epochs), 87..99 no prune (13 epochs).
        if self.method == "standard":
            use_pruning = False
        elif self.method == "infobatch":
            use_pruning = epoch < self.switch_epoch
        else:  # ib_prw_norm
            use_pruning = epoch < self.switch_epoch  # phase 1
            if epoch == self.switch_epoch:
                # We entered phase 2: freeze PRW weights exactly once.
                if self.prw_weights is None:
                    self._freeze_prw_weights()

        if self.method in {"infobatch", "ib_prw_norm"}:
            active = self._select_active_infobatch(use_pruning)
        else:
            active = self.allowed_train_indices

        if self.track_extras:
            if use_pruning:
                self.num_prune_epochs += 1
            allowed_n = max(1, self.allowed_train_indices.numel())
            self._last_epoch_extras = {
                "active_size": int(active.numel()),
                "effective_pruning_rate": 1.0 - float(active.numel()) / float(allowed_n),
                "phase_is_prune": 1.0 if use_pruning else 0.0,
            }
            if self.scores is not None:
                allowed_scores = self.scores[self.allowed_train_indices]
                self._last_epoch_extras["scores/mean"] = float(allowed_scores.mean().item())
                self._last_epoch_extras["scores/std"] = float(allowed_scores.std().item())

        # Static active set → reuse a cached Loader so FFCV shuffles differently
        # each epoch. Active set changes only while we're in the pruning phase
        # of infobatch / ib_prw_norm; otherwise active == allowed_train_indices.
        active_is_static = not use_pruning

        # Per-sample weights for this epoch.
        # Mirrors CIFAR reference: reset self._epoch_weights to 1.0 each epoch,
        # then overwrite well-learned rows to 1/ib_ratio during pruning.
        # ib_prw_norm phase 2 uses the frozen prw_weights tensor directly.
        # standard: weights unused inside the loop (scalar-reduction loss).
        if self.method == "standard":
            epoch_weights = None
        elif self.method == "ib_prw_norm" and not use_pruning:
            epoch_weights = self.prw_weights
        else:
            self._epoch_weights.fill_(1.0)
            if use_pruning:
                thresh = self.scores[self.allowed_train_indices].mean()
                well_mask = self.scores < thresh
                self._epoch_weights[well_mask] = 1.0 / self.ib_ratio
            epoch_weights = self._epoch_weights

        if active_is_static and self._cached_train_loader is not None \
                and self._cached_loader_size == int(active.numel()):
            train_loader = self._cached_train_loader
            loader_rebuild_ms = 0.0
        else:
            t0 = time.perf_counter()
            train_loader = self._build_train_loader_full(active_indices=active)
            loader_rebuild_ms = (time.perf_counter() - t0) * 1000.0
            if active_is_static:
                self._cached_train_loader = train_loader
                self._cached_loader_size = int(active.numel())
            else:
                self._cached_train_loader = None
                self._cached_loader_size = None

        loss_sum = torch.zeros(1, device=self.device, dtype=torch.float64)
        correct = torch.zeros(1, device=self.device, dtype=torch.float64)
        total = torch.zeros(1, device=self.device, dtype=torch.float64)
        need_score_update = self.method in {"infobatch", "ib_prw_norm"} and use_pruning

        first_batch_ms = None
        iter_start = time.perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"train ep{epoch}",
            disable=self.gpu_id != 0,
            dynamic_ncols=True,
            leave=False,
        )
        for step_idx, (images, labels, indices) in enumerate(pbar):
            if step_idx == 0:
                first_batch_ms = (time.perf_counter() - iter_start) * 1000.0
            self.optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=self.precision, enabled=self.use_amp):
                outputs = self.model(images.contiguous())
                raw_loss = self.criterion(outputs, labels)  # [B] per-sample, or scalar when standard

                if self.method == "standard":
                    loss = raw_loss if raw_loss.ndim == 0 else raw_loss.mean()
                else:
                    per_sample_w = epoch_weights[indices]
                    loss = (raw_loss * per_sample_w).mean()
                    if need_score_update:
                        self._record_local_scores(raw_loss, indices)

            loss.backward()
            self.optimizer.step()

            bs = labels.size(0)
            loss_sum += loss.detach().float() * bs
            correct += outputs.argmax(dim=1).eq(labels).sum()
            total += bs
            if self.gpu_id == 0 and step_idx % 20 == 0:
                pbar.set_postfix(loss=f"{(loss_sum.item() / max(total.item(), 1)):.3f}",
                                 acc=f"{(100.0 * correct.item() / max(total.item(), 1)):.2f}")

        if need_score_update:
            self._finalize_scores_epoch()

        if self.gpu_id == 0:
            fb = f"{first_batch_ms:.0f}" if first_batch_ms is not None else "n/a"
            rb = f"{loader_rebuild_ms:.0f}" if loader_rebuild_ms > 0 else "cached"
            print(f"[timing] epoch={epoch} loader_build_ms={rb} first_batch_ms={fb}")

        if self.track_extras and epoch in self.extras_snapshot_epochs:
            self._save_extras_snapshot(epoch)

        return loss_sum, correct, total

    def _save_extras_snapshot(self, epoch: int):
        # sample_prune_counts is only written on rank 0, so no all_reduce.
        if self.gpu_id != 0:
            return
        snap_dir = os.path.join(self.expt_dir, "extras", f"epoch_{epoch}")
        os.makedirs(snap_dir, exist_ok=True)
        prune_denom = float(max(1, self.num_prune_epochs))
        prune_rate = self.sample_prune_counts / prune_denom
        np.save(os.path.join(snap_dir, "per_sample_prune_rate.npy"), prune_rate.cpu().numpy())
        print(f"[extras] snapshot at epoch {epoch} -> {snap_dir} "
              f"(num_prune_epochs_so_far={self.num_prune_epochs})")

    def test(self):
        self.model.eval()

        loss_sum = torch.zeros(1, device=self.device, dtype=torch.float64)
        correct = torch.zeros(1, device=self.device, dtype=torch.float64)
        total = torch.zeros(1, device=self.device, dtype=torch.float64)

        test_crit = nn.CrossEntropyLoss()
        test_iter = tqdm(
            self.test_loader,
            desc="test",
            disable=self.gpu_id != 0,
            dynamic_ncols=True,
            leave=False,
        )
        with torch.no_grad():
            for images, labels, _ in test_iter:
                with autocast(device_type="cuda", dtype=self.precision, enabled=self.use_amp):
                    outputs = self.model(images.contiguous())
                    loss = test_crit(outputs, labels)
                bs = labels.size(0)
                loss_sum += loss.detach().float() * bs
                correct += outputs.argmax(dim=1).eq(labels).sum()
                total += bs
        return loss_sum, correct, total

    def train(self):
        for epoch in range(self.epochs):
            tr_l, tr_c, tr_t = self.train_one_epoch(epoch)
            te_l, te_c, te_t = self.test()

            metrics = torch.stack([tr_l.squeeze(), tr_c.squeeze(), tr_t.squeeze(), te_l.squeeze(), te_c.squeeze(), te_t.squeeze()])
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            tr_loss = (metrics[0] / metrics[2]).item()
            tr_acc = (100.0 * metrics[1] / metrics[2]).item()
            te_loss = (metrics[3] / metrics[5]).item()
            te_acc = (100.0 * metrics[4] / metrics[5]).item()

            if self.gpu_id == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch} | lr {lr:.4g} | train {tr_loss:.4f}/{tr_acc:.2f} | test {te_loss:.4f}/{te_acc:.2f}")
                try:
                    import wandb
                    if wandb.run is not None:
                        log_payload = {"epoch": epoch, "lr": lr, "train_loss": tr_loss, "train_acc": tr_acc,
                                       "test_loss": te_loss, "test_acc": te_acc}
                        if self.track_extras and self._last_epoch_extras:
                            log_payload.update(self._last_epoch_extras)
                        wandb.log(log_payload)
                except ImportError:
                    pass

                # Default: save only the final epoch. If save_epochs is set
                # (comma-separated list via loo.save_epochs), save those instead.
                if self.save_epochs:
                    save_now = epoch in self.save_epochs
                else:
                    save_now = epoch == self.epochs - 1
                if save_now:
                    ckpt_dir = os.path.join(self.expt_dir, "checkpoints")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(self.model.module.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))

