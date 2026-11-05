"""
Single-run training entry point. Spawns DDP workers across all visible GPUs.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
        --config-file configs/resnet50_standard.yaml \
        --experiment_params.expt_name r50_standard_full
"""
from __future__ import annotations

import datetime
import os
import random
import sys
import uuid
from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fastargs import get_current_config

# Ensure `utils` is importable when train.py is invoked directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # registers fastargs sections  # noqa: E402
from utils.models import build_model  # noqa: E402


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(rank: int, world_size: int, address: str, port: int):
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def worker(rank: int, world_size: int, expt_dir: str):
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")

    setup_distributed(rank, world_size, config["dist_params.address"], config["dist_params.port"])
    set_seed(config["experiment_params.seed"] + rank)

    if rank == 0 and config["experiment_params.wandb_project"]:
        try:
            import wandb
            wandb.init(
                name=config["experiment_params.expt_name"],
                project=config["experiment_params.wandb_project"],
                config=dict(config.get_all_config()) if hasattr(config, "get_all_config") else None,
                settings=wandb.Settings(save_code=False, disable_git=True),
                dir=os.environ.get("WANDB_DIR", "/tmp/wandb"),
            )
        except ImportError:
            pass

    model = build_model(
        model_name=config["model_params.model_name"],
        num_classes=config["dataset.num_classes"],
        pretrained=harness._as_bool(config["model_params.pretrained"]),
    )
    h = harness.Harness(gpu_id=rank, expt_dir=expt_dir, model=model)
    h.train()

    if rank == 0:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass
    dist.destroy_process_group()


def main():
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")

    world_size = torch.cuda.device_count()
    expected = config["dist_params.gpus_per_run"]
    if world_size != expected:
        raise RuntimeError(f"Expected {expected} visible GPUs, found {world_size}. "
                           "Slice with CUDA_VISIBLE_DEVICES.")

    base_dir = config["experiment_params.base_dir"]
    family_dir = os.path.join(base_dir, config["experiment_params.expt_name"])

    run_tag = config["loo.run_tag"]
    if run_tag:
        sub = run_tag if run_tag.startswith("run_") else f"run_{run_tag}"
    else:
        uid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        sub = f"run_{uid}"
    expt_dir = os.path.join(family_dir, sub)

    if os.path.exists(os.path.join(expt_dir, "checkpoints")):
        raise FileExistsError(f"Run directory already populated: {expt_dir}")

    os.makedirs(os.path.join(expt_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(expt_dir, "resolved_config.txt"), "w") as f:
        f.write(str(config))

    mp.spawn(worker, args=(world_size, expt_dir), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
