"""LR schedulers. Both epoch-stepped to match the harness's `scheduler.step()` per epoch."""
import numpy as np
import torch
from fastargs.decorators import param
from torch.optim.optimizer import Optimizer


class _EpochStepScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def get_lr(self) -> float:
        raise NotImplementedError


class CosineLRWarmup(_EpochStepScheduler):
    """Cosine decay with linear warmup. Used for DeiT-small + AdamW."""

    @param("experiment_params.epochs")
    @param("optimizer.warmup_steps")
    @param("optimizer.lr")
    @param("optimizer.lr_min")
    def __init__(self, epochs, warmup_steps, lr, lr_min, optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.epochs = epochs
        self.warmup = warmup_steps
        self.lr = lr
        self.lr_min = lr_min

    def get_lr(self) -> float:
        e = self.last_epoch
        if e < self.warmup:
            return self.lr * (e + 1) / max(1, self.warmup)
        t = (e - self.warmup) / max(1, self.epochs - self.warmup)
        return self.lr_min + 0.5 * (1.0 + np.cos(np.pi * t)) * (self.lr - self.lr_min)


class TriangularSchedule(_EpochStepScheduler):
    """Triangular LR: linear warmup to peak, linear decay to 0. Used for ResNet50 + SGD."""

    @param("experiment_params.epochs")
    @param("optimizer.warmup_steps")
    @param("optimizer.lr")
    def __init__(self, epochs, warmup_steps, lr, optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.epochs = epochs
        self.warmup = warmup_steps
        self.lr = lr
        xs = [0, self.warmup, self.epochs]
        ys = [self.lr * 0.1, self.lr, 0.0]
        self._table = np.interp(np.arange(self.epochs + 1), xs, ys)

    def get_lr(self) -> float:
        idx = min(max(0, self.last_epoch), self.epochs)
        return float(self._table[idx])
