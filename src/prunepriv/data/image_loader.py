"""ImageLoader: a CifarLoader-compatible GPU loader for larger (224px) image datasets.

Subclasses CifarLoader to inherit the InfoBatch/WLIB selection machinery (scores/weights,
perform_infobatch, __len__, batches_for_indices, apply_subset), but:
  * loads a CHW uint8 blob at a stored size (e.g. 256) and keeps it uint8 on GPU (memory),
  * augments PER BATCH (RandomCrop(crop)+flip for train, CenterCrop(crop) for eval) instead of the
    CifarLoader whole-set precompute (which is impractical at 256px),
  * uses arbitrary mean/std (e.g. ImageNet).
Yields the same (images, labels, idxs) / (images, labels) tuples the training engine expects.
"""
from __future__ import annotations

import torch

from ..training.engine import CifarLoader, batch_crop, batch_flip_lr


class ImageLoader(CifarLoader):
    def __init__(self, path, dataset_name, train=True, batch_size=128, aug=None, mean=None, std=None,
                 crop_size=224, drop_last=None, shuffle=None, altflip=False, device="cuda",
                 blob_device=None, **kw):
        import os
        import torchvision.transforms as T
        self.device = torch.device(device)                        # compute device (GPU)
        # blob_device: where the big uint8 image tensor lives. Default = compute device (GPU-resident,
        # like waterbirds). Set "cpu" for large datasets (e.g. celeba, 162k imgs) -> batches are moved
        # to the GPU per-batch, avoiding a multi-GB GPU-resident blob.
        self.blob_device = torch.device(blob_device) if blob_device is not None else self.device
        data_path = os.path.join(path, f"{dataset_name}_train.pt" if train else f"{dataset_name}_test.pt")
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        self.train = train
        self.epoch = 0
        # keep images uint8 CHW on blob_device; normalize/crop per batch on the compute device
        imgs = data["images"]
        if imgs.dim() == 4 and imgs.shape[1] not in (1, 3):  # NHWC -> NCHW
            imgs = imgs.permute(0, 3, 1, 2)
        self.images = imgs.to(self.blob_device).contiguous()      # uint8 [N,3,S,S]
        self.labels = data["labels"].to(self.device)
        self.classes = data.get("classes", list(range(int(self.labels.max()) + 1)))
        self.sample_ids = torch.arange(len(self.images), device=self.device)
        self.clean_labels = self.labels.clone() if train else None
        self.noisy_labels = None
        self.noise_mask = None
        self.crop_size = crop_size
        mean = mean or (0.485, 0.456, 0.406)
        std = std or (0.229, 0.224, 0.225)
        self.normalize = T.Normalize(mean, std)
        self.proc_images = {}
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle
        self.altflip = altflip
        self.scores = torch.ones(len(self.images), device=self.device) * 3
        self.weights = torch.ones(len(self.images), device=self.device)
        self.static_three_segment_bins = None

    # -- per-batch preprocessing (u8 may be on blob_device; move to compute device first) --
    # Crop on the raw uint8 with a SINGLE random offset per batch (one cheap slice), not per-image
    # masked copies (batch_crop is O(r) masked ops/batch -> the CelebA bottleneck at r=8). One offset
    # per batch still gives ample crop augmentation across the many batches/epochs of a large dataset.
    def _train_batch(self, u8):
        u8 = u8.to(self.device, non_blocking=True)
        m = u8.shape[-1] - self.crop_size
        if self.aug.get("crop", True) and m > 0:
            dy = int(torch.randint(0, m + 1, (1,)).item())
            dx = int(torch.randint(0, m + 1, (1,)).item())
            u8 = u8[:, :, dy:dy + self.crop_size, dx:dx + self.crop_size]
        elif m > 0:
            r = m // 2
            u8 = u8[:, :, r:r + self.crop_size, r:r + self.crop_size]
        x = u8.to(memory_format=torch.channels_last).float().div_(255.0)
        if self.aug.get("flip", True):
            x = batch_flip_lr(x)
        return self.normalize(x)

    def _center(self, x):
        r = (x.shape[-1] - self.crop_size) // 2
        return x[:, :, r:r + self.crop_size, r:r + self.crop_size]

    def _eval_batch(self, u8):
        u8 = u8.to(self.device, non_blocking=True)
        r = (u8.shape[-1] - self.crop_size) // 2
        if r > 0:
            u8 = u8[:, :, r:r + self.crop_size, r:r + self.crop_size]
        x = u8.to(memory_format=torch.channels_last).float().div_(255.0)
        return self.normalize(x)

    def _order(self):
        return (torch.randperm if self.shuffle else torch.arange)(len(self.images), device=self.device)

    def _img(self, idxs):
        # gather image rows on the blob's device (idxs live on the compute device)
        return self.images[idxs.to(self.blob_device)]

    def iterate(self, indices):
        self.epoch += 1
        prep = self._train_batch if self.train else self._eval_batch
        if self.train:
            if indices is None:
                indices = self._order()
            for i in range(self.batches_for_indices(indices)):
                idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
                yield prep(self._img(idxs)), self.labels[idxs], idxs
        else:
            order = self._order()
            for i in range(len(self)):
                idxs = order[i * self.batch_size:(i + 1) * self.batch_size]
                yield prep(self._img(idxs)), self.labels[idxs]

    def iterate_wo_aug(self, indices):
        if self.train:
            if indices is None:
                indices = self._order()
            for i in range(self.batches_for_indices(indices)):
                idxs = indices[i * self.batch_size:(i + 1) * self.batch_size]
                yield self._eval_batch(self._img(idxs)), self.labels[idxs], idxs
        else:
            order = self._order()
            for i in range(len(self)):
                idxs = order[i * self.batch_size:(i + 1) * self.batch_size]
                yield self._eval_batch(self._img(idxs)), self.labels[idxs]

    def apply_subset(self, subset_indices):
        self.images = self.images[subset_indices.to(self.blob_device)]
        self.labels = self.labels[subset_indices]
        self.sample_ids = self.sample_ids[subset_indices]
        self.scores = self.scores[subset_indices]
        self.weights = self.weights[subset_indices]
        if self.clean_labels is not None:
            self.clean_labels = self.clean_labels[subset_indices]
        self.epoch = 0
        self.proc_images = {}
        self.static_three_segment_bins = None
