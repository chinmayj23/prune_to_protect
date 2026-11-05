# Architecture

Dataset, model and method are three independent choices. A registry composes them into one config, and
training, attacks and memorization all read that config.

```
src/prunepriv/
  configs/       the registry: datasets/, models/, methods/build.py, base.py
  training/      engine.py (trainer) and api.py (train_full)
  attacks/       runner.py (end to end), rmia/ (the attack)
  memorization/  loo.py (leave-one-out scorer)
  data/          image_loader.py (CelebA, Waterbirds)
  analysis/      figure code
  imagenet/      ImageNet pipeline (FFCV + DDP)
  paths.py       all filesystem locations
scripts/         command-line entry points: train, attack, memorize
studies/         drivers for the paper's experiments
```

## The registry (`configs/`)

- **`datasets/`**: one `DatasetSpec` per dataset (classes, size, normalization, loader backend, default
  model). Look one up with `get_dataset_spec(name)`.
- **`models/`**: `build_model(name, num_classes, **kw)` over builders added with `register_model`.
- **`methods/build.py`**: `VERIFIED_METHODS` lists the available methods and `TRAIN_DEFAULTS` holds the
  per-dataset recipe. `build_method_config` applies both to an `ExptConfig` from `base.py`.
- **`build_config(dataset, model, method, **overrides)`** ties these together.

`wlib` is an alias of `ib_prw` (pruning-rate weighting) and `ib` of `infobatch`. Result directories use
the canonical names.

## Training (`training/`)

`engine.py` holds the data loaders, the InfoBatch pruning logic and the training loop, and dispatches on
`cfg.training_method` and `cfg.model_name`. `run_single_training(cfg, seed, subset_indices, unit_name,
output_dir)` trains one model. `train_full` wraps it for full-data training and returns test accuracy.

WLIB follows InfoBatch until annealing starts, counting how often each sample is pruned. At the switch
epoch, `compute_prw_weights` turns the counts into fixed per-sample loss weights (Eq. 6 in the paper).

The engine contains code paths for methods outside the paper. They are not reachable from the CLI.

## Loader backends

- `fast_pt`: GPU-resident CIFAR loader with on-GPU augmentation, which tracks per-sample ids and
  InfoBatch scores and weights.
- `image_pt`: preprocessed image tensors for CelebA and Waterbirds.
- `ffcv`: ImageNet `.beton` files, with an `index` field as the canonical sample id.

## Attacks (`attacks/`)

`runner.py` trains the shadow and target banks, exports logits, writes an attack config, and runs
`rmia/main.py` as a subprocess. Banks live under `banks/`, reports under
`results/attacks/<tag>/report_<attack>_<mode>/`.

| Choice | Setting |
|---|---|
| RMIA or LiRA | `--attack`, which sets `audit.signal` |
| online or offline | `--online`/`--offline`, which sets `audit.offline` |
| defense-unaware or defense-aware | shadows use standard training, or `--defense-aware` matches the target's method |

`studies/compute_privacy.py` computes AUC and TPR per target model and aggregates them over the target
bank.

## Memorization (`memorization/`)

`compute_loo_scores` estimates memorization as mean correctness over models trained with a sample minus
mean correctness over models trained without it, using the subset membership file.

## ImageNet (`imagenet/`)

A separate pipeline built on fastargs configs, FFCV loaders and DDP training, with its own InfoBatch and
WLIB implementation in `harness.py` and an offline RMIA attack in `rmia/`. It needs its own environment
because FFCV's compiled dependencies do not coexist with the CIFAR environment. See its README.
