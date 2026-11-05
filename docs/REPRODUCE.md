# Reproducing the paper

Every table and figure, the command that produces it, and the expected numbers. All expected numbers
are from the paper.

## Setup

```bash
bash env/setup_cifar.sh && source env/venv/bin/activate
export MPLBACKEND=Agg
```

Data downloads automatically on first use for CIFAR-10 and CIFAR-100. Outputs go to `data/`, `banks/` and
`results/` in the repo root; set `PRUNEPRIV_DATA`, `PRUNEPRIV_BANKS` and `PRUNEPRIV_RESULTS` to move
them. The study scripts expect the package to be installed (`env/setup_cifar.sh` does this).

CIFAR experiments in the paper ran on a single NVIDIA A6000, and ImageNet on 4× NVIDIA A100 (40 GB).

## What to expect

- **Retraining will not match to the decimal.** Training uses bf16/TF32 and nondeterministic cuDNN
  kernels, so results vary between runs even at a fixed seed. InfoBatch and WLIB vary more than standard
  training; an occasional run lands about 2 points low, so compare means over seeds rather than single runs.
- **Attack metrics are per target.** AUC and TPR are computed for each target model, then averaged over
  the bank, by `studies/compute_privacy.py`. The pooled AUC that `scripts/attack.py` prints is only a
  quick check and does not match the paper's numbers.
- **Status.** CIFAR training, the RMIA/LiRA attack and the memorization scorer were checked against the
  experiments behind the paper. `studies/train_full_models.py` and `cifar10_loo.py` are thin wrappers
  around the same training engine. The ImageNet pipeline, the figure scripts and the `studies/` scripts
  that read attack and checkpoint outputs have not been run end to end in this release.

## Map

| Paper | Produced by |
|---|---|
| Table 1, CIFAR accuracy | `scripts/train.py` |
| Table 1, CIFAR privacy | `scripts/attack.py`, `studies/compute_privacy.py` |
| Table 1, ImageNet | `src/prunepriv/imagenet/README.md` |
| Table 2 (ablations) | `studies/run_accuracy.py`, `scripts/attack.py`, `studies/compute_privacy.py` |
| Table 3 (configurations) | `TRAIN_DEFAULTS` in `configs/methods/build.py`, `imagenet/configs/*.yaml` |
| Table 4 (training cost) | `studies/train_full_models.py` (retained, samples); no script for time |
| Table 5 (LiRA, defense-aware) | `scripts/attack.py`, `studies/compute_privacy.py` |
| Table 6 (per-class) | `studies/compute_group_accuracy.py`, `studies/compute_subgroup.py` |
| Table 7 (CelebA) | `studies/prepare_celeba.py`, `celeba_5seed.py`, `celeba_subgroup_rmia.py` |
| Table 8 (InfoMax) | `studies/infomax_select.py`, `infomax_rmia.py` |
| Table 9 (hard group) | `studies/compute_group_accuracy.py` |
| Figures | `studies/cifar10_loo.py`, then `src/prunepriv/analysis/README.md` |

## Table 1: privacy and accuracy

Accuracy comes from full-data models, 5 seeds:

```bash
for m in standard infobatch wlib; do
  python scripts/train.py --dataset cifar10  --model resnet20 --method $m --seeds 0 1 42 1234 3407
  python scripts/train.py --dataset cifar100 --model wrn28_2  --method $m --seeds 0 1 42 1234 3407
done
```

Privacy comes from banks of 64 shadow and 20 target models trained on 50% subsets. Defense-unaware
means the shadows use standard training, so the shadow bank is trained once and shared:

```bash
for m in standard infobatch wlib; do
  python scripts/attack.py --dataset cifar10  --model resnet20 --target-method $m --attack rmia --online --num-targets 20
  python scripts/attack.py --dataset cifar100 --model wrn28_2  --target-method $m --attack rmia --online --num-targets 20
done
python studies/compute_privacy.py     # all attacks so far -> results/studies/privacy/privacy_metrics.json
```

| Dataset (model) | Method | AUC | TPR@0% | TPR@0.01% | TPR@1% | Test acc |
|---|---|---|---|---|---|---|
| CIFAR-10 (ResNet-20) | Standard | 71.82 ± 2.95 | 1.87 ± 1.08 | 3.15 ± 1.17 | 13.76 ± 2.87 | 92.21 |
| | InfoBatch | 66.69 ± 3.82 | 0.74 ± 0.55 | 1.25 ± 0.67 | 8.47 ± 2.88 | 91.74 |
| | WLIB | 61.43 ± 4.61 | 0.20 ± 0.35 | 0.40 ± 0.43 | 4.58 ± 2.43 | 91.64 |
| CIFAR-100 (WRN-28-2) | Standard | 91.02 ± 0.25 | 4.89 ± 2.52 | 7.80 ± 2.03 | 34.74 ± 0.78 | 74.27 |
| | InfoBatch | 88.56 ± 0.28 | 1.81 ± 1.09 | 3.57 ± 1.39 | 25.20 ± 1.22 | 73.63 |
| | WLIB | 88.06 ± 0.27 | 1.22 ± 0.70 | 2.39 ± 0.70 | 21.59 ± 1.06 | 74.34 |
| ImageNet (ResNet-50) | Standard | 73.7 ± 0.15 | 0.0 | 1.50 ± 1.7 | 17.4 ± 0.11 | 76.53 |
| | InfoBatch | 73.3 ± 0.04 | 0.0 | 1.13 ± 1.3 | 15.5 ± 0.05 | 75.95 |
| | WLIB | 72.9 ± 0.16 | 0.0 | 0.60 ± 0.7 | 12.9 ± 0.13 | 75.85 |
| ImageNet (DeiT-Small) | Standard | 78.34 ± 0.72 | 0.0 | 0.00 | 19.4 ± 0.60 | 72.9 |
| | InfoBatch | 79.30 ± 0.09 | 0.0 | 0.00 | 19.5 ± 0.09 | 72.2 |
| | WLIB | 77.12 ± 0.08 | 0.0 | 0.00 | 17.3 ± 0.14 | 72.5 |

## Table 2: ablations (CIFAR-10, ResNet-20)

| Paper name | `--method` |
|---|---|
| No annealing | `ib_noanneal` |
| Early stopping | `ib_earlystop_cooled` |
| Loss weights | `wlib_loss` |
| Shuffled ρ weights | `wlib_shuffle` |
| Constant hard weights | `wlib_hardconst` |

```bash
python studies/run_accuracy.py        # accuracy for all methods and ablations
for m in ib_noanneal ib_earlystop_cooled wlib_loss wlib_shuffle wlib_hardconst; do
  python scripts/attack.py --dataset cifar10 --model resnet20 --target-method $m --attack rmia --online --num-targets 20
  python scripts/attack.py --dataset cifar10 --model resnet20 --target-method $m --attack rmia --online --num-targets 20 --defense-aware
done
```

`--defense-aware` trains a shadow bank with the target's method, so each ablation adds 64 models. The
paper averages No annealing over 5 seeds.

| Method | Test acc | Unaware AUC | TPR@0% | TPR@0.01% | TPR@1% | Aware AUC | TPR@0% | TPR@0.01% | TPR@1% |
|---|---|---|---|---|---|---|---|---|---|
| Standard | 92.21 | 71.82 | 1.87 | 3.15 | 13.76 | 71.82 | 1.87 | 3.15 | 13.76 |
| InfoBatch | 91.74 | 66.69 | 0.74 | 1.25 | 8.47 | 69.23 | 1.01 | 1.90 | 10.78 |
| No annealing | 91.10 | 65.44 | 0.47 | 0.98 | 7.54 | 68.20 | 0.92 | 1.59 | 10.03 |
| Early stopping | 90.42 | 63.19 | 0.33 | 0.62 | 5.91 | 66.69 | 0.61 | 1.25 | 8.74 |
| Loss weights | 91.26 | 65.95 | 0.58 | 1.15 | 8.00 | 68.27 | 1.12 | 1.68 | 10.10 |
| Shuffled ρ weights | 91.26 | 66.20 | 0.58 | 1.22 | 8.04 | 68.97 | 0.93 | 1.75 | 10.47 |
| Constant hard weights | 91.17 | 60.84 | 0.19 | 0.34 | 4.25 | 66.12 | 0.65 | 1.13 | 8.11 |
| WLIB | 91.64 | 61.43 | 0.20 | 0.40 | 4.58 | 65.76 | 0.56 | 1.02 | 7.88 |

## Table 3: configurations

| | CIFAR-10 ResNet-20 | CIFAR-100 WRN-28-2 | ImageNet ResNet-50 | ImageNet DeiT-Small |
|---|---|---|---|---|
| Optimizer | SGD | SGD | SGD | AdamW |
| Momentum | 0.9 | 0.9 | 0.9 | 0.9 |
| Learning rate | 0.2 | 0.1 | 0.25 | 5 × 10⁻³ |
| Weight decay | 5 × 10⁻⁴ | 3 × 10⁻⁴ | 5 × 10⁻⁵ | 0.1 |
| Batch size | 512 | 128 | 1024 | 1024 |
| Epochs | 100 | 100 | 100 | 300 |
| Warmup steps | 20 | 20 | 10 | 10 |

All runs use a triangular learning-rate schedule. InfoBatch and WLIB use r = 0.5 and δ = 0.875, and WLIB
clamps ρ at ρ_min = 0.05.

## Full-data models (Tables 4, 6, 9 and the figures)

These analyses need checkpoints at matched optimizer steps, per-sample pruning rates and evolution logs.
One script writes all of them to `results/full_models/`:

```bash
python studies/train_full_models.py --dataset cifar10
python studies/train_full_models.py --dataset cifar100 --steps
```

## Table 4: training cost

Each method's `seed_avg.pt` stores `avg/mean_effective_pruning_rate`: the retained fraction is one minus
this, and processed samples are the retained fraction × training set size × epochs. Compute scales with
processed samples. No benchmark script is included for wall-clock time, which also depends on
hardware; the paper's values were measured on one A6000.

| Dataset | Method | Retained (%) | Samples (M) | TFLOPs | Time (s) | Speedup |
|---|---|---|---|---|---|---|
| CIFAR-10 (ResNet-20) | Standard | 100 | 5.00 | 1224 | 144 | 1.00× |
| | InfoBatch | 65 | 3.23 | 791 | 97 | 1.48× |
| | WLIB | 65 | 3.23 | 791 | 98 | 1.47× |
| CIFAR-100 (WRN-28-2) | Standard | 100 | 5.00 | 6431 | 824 | 1.00× |
| | InfoBatch | 68.5 | 3.42 | 4402 | 564 | 1.46× |
| | WLIB | 68.5 | 3.42 | 4402 | 563 | 1.46× |

## Table 5: RMIA and LiRA, defense-unaware and defense-aware (CIFAR-10)

```bash
for m in infobatch wlib; do
  python scripts/attack.py --dataset cifar10 --model resnet20 --target-method $m --attack rmia --online --num-targets 20 --defense-aware
  python scripts/attack.py --dataset cifar10 --model resnet20 --target-method $m --attack lira --online --num-targets 20 --defense-aware
done
for m in standard infobatch wlib; do
  python scripts/attack.py --dataset cifar10 --model resnet20 --target-method $m --attack lira --online --num-targets 20
done
```

| Attack | Method | Setting | AUC | TPR@0% | TPR@0.01% | TPR@1% |
|---|---|---|---|---|---|---|
| RMIA | Standard | – | 71.82 | 1.87 | 3.15 | 13.76 |
| | InfoBatch | Unaware | 66.69 | 0.74 | 1.25 | 8.47 |
| | InfoBatch | Aware | 69.23 | 1.01 | 1.90 | 10.78 |
| | WLIB | Unaware | 61.43 | 0.20 | 0.40 | 4.58 |
| | WLIB | Aware | 65.76 | 0.56 | 1.02 | 7.88 |
| LiRA | Standard | – | 71.82 | 2.04 | 3.43 | 14.10 |
| | InfoBatch | Unaware | 66.44 | 0.84 | 1.39 | 8.28 |
| | InfoBatch | Aware | 67.86 | 1.09 | 1.66 | 10.45 |
| | WLIB | Unaware | 60.97 | 0.22 | 0.45 | 4.39 |
| | WLIB | Aware | 64.95 | 0.40 | 0.80 | 7.47 |

## Table 6: per-class difference, WLIB minus InfoBatch (CIFAR-10)

```bash
python studies/compute_group_accuracy.py   # per-class test accuracy
python studies/compute_subgroup.py         # per-class RMIA AUC
```

The accuracy script reads one seed (3407); the AUC script averages over the target bank.

| Class | Test acc (pp) | AUC (pp) |
|---|---|---|
| Airplane | +0.40 | −4.41 |
| Automobile | +0.10 | −2.42 |
| Bird | −0.10 | −6.66 |
| Cat | −0.40 | −8.88 |
| Deer | +0.70 | −5.03 |
| Dog | −0.10 | −8.06 |
| Frog | +0.40 | −4.03 |
| Horse | +0.30 | −4.57 |
| Ship | 0.00 | −2.34 |
| Truck | +0.40 | −2.97 |

## Table 7: CelebA (ResNet-10t)

Target Blond_Hair with Male as the spurious attribute, giving four groups (group = 2·Blond_Hair + Male).
Blonde males are the minority at 0.85% of the training data. The recipe is 50 epochs, δ = 0.875, no
CutMix/MixUp.

```bash
python studies/prepare_celeba.py           # one-time: download and preprocess
python studies/celeba_5seed.py             # accuracy per group
for m in standard infobatch wlib; do
  python scripts/attack.py --dataset celeba --model resnet10t --target-method $m --attack rmia --online --num-targets 20 \
      --population-cap 20000               # caps the RMIA population to fit in memory
done
python studies/celeba_subgroup_rmia.py     # RMIA per group
```

| Metric | Method | Overall | Non-blonde female | Non-blonde male | Blonde female | Blonde male |
|---|---|---|---|---|---|---|
| AUC | Standard | 53.3 | 53.0 | 50.4 | 59.8 | 75.6 |
| | InfoBatch | 54.5 | 54.2 | 50.9 | 62.3 | 80.6 |
| | WLIB | 52.9 | 52.8 | 50.6 | 57.4 | 68.5 |
| TPR@0% | Standard | 0.00 | 0.01 | 0.02 | 0.00 | 0.51 |
| | InfoBatch | 0.00 | 0.01 | 0.02 | 0.01 | 0.34 |
| | WLIB | 0.00 | 0.00 | 0.02 | 0.00 | 0.06 |
| TPR@0.01% | Standard | 0.02 | 0.04 | 0.07 | 0.01 | 0.51 |
| | InfoBatch | 0.02 | 0.03 | 0.07 | 0.02 | 0.34 |
| | WLIB | 0.00 | 0.01 | 0.05 | 0.00 | 0.06 |
| TPR@1% | Standard | 2.9 | 2.8 | 1.4 | 4.6 | 4.2 |
| | InfoBatch | 3.5 | 3.4 | 1.6 | 5.8 | 4.7 |
| | WLIB | 2.2 | 2.3 | 1.5 | 2.6 | 0.37 |
| Test acc | Standard | 95.63 | 96.0 | 99.6 | 86.8 | 36.7 |
| | InfoBatch | 95.56 | 95.7 | 99.6 | 87.0 | 39.3 |
| | WLIB | 95.71 | 95.9 | 99.7 | 87.2 | 37.0 |

Group-adjusted test accuracy: Standard 95.69, InfoBatch 95.61, WLIB 95.76.

## Table 8: InfoMax (CIFAR-10)

InfoMax selects a static coreset at InfoBatch's retained fraction (0.645). Selection needs a full-data
standard model, and the defense-unaware attack needs the standard shadow bank from Table 1:

```bash
python studies/infomax_select.py
python scripts/train.py --dataset cifar10 --model resnet20 --method infomax --seeds 0 1 42 1234 3407
python studies/infomax_rmia.py
```

| Method | AUC | TPR@0% | TPR@0.01% | TPR@1% | Test acc |
|---|---|---|---|---|---|
| Standard | 71.82 | 1.87 | 3.15 | 13.76 | 92.21 |
| InfoMax | 71.47 | 1.78 | 3.03 | 13.72 | 91.55 |
| InfoBatch | 66.69 | 0.74 | 1.25 | 8.47 | 91.74 |
| WLIB | 61.43 | 0.20 | 0.40 | 4.58 | 91.64 |

## Table 9: hard-group training accuracy at matched steps (CIFAR-10)

`compute_group_accuracy.py` above writes this to `results/studies/perstep/group_accuracy.json`. The
table is `per_step.<method>.<step>.train.hard`, for seed 3407. The hard group is the lowest of three
equal-width ρ bins. The `test` entries in that file are a separate measurement, grouped by test loss,
and are not reported in the paper.

| Step | Standard | InfoBatch | WLIB |
|---|---|---|---|
| 1000 | 54.5 | 50.6 | 50.6 |
| 2000 | 47.7 | 60.6 | 60.6 |
| 3000 | 64.2 | 67.4 | 67.4 |
| 4000 | 65.9 | 77.8 | 77.8 |
| 5000 | 64.2 | 90.3 | 90.3 |
| 6300 | 75.5 | 98.7 | 96.6 |

## Memorization and figures

Samples are grouped into three equal-width bins of InfoBatch's ρ: hard (ρ ∈ [0.00, 0.21],
n = 6,303), medium (n = 25,789) and easy (n = 17,908).

The memorization bank is 400 models per method on independent 70% subsets (Appendix A.2):

```bash
python studies/cifar10_loo.py --methods standard infobatch wlib
```

To score a bank directly:

```bash
python scripts/memorize.py --dataset cifar10 --model resnet20 \
    --runs-dir results/studies/cifar10_loo/ib_prw --num-runs 400 \
    --subset-file data/subsets_cifar_400.pt --epochs 99 --out <dir>
```

Then follow `src/prunepriv/analysis/README.md` to build the figures.
