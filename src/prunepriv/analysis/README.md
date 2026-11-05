# Figures

`clean_paper_plots.py` builds the paper's figures and tables from experiment outputs.
`report_plot_utils.py` is the data layer: it maps dataset × method to input files.
`paper_plot_utils.py` holds shared plotting helpers.

## Inputs

Run these first (see `docs/REPRODUCE.md`):

```bash
python studies/train_full_models.py --dataset cifar10
python studies/train_full_models.py --dataset cifar100 --steps
python studies/cifar10_loo.py --methods standard infobatch wlib
for m in standard infobatch wlib; do
  python scripts/attack.py --dataset cifar10 --model resnet20 --target-method $m --attack rmia --online --num-targets 20
done
python studies/collect_analysis_inputs.py
```

The last command stages everything under `results/analysis/`, or `PRUNEPRIV_ANALYSIS_ROOT` if set.

The ImageNet accuracy curves in Figure 6 read `imagenet/test_accuracy_{standard,infobatch,wlib}.csv` under
the same root. Each CSV needs a `Step` column and a column whose name contains `test_acc`, the format
of a Weights & Biases export (`experiment_params.wandb_project` turns logging on).

## Run

```bash
export MPLBACKEND=Agg
python -m prunepriv.analysis.clean_paper_plots
```

Figures and tables go to `outputs/paper/{figures,tables}/` under the analysis root.

| Paper | Output |
|---|---|
| Figure 3 | `cifar10_tail_histogram` |
| Figure 7 | `cifar10_pruning_anchor_vs_loss_paper_epoch_{10,50}` |
| Figure 11 | `entropy_memorization_summary` |

The other memorization figures come from the `pruning_difficulty`, `phase_memorization_growth` and
`pruning_rate_vs_memorization` outputs of `run_all`. For Figure 6, call
`plot_test_accuracy_trajectory(dataset_key)` with `"cifar10"`, `"cifar100_wrn28_2"` or `"imagenet"`.
