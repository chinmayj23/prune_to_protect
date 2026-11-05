# Waterbirds

Waterbirds is included as an additional dataset for studying group robustness. It is **not part of the
paper**, so no expected numbers are given here.

The setup is ResNet from scratch, SGD (lr 0.1, weight decay 5 × 10⁻⁴), triangular schedule, RandomCrop and
flip, 100 epochs. The four groups are label × background: 0 landbird/land, 1 landbird/water, 2 waterbird/land,
3 waterbird/water. Groups 1 and 2 are the minorities.

```bash
python studies/prepare_waterbirds.py         # one-time: download and preprocess
python studies/waterbirds_multiseed.py       # accuracy per group over 5 seeds
python scripts/attack.py --dataset waterbirds --model resnet18_224 --target-method wlib --attack rmia --online --num-targets 8
python studies/waterbirds_subgroup_rmia.py   # RMIA per group
```

## Memorization

```bash
python studies/waterbirds_loo.py --num-runs 200 --epochs 100 --score-every 20
```

This downloads the data if needed, draws balanced subsets (each sample is in exactly half of the models),
trains the bank for each method and writes `results/studies/waterbirds_loo/waterbirds_loo_scores.pt`:

```python
s = torch.load("results/studies/waterbirds_loo/waterbirds_loo_scores.pt")
s["ib_prw"]["memorization"]   # [n_epochs, 4795], one row per tracked epoch
s["ib_prw"]["epochs"]         # tracked epochs
s["ib_prw"]["labels"]         # 0 landbird, 1 waterbird
s["ib_prw"]["groups"]         # 0..3 as above
```

Epochs 87 and 88 are always tracked because they straddle the switch to WLIB's weighting. Runs resume:
a run with an existing `correctness.npy` is skipped. Checkpoints are deleted after scoring unless you pass
`--keep-ckpts`. `--num-runs` must be even.
