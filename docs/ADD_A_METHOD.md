# Add a pruning or training method

1. Add the name to `VERIFIED_METHODS` in `configs/methods/build.py`.
2. In `build_method_config`, pick its base config class and set any method-specific fields.
3. If it needs new per-batch selection logic, add it to the pruning dispatch in `training/engine.py`.
4. Add a seed offset to `_SEED_BONUS` in `attacks/runner.py` so its banks use distinct seeds.
5. Smoke test: `python scripts/train.py --dataset cifar10 --method <name> --epochs 2`.

The method is then available to training, RMIA (as a target or matched shadow) and memorization, because
they all consume the same config.
