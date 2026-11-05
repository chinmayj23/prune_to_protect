# Add a dataset

1. Create `src/prunepriv/configs/datasets/<name>.py` with a `SPEC = DatasetSpec(...)` (classes, input
   size, mean/std, train/test size, `loader_backend`, `default_model`).
2. Add it to `_REGISTRY` in `configs/datasets/__init__.py`.
3. Add its training recipe to `TRAIN_DEFAULTS` in `configs/methods/build.py`.
4. If it needs a new loader, add a backend under `data/` and handle it in `TrainingHarness.setup_loaders`
   in `training/engine.py`.
5. Smoke test: `python scripts/train.py --dataset <name> --model <m> --method standard --epochs 2`.

Models, methods, attacks and memorization then work through `build_config`.
