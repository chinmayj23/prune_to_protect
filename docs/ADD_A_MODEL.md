# Add a model

1. Add a builder `(num_classes, **kw) -> nn.Module` in `configs/models/` and register it in
   `_register_builtin_models` (`configs/models/__init__.py`).
2. Add the name to `_MODEL_TOKEN` in `configs/methods/build.py`.
3. List it in the relevant `DatasetSpec.models` and select it with `--model <name>`.

The engine builds any registered model through `build_model`, subject to input-size compatibility.
