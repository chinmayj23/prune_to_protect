# ImageNet pipeline (FFCV + DDP)

Trains ResNet-50 and DeiT-Small with `standard`, `infobatch` and WLIB (`ib_prw_norm` in the configs),
and runs offline RMIA. Pruning is implemented in `harness.py`.

This pipeline supports offline RMIA only: no online RMIA, LiRA or memorization.

## Setup

FFCV needs its own conda environment:

```bash
bash env/setup_imagenet.sh
conda activate prunepriv_imagenet
```

## Data

```bash
python -m prunepriv.imagenet.write_indexed_betons \
    --imagenet-root /path/to/imagenet_1k --out-dir /path/to/betons
```

This writes `.beton` files with fields `image`, `label` and `index`. The `index` is the sample id used by
pruning and RMIA. Standard and Kaggle ImageNet layouts both work.

## Train

Configs are in `configs/{resnet50,deit_small}_{standard,infobatch,ib_prw_norm}.yaml`. Set
`dataset.data_root` to the beton directory and `experiment_params.base_dir` to the output directory.
Runs use 4 GPUs (`dist_params.gpus_per_run`) with a per-GPU batch size of 256.

```bash
python -m prunepriv.imagenet.train --config-file src/prunepriv/imagenet/configs/resnet50_standard.yaml
```

## RMIA

Generate membership masks, in which each sample is in exactly half of the models of a bank. The default
is 4 shadows and 4 targets. Offline RMIA with a = 1.0 uses only the shadows a sample is not a member of,
which is 2 per sample:

```bash
python -m prunepriv.imagenet.utils.membership --out-dir <membership_dir>
```

Train each shadow and target model on its mask by setting `loo.include_mask_path` to
`<membership_dir>/{shadows,targets}/run_<i>/keep.npy` and `loo.run_tag` to the run id. Standard targets
reuse the shadow runs. Then compute signals for each model and attack:

```bash
python -m prunepriv.imagenet.rmia.infer \
    --config-file src/prunepriv/imagenet/configs/resnet50_standard.yaml \
    --ckpt <model.pt> --out <signal_train.npy>

python -m prunepriv.imagenet.rmia.attack \
    --shadow-dir <base>/<arch>_standard_shadow_bank \
    --target-dir <base>/<arch>_<method>_target_bank \
    --membership-dir <membership_dir> --out-dir <out> --gamma 2.0
```

`summary.json` reports per-target metrics and their means. Paper values are in `docs/REPRODUCE.md`.
