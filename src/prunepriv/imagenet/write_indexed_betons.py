#!/usr/bin/env python3
"""
Write ImageNet FFCV .beton files with an explicit sample index field.

The output schema is:
    image: RGBImageField
    label: IntField
    index: IntField

`index` is the row id in the written beton. For the training split this is
the id that InfoBatch, LOO subsets, memorization scores, and RMIA artifacts
should use as their canonical sample key.

Expected source layout:
    IMAGENET_ROOT/
      train/<class_id>/*.JPEG
      val/<class_id>/*.JPEG

Kaggle ImageNet layout is also supported:
    IMAGENET_ROOT/
      ILSVRC/Data/CLS-LOC/train/<class_id>/*.JPEG
      ILSVRC/Data/CLS-LOC/val/*.JPEG
      LOC_val_solution.csv

Example:
    python imagenet/write_indexed_betons.py \
        --imagenet-root /path/to/imagenet_1k \
        --out-dir /path/to/ffcv_imagenet_indexed \
        --num-workers 16
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional


class IndexedImageFolder:
    """ImageFolder wrapper that returns (image, label, row_index)."""

    def __init__(self, root: Path):
        ImageFolder, _ = import_torchvision()
        self.root = root
        self.dataset = ImageFolder(str(root))
        self.samples = self.dataset.samples
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        image, label = self.dataset[index]
        return image, int(label), int(index)


class IndexedFlatValFolder:
    """Flat Kaggle val/ folder wrapper with labels from LOC_val_solution.csv."""

    def __init__(self, root: Path, solution_csv: Path, class_to_idx: dict[str, int]):
        self.root = root
        self.classes = sorted(class_to_idx, key=class_to_idx.get)
        self.class_to_idx = dict(class_to_idx)

        labels_by_stem = read_val_solution(solution_csv)
        samples = []
        for path in sorted(root.glob("*.JPEG")):
            class_id = labels_by_stem.get(path.stem)
            if class_id is None:
                raise ValueError(f"No validation label found for {path.name} in {solution_csv}")
            if class_id not in self.class_to_idx:
                raise ValueError(f"Validation class {class_id!r} is not present in train class_to_idx")
            samples.append((str(path), self.class_to_idx[class_id]))

        if not samples:
            raise ValueError(f"No .JPEG validation images found in {root}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        path, label = self.samples[index]
        _, default_loader = import_torchvision()
        image = default_loader(path)
        return image, int(label), int(index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate ImageNet FFCV betons with an explicit index field."
    )
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        required=True,
        help="Root containing train/ and val/, or the Kaggle imagenet_1k root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where the indexed betons and manifests will be written.",
    )
    parser.add_argument(
        "--train-output",
        default="train_500_0.50_90_indexed.beton",
        help="Output filename for the train split.",
    )
    parser.add_argument(
        "--val-output",
        default="val_500_0.50_90_indexed.beton",
        help="Output filename for the validation split.",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=500,
        help="FFCV RGBImageField max_resolution.",
    )
    parser.add_argument(
        "--write-mode",
        default="proportion",
        choices=("raw", "jpg", "smart", "proportion"),
        help="FFCV RGBImageField write_mode.",
    )
    parser.add_argument(
        "--compress-probability",
        type=float,
        default=0.50,
        help="Compression probability used by proportion write mode.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for compressed images.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of DatasetWriter worker processes.",
    )
    parser.add_argument(
        "--val-solution-csv",
        type=Path,
        default=None,
        help="Optional path to Kaggle LOC_val_solution.csv for flat val/ folders.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only write the validation split.",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Only write the training split.",
    )
    return parser.parse_args()


def read_val_solution(path: Path) -> dict[str, str]:
    labels = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["ImageId"].strip()
            prediction = row["PredictionString"].strip()
            if not prediction:
                raise ValueError(f"Missing PredictionString for {image_id}")
            labels[image_id] = prediction.split()[0]
    if not labels:
        raise ValueError(f"No labels read from {path}")
    return labels


def import_torchvision():
    try:
        from torchvision.datasets import ImageFolder
        from torchvision.datasets.folder import default_loader
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required. Run this inside the same environment/container "
            "used for ImageNet training."
        ) from exc
    return ImageFolder, default_loader


def resolve_layout(imagenet_root: Path, val_solution_csv: Optional[Path]):
    root = imagenet_root.resolve()
    cls_loc_root = root / "ILSVRC" / "Data" / "CLS-LOC"
    data_root = cls_loc_root if cls_loc_root.exists() else root

    if val_solution_csv is not None:
        solution_csv = val_solution_csv.resolve()
    else:
        solution_csv = None
        for candidate_root in [root, *root.parents]:
            candidate = candidate_root / "LOC_val_solution.csv"
            if candidate.exists():
                solution_csv = candidate.resolve()
                break

    return data_root.resolve(), solution_csv


def import_ffcv():
    try:
        from ffcv.fields import IntField, RGBImageField
        from ffcv.writer import DatasetWriter
    except ImportError as exc:
        raise SystemExit(
            "FFCV is required. Run this inside the same environment/container "
            "used for ImageNet FFCV training."
        ) from exc
    return DatasetWriter, RGBImageField, IntField


def build_fields(args: argparse.Namespace):
    _, RGBImageField, IntField = import_ffcv()
    return {
        "image": RGBImageField(
            max_resolution=args.max_resolution,
            write_mode=args.write_mode,
            compress_probability=args.compress_probability,
            jpeg_quality=args.jpeg_quality,
        ),
        "label": IntField(),
        "index": IntField(),
    }


def write_manifest(dataset, split_name: str, output_path: Path) -> None:
    manifest_path = output_path.with_suffix(output_path.suffix + ".jsonl")
    class_map_path = output_path.with_suffix(output_path.suffix + ".classes.json")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row_index, (path, label) in enumerate(dataset.samples):
            record = {
                "index": row_index,
                "label": int(label),
                "class_id": dataset.classes[int(label)],
                "path": str(Path(path).relative_to(dataset.root)),
                "split": split_name,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    with class_map_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset.class_to_idx, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_split(
    split_name: str,
    split_dir: Path,
    output_path: Path,
    args: argparse.Namespace,
    expected_class_to_idx: Optional[dict[str, int]] = None,
    val_solution_csv: Optional[Path] = None,
) -> dict[str, int]:
    DatasetWriter, _, _ = import_ffcv()

    if not split_dir.exists():
        raise FileNotFoundError(f"{split_name} directory does not exist: {split_dir}")

    is_flat_val = split_name == "val" and any(split_dir.glob("*.JPEG"))
    if is_flat_val:
        if expected_class_to_idx is None:
            raise ValueError("Flat validation folders require train class_to_idx")
        if val_solution_csv is None or not val_solution_csv.exists():
            raise FileNotFoundError(
                "Flat validation folder detected, but LOC_val_solution.csv was not found. "
                "Pass --val-solution-csv explicitly."
            )
        dataset = IndexedFlatValFolder(split_dir, val_solution_csv, expected_class_to_idx)
    else:
        dataset = IndexedImageFolder(split_dir)

    if expected_class_to_idx is not None and dataset.class_to_idx != expected_class_to_idx:
        raise ValueError(
            f"{split_name} class_to_idx does not match the train split. "
            "The index field would still be valid, but labels would not be "
            "consistent between train and validation."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = build_fields(args)

    print(f"Writing {split_name}: {len(dataset)} samples")
    print(f"  source: {split_dir}")
    print(f"  output: {output_path}")

    writer = DatasetWriter(str(output_path), fields, num_workers=args.num_workers)
    writer.from_indexed_dataset(dataset)

    write_manifest(dataset, split_name=split_name, output_path=output_path)
    print(f"  manifest: {output_path}.jsonl")
    print(f"  class map: {output_path}.classes.json")

    return dataset.class_to_idx


def main() -> None:
    args = parse_args()
    imagenet_root, val_solution_csv = resolve_layout(args.imagenet_root, args.val_solution_csv)
    out_dir = args.out_dir.resolve()

    train_dir = imagenet_root / "train"
    val_dir = imagenet_root / "val"

    train_class_to_idx = None
    if args.skip_train and not args.skip_val:
        train_class_to_idx = IndexedImageFolder(train_dir).class_to_idx
    elif not args.skip_train:
        train_class_to_idx = write_split(
            split_name="train",
            split_dir=train_dir,
            output_path=out_dir / args.train_output,
            args=args,
        )

    if not args.skip_val:
        write_split(
            split_name="val",
            split_dir=val_dir,
            output_path=out_dir / args.val_output,
            args=args,
            expected_class_to_idx=train_class_to_idx,
            val_solution_csv=val_solution_csv,
        )

    print("Done.")
    print("Use the emitted `index` field as the canonical sample id.")


if __name__ == "__main__":
    main()
