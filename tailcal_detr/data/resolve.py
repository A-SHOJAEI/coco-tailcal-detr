from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tailcal_detr.data.coco_dataset import CategoryMapping, SimpleCOCODetectionDataset, build_category_mapping
from tailcal_detr.data.smoke_coco import generate_smoke_coco


@dataclass(frozen=True)
class ResolvedData:
    train_ann: Path
    val_ann: Path
    train_images_dir: Path
    val_images_dir: Path
    mapping: CategoryMapping


def resolve_data(cfg: Dict[str, Any]) -> ResolvedData:
    """Ensure data exists and return paths + category mapping."""
    mode = cfg["project"]["mode"]

    if mode == "smoke":
        sd = cfg.get("smoke_data", {})
        root = Path(sd["root"])
        train_ann = root / "annotations" / "instances_train.json"
        val_ann = root / "annotations" / "instances_val.json"

        if not train_ann.exists():
            generate_smoke_coco(
                root=root,
                num_train=int(sd.get("num_train_images", 8)),
                num_val=int(sd.get("num_val_images", 4)),
                image_size=tuple(sd.get("image_size", [320, 320])),
                num_classes=int(sd.get("num_classes", 3)),
                seed=int(sd.get("seed", 0)),
            )

        mapping = build_category_mapping(train_ann)
        return ResolvedData(
            train_ann=train_ann,
            val_ann=val_ann,
            train_images_dir=root / "images" / "train",
            val_images_dir=root / "images" / "val",
            mapping=mapping,
        )

    if mode == "coco":
        data_root = Path(cfg["paths"]["data_root"])
        coco_root = data_root / "coco"
        train_ann = coco_root / "annotations" / "instances_train2017.json"
        val_ann = coco_root / "annotations" / "instances_val2017.json"
        mapping = build_category_mapping(train_ann)
        return ResolvedData(
            train_ann=train_ann,
            val_ann=val_ann,
            train_images_dir=coco_root / "train2017",
            val_images_dir=coco_root / "val2017",
            mapping=mapping,
        )

    raise ValueError(f"Unknown project.mode: {mode}")


def build_datasets(resolved: ResolvedData) -> Tuple[SimpleCOCODetectionDataset, SimpleCOCODetectionDataset]:
    train_ds = SimpleCOCODetectionDataset(
        images_dir=resolved.train_images_dir,
        ann_path=resolved.train_ann,
        mapping=resolved.mapping,
    )
    val_ds = SimpleCOCODetectionDataset(
        images_dir=resolved.val_images_dir,
        ann_path=resolved.val_ann,
        mapping=resolved.mapping,
    )
    return train_ds, val_ds
