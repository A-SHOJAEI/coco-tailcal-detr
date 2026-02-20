from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


def generate_smoke_coco(
    root: str | Path,
    num_train: int = 8,
    num_val: int = 4,
    image_size: tuple[int, int] = (320, 320),
    num_classes: int = 3,
    seed: int = 0,
) -> None:
    """Generate a tiny COCO-format dataset for smoke testing."""
    root = Path(root)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    categories = [{"id": i, "name": f"class_{i}", "supercategory": "thing"} for i in range(num_classes)]

    for split_name, n_images in [("train", num_train), ("val", num_val)]:
        images_dir = root / f"images/{split_name}"
        images_dir.mkdir(parents=True, exist_ok=True)
        ann_dir = root / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)

        images_list = []
        annotations_list = []
        ann_id = 1

        for img_idx in range(n_images):
            h, w = image_size
            img_arr = np_rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
            fname = f"{split_name}_{img_idx:04d}.png"
            Image.fromarray(img_arr).save(str(images_dir / fname))

            images_list.append({
                "id": img_idx + 1,
                "file_name": fname,
                "height": h,
                "width": w,
            })

            # Add 1-3 random box annotations.
            n_anns = rng.randint(1, 3)
            for _ in range(n_anns):
                bx = rng.randint(0, w - 40)
                by = rng.randint(0, h - 40)
                bw = rng.randint(20, min(80, w - bx))
                bh = rng.randint(20, min(80, h - by))
                cat_id = rng.randint(0, num_classes - 1)
                annotations_list.append({
                    "id": ann_id,
                    "image_id": img_idx + 1,
                    "category_id": cat_id,
                    "bbox": [bx, by, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

        coco_ann = {
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories,
        }
        ann_path = ann_dir / f"instances_{split_name}.json"
        ann_path.write_text(json.dumps(coco_ann, indent=2))
