from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


@dataclass(frozen=True)
class CategoryMapping:
    cat_id_to_contig: Dict[int, int]
    contig_to_cat_id: Dict[int, int]


def build_category_mapping(ann_path: Path) -> CategoryMapping:
    ann = json.loads(Path(ann_path).read_text())
    cat_ids = sorted(set(int(c["id"]) for c in ann["categories"]))
    cat_id_to_contig = {cid: i for i, cid in enumerate(cat_ids)}
    contig_to_cat_id = {i: cid for cid, i in cat_id_to_contig.items()}
    return CategoryMapping(cat_id_to_contig=cat_id_to_contig, contig_to_cat_id=contig_to_cat_id)


class SimpleCOCODetectionDataset(Dataset):
    """COCO detection dataset returning (image_tensor, target_dict) for DETR-style models."""

    def __init__(
        self,
        images_dir: Path,
        ann_path: Path,
        mapping: CategoryMapping,
        max_images: Optional[int] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.images_dir = images_dir
        ann = json.loads(ann_path.read_text())
        self.img_infos = {int(im["id"]): im for im in ann["images"]}
        self.mapping = mapping
        self.transform = transform or T.Compose([T.ToTensor()])

        # Group annotations by image.
        self.anns_by_img: Dict[int, List[Dict]] = {}
        for a in ann["annotations"]:
            img_id = int(a["image_id"])
            if img_id not in self.anns_by_img:
                self.anns_by_img[img_id] = []
            self.anns_by_img[img_id].append(a)

        self.img_ids = sorted(self.img_infos.keys())
        if max_images is not None:
            self.img_ids = self.img_ids[:max_images]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.img_ids[idx]
        info = self.img_infos[img_id]
        img_path = self.images_dir / info["file_name"]
        img = Image.open(str(img_path)).convert("RGB")

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = T.ToTensor()(img)

        anns = self.anns_by_img.get(img_id, [])
        boxes = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])
            contig = self.mapping.cat_id_to_contig.get(int(a["category_id"]))
            if contig is not None:
                labels.append(contig)
            else:
                labels.append(0)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        return img_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets
