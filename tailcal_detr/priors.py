from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tailcal_detr.data.coco_dataset import build_category_mapping
from tailcal_detr.io_utils import now_utc_iso, write_json


@dataclass(frozen=True)
class ClassPriors:
    ann_path: str
    counts_by_cat_id: dict[int, int]
    priors_by_cat_id: dict[int, float]
    buckets: dict[str, list[int]]
    cat_id_to_contig: dict[int, int]
    contig_to_cat_id: dict[int, int]


def compute_class_priors(ann_path: Path) -> ClassPriors:
    ann = json.loads(ann_path.read_text())
    mapping = build_category_mapping(ann_path)

    counts = {cid: 0 for cid in mapping.cat_id_to_contig.keys()}
    for a in ann["annotations"]:
        cid = int(a["category_id"])
        if cid in counts:
            counts[cid] += 1
    total = sum(counts.values())
    priors = {cid: (counts[cid] / total if total > 0 else 0.0) for cid in counts}

    # Frequency buckets from train counts: bottom 20% tail, top 20% head.
    cat_ids_sorted = sorted(counts.keys(), key=lambda c: (counts[c], c))
    k = len(cat_ids_sorted)
    tail_n = max(1, int(round(0.2 * k)))
    head_n = max(1, int(round(0.2 * k)))
    tail = cat_ids_sorted[:tail_n]
    head = cat_ids_sorted[-head_n:]
    medium = [c for c in cat_ids_sorted if c not in set(tail) and c not in set(head)]

    return ClassPriors(
        ann_path=str(ann_path),
        counts_by_cat_id=counts,
        priors_by_cat_id=priors,
        buckets={"tail": tail, "medium": medium, "head": head},
        cat_id_to_contig=mapping.cat_id_to_contig,
        contig_to_cat_id=mapping.contig_to_cat_id,
    )


def save_class_priors(priors: ClassPriors, out_path: Path) -> None:
    payload: dict[str, Any] = {
        "generated_at": now_utc_iso(),
        "ann_path": priors.ann_path,
        "counts_by_cat_id": {str(k): int(v) for k, v in priors.counts_by_cat_id.items()},
        "priors_by_cat_id": {str(k): float(v) for k, v in priors.priors_by_cat_id.items()},
        "buckets": {k: [int(x) for x in v] for k, v in priors.buckets.items()},
        "cat_id_to_contig": {str(k): int(v) for k, v in priors.cat_id_to_contig.items()},
        "contig_to_cat_id": {str(k): int(v) for k, v in priors.contig_to_cat_id.items()},
    }
    write_json(out_path, payload)


def load_class_priors(path: Path) -> ClassPriors:
    obj = json.loads(path.read_text())
    return ClassPriors(
        ann_path=str(obj["ann_path"]),
        counts_by_cat_id={int(k): int(v) for k, v in obj["counts_by_cat_id"].items()},
        priors_by_cat_id={int(k): float(v) for k, v in obj["priors_by_cat_id"].items()},
        buckets={k: [int(x) for x in v] for k, v in obj["buckets"].items()},
        cat_id_to_contig={int(k): int(v) for k, v in obj["cat_id_to_contig"].items()},
        contig_to_cat_id={int(k): int(v) for k, v in obj["contig_to_cat_id"].items()},
    )


def priors_contiguous(priors: ClassPriors) -> np.ndarray:
    k = len(priors.cat_id_to_contig)
    arr = np.zeros((k,), dtype=np.float64)
    for cat_id, contig in priors.cat_id_to_contig.items():
        arr[contig] = priors.priors_by_cat_id.get(cat_id, 0.0)
    s = arr.sum()
    if s > 0:
        arr /= s
    else:
        arr[:] = 1.0 / max(1, k)
    return arr

