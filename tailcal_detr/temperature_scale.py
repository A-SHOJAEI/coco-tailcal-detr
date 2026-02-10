from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tailcal_detr.config import ensure_dir, load_yaml
from tailcal_detr.data.coco_dataset import collate_fn
from tailcal_detr.data.resolve import build_datasets, resolve_data
from tailcal_detr.eval_calibration import _iou_xywh
from tailcal_detr.io_utils import now_utc_iso, write_json
from tailcal_detr.model import build_model_for_checkpoint
from tailcal_detr.repro import ReproConfig, seed_everything


@dataclass(frozen=True)
class TemperatureResult:
    temperature: float
    n: int
    nll: float


def _device_from_cfg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _collect_conf_correct(
    model,
    device: torch.device,
    dl: DataLoader,
    gt_ann_path: Path,
    contig_to_cat_id: dict[int, int],
    score_thr: float,
    top_k: int,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    gt = json.loads(gt_ann_path.read_text())
    gt_by_image_cat: dict[int, dict[int, list[np.ndarray]]] = {}
    for a in gt["annotations"]:
        if int(a.get("iscrowd", 0)) == 1:
            continue
        img_id = int(a["image_id"])
        cat_id = int(a["category_id"])
        gt_by_image_cat.setdefault(img_id, {}).setdefault(cat_id, []).append(np.array(a["bbox"], dtype=np.float32))

    all_scores: list[float] = []
    all_correct: list[int] = []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(dl, desc="temperature_scale"):
            images = [im.to(device) for im in images]
            outputs = model(images)  # type: ignore[call-arg]
            for out, tgt in zip(outputs, targets):
                image_id = int(tgt["image_id"].item())
                # Convert boxes to xywh for matching.
                boxes_xyxy = out["boxes"].detach().cpu().numpy().astype(np.float32)
                x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
                boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
                labels = out["labels"].detach().cpu().numpy().astype(np.int64)
                scores = out["scores"].detach().cpu().numpy().astype(np.float64)

                keep = scores >= score_thr
                boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
                if scores.size > top_k:
                    idx = np.argsort(-scores)[:top_k]
                    boxes, labels, scores = boxes[idx], labels[idx], scores[idx]

                # Match per category greedily.
                matched: dict[int, set[int]] = {}
                for b, lc, s in zip(boxes, labels, scores):
                    # torchvision DETR outputs contiguous class indices; map back to annotation category_id.
                    cat_id = int(contig_to_cat_id[int(lc)])
                    gts = gt_by_image_cat.get(image_id, {}).get(cat_id, [])
                    matched.setdefault(cat_id, set())
                    best_j = -1
                    best_iou = 0.0
                    for j, gb in enumerate(gts):
                        if j in matched[cat_id]:
                            continue
                        iou = _iou_xywh(b, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    ok = 1 if (best_j >= 0 and best_iou >= iou_thr) else 0
                    if ok:
                        matched[cat_id].add(best_j)
                    all_scores.append(float(s))
                    all_correct.append(ok)

    return np.asarray(all_scores, dtype=np.float64), np.asarray(all_correct, dtype=np.float64)


def _nll_temperature(T: float, scores: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-7
    s = np.clip(scores, eps, 1.0 - eps)
    z = np.log(s) - np.log1p(-s)
    z = z / T
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, eps, 1.0 - eps)
    nll = -(y * np.log(p) + (1.0 - y) * np.log1p(-p)).mean()
    return float(nll)


def fit_temperature(scores: np.ndarray, correct: np.ndarray) -> TemperatureResult:
    if scores.size == 0:
        return TemperatureResult(temperature=float("nan"), n=0, nll=float("nan"))
    res = minimize_scalar(
        lambda t: _nll_temperature(float(t), scores, correct),
        bounds=(0.05, 5.0),
        method="bounded",
        options={"xatol": 1e-3},
    )
    T = float(res.x)
    nll = _nll_temperature(T, scores, correct)
    return TemperatureResult(temperature=T, n=int(scores.size), nll=float(nll))


def temperature_scale(cfg: dict[str, Any], checkpoint: Path, out_path: Path) -> dict[str, Any]:
    seed_everything(ReproConfig(seed=int(cfg["repro"]["seed"]), deterministic=bool(cfg["repro"]["deterministic"])))
    ensure_dir(out_path.parent)

    resolved = resolve_data(cfg)
    train_ds, _ = build_datasets(resolved)

    # Deterministic calibration subset from train.
    n_calib = min(32 if cfg["project"]["mode"] == "smoke" else 5000, len(train_ds))
    calib_idx = list(range(n_calib))
    calib_ds = Subset(train_ds, calib_idx)

    device = _device_from_cfg(str(cfg["train_baseline"]["device"]))
    num_classes = len(resolved.mapping.cat_id_to_contig)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_for_checkpoint(num_classes=num_classes, ckpt=ckpt)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    dl = DataLoader(calib_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    scores, correct = _collect_conf_correct(
        model=model,
        device=device,
        dl=dl,
        gt_ann_path=resolved.train_ann,
        contig_to_cat_id=resolved.mapping.contig_to_cat_id,
        score_thr=0.05,
        top_k=50,
        iou_thr=0.5,
    )
    fit = fit_temperature(scores, correct)
    payload = {
        "created_at": now_utc_iso(),
        "checkpoint": str(checkpoint),
        "n": fit.n,
        "temperature": fit.temperature,
        "nll": fit.nll,
    }
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    temperature_scale(cfg, checkpoint=Path(args.checkpoint), out_path=Path(args.out))


if __name__ == "__main__":
    main()
