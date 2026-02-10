from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tailcal_detr.config import ensure_dir, load_yaml
from tailcal_detr.data.coco_dataset import collate_fn
from tailcal_detr.data.resolve import build_datasets, resolve_data
from tailcal_detr.io_utils import now_utc_iso, read_json, write_json
from tailcal_detr.model import build_model_for_checkpoint


def _device_from_cfg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _score_temp_scale(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    # Apply temperature scaling in logit space for probabilities.
    eps = 1e-7
    s = torch.clamp(scores, min=eps, max=1.0 - eps)
    z = torch.log(s) - torch.log1p(-s)
    z = z / float(temperature)
    return torch.sigmoid(z)


def _to_coco_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)


def eval_coco(cfg: dict[str, Any], run_dir: Path, checkpoint: Path, temperature_path: Path | None = None) -> dict[str, Any]:
    ensure_dir(run_dir)
    resolved = resolve_data(cfg)
    _, val_ds = build_datasets(resolved)

    num_classes = len(resolved.mapping.cat_id_to_contig)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_for_checkpoint(num_classes=num_classes, ckpt=ckpt)
    model.load_state_dict(ckpt["model_state"], strict=True)

    device = _device_from_cfg(str(cfg["train_baseline"]["device"]))
    model.to(device)
    model.eval()

    batch_size = int(cfg["eval"]["batch_size"])
    num_workers = int(cfg["eval"]["num_workers"])
    dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    score_thr = float(cfg["eval"]["score_threshold"])
    top_k = int(cfg["eval"]["top_k"])

    temperature = None
    if temperature_path is not None and temperature_path.exists():
        tval = float(read_json(temperature_path).get("temperature"))
        if np.isfinite(tval) and tval > 0:
            temperature = tval

    preds: list[dict[str, Any]] = []
    peak_vram_bytes = None
    t0 = time.time()
    n_images = 0
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
        for images, targets in tqdm(dl, desc=f"eval {run_dir.name}"):
            images = [im.to(device) for im in images]
            outputs = model(images)  # type: ignore[call-arg]
            for out, tgt in zip(outputs, targets):
                image_id = int(tgt["image_id"].item())
                boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
                labels_contig = out["labels"].detach().cpu().numpy().astype(np.int64)
                scores = out["scores"].detach().cpu()

                if temperature is not None:
                    scores = _score_temp_scale(scores, temperature)
                scores_np = scores.numpy().astype(np.float32)

                # Filter and top-k.
                keep = scores_np >= score_thr
                boxes = boxes[keep]
                labels_contig = labels_contig[keep]
                scores_np = scores_np[keep]
                if scores_np.size > top_k:
                    idx = np.argsort(-scores_np)[:top_k]
                    boxes = boxes[idx]
                    labels_contig = labels_contig[idx]
                    scores_np = scores_np[idx]

                coco_boxes = _to_coco_xywh(boxes)
                for b, lc, s in zip(coco_boxes, labels_contig, scores_np):
                    cat_id = int(resolved.mapping.contig_to_cat_id[int(lc)])
                    preds.append(
                        {
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": [float(x) for x in b.tolist()],
                            "score": float(s),
                        }
                    )
            n_images += len(images)
        if device.type == "cuda":
            peak_vram_bytes = int(torch.cuda.max_memory_allocated(device=device))
    t1 = time.time()

    pred_path = run_dir / "preds_val.json"
    write_json(pred_path, preds)

    ap_per_class = []
    coco_stats = {"mAP": float("nan"), "AP50": float("nan"), "AP75": float("nan"), "AP_small": float("nan"), "AP_medium": float("nan"), "AP_large": float("nan")}

    if len(preds) > 0:
        # COCOeval
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt = COCO(str(resolved.val_ann))
        dt = gt.loadRes(str(pred_path))
        coco_eval = COCOeval(gt, dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats.tolist()  # 12 numbers
        coco_stats = {
            "mAP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AP_medium": float(stats[4]),
            "AP_large": float(stats[5]),
        }
        # Per-class AP for IoU=0.5:0.95, area=all, maxDets=100.
        # precision dims: [T, R, K, A, M]
        precision = coco_eval.eval["precision"]
        for k_i, cat_id in enumerate(coco_eval.params.catIds):
            p = precision[:, :, k_i, 0, -1]
            p = p[p > -1]
            ap = float(np.mean(p)) if p.size else float("nan")
            ap_per_class.append({"category_id": int(cat_id), "ap": ap})

    out = {
        "created_at": now_utc_iso(),
        "checkpoint": str(checkpoint),
        "temperature": temperature,
        "timing": {
            "images": n_images,
            "seconds": t1 - t0,
            "images_per_sec": (n_images / max(1e-9, (t1 - t0))),
            "peak_vram_bytes": peak_vram_bytes,
        },
        "coco_stats": coco_stats,
        "ap_per_class": ap_per_class,
    }
    write_json(run_dir / "metrics_coco.json", out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--temperature", default=None)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    eval_coco(cfg, run_dir=Path(args.run_dir), checkpoint=Path(args.checkpoint), temperature_path=(Path(args.temperature) if args.temperature else None))


if __name__ == "__main__":
    main()
