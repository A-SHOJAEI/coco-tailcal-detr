from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tailcal_detr.io_utils import now_utc_iso, write_json


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    # a, b: [x, y, w, h]
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass(frozen=True)
class CalibResult:
    n: int
    ece: float
    aurc: float
    curve: list[dict[str, float]]


def compute_ece(scores: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (scores >= lo) & (scores < hi) if i < n_bins - 1 else (scores >= lo) & (scores <= hi)
        if not np.any(m):
            continue
        conf = float(scores[m].mean())
        acc = float(correct[m].mean())
        ece += abs(acc - conf) * (float(m.sum()) / float(n))
    return float(ece)


def compute_risk_coverage(scores: np.ndarray, correct: np.ndarray, n_points: int = 50) -> tuple[float, list[dict[str, float]]]:
    idx = np.argsort(-scores)
    scores = scores[idx]
    correct = correct[idx]
    n = len(scores)
    cum_correct = np.cumsum(correct)
    coverage = (np.arange(1, n + 1) / n).astype(np.float64)
    acc = cum_correct / np.arange(1, n + 1)
    risk = 1.0 - acc

    # Sample points for a compact curve.
    xs = np.linspace(0, n - 1, n_points).astype(int) if n > 0 else np.array([], dtype=int)
    curve = [{"coverage": float(coverage[i]), "risk": float(risk[i])} for i in xs]
    aurc = float(np.trapz(risk, coverage)) if n > 1 else float("nan")
    return aurc, curve


def eval_calibration(pred_path: Path, gt_ann_path: Path, out_path: Path, iou_thr: float = 0.5) -> CalibResult:
    preds = json.loads(pred_path.read_text())
    gt = json.loads(gt_ann_path.read_text())

    gt_by_image_cat: dict[int, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for a in gt["annotations"]:
        if int(a.get("iscrowd", 0)) == 1:
            continue
        img_id = int(a["image_id"])
        cat_id = int(a["category_id"])
        gt_by_image_cat[img_id][cat_id].append(np.array(a["bbox"], dtype=np.float32))

    preds_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        preds_by_image[int(p["image_id"])].append(p)
    for img_id in preds_by_image:
        preds_by_image[img_id].sort(key=lambda x: float(x["score"]), reverse=True)

    all_scores: list[float] = []
    all_correct: list[int] = []

    for img_id, plist in preds_by_image.items():
        matched: dict[int, set[int]] = defaultdict(set)  # cat_id -> matched gt indices
        for p in plist:
            cat_id = int(p["category_id"])
            pb = np.array(p["bbox"], dtype=np.float32)
            gts = gt_by_image_cat.get(img_id, {}).get(cat_id, [])
            best_j = -1
            best_iou = 0.0
            for j, gb in enumerate(gts):
                if j in matched[cat_id]:
                    continue
                iou = _iou_xywh(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            ok = 1 if (best_j >= 0 and best_iou >= iou_thr) else 0
            if ok:
                matched[cat_id].add(best_j)
            all_scores.append(float(p["score"]))
            all_correct.append(ok)

    scores = np.asarray(all_scores, dtype=np.float64)
    correct = np.asarray(all_correct, dtype=np.float64)
    if scores.size == 0:
        res = CalibResult(n=0, ece=float("nan"), aurc=float("nan"), curve=[])
    else:
        ece = compute_ece(scores, correct, n_bins=10)
        aurc, curve = compute_risk_coverage(scores, correct, n_points=50)
        res = CalibResult(n=int(scores.size), ece=float(ece), aurc=float(aurc), curve=curve)

    write_json(
        out_path,
        {
            "created_at": now_utc_iso(),
            "pred_path": str(pred_path),
            "gt_ann_path": str(gt_ann_path),
            "iou_thr": float(iou_thr),
            "n": res.n,
            "ece": res.ece,
            "aurc": res.aurc,
            "curve": res.curve,
        },
    )
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args(argv)
    eval_calibration(Path(args.pred), Path(args.gt), Path(args.out), iou_thr=float(args.iou))


if __name__ == "__main__":
    main()

