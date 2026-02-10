from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from tailcal_detr.config import ensure_dir, load_yaml
from tailcal_detr.data.coco_dataset import collate_fn
from tailcal_detr.data.resolve import build_datasets, resolve_data
from tailcal_detr.ddp import init_distributed_if_needed, is_main_process
from tailcal_detr.io_utils import now_utc_iso, write_json
from tailcal_detr.model import (
    ModelSpec,
    apply_logit_adjustment,
    build_detr_r50,
    count_trainable_params,
    freeze_all,
    unfreeze_classification_head,
    unfreeze_bbox_head,
    unfreeze_last_encoder_block,
)
from tailcal_detr.priors import load_class_priors, priors_contiguous
from tailcal_detr.repro import ReproConfig, seed_everything, seed_worker


@dataclass(frozen=True)
class TailCalOutput:
    out_dir: Path
    best_ckpt: Path


def _device_from_cfg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_image_weights(train_ds, counts_contig: np.ndarray) -> np.ndarray:
    # Weight an image by the mean inverse frequency of classes present.
    # This is a simple, effective heuristic for class-balanced image sampling.
    eps = 1.0
    inv = 1.0 / (counts_contig + eps)
    weights = np.zeros((len(train_ds),), dtype=np.float64)
    for i in range(len(train_ds)):
        _, tgt = train_ds[i]
        labels = tgt["labels"].numpy() if torch.is_tensor(tgt["labels"]) else np.array([])
        if labels.size == 0:
            weights[i] = inv.mean()
        else:
            weights[i] = float(inv[labels].mean())
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    return weights


def train_tailcal(cfg: dict[str, Any], out_dir: Path, init_ckpt: Path, priors_path: Path, exp_overrides: dict[str, Any]) -> TailCalOutput:
    info = init_distributed_if_needed()
    seed_everything(ReproConfig(seed=int(cfg["repro"]["seed"]), deterministic=bool(cfg["repro"]["deterministic"])))

    resolved = resolve_data(cfg)
    train_ds, _ = build_datasets(resolved)

    ensure_dir(out_dir / "checkpoints")

    num_classes = len(resolved.mapping.cat_id_to_contig)
    model = build_detr_r50(num_classes=num_classes)

    ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    # Freeze baseline weights then selectively unfreeze.
    freeze_all(model)
    unfreeze_classification_head(model)
    if bool(exp_overrides.get("unfreeze_bbox_head", cfg["tailcal"].get("unfreeze_bbox_head", False))):
        unfreeze_bbox_head(model)
    if bool(exp_overrides.get("unfreeze_last_encoder_block", cfg["tailcal"].get("unfreeze_last_encoder_block", False))):
        unfreeze_last_encoder_block(model)

    pri = load_class_priors(priors_path)
    pri_contig = priors_contiguous(pri)
    logit_adj = bool(exp_overrides.get("logit_adjustment", cfg["tailcal"].get("logit_adjustment", False)))
    if logit_adj:
        apply_logit_adjustment(
            model,
            ModelSpec(
                num_classes=num_classes,
                logit_adjustment=True,
                logit_adjustment_tau=float(exp_overrides.get("logit_adjustment_tau", cfg["tailcal"].get("logit_adjustment_tau", 1.0))),
                log_priors=torch.tensor(pri_contig, dtype=torch.float32),
            ),
        )

    device = _device_from_cfg(str(cfg["train_baseline"]["device"]))
    model.to(device)

    if info.enabled:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[info.local_rank] if device.type == "cuda" else None)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["tailcal"]["lr"]), weight_decay=float(cfg["tailcal"]["weight_decay"]))

    batch_size = int(cfg["tailcal"]["batch_size"])
    num_workers = int(cfg["tailcal"]["num_workers"])

    balanced = bool(exp_overrides.get("balanced_sampling", cfg["tailcal"].get("balanced_sampling", False)))
    if info.enabled:
        # Weighted sampling + DDP sharding is non-trivial; default to DistributedSampler.
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        shuffle = False
    else:
        if balanced:
            # Build weights from train counts.
            counts_contig = np.zeros((num_classes,), dtype=np.float64)
            for cat_id, contig in resolved.mapping.cat_id_to_contig.items():
                counts_contig[contig] = float(pri.counts_by_cat_id.get(cat_id, 0))
            w = _build_image_weights(train_ds, counts_contig)
            sampler = WeightedRandomSampler(weights=torch.tensor(w, dtype=torch.double), num_samples=len(train_ds), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

    dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )

    epochs = int(cfg["tailcal"]["epochs"])
    max_steps = cfg["tailcal"].get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None

    best_loss = float("inf")
    best_ckpt = out_dir / "checkpoints" / "best.pt"
    history: list[dict[str, Any]] = []

    if is_main_process():
        write_json(
            out_dir / "tailcal_setup.json",
            {
                "created_at": now_utc_iso(),
                "init_checkpoint": str(init_ckpt),
                "trainable_params": count_trainable_params(model.module if hasattr(model, "module") else model),
                "balanced_sampling": balanced,
                "logit_adjustment": logit_adj,
                "unfreeze_bbox_head": bool(exp_overrides.get("unfreeze_bbox_head", cfg["tailcal"].get("unfreeze_bbox_head", False))),
                "unfreeze_last_encoder_block": bool(exp_overrides.get("unfreeze_last_encoder_block", cfg["tailcal"].get("unfreeze_last_encoder_block", False))),
            },
        )

    for epoch in range(1, epochs + 1):
        if info.enabled and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        model.train()
        pbar = tqdm(dl, disable=not is_main_process(), desc=f"tailcal epoch {epoch}/{epochs}")
        total_loss = 0.0
        n_steps = 0

        for step, (images, targets) in enumerate(pbar, start=1):
            if max_steps is not None and step > max_steps:
                break
            images = [im.to(device) for im in images]
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)  # type: ignore[call-arg]
            loss = sum(loss_dict.values())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu())
            n_steps += 1
            if is_main_process():
                pbar.set_postfix(loss=float(loss.detach().cpu()))

        avg_loss = total_loss / max(1, n_steps)
        history.append({"epoch": epoch, "avg_loss": avg_loss})

        if is_main_process():
            ckpt_out = {
                "created_at": now_utc_iso(),
                "epoch": epoch,
                "model_state": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                "optimizer_state": opt.state_dict(),
                "num_classes": num_classes,
                "tailcal": {
                    "balanced_sampling": balanced,
                    "logit_adjustment": logit_adj,
                    "logit_adjustment_tau": float(exp_overrides.get("logit_adjustment_tau", cfg["tailcal"].get("logit_adjustment_tau", 1.0))),
                    "unfreeze_bbox_head": bool(exp_overrides.get("unfreeze_bbox_head", cfg["tailcal"].get("unfreeze_bbox_head", False))),
                    "unfreeze_last_encoder_block": bool(exp_overrides.get("unfreeze_last_encoder_block", cfg["tailcal"].get("unfreeze_last_encoder_block", False))),
                },
                "mapping": {
                    "cat_id_to_contig": resolved.mapping.cat_id_to_contig,
                    "contig_to_cat_id": resolved.mapping.contig_to_cat_id,
                },
            }
            torch.save(ckpt_out, out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt_out, best_ckpt)
            write_json(out_dir / "train_history.json", {"history": history, "best_loss": best_loss})

    return TailCalOutput(out_dir=out_dir, best_ckpt=best_ckpt)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init_checkpoint", required=True)
    ap.add_argument("--class_priors", required=True)
    ap.add_argument("--balanced_sampling", type=str, default=None)
    ap.add_argument("--logit_adjustment", type=str, default=None)
    ap.add_argument("--temperature_scaling", type=str, default=None)
    ap.add_argument("--unfreeze_bbox_head", type=str, default=None)
    ap.add_argument("--unfreeze_last_encoder_block", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    overrides: dict[str, Any] = {}
    for k in ["balanced_sampling", "logit_adjustment", "temperature_scaling", "unfreeze_bbox_head", "unfreeze_last_encoder_block"]:
        v = getattr(args, k)
        if v is None:
            continue
        overrides[k] = v.lower() in ("1", "true", "yes", "y")

    train_tailcal(
        cfg=cfg,
        out_dir=Path(args.out),
        init_ckpt=Path(args.init_checkpoint),
        priors_path=Path(args.class_priors),
        exp_overrides=overrides,
    )


if __name__ == "__main__":
    main()
