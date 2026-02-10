from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tailcal_detr.config import ensure_dir, load_yaml
from tailcal_detr.data.coco_dataset import collate_fn
from tailcal_detr.data.resolve import build_datasets, resolve_data
from tailcal_detr.ddp import init_distributed_if_needed, is_main_process
from tailcal_detr.io_utils import now_utc_iso, write_json
from tailcal_detr.model import build_detr_r50
from tailcal_detr.repro import ReproConfig, seed_everything, seed_worker


@dataclass(frozen=True)
class TrainOutput:
    out_dir: Path
    best_ckpt: Path


def _device_from_cfg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_baseline(cfg: dict[str, Any]) -> TrainOutput:
    info = init_distributed_if_needed()
    seed_everything(ReproConfig(seed=int(cfg["repro"]["seed"]), deterministic=bool(cfg["repro"]["deterministic"])))

    resolved = resolve_data(cfg)
    train_ds, _ = build_datasets(resolved)

    out_dir = Path(cfg["train_baseline"]["out"])
    ensure_dir(out_dir / "checkpoints")

    num_classes = len(resolved.mapping.cat_id_to_contig)
    model = build_detr_r50(num_classes=num_classes)
    device = _device_from_cfg(str(cfg["train_baseline"]["device"]))
    model.to(device)

    # DDP wrapping if launched via torchrun.
    if info.enabled:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[info.local_rank] if device.type == "cuda" else None)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["train_baseline"]["lr"]), weight_decay=float(cfg["train_baseline"]["weight_decay"]))

    batch_size = int(cfg["train_baseline"]["batch_size"])
    num_workers = int(cfg["train_baseline"]["num_workers"])

    if info.enabled:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
    else:
        sampler = None

    dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )

    epochs = int(cfg["train_baseline"]["epochs"])
    max_steps = cfg["train_baseline"].get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None

    best_loss = float("inf")
    best_ckpt = out_dir / "checkpoints" / "best.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        if info.enabled:
            assert sampler is not None
            sampler.set_epoch(epoch)

        model.train()
        pbar = tqdm(dl, disable=not is_main_process(), desc=f"baseline epoch {epoch}/{epochs}")
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
        rec = {"epoch": epoch, "avg_loss": avg_loss}
        history.append(rec)

        if is_main_process():
            ckpt = {
                "created_at": now_utc_iso(),
                "epoch": epoch,
                "model_state": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                "optimizer_state": opt.state_dict(),
                "num_classes": num_classes,
                "mapping": {
                    "cat_id_to_contig": resolved.mapping.cat_id_to_contig,
                    "contig_to_cat_id": resolved.mapping.contig_to_cat_id,
                },
            }
            torch.save(ckpt, out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, best_ckpt)

            write_json(out_dir / "train_history.json", {"history": history, "best_loss": best_loss})

    return TrainOutput(out_dir=out_dir, best_ckpt=best_ckpt)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    train_baseline(cfg)


if __name__ == "__main__":
    main()

