from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from tailcal_detr.config import ensure_dir, load_yaml
from tailcal_detr.data.resolve import resolve_data
from tailcal_detr.eval_calibration import eval_calibration
from tailcal_detr.eval_coco import eval_coco
from tailcal_detr.io_utils import now_utc_iso, read_json, write_json, write_text
from tailcal_detr.precompute_class_priors import main as _priors_main
from tailcal_detr.temperature_scale import temperature_scale
from tailcal_detr.train import train_baseline
from tailcal_detr.train_tailcal import train_tailcal


def _class_priors_path(cfg: dict[str, Any]) -> Path:
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    return artifacts / "class_priors.json"


def step_data(cfg: dict[str, Any]) -> None:
    artifacts = ensure_dir(cfg["paths"]["artifacts_dir"])
    _ = resolve_data(cfg)  # ensures data exists (smoke creates; coco downloads+extracts)
    resolved = resolve_data(cfg)
    pri_path = _class_priors_path(cfg)
    _priors_main(["--ann", str(resolved.train_ann), "--out", str(pri_path)])
    write_json(artifacts / "data_ready.json", {"created_at": now_utc_iso(), "mode": cfg["project"]["mode"]})


def _baseline_checkpoint(cfg: dict[str, Any]) -> Path:
    baseline_out = Path(cfg["train_baseline"]["out"])
    return baseline_out / "checkpoints" / "best.pt"


def step_train(cfg: dict[str, Any]) -> None:
    # Baseline
    baseline_out = train_baseline(cfg)
    base_ckpt = baseline_out.best_ckpt

    pri_path = _class_priors_path(cfg)

    # TailCal + ablations described in plan.
    for exp in cfg.get("experiments", []):
        if exp["kind"] == "baseline":
            continue
        out_dir = Path(exp["out"])
        ensure_dir(out_dir)

        # Temperature-only ablation: no fine-tuning, just inherit baseline weights.
        if exp.get("balanced_sampling") is False and exp.get("logit_adjustment") is False and exp.get("temperature_scaling") is True:
            ensure_dir(out_dir / "checkpoints")
            shutil.copy2(base_ckpt, out_dir / "checkpoints" / "best.pt")
            write_json(out_dir / "tailcal_setup.json", {"created_at": now_utc_iso(), "note": "temp-only: reused baseline checkpoint"})
            continue

        train_tailcal(
            cfg=cfg,
            out_dir=out_dir,
            init_ckpt=base_ckpt,
            priors_path=pri_path,
            exp_overrides={
                "balanced_sampling": exp.get("balanced_sampling", cfg["tailcal"].get("balanced_sampling", False)),
                "logit_adjustment": exp.get("logit_adjustment", cfg["tailcal"].get("logit_adjustment", False)),
                "logit_adjustment_tau": exp.get("logit_adjustment_tau", cfg["tailcal"].get("logit_adjustment_tau", 1.0)),
                "temperature_scaling": exp.get("temperature_scaling", cfg["tailcal"].get("temperature_scaling", False)),
                "unfreeze_bbox_head": exp.get("unfreeze_bbox_head", cfg["tailcal"].get("unfreeze_bbox_head", False)),
                "unfreeze_last_encoder_block": exp.get("unfreeze_last_encoder_block", cfg["tailcal"].get("unfreeze_last_encoder_block", False)),
            },
        )


def step_eval(cfg: dict[str, Any]) -> None:
    resolved = resolve_data(cfg)
    pri_path = _class_priors_path(cfg)
    if not pri_path.exists():
        raise FileNotFoundError(f"Missing priors file, run `make data` first: {pri_path}")

    # Evaluate each experiment; if temperature scaling enabled, fit temperature from train-calib subset.
    for exp in cfg.get("experiments", []):
        run_dir = Path(exp["out"])
        ckpt = run_dir / "checkpoints" / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {exp['name']}: {ckpt}")

        temp_path = run_dir / "temperature.json"
        if bool(exp.get("temperature_scaling", False)):
            if not temp_path.exists():
                temperature_scale(cfg, checkpoint=ckpt, out_path=temp_path)

        eval_coco(cfg, run_dir=run_dir, checkpoint=ckpt, temperature_path=(temp_path if temp_path.exists() else None))
        pred_path = run_dir / "preds_val.json"
        eval_calibration(pred_path, resolved.val_ann, run_dir / "calibration.json", iou_thr=0.5)


def step_report(cfg: dict[str, Any]) -> None:
    artifacts = ensure_dir(cfg["paths"]["artifacts_dir"])
    pri = read_json(_class_priors_path(cfg))
    buckets: dict[str, list[int]] = {k: [int(x) for x in v] for k, v in pri.get("buckets", {}).items()}

    # Aggregate key metrics from each run.
    rows = []
    for exp in cfg.get("experiments", []):
        run_dir = Path(exp["out"])
        m_coco = read_json(run_dir / "metrics_coco.json")
        m_cal = read_json(run_dir / "calibration.json")
        ap_pc = {int(x["category_id"]): float(x["ap"]) for x in m_coco.get("ap_per_class", []) if "category_id" in x and "ap" in x}

        grouped_ap: dict[str, float] = {}
        for bname, cat_ids in buckets.items():
            vals = [ap_pc.get(int(cid)) for cid in cat_ids]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and (v != v))]  # drop NaNs
            grouped_ap[bname] = float(sum(vals) / len(vals)) if vals else float("nan")

        row = {
            "name": exp["name"],
            "kind": exp["kind"],
            "run_dir": str(run_dir),
            "coco": m_coco.get("coco_stats", {}),
            "grouped_ap": grouped_ap,
            "calibration": {"ece": m_cal.get("ece"), "aurc": m_cal.get("aurc"), "n": m_cal.get("n")},
        }
        if (run_dir / "temperature.json").exists():
            row["temperature"] = read_json(run_dir / "temperature.json").get("temperature")
        rows.append(row)

    results = {
        "created_at": now_utc_iso(),
        "config": cfg,
        "runs": rows,
    }
    write_json(artifacts / "results.json", results)

    # Human report.
    def _fmt(x):
        return "nan" if x is None else (f"{x:.4f}" if isinstance(x, (int, float)) else str(x))

    lines = []
    lines.append(f"# TailCal-DETR Report")
    lines.append("")
    lines.append(f"Generated at: `{results['created_at']}`")
    lines.append(f"Mode: `{cfg['project']['mode']}`")
    lines.append("")
    lines.append("| Run | mAP | AP50 | AP75 | ECE | AURC | T |")
    lines.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:|")
    for r in rows:
        coco = r["coco"]
        cal = r["calibration"]
        lines.append(
            "| {name} | {mAP} | {AP50} | {AP75} | {ECE} | {AURC} | {T} |".format(
                name=r["name"],
                mAP=_fmt(coco.get("mAP")),
                AP50=_fmt(coco.get("AP50")),
                AP75=_fmt(coco.get("AP75")),
                ECE=_fmt(cal.get("ece")),
                AURC=_fmt(cal.get("aurc")),
                T=_fmt(r.get("temperature")),
            )
        )
    lines.append("")
    lines.append("## Grouped AP (By Train Frequency Buckets)")
    lines.append("")
    lines.append("| Run | Tail AP | Medium AP | Head AP |")
    lines.append("| --- | ---:| ---:| ---:|")
    for r in rows:
        ga = r.get("grouped_ap", {})
        lines.append(
            "| {name} | {tail} | {med} | {head} |".format(
                name=r["name"],
                tail=_fmt(ga.get("tail")),
                med=_fmt(ga.get("medium")),
                head=_fmt(ga.get("head")),
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- COCO metrics computed via `pycocotools` COCOeval.")
    lines.append("- Calibration metrics (ECE, risk-coverage/AURC) computed from COCO-format predictions using greedy IoU matching at 0.5 IoU.")
    write_text(artifacts / "report.md", "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["data", "train", "eval", "report", "all"])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)

    if args.step == "data":
        step_data(cfg)
    elif args.step == "train":
        step_train(cfg)
    elif args.step == "eval":
        step_eval(cfg)
    elif args.step == "report":
        step_report(cfg)
    elif args.step == "all":
        step_data(cfg)
        step_train(cfg)
        step_eval(cfg)
        step_report(cfg)
    else:
        raise ValueError(args.step)


if __name__ == "__main__":
    main()
