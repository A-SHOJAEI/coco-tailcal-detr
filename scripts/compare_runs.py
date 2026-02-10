from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tailcal_detr.io_utils import read_json, write_json, write_text


def compare(run_dirs: list[Path], out_json: Path, out_md: Path) -> None:
    rows: list[dict[str, Any]] = []
    for rd in run_dirs:
        coco = read_json(rd / "metrics_coco.json")
        cal = read_json(rd / "calibration.json")
        temp = read_json(rd / "temperature.json")["temperature"] if (rd / "temperature.json").exists() else None
        rows.append(
            {
                "run": rd.name,
                "dir": str(rd),
                "coco": coco.get("coco_stats", {}),
                "calibration": {"ece": cal.get("ece"), "aurc": cal.get("aurc"), "n": cal.get("n")},
                "temperature": temp,
            }
        )
    write_json(out_json, {"runs": rows})

    lines = []
    lines.append("| Run | mAP | AP50 | AP75 | ECE | AURC | T |")
    lines.append("| --- | ---:| ---:| ---:| ---:| ---:| ---:|")
    for r in rows:
        c = r["coco"]
        ca = r["calibration"]
        lines.append(
            f"| {r['run']} | {c.get('mAP')} | {c.get('AP50')} | {c.get('AP75')} | {ca.get('ece')} | {ca.get('aurc')} | {r.get('temperature')} |"
        )
    write_text(out_md, "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories (each must contain metrics_coco.json)")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args(argv)
    compare([Path(x) for x in args.runs], Path(args.out_json), Path(args.out_md))


if __name__ == "__main__":
    main()

