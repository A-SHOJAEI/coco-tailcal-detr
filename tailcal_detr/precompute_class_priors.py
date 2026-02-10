from __future__ import annotations

import argparse
from pathlib import Path

from tailcal_detr.config import ensure_dir
from tailcal_detr.priors import compute_class_priors, save_class_priors


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO instances_train*.json")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args(argv)

    ann = Path(args.ann)
    out = Path(args.out)
    ensure_dir(out.parent)
    priors = compute_class_priors(ann)
    save_class_priors(priors, out)


if __name__ == "__main__":
    main()

