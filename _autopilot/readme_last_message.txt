Overwrote `README.md` to be specific to this repo’s actual pipeline, configs, and outputs, grounded in `artifacts/results.json`, `artifacts/report.md`, and the implemented code.

It now includes:
- Problem statement
- Dataset provenance (smoke synthetic generator + optional COCO 2017 downloader with SHA256 manifest)
- Methodology (baseline, priors/bucketing, TailCal knobs, temperature fitting, eval definitions)
- Baselines/ablations exactly as configured in `configs/smoke.yaml` / `configs/full.yaml`
- Exact result tables copied from `artifacts/report.md` (with timestamp and config)
- Repro commands (`make all`, stepwise targets) and output file locations
- Limitations (including the `torchvision==0.20.1` TinyDETR fallback and calibration sample-size dependence)
- Concrete next research steps