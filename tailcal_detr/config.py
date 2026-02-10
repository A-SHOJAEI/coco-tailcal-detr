from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got: {type(data)}")
    return data


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class CliArgs:
    config: Path


def parse_config_arg(argv: list[str] | None = None) -> CliArgs:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--config", required=True, type=Path)
    ns = ap.parse_args(argv)
    return CliArgs(config=ns.config)


def deep_get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(d: dict[str, Any], key: str, value: Any) -> None:
    cur: Any = d
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

