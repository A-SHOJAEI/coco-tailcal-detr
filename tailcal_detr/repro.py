from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReproConfig:
    seed: int
    deterministic: bool


def seed_everything(cfg: ReproConfig) -> None:
    # Hash seed must be set before Python starts to be fully effective, but setting it here
    # still helps for subprocesses or if users call into this module early.
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    # PyTorch CUDA determinism: some ops (e.g. attention) require cuBLAS workspace config.
    # Setting this does nothing on CPU-only machines.
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    try:
        import torch

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # This can raise on unsupported ops; callers can disable deterministic in config.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        # Allow config parsing / non-torch steps to work without torch installed.
        return


def seed_worker(worker_id: int) -> None:
    # Deterministic DataLoader workers (PyTorch uses base_seed + worker_id).
    try:
        import torch

        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    except Exception:
        return
