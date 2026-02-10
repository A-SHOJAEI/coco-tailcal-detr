from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def get_dist_info() -> DistInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistInfo(enabled=world_size > 1, rank=rank, world_size=world_size, local_rank=local_rank)


def init_distributed_if_needed() -> DistInfo:
    info = get_dist_info()
    if not info.enabled:
        return info

    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if _has_cuda() else "gloo", init_method="env://")
    return info


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False

