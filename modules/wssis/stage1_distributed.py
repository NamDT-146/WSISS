"""Helpers for Stage-1 GNN distributed training."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def resolve_stage1_num_gpus(explicit: Optional[int] = None) -> int:
    if explicit is not None and explicit > 0:
        return int(explicit)
    env = os.environ.get("WSSIS_NUM_GPUS", "").strip()
    if env.isdigit():
        return max(1, int(env))
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def init_stage1_process_group(local_rank: int, world_size: int) -> torch.device:
    """Used when launched via ``stage1_launch`` (process group already initialized)."""
    if world_size > 1 and not dist.is_initialized():
        raise RuntimeError("Expected torch.distributed to be initialized by stage1_launch")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def stage1_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def stage1_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def stage1_is_main() -> bool:
    return stage1_rank() == 0


def barrier() -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def wrap_stage1_refiner(refiner: torch.nn.Module, local_rank: int) -> torch.nn.Module:
    if stage1_world_size() <= 1:
        return refiner
    return DDP(
        refiner,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=False,
    )


def build_stage1_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    collate_fn,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    sampler = None
    if stage1_world_size() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=stage1_world_size(),
            rank=stage1_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader, sampler


def unwrap_refiner(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def state_dict_for_save(model: torch.nn.Module) -> dict:
    return unwrap_refiner(model).state_dict()
