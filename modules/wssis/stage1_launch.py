"""
Multi-GPU launch for Stage-1 GNN (PyTorch DDP, no Detectron2 dependency).
"""

from __future__ import annotations

import logging
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modules.wssis.proc_utils import cleanup_distributed, install_worker_signal_handlers

DEFAULT_TIMEOUT = timedelta(minutes=30)
logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _distributed_worker(
    local_rank: int,
    main_func,
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int,
    dist_url: str,
    args: tuple,
    timeout=DEFAULT_TIMEOUT,
) -> None:
    install_worker_signal_handlers()
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        assert num_gpus_per_machine <= torch.cuda.device_count()
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL" if has_gpu else "GLOO",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
        if has_gpu:
            torch.cuda.set_device(local_rank)
        dist.barrier()
        main_func(local_rank, world_size, *args)
    finally:
        cleanup_distributed()


def launch(
    main_func,
    num_gpus_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str | None = None,
    args: tuple = (),
    timeout=DEFAULT_TIMEOUT,
) -> None:
    """Spawn one process per GPU and call ``main_func(local_rank, world_size, *args)``."""
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url is None or dist_url == "auto":
            assert num_machines == 1, "dist_url=auto supports single machine only."
            dist_url = f"tcp://127.0.0.1:{_find_free_port()}"
        mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
                timeout,
            ),
            daemon=False,
        )
    else:
        install_worker_signal_handlers()
        try:
            main_func(0, 1, *args)
        finally:
            cleanup_distributed()
