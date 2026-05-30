"""
Detectron2 multi-GPU launch with WSSIS GPU / process-group cleanup on exit.

Use this instead of ``detectron2.engine.launch`` for all WSSIS Mask2Former training.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from detectron2.utils import comm

from modules.wssis.proc_utils import cleanup_distributed, install_worker_signal_handlers

DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port() -> int:
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    install_worker_signal_handlers()
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        assert num_gpus_per_machine <= torch.cuda.device_count()
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        try:
            dist.init_process_group(
                backend="NCCL" if has_gpu else "GLOO",
                init_method=dist_url,
                world_size=world_size,
                rank=global_rank,
                timeout=timeout,
            )
        except Exception as e:
            logging.getLogger(__name__).error("Process group URL: %s", dist_url)
            raise e

        comm.create_local_process_group(num_gpus_per_machine)
        if has_gpu:
            torch.cuda.set_device(local_rank)
        comm.synchronize()
        main_func(*args)
    finally:
        cleanup_distributed()


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
            dist_url = f"tcp://127.0.0.1:{_find_free_port()}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logging.getLogger(__name__).warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

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
            main_func(*args)
        finally:
            cleanup_distributed()
