"""Subprocess and distributed-training cleanup (GPU memory on interrupt)."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Callable, Mapping, Optional, Sequence


def cleanup_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def cleanup_distributed() -> None:
    try:
        import torch.distributed as dist
    except ImportError:
        cleanup_cuda()
        return
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    cleanup_cuda()


def install_worker_signal_handlers(cleanup: Callable[[], None] = cleanup_distributed) -> None:
    """Register SIGINT/SIGTERM handlers in a training worker process."""

    def _handler(signum, _frame):
        try:
            cleanup()
        finally:
            raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def _terminate_process_tree(proc: subprocess.Popen, *, grace_sec: float = 15.0) -> None:
    if proc.poll() is not None:
        return

    if sys.platform == "win32":
        proc.terminate()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()

    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    if proc.poll() is not None:
        return

    if sys.platform == "win32":
        proc.kill()
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """
    Run ``cmd`` in its own process group/session and tear down the full tree on
    interrupt or failure so spawned GPU workers do not leak memory.
    """
    popen_kw: dict = {"cwd": cwd, "env": env}
    if sys.platform != "win32":
        popen_kw["start_new_session"] = True

    proc = subprocess.Popen(list(cmd), **popen_kw)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("[wssis] Interrupted — terminating training process tree...", file=sys.stderr)
        _terminate_process_tree(proc)
        raise
    finally:
        if proc.poll() is None:
            _terminate_process_tree(proc)


def kill_stale_training_workers(*, dry_run: bool = False) -> list[int]:
    """
    Best-effort cleanup of orphaned Mask2Former / experiment processes.
    Returns PIDs that were signalled.
    """
    patterns = (
        "train_net.py",
        "modules.wssis.run_experiment",
    )
    killed: list[int] = []
    if sys.platform == "win32":
        return killed

    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return killed

    my_pid = os.getpid()
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, args = parts
        if not any(p in args for p in patterns):
            continue
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == my_pid:
            continue
        if dry_run:
            killed.append(pid)
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

    if killed and not dry_run:
        time.sleep(2.0)
        for pid in killed:
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return killed


def _main() -> None:
    parser = argparse.ArgumentParser(description="Kill stale WSSIS GPU training workers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pids = kill_stale_training_workers(dry_run=args.dry_run)
    if not pids:
        print("No stale training workers found.")
        return
    if args.dry_run:
        print("Would terminate:", " ".join(str(p) for p in pids))
    else:
        print("Terminated stale training workers:", " ".join(str(p) for p in pids))


if __name__ == "__main__":
    _main()
