"""
Unified run directory: logs, metrics, checkpoints, progress, visualizations.

Layout:
  outputs/runs/<run_id>/
    progress.json
    config.json
    logs/{train.log, metrics.jsonl, tensorboard/}
    checkpoints/{last.pt, best.pt, epoch_XXX.pt}
    visualizations/
    eval/
    report/          # bundle for upload (copied/symlinked artifacts)
    experiments/<ID>/  # stage-2 per-experiment outputs
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from modules.wssis.paths import outputs_dir, repo_root


def default_run_id() -> str:
    env = os.environ.get("WSSIS_RUN_ID")
    if env:
        return env.strip()
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(run_id: Optional[str] = None, run_dir: Optional[Path | str] = None) -> Path:
    if run_dir is not None:
        p = Path(run_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    rid = run_id or default_run_id()
    p = outputs_dir() / "runs" / rid
    p.mkdir(parents=True, exist_ok=True)
    return p


class RunContext:
    """Single run bundle for training, logging, resume, and report export."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        run_dir: Optional[Path | str] = None,
        task: str = "stage1_gnn",
        experiment_id: Optional[str] = None,
    ):
        self.run_id = run_id or (Path(run_dir).name if run_dir else default_run_id())
        self.root = resolve_run_dir(self.run_id if run_dir is None else None, run_dir)
        if run_dir is not None:
            self.run_id = self.root.name

        self.task = task
        self.experiment_id = experiment_id
        self.logs_dir = self.root / "logs"
        self.ckpt_dir = self.root / "checkpoints"
        self.viz_dir = self.root / "visualizations"
        self.eval_dir = self.root / "eval"
        self.report_dir = self.root / "report"
        self.exp_dir = (
            self.root / "experiments" / experiment_id
            if experiment_id
            else self.root / "experiments"
        )

        for d in (
            self.logs_dir,
            self.ckpt_dir,
            self.viz_dir,
            self.eval_dir,
            self.report_dir,
            self.exp_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        self.progress_path = self.root / "progress.json"
        self.metrics_jsonl = self.logs_dir / "metrics.jsonl"
        self.config_path = self.root / "config.json"
        self._progress = self._load_progress()
        self._tb_writer = None
        self._wandb_run = None
        self._logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"wssis.{self.run_id}.{self.task}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(self.logs_dir / "train.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def _load_progress(self) -> dict:
        if self.progress_path.exists():
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        return {
            "run_id": self.run_id,
            "root": str(self.root),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "steps": {},
        }

    def save_progress(self) -> None:
        self._progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.progress_path.write_text(
            json.dumps(self._progress, indent=2), encoding="utf-8"
        )

    def update_step(self, step: str, state: Any) -> None:
        self._progress.setdefault("steps", {})[step] = state
        self.save_progress()

    def step_status(self, step: str) -> Any:
        return self._progress.get("steps", {}).get(step)

    def is_step_done(self, step: str) -> bool:
        s = self.step_status(step)
        return s == "done" or (isinstance(s, dict) and s.get("status") == "done")

    def save_config(self, config: dict) -> None:
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        self._copy_to_report(self.config_path, "config.json")

    def log(self, msg: str, *args) -> None:
        self._logger.info(msg, *args)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        row = {"timestamp": time.time(), **metrics}
        if step is not None:
            row["step"] = step
        with open(self.metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if self._tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step or metrics.get("epoch", 0))

        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=str(self.logs_dir / "tensorboard"))
            self.log("TensorBoard: %s", self.logs_dir / "tensorboard")
        except ImportError:
            self.log("TensorBoard not available (install tensorboard)")

    def init_wandb(self, config: Optional[dict] = None, job_type: str = "train") -> None:
        if not os.environ.get("WANDB_PROJECT"):
            return
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=os.environ["WANDB_PROJECT"],
                name=f"{self.run_id}_{self.task}",
                config=config,
                dir=str(self.logs_dir),
                job_type=job_type,
                reinit=True,
            )
            self.log("WandB run started: %s", self._wandb_run.url)
        except Exception as e:
            self.log("WandB init failed: %s", e)

    def save_checkpoint(
        self,
        payload: dict,
        name: str = "last.pt",
        copy_to_legacy: Optional[Path] = None,
    ) -> Path:
        path = self.ckpt_dir / name
        torch = __import__("torch")
        torch.save(payload, path)
        if name == "best.pt":
            self._copy_to_report(path, "best_checkpoint.pt")
        if copy_to_legacy and name in ("best.pt", "last.pt"):
            shutil.copy2(path, copy_to_legacy)
        return path

    def load_checkpoint(self, path: Optional[Path] = None) -> Optional[dict]:
        torch = __import__("torch")
        ckpt_path = path or (self.ckpt_dir / "last.pt")
        if not ckpt_path.exists():
            alt = self.ckpt_dir / "best.pt"
            ckpt_path = alt if alt.exists() else ckpt_path
        if not ckpt_path.exists():
            return None
        self.log("Resuming from %s", ckpt_path)
        try:
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location="cpu")

    def _copy_to_report(self, src: Path, dest_name: str) -> None:
        dest = self.report_dir / dest_name
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)

    def bundle_visualizations(self) -> None:
        if self.viz_dir.exists() and any(self.viz_dir.iterdir()):
            dest = self.report_dir / "visualizations"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(self.viz_dir, dest)

    def bundle_logs(self) -> None:
        for name in ("metrics.jsonl", "train.log", "metrics_history.json"):
            p = self.logs_dir / name
            if p.exists():
                shutil.copy2(p, self.report_dir / name)

    def finalize_report_bundle(self, extra_files: Optional[Dict[str, Path]] = None) -> Path:
        """Collect artifacts for report upload into report/."""
        self.bundle_logs()
        self.bundle_visualizations()
        if self.config_path.exists():
            self._copy_to_report(self.config_path, "config.json")
        if (self.ckpt_dir / "best.pt").exists():
            self._copy_to_report(self.ckpt_dir / "best.pt", "best_checkpoint.pt")
        if extra_files:
            for name, src in extra_files.items():
                if src.exists():
                    shutil.copy2(src, self.report_dir / name)
        readme = self.report_dir / "README.txt"
        readme.write_text(
            f"WSSIS run bundle\nrun_id={self.run_id}\ntask={self.task}\n"
            f"experiment={self.experiment_id}\nroot={self.root}\n",
            encoding="utf-8",
        )
        self.log("Report bundle ready: %s", self.report_dir)
        return self.report_dir

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            import wandb

            wandb.finish()


class EarlyStopping:
    def __init__(
        self,
        patience: int = 3,
        monitor: str = "val_iou",
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, metrics: dict) -> bool:
        if self.patience <= 0:
            return False
        value = metrics.get(self.monitor)
        if value is None:
            return False

        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def gpu_memory_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:
        pass
    return 0.0
