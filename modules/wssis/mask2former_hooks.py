"""Detectron2 training hooks for WSSIS Stage-2 eval policy."""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

from detectron2.engine.hooks import EvalHook, HookBase
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.utils import comm

from modules.wssis.run_context import EarlyStopping


def resolve_eval_metric(
    results: dict,
    monitor_suffix: str = "segm/AP",
) -> Tuple[Optional[str], Optional[float]]:
    """Pick scalar metric from nested COCO eval results (e.g. segm/AP)."""
    if not results:
        return None, None
    flat = flatten_results_dict(results)
    if monitor_suffix in flat:
        return monitor_suffix, float(flat[monitor_suffix])
    for key, value in flat.items():
        if key.endswith(monitor_suffix) or key.endswith(f"/{monitor_suffix}"):
            try:
                return key, float(value)
            except (TypeError, ValueError):
                continue
    return None, None


class WssisEvalHook(EvalHook):
    """
    Per-epoch eval on val_sample_20pct; final eval (after_train) on full val_all.

    Matches PLAN / RUNBOOK: fast subset during training, full val only at the end.
    """

    def __init__(
        self,
        eval_period: int,
        eval_function_subset,
        eval_function_full,
        eval_after_train: bool = True,
    ):
        super().__init__(eval_period, eval_function_subset, eval_after_train=eval_after_train)
        self._func_full = eval_function_full

    def _store_results(self, results) -> None:
        if not results:
            return
        assert isinstance(results, dict), f"Eval function must return a dict, got {type(results)}"
        flattened = flatten_results_dict(results)
        for k, v in flattened.items():
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(
                    f"[WssisEvalHook] eval_function should return nested float dict; "
                    f"got {k}: {v!r}"
                ) from e
        self.trainer.storage.put_scalars(**flattened, smoothing_hint=False)

    def after_train(self):
        if self._eval_after_train and self.trainer.iter + 1 >= self.trainer.max_iter:
            self._store_results(self._func_full())
            comm.synchronize()
        del self._func
        del self._func_full


class WssisEarlyStoppingHook(HookBase):
    """
    Stop Mask2Former training when subset-val segm/AP stalls (patience in epochs).

    Must be registered immediately after the eval hook. On trigger, sets
    ``trainer.max_iter`` so the loop ends and ``WssisEvalHook.after_train`` runs
    full-val eval.
    """

    def __init__(
        self,
        eval_period: int,
        patience: int = 3,
        monitor_suffix: str = "segm/AP",
        mode: str = "max",
        min_delta: float = 1e-4,
    ):
        self._period = eval_period
        self._monitor_suffix = monitor_suffix
        self._logger = logging.getLogger(__name__)
        self._es = EarlyStopping(
            patience=patience,
            monitor=monitor_suffix,
            mode=mode,
            min_delta=min_delta,
        )

    def after_step(self):
        if self._es.patience <= 0 or self._period <= 0:
            return

        next_iter = self.trainer.iter + 1
        if next_iter % self._period != 0 or next_iter >= self.trainer.max_iter:
            return

        results = getattr(self.trainer, "_last_eval_results", None)
        metric_key, value = resolve_eval_metric(results, self._monitor_suffix)
        if value is None or not math.isfinite(value):
            self._logger.warning(
                "Early stopping: no metric matching %r in eval results", self._monitor_suffix
            )
            return

        improved = self._es.step({self._es.monitor: value})
        if improved:
            self._logger.info(
                "Early stopping: new best %s=%.4f (%s)",
                self._monitor_suffix,
                value,
                metric_key,
            )
        elif self._es.should_stop:
            epoch = next_iter // self._period
            self._logger.info(
                "Early stopping at epoch %d / iter %d (patience=%d, best %s=%.4f)",
                epoch,
                self.trainer.iter,
                self._es.patience,
                self._monitor_suffix,
                self._es.best,
            )
            self.trainer.max_iter = next_iter
