"""Detectron2 training hooks for WSSIS Stage-2 eval policy."""

from __future__ import annotations

import logging
import math
import operator
from typing import List, Optional, Tuple

from detectron2.engine.hooks import EvalHook, HookBase
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.utils import comm

from modules.wssis.run_context import EarlyStopping


def candidate_eval_metric_keys(
    monitor_suffix: str = "segm/AP",
    dataset_names: Optional[List[str]] = None,
) -> List[str]:
    """Keys to try for COCO mask AP in storage / eval results (unwrapped or prefixed)."""
    keys = [monitor_suffix]
    if dataset_names:
        for name in dataset_names:
            prefixed = f"{name}/{monitor_suffix}"
            if prefixed not in keys:
                keys.append(prefixed)
    return keys


def resolve_eval_metric(
    results: dict,
    monitor_suffix: str = "segm/AP",
    dataset_names: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[float]]:
    """Pick scalar metric from nested COCO eval results (e.g. segm/AP)."""
    if not results:
        return None, None
    flat = flatten_results_dict(results)
    for cand in candidate_eval_metric_keys(monitor_suffix, dataset_names):
        if cand in flat:
            try:
                v = float(flat[cand])
                if math.isfinite(v):
                    return cand, v
            except (TypeError, ValueError):
                continue
    for key, value in flat.items():
        if key.endswith(monitor_suffix) or key.endswith(f"/{monitor_suffix}"):
            try:
                v = float(value)
                if math.isfinite(v):
                    return key, v
            except (TypeError, ValueError):
                continue
    return None, None


def resolve_metric_from_trainer(
    trainer,
    monitor_suffix: str = "segm/AP",
    dataset_names: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Resolve eval metric after EvalHook runs.

    ``DefaultTrainer.test()`` usually unwraps single-dataset results to
    ``{segm: {AP: ...}}`` (storage key ``segm/AP``). If unwrapping did not happen,
    keys look like ``wssis_val_1A/segm/AP``.
    """
    cached = getattr(trainer, "_wssis_eval_metric", None)
    if isinstance(cached, dict):
        key = cached.get("key")
        val = cached.get("value")
        if key is not None and val is not None and math.isfinite(float(val)):
            return key, float(val)

    names = dataset_names
    if names is None and getattr(trainer, "cfg", None) is not None:
        names = list(getattr(trainer.cfg.DATASETS, "TEST", ()) or ())

    key, val = resolve_eval_metric(
        getattr(trainer, "_last_eval_results", None) or {},
        monitor_suffix,
        names,
    )
    if val is not None:
        return key, val

    storage = getattr(trainer, "storage", None)
    if storage is None:
        return None, None

    latest = storage.latest()
    ordered_keys = candidate_eval_metric_keys(monitor_suffix, names)
    for k in sorted(latest.keys()):
        if k not in ordered_keys and (
            k == monitor_suffix
            or k.endswith(f"/{monitor_suffix}")
            or k.endswith(monitor_suffix)
        ):
            ordered_keys.append(k)

    for cand in ordered_keys:
        tup = latest.get(cand)
        if tup is None:
            continue
        try:
            v = float(tup[0])
            if math.isfinite(v):
                return cand, v
        except (TypeError, ValueError, IndexError):
            continue
    return None, None


def cache_eval_metric_on_trainer(trainer, monitor_suffix: str = "segm/AP") -> None:
    """Store resolved AP on trainer for early stopping / best checkpoint hooks."""
    key, val = resolve_metric_from_trainer(trainer, monitor_suffix)
    if val is not None:
        trainer._wssis_eval_metric = {"key": key, "value": val}
    else:
        trainer._wssis_eval_metric = None


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
        early_stopping_hook: Optional["WssisEarlyStoppingHook"] = None,
    ):
        super().__init__(eval_period, eval_function_subset, eval_after_train=eval_after_train)
        self._func_full = eval_function_full
        self._early_stopping_hook = early_stopping_hook

    def _do_eval(self):
        results = self._func()
        self.trainer._last_eval_results = results
        if results:
            assert isinstance(results, dict), f"Eval function must return a dict, got {type(results)}"
            flattened = flatten_results_dict(results)
            scalars = {}
            for k, v in flattened.items():
                try:
                    scalars[k] = float(v)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"[WssisEvalHook] eval_function should return float scalars. Got {k!r}: {v!r}"
                    ) from e
            self.trainer.storage.put_scalars(**scalars, smoothing_hint=False)
        comm.synchronize()
        cache_eval_metric_on_trainer(
            self.trainer,
            self._early_stopping_hook._monitor_suffix
            if self._early_stopping_hook is not None
            else "segm/AP",
        )
        if self._early_stopping_hook is not None:
            self._early_stopping_hook.on_eval_complete(self.trainer)

    def after_train(self):
        if self._eval_after_train and self.trainer.iter + 1 >= self.trainer.max_iter:
            results = self._func_full()
            if results:
                assert isinstance(results, dict)
                flattened = flatten_results_dict(results)
                for k, v in flattened.items():
                    self.trainer.storage.put_scalars(**{k: float(v)}, smoothing_hint=False)
            comm.synchronize()
        del self._func
        del self._func_full


class WssisEarlyStoppingHook(HookBase):
    """
    Stop Mask2Former training when subset-val segm/AP stalls (patience in epochs).

    Prefer registering with :class:`WssisEvalHook` (``on_eval_complete`` after eval).
    ``after_step`` remains as a fallback for a plain Detectron2 ``EvalHook``.
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
        self._last_checked_iter = -1

    def on_eval_complete(self, trainer) -> None:
        """Called by WssisEvalHook immediately after subset eval (main process only)."""
        if self._es.patience <= 0 or self._period <= 0:
            return
        if not comm.is_main_process():
            return

        next_iter = trainer.iter + 1
        if next_iter % self._period != 0 or next_iter >= trainer.max_iter:
            return
        if next_iter == self._last_checked_iter:
            return
        self._last_checked_iter = next_iter
        self._step_early_stop(trainer, next_iter)

    def after_step(self):
        """Fallback when paired with a plain EvalHook (not WssisEvalHook)."""
        if self._es.patience <= 0 or self._period <= 0:
            return
        if not comm.is_main_process():
            return

        next_iter = self.trainer.iter + 1
        if next_iter % self._period != 0 or next_iter >= self.trainer.max_iter:
            return
        if next_iter == self._last_checked_iter:
            return
        self._last_checked_iter = next_iter
        self._step_early_stop(self.trainer, next_iter)

    def _step_early_stop(self, trainer, next_iter: int) -> None:
        metric_key, value = resolve_metric_from_trainer(trainer, self._monitor_suffix)
        if value is None:
            sample_keys = []
            if getattr(trainer, "storage", None) is not None:
                sample_keys = [
                    k
                    for k in trainer.storage.latest().keys()
                    if "AP" in k or "segm" in k
                ][:16]
            self._logger.warning(
                "Early stopping: no metric matching %r in eval results or storage "
                "(AP-related keys: %s)",
                self._monitor_suffix,
                sample_keys or "(none)",
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
                trainer.iter,
                self._es.patience,
                self._monitor_suffix,
                self._es.best,
            )
            trainer.max_iter = next_iter


class WssisBestCheckpointer(HookBase):
    """
    Save ``model_best`` when validation mask AP improves.

    Resolves the storage key dynamically (``segm/AP`` vs ``<dataset>/segm/AP``)
    so it stays aligned with :class:`WssisEvalHook` and early stopping.
    """

    def __init__(
        self,
        eval_period: int,
        checkpointer,
        monitor_suffix: str = "segm/AP",
        mode: str = "max",
        file_prefix: str = "model_best",
    ):
        self._period = eval_period
        self._checkpointer = checkpointer
        self._monitor_suffix = monitor_suffix
        self._file_prefix = file_prefix
        self._logger = logging.getLogger(__name__)
        assert mode in ("max", "min")
        self._compare = operator.gt if mode == "max" else operator.lt
        self.best_metric: Optional[float] = None
        self.best_iter: Optional[int] = None

    def _best_checking(self) -> None:
        metric_key, latest_metric = resolve_metric_from_trainer(
            self.trainer, self._monitor_suffix
        )
        if latest_metric is None:
            sample_keys = []
            if getattr(self.trainer, "storage", None) is not None:
                sample_keys = [
                    k
                    for k in self.trainer.storage.latest().keys()
                    if "AP" in k or "segm" in k
                ][:16]
            self._logger.warning(
                "BestCheckpointer: no metric matching %r in eval results or storage "
                "(AP-related keys: %s). Skipping best checkpoint.",
                self._monitor_suffix,
                sample_keys or "(none)",
            )
            return

        metric_iter = self.trainer.iter + 1
        if math.isnan(latest_metric) or math.isinf(latest_metric):
            return

        if self.best_metric is None:
            self.best_metric = latest_metric
            self.best_iter = metric_iter
            self._checkpointer.save(
                self._file_prefix, iteration=metric_iter
            )
            self._logger.info(
                "Saved first model at %s=%.4f (%s) @ iter %d",
                self._monitor_suffix,
                latest_metric,
                metric_key,
                metric_iter,
            )
        elif self._compare(latest_metric, self.best_metric):
            self._checkpointer.save(
                self._file_prefix, iteration=metric_iter
            )
            self._logger.info(
                "Saved best model: %s=%.4f (%s) > %.4f @ iter %d",
                self._monitor_suffix,
                latest_metric,
                metric_key,
                self.best_metric,
                self.best_iter,
            )
            self.best_metric = latest_metric
            self.best_iter = metric_iter
        else:
            self._logger.info(
                "Not saving best: %s=%.4f (%s) vs best %.4f @ iter %d",
                self._monitor_suffix,
                latest_metric,
                metric_key,
                self.best_metric,
                self.best_iter,
            )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self._best_checking()

    def after_train(self):
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._best_checking()
