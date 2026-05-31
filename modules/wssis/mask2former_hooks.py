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


def _dataset_names_for_trainer(trainer, dataset_names: Optional[List[str]]) -> Optional[List[str]]:
    if dataset_names is not None:
        return dataset_names
    if getattr(trainer, "cfg", None) is not None:
        return list(getattr(trainer.cfg.DATASETS, "TEST", ()) or ())
    return None


def _resolve_metric_from_storage(
    trainer,
    monitor_suffix: str,
    dataset_names: Optional[List[str]],
) -> Tuple[Optional[str], Optional[float]]:
    storage = getattr(trainer, "storage", None)
    if storage is None:
        return None, None

    names = _dataset_names_for_trainer(trainer, dataset_names)
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


def resolve_fresh_eval_metric(
    trainer,
    monitor_suffix: str = "segm/AP",
    dataset_names: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Resolve metric from the eval that just finished.

    Uses ``_last_eval_results`` and event storage only — never a stale
    ``_wssis_eval_metric`` cache (see ``cache_eval_metric_on_trainer``).
    """
    names = _dataset_names_for_trainer(trainer, dataset_names)
    key, val = resolve_eval_metric(
        getattr(trainer, "_last_eval_results", None) or {},
        monitor_suffix,
        names,
    )
    if val is not None:
        return key, val
    return _resolve_metric_from_storage(trainer, monitor_suffix, names)


def resolve_metric_from_trainer(
    trainer,
    monitor_suffix: str = "segm/AP",
    dataset_names: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Resolve eval metric after EvalHook runs.

    Prefers fresh eval results; falls back to cached metric only when fresh
    sources are unavailable (e.g. outside ``WssisEvalHook._do_eval``).
    """
    key, val = resolve_fresh_eval_metric(trainer, monitor_suffix, dataset_names)
    if val is not None:
        return key, val

    cached = getattr(trainer, "_wssis_eval_metric", None)
    if isinstance(cached, dict):
        ckey = cached.get("key")
        cval = cached.get("value")
        if ckey is not None and cval is not None and math.isfinite(float(cval)):
            return ckey, float(cval)
    return None, None


def cache_eval_metric_on_trainer(trainer, monitor_suffix: str = "segm/AP") -> None:
    """Store resolved AP on trainer for early stopping / best checkpoint hooks."""
    key, val = resolve_fresh_eval_metric(trainer, monitor_suffix)
    if val is not None:
        trainer._wssis_eval_metric = {"key": key, "value": val}
    else:
        trainer._wssis_eval_metric = None


def _monitor_suffix_from_hooks(
    early_stopping_hook: Optional["WssisEarlyStoppingHook"],
    best_checkpointer_hook: Optional["WssisBestCheckpointer"],
) -> str:
    if early_stopping_hook is not None:
        return early_stopping_hook._monitor_suffix
    if best_checkpointer_hook is not None:
        return best_checkpointer_hook._monitor_suffix
    return "segm/AP"


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
        best_checkpointer_hook: Optional["WssisBestCheckpointer"] = None,
    ):
        super().__init__(eval_period, eval_function_subset, eval_after_train=eval_after_train)
        self._func_full = eval_function_full
        self._early_stopping_hook = early_stopping_hook
        self._best_checkpointer_hook = best_checkpointer_hook

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
        monitor_suffix = _monitor_suffix_from_hooks(
            self._early_stopping_hook,
            self._best_checkpointer_hook,
        )
        self.trainer._wssis_best_saved_this_eval = False
        self.trainer._wssis_best_ckpt_note = ""
        cache_eval_metric_on_trainer(self.trainer, monitor_suffix)
        if self._best_checkpointer_hook is not None:
            self._best_checkpointer_hook.on_eval_complete(self.trainer)
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
        patience: int = 5,
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
        metric_key, value = resolve_fresh_eval_metric(trainer, self._monitor_suffix)
        if value is None:
            sample_keys = []
            if getattr(trainer, "storage", None) is not None:
                sample_keys = [
                    k
                    for k in trainer.storage.latest().keys()
                    if "AP" in k or "segm" in k
                ][:16]
            self._logger.warning(
                "[EarlyStop] iter %d: no metric matching %r in eval results or storage "
                "(AP-related keys: %s)",
                next_iter,
                self._monitor_suffix,
                sample_keys or "(none)",
            )
            return

        epoch = next_iter // self._period
        prev_best = self._es.best
        improved = self._es.step({self._es.monitor: value})

        ckpt_note = getattr(trainer, "_wssis_best_ckpt_note", "")
        if ckpt_note:
            self._logger.info("[EarlyStop] iter %d (epoch %d): %s", next_iter, epoch, ckpt_note)

        if improved:
            if prev_best is None:
                self._logger.info(
                    "[EarlyStop] iter %d (epoch %d): new best %s=%.4f (%s) — "
                    "patience reset (0/%d)",
                    next_iter,
                    epoch,
                    self._monitor_suffix,
                    value,
                    metric_key,
                    self._es.patience,
                )
            else:
                self._logger.info(
                    "[EarlyStop] iter %d (epoch %d): new best %s=%.4f (%s), "
                    "was %.4f — patience reset (0/%d)",
                    next_iter,
                    epoch,
                    self._monitor_suffix,
                    value,
                    metric_key,
                    prev_best,
                    self._es.patience,
                )
        elif self._es.should_stop:
            self._logger.info(
                "[EarlyStop] iter %d (epoch %d): STOP — no improvement for %d eval(s); "
                "last %s=%.4f (%s), best=%.4f",
                next_iter,
                epoch,
                self._es.patience,
                self._monitor_suffix,
                value,
                metric_key,
                self._es.best,
            )
            trainer.max_iter = min(trainer.max_iter, next_iter)
            trainer._wssis_stop_training = True
        else:
            remaining = self._es.patience - self._es.counter
            self._logger.info(
                "[EarlyStop] iter %d (epoch %d): no improvement — %s=%.4f (%s), "
                "best=%.4f, patience %d/%d (%d eval(s) until stop)",
                next_iter,
                epoch,
                self._monitor_suffix,
                value,
                metric_key,
                self._es.best,
                self._es.counter,
                self._es.patience,
                remaining,
            )


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
        self._last_checked_iter = -1

    def on_eval_complete(self, trainer) -> None:
        """Called by WssisEvalHook immediately after subset eval (main process only)."""
        if not comm.is_main_process():
            return

        next_iter = trainer.iter + 1
        if next_iter == self._last_checked_iter:
            return
        self._last_checked_iter = next_iter
        self._update_best(trainer)

    def _update_best(self, trainer) -> None:
        metric_key, latest_metric = resolve_fresh_eval_metric(
            trainer, self._monitor_suffix
        )
        if latest_metric is None:
            sample_keys = []
            if getattr(trainer, "storage", None) is not None:
                sample_keys = [
                    k
                    for k in trainer.storage.latest().keys()
                    if "AP" in k or "segm" in k
                ][:16]
            trainer._wssis_best_ckpt_note = (
                f"model_best skipped (no {self._monitor_suffix} in eval results)"
            )
            self._logger.warning(
                "[BestCkpt] iter %d: no metric matching %r in eval results or storage "
                "(AP-related keys: %s). Skipping best checkpoint.",
                trainer.iter + 1,
                self._monitor_suffix,
                sample_keys or "(none)",
            )
            return

        metric_iter = trainer.iter + 1
        if math.isnan(latest_metric) or math.isinf(latest_metric):
            trainer._wssis_best_ckpt_note = (
                f"model_best skipped (non-finite {self._monitor_suffix}={latest_metric})"
            )
            return

        if self.best_metric is None:
            self.best_metric = latest_metric
            self.best_iter = metric_iter
            self._checkpointer.save(
                self._file_prefix, iteration=metric_iter
            )
            trainer._wssis_best_saved_this_eval = True
            trainer._wssis_best_ckpt_note = (
                f"saved model_best ({self._monitor_suffix}={latest_metric:.4f}, first eval)"
            )
            self._logger.info(
                "[BestCkpt] iter %d: saved %s (first eval, %s=%.4f, key=%s)",
                metric_iter,
                self._file_prefix,
                self._monitor_suffix,
                latest_metric,
                metric_key,
            )
        elif self._compare(latest_metric, self.best_metric):
            prev_best = self.best_metric
            prev_iter = self.best_iter
            self._checkpointer.save(
                self._file_prefix, iteration=metric_iter
            )
            self.best_metric = latest_metric
            self.best_iter = metric_iter
            trainer._wssis_best_saved_this_eval = True
            trainer._wssis_best_ckpt_note = (
                f"saved model_best ({self._monitor_suffix}={latest_metric:.4f} "
                f"> {prev_best:.4f} @ iter {prev_iter})"
            )
            self._logger.info(
                "[BestCkpt] iter %d: saved %s (%s=%.4f > %.4f @ iter %d, key=%s)",
                metric_iter,
                self._file_prefix,
                self._monitor_suffix,
                latest_metric,
                prev_best,
                prev_iter,
                metric_key,
            )
        else:
            trainer._wssis_best_ckpt_note = (
                f"model_best not saved ({self._monitor_suffix}={latest_metric:.4f} "
                f"<= best {self.best_metric:.4f} @ iter {self.best_iter})"
            )
            self._logger.info(
                "[BestCkpt] iter %d: not saved (%s=%.4f, key=%s) — "
                "current best %.4f @ iter %d",
                metric_iter,
                self._monitor_suffix,
                latest_metric,
                metric_key,
                self.best_metric,
                self.best_iter,
            )

    def after_step(self):
        """Fallback when paired with a plain Detectron2 EvalHook."""
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self.on_eval_complete(self.trainer)


class WssisTrainProgressHook(HookBase):
    """tqdm progress bar on the main process (WSSIS custom train loop)."""

    def __init__(self):
        self._pbar = None

    def before_train(self):
        if not comm.is_main_process():
            return
        try:
            from tqdm import tqdm
        except ImportError:
            return

        self._pbar = tqdm(
            total=self.trainer.max_iter,
            initial=self.trainer.start_iter,
            desc="Mask2Former",
            unit="iter",
            dynamic_ncols=True,
        )

    def after_step(self):
        if self._pbar is None:
            return

        storage = getattr(self.trainer, "storage", None)
        postfix = {}
        if storage is not None:
            for key in ("total_loss", "loss_distill", "lr"):
                try:
                    val = storage.history(key).latest()
                    if val is not None:
                        postfix[key] = f"{float(val):.4g}"
                except KeyError:
                    pass
        if postfix:
            self._pbar.set_postfix(postfix, refresh=False)

        self._pbar.update(1)

    def after_train(self):
        if self._pbar is not None:
            if self._pbar.n < self.trainer.iter:
                self._pbar.update(self.trainer.iter - self._pbar.n)
            self._pbar.close()
            self._pbar = None
