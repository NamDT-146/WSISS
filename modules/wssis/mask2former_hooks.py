"""Detectron2 training hooks for WSSIS Stage-2 eval policy."""

from __future__ import annotations

from detectron2.engine.hooks import EvalHook
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.utils import comm


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
