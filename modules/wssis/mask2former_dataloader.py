"""Semi-weak paired dataset for Detectron2 (50/50 labeled / weak)."""

from __future__ import annotations

import copy
from typing import List

from torch.utils.data import Dataset


class WssisSemiWeakDataset(Dataset):
    """
    Alternates labeled (full GT) and weak (no student GT; teacher annos attached).

    Each index maps to one image. Weak samples carry ``wssis_teacher_anns`` for
    oracle-jitter prompt generation only.
    """

    def __init__(self, labeled_records: List[dict], weak_records: List[dict]):
        self.labeled = labeled_records
        self.weak = weak_records
        if not self.labeled:
            raise ValueError("WssisSemiWeakDataset requires labeled records")
        if not self.weak:
            raise ValueError("WssisSemiWeakDataset requires weak records")
        self._pairs = min(len(self.labeled), len(self.weak))

    def __len__(self) -> int:
        return 2 * self._pairs

    def __getitem__(self, idx: int) -> dict:
        pair_idx = idx // 2
        if idx % 2 == 0:
            rec = copy.deepcopy(self.labeled[pair_idx % len(self.labeled)])
            rec["wssis_is_labeled"] = True
            rec["wssis_teacher_anns"] = rec.get("annotations", [])
            return rec
        rec = copy.deepcopy(self.weak[pair_idx % len(self.weak)])
        rec["wssis_is_labeled"] = False
        rec["wssis_teacher_anns"] = copy.deepcopy(rec.get("annotations", []))
        rec["annotations"] = []
        return rec
