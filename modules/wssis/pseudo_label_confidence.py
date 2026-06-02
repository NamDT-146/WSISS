"""
Pseudo-label confidence gating for GNN-refined masks (Stage 1 logging + Stage 2 student).

Threshold modes (FixMatch / AdaMatch / FreeMatch ideas, adapted for per-pixel mask logits):
  - ``fixed``: global cutoff (default 0.9, FixMatch-style ``p_cutoff``)
  - ``adamatch``: batch-relative cutoff = mean(max prob on reference batch) * ``p_cutoff``
  - ``freematch``: EMA of high-confidence statistics (SAT-style adaptive cutoff)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

ThresholdMode = Literal["fixed", "adamatch", "freematch"]

DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_THRESHOLD_MODE: ThresholdMode = "fixed"
PSEUDO_VOTE_MIN = 2


@dataclass
class FreeMatchThresholdConfig:
    ema_momentum: float = 0.999
    use_quantile: bool = True
    quantile: float = 0.8
    clip_max: float = 0.95


@dataclass
class PseudoLabelThresholdConfig:
    threshold_mode: ThresholdMode = DEFAULT_THRESHOLD_MODE
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD
    freematch: FreeMatchThresholdConfig = field(default_factory=FreeMatchThresholdConfig)

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "PseudoLabelThresholdConfig":
        if not raw:
            return cls()
        mode = str(raw.get("threshold_mode", DEFAULT_THRESHOLD_MODE)).lower()
        if mode not in ("fixed", "adamatch", "freematch"):
            mode = DEFAULT_THRESHOLD_MODE
        fm = raw.get("freematch") or {}
        return cls(
            threshold_mode=mode,  # type: ignore[arg-type]
            confidence_threshold=float(
                raw.get("confidence_threshold", DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD)
            ),
            freematch=FreeMatchThresholdConfig(
                ema_momentum=float(fm.get("ema_momentum", 0.999)),
                use_quantile=bool(fm.get("use_quantile", True)),
                quantile=float(fm.get("quantile", 0.8)),
                clip_max=float(fm.get("clip_max", 0.95)),
            ),
        )


def parse_pseudo_label_config(config: Optional[dict] = None) -> PseudoLabelThresholdConfig:
    if not config:
        return PseudoLabelThresholdConfig()
    pl = config.get("pseudo_label")
    if isinstance(pl, dict):
        return PseudoLabelThresholdConfig.from_dict(pl)
    if "pseudo_confidence_threshold" in config:
        return PseudoLabelThresholdConfig(
            confidence_threshold=float(config["pseudo_confidence_threshold"])
        )
    return PseudoLabelThresholdConfig()


def resolve_pseudo_confidence_threshold(
    config: Optional[dict] = None,
    *,
    fallback: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
) -> float:
    """Fixed cutoff only (for YAML scalar / backward compat)."""
    return parse_pseudo_label_config(config).confidence_threshold or fallback


def resolve_pseudo_threshold_mode(config: Optional[dict] = None) -> ThresholdMode:
    return parse_pseudo_label_config(config).threshold_mode


def max_prob_over_heads(probs: torch.Tensor) -> torch.Tensor:
    """Per-pixel max probability across GNN heads. Shape [B, H, W]."""
    if probs.dim() == 3:
        probs = probs.unsqueeze(0)
    return probs.max(dim=1).values


class PseudoThresholdPolicy:
    """Computes an effective scalar threshold for one pseudo-label forward."""

    def __init__(self, cfg: Optional[PseudoLabelThresholdConfig] = None):
        self.cfg = cfg or PseudoLabelThresholdConfig()
        self._time_p: float = self.cfg.confidence_threshold

    @classmethod
    def from_run_config(cls, config: Optional[dict] = None) -> "PseudoThresholdPolicy":
        return cls(parse_pseudo_label_config(config))

    @property
    def mode(self) -> ThresholdMode:
        return self.cfg.threshold_mode

    @property
    def p_cutoff(self) -> float:
        return self.cfg.confidence_threshold

    def state_dict(self) -> dict[str, Any]:
        return {"time_p": self._time_p, "cfg": self.cfg}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._time_p = float(state.get("time_p", self._time_p))

    @torch.no_grad()
    def effective_threshold(
        self,
        refined_masks_logits: torch.Tensor,
        *,
        reference_logits: Optional[torch.Tensor] = None,
        update: bool = True,
    ) -> float:
        probs = refined_probs_from_logits(refined_masks_logits)
        ref_probs = (
            refined_probs_from_logits(reference_logits)
            if reference_logits is not None
            else probs
        )

        if self.cfg.threshold_mode == "fixed":
            return self.cfg.confidence_threshold

        if self.cfg.threshold_mode == "adamatch":
            ref_max = max_prob_over_heads(ref_probs)
            thresh = float(ref_max.mean().item()) * self.cfg.confidence_threshold
            return min(max(thresh, 0.0), 1.0)

        # freematch — SAT-style EMA on batch max-over-heads map
        fm = self.cfg.freematch
        max_probs = max_prob_over_heads(probs)
        flat = max_probs.reshape(-1)
        if flat.numel() == 0:
            return self._time_p
        if fm.use_quantile:
            stat = torch.quantile(flat, fm.quantile)
        else:
            stat = flat.mean()
        stat_f = float(stat.item())
        if update:
            self._time_p = fm.ema_momentum * self._time_p + (1.0 - fm.ema_momentum) * stat_f
            if fm.clip_max is not None:
                self._time_p = min(self._time_p, fm.clip_max)
        return self._time_p

    @torch.no_grad()
    def binary_masks_from_logits(
        self,
        refined_masks_logits: torch.Tensor,
        *,
        reference_logits: Optional[torch.Tensor] = None,
        update: bool = True,
    ) -> torch.Tensor:
        thresh = self.effective_threshold(
            refined_masks_logits,
            reference_logits=reference_logits,
            update=update,
        )
        probs = refined_probs_from_logits(refined_masks_logits)
        return binary_masks_from_probs(probs, thresh)

    @torch.no_grad()
    def generate_pseudo_label(
        self,
        refined_masks_logits: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        *,
        reference_logits: Optional[torch.Tensor] = None,
        update: bool = True,
        vote_min: int = PSEUDO_VOTE_MIN,
    ) -> torch.Tensor:
        if refined_masks_logits.dim() == 3:
            refined_masks_logits = refined_masks_logits.unsqueeze(0)
        binary = self.binary_masks_from_logits(
            refined_masks_logits,
            reference_logits=reference_logits,
            update=update,
        )
        votes = binary.sum(dim=1, keepdim=True)
        agreed = (votes >= vote_min).float()
        if target_size is not None and agreed.shape[-2:] != target_size:
            agreed = F.interpolate(agreed, size=target_size, mode="nearest")
        return agreed


def build_threshold_policy(
    config: Optional[dict] = None,
    *,
    mode: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
) -> PseudoThresholdPolicy:
    """Build policy from run config with optional CLI overrides."""
    cfg = parse_pseudo_label_config(config)
    if mode is not None:
        cfg.threshold_mode = mode.lower()  # type: ignore[assignment]
    if confidence_threshold is not None:
        cfg.confidence_threshold = float(confidence_threshold)
    return PseudoThresholdPolicy(cfg)


def refined_probs_from_logits(refined_masks_logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid probabilities; accepts logits or probabilities in [0, 1]."""
    if refined_masks_logits.numel() == 0:
        return refined_masks_logits
    if refined_masks_logits.min() >= 0.0 and refined_masks_logits.max() <= 1.0:
        return refined_masks_logits.float()
    return torch.sigmoid(refined_masks_logits)


def binary_masks_from_probs(
    probs: torch.Tensor,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
) -> torch.Tensor:
    return (probs > confidence_threshold).float()


def binary_masks_from_logits(
    refined_masks_logits: torch.Tensor,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
    *,
    threshold_policy: Optional[PseudoThresholdPolicy] = None,
    reference_logits: Optional[torch.Tensor] = None,
    update: bool = True,
) -> torch.Tensor:
    if threshold_policy is not None:
        return threshold_policy.binary_masks_from_logits(
            refined_masks_logits,
            reference_logits=reference_logits,
            update=update,
        )
    return binary_masks_from_probs(
        refined_probs_from_logits(refined_masks_logits),
        confidence_threshold,
    )


def over_threshold_ratio(
    refined_masks_logits: torch.Tensor,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
    *,
    threshold_policy: Optional[PseudoThresholdPolicy] = None,
    reference_logits: Optional[torch.Tensor] = None,
    update: bool = False,
) -> float:
    if refined_masks_logits.numel() == 0:
        return 0.0
    probs = refined_probs_from_logits(refined_masks_logits)
    if threshold_policy is not None:
        thresh = threshold_policy.effective_threshold(
            refined_masks_logits,
            reference_logits=reference_logits,
            update=update,
        )
    else:
        thresh = confidence_threshold
    return float((probs > thresh).float().mean().item())


def agreement_rate(
    refined_masks_logits: torch.Tensor,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
    vote_min: int = PSEUDO_VOTE_MIN,
    *,
    threshold_policy: Optional[PseudoThresholdPolicy] = None,
    reference_logits: Optional[torch.Tensor] = None,
    update: bool = False,
) -> float:
    if refined_masks_logits.numel() == 0:
        return 0.0
    binary = binary_masks_from_logits(
        refined_masks_logits,
        confidence_threshold,
        threshold_policy=threshold_policy,
        reference_logits=reference_logits,
        update=update,
    )
    if binary.dim() == 3:
        binary = binary.unsqueeze(0)
    votes = binary.sum(dim=1)
    return float((votes >= vote_min).float().mean().item())


def generate_pseudo_label_from_logits(
    refined_masks_logits: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
    *,
    confidence_threshold: float = DEFAULT_PSEUDO_CONFIDENCE_THRESHOLD,
    vote_min: int = PSEUDO_VOTE_MIN,
    threshold_policy: Optional[PseudoThresholdPolicy] = None,
    reference_logits: Optional[torch.Tensor] = None,
    update: bool = True,
) -> torch.Tensor:
    if threshold_policy is not None:
        return threshold_policy.generate_pseudo_label(
            refined_masks_logits,
            target_size,
            reference_logits=reference_logits,
            update=update,
            vote_min=vote_min,
        )
    if refined_masks_logits.dim() == 3:
        refined_masks_logits = refined_masks_logits.unsqueeze(0)
    effective_vote_min = vote_min
    if refined_masks_logits.shape[1] == 1:
        effective_vote_min = 1
    binary = binary_masks_from_logits(refined_masks_logits, confidence_threshold)
    votes = binary.sum(dim=1, keepdim=True)
    agreed = (votes >= effective_vote_min).float()
    if target_size is not None and agreed.shape[-2:] != target_size:
        agreed = F.interpolate(agreed, size=target_size, mode="nearest")
    return agreed
