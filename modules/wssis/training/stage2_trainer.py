"""
Stage-2 joint teacher-student training step for Mask2Former (Exp 1C).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from detectron2.data import detection_utils as utils
from detectron2.modeling import ImageList
from detectron2.structures import BitMasks, Boxes, Instances

from modules.wssis.mask2former_datasets import coco_anns_to_masks_for_image
from modules.wssis.pseudo_label_confidence import refined_probs_from_logits
from modules.wssis.training.stage2_augment import apply_geom_transform_to_mask, build_dual_views
from modules.wssis.training.stage2_losses import (
    LossWeightSchedule,
    build_pce_valid_mask,
    partial_bce_loss,
    partial_dice_loss,
    student_feedback_loss,
    symmetric_sam_triplet_loss,
    voting_pseudo_mask,
)
from modules.vig_refinenet.sam_stage1_common import forward_teacher_objects_impl


def _masks_to_instances(
    masks: List[np.ndarray],
    cats: List[int],
    image_size: Tuple[int, int],
    device: torch.device,
) -> Instances:
    h, w = image_size
    if not masks:
        return Instances((h, w))
    boxes = []
    bitmask_list = []
    classes = []
    for mask, cat in zip(masks, cats):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        boxes.append([x0, y0, x1 + 1, y1 + 1])
        bitmask_list.append(mask.astype(np.uint8))
        classes.append(int(cat) if cat else 1)
    if not boxes:
        return Instances((h, w))
    inst = Instances((h, w))
    inst.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32, device=device))
    inst.gt_classes = torch.tensor(classes, dtype=torch.int64, device=device)
    inst.gt_masks = BitMasks(
        torch.stack([torch.from_numpy(m) for m in bitmask_list], dim=0).to(device)
    )
    return inst


def _student_head_outputs(model, batched_inputs: List[dict]) -> List[dict]:
    """Run backbone + sem_seg_head (training tensors, no criterion)."""
    images = [x["image"].to(model.device) for x in batched_inputs]
    images = [(x - model.pixel_mean) / model.pixel_std for x in images]
    images = ImageList.from_tensors(images, model.size_divisibility)
    features = model.backbone(images.tensor)
    outputs = model.sem_seg_head(features)
    return outputs


def _best_query_mask(pred_masks: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """pred_masks [Q,H,W], target [H,W] -> best query [H,W] logits."""
    tgt = target_mask.float()
    if tgt.dim() == 3:
        tgt = tgt[0]
    best_iou = -1.0
    best = pred_masks[0]
    for i in range(pred_masks.shape[0]):
        pm = pred_masks[i].sigmoid()
        t = F.interpolate(
            tgt.unsqueeze(0).unsqueeze(0),
            size=pm.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        inter = (pm * t).sum()
        union = pm.sum() + t.sum() - inter + 1e-6
        iou = inter / union
        if float(iou.item()) > best_iou:
            best_iou = float(iou.item())
            best = pred_masks[i]
    return best


class WssisStage2TrainerMixin:
    """Mixin for Detectron2 Trainer: joint GNN teacher + Mask2Former student."""

    def _wssis_joint_enabled(self) -> bool:
        wssis = getattr(self.cfg, "WSSIS", None)
        return (
            wssis is not None
            and getattr(wssis, "USE_SEMI_WEAK", False)
            and getattr(wssis, "USE_STAGE2_JOINT_LOSS", False)
        )

    def _wssis_loss_schedule(self) -> LossWeightSchedule:
        w = self.cfg.WSSIS
        return LossWeightSchedule(
            warmup_frac=float(getattr(w, "LOSS_WARMUP_FRAC", 0.2)),
            lambda_t_pce=float(getattr(w, "LAMBDA_T_PCE", 1.0)),
            lambda_t_sym=float(getattr(w, "LAMBDA_T_SYM", 0.1)),
            lambda_t_feedback=float(getattr(w, "LAMBDA_T_FEEDBACK", 0.05)),
            lambda_s_sup=float(getattr(w, "LAMBDA_S_SUP", 1.0)),
            lambda_s_unsup=float(getattr(w, "LAMBDA_S_UNSUP", 1.0)),
            lambda_s_semi=float(getattr(w, "LAMBDA_S_SEMI", 0.5)),
        )

    def _wssis_prepare_joint_batch(self, data: List[dict]) -> Tuple[List[dict], Dict[str, torch.Tensor]]:
        teacher = getattr(self, "_wssis_teacher", None)
        if teacher is None:
            return data, {}

        device = self.model.device
        schedule = self._wssis_loss_schedule()
        weights = schedule.weights(self.iter, self.max_iter)
        thresh = float(getattr(self.cfg.WSSIS, "PSEUDO_CONFIDENCE_THRESHOLD", 0.9))
        vote_min = int(getattr(self.cfg.WSSIS, "PSEUDO_VOTE_MIN", 2))

        t_pce_sum = None
        t_sym_sum = None
        n_teacher = 0

        for rec in data:
            is_labeled = rec.get("wssis_is_labeled", True)
            file_name = rec.get("file_name")
            if file_name is None:
                continue

            image_rgb = utils.read_image(file_name, format=getattr(self, "img_format", "RGB"))
            dual = build_dual_views(image_rgb)
            h_s, w_s = dual.geom.out_size, dual.geom.out_size
            rec["image"] = dual.image_strong.to(device)
            rec["height"] = h_s
            rec["width"] = w_s

            if is_labeled and "instances" in rec:
                inst = rec["instances"].to(device)
                if inst.has("gt_masks") and len(inst) > 0:
                    masks_np = inst.gt_masks.tensor.cpu().numpy()
                    warped = [apply_geom_transform_to_mask(m, dual.geom) for m in masks_np]
                    cats = inst.gt_classes.cpu().tolist()
                    rec["instances"] = _masks_to_instances(warped, cats, (h_s, w_s), device)
                continue

            anns = rec.get("wssis_teacher_anns") or []
            if not anns:
                continue
            mask_np_list, ann_ids, cats = coco_anns_to_masks_for_image(
                anns, image_rgb.shape[0], image_rgb.shape[1], max_objects=8
            )
            if not mask_np_list:
                continue
            warped_masks = [apply_geom_transform_to_mask(m, dual.geom) for m in mask_np_list]
            masks_t = torch.stack(
                [torch.from_numpy(m.astype(np.float32)) for m in warped_masks],
                dim=0,
            ).to(device)

            weak_sig = rec.get("wssis_weak_signal_type", "points_only")
            meta = {
                "image_id": rec.get("image_id", 0),
                "ann_ids": ann_ids,
                "category_ids": cats,
                "split": rec.get("split", "train"),
            }
            img_weak = dual.image_weak.to(device)
            _, _, refined, sam3, weak_signal = forward_teacher_objects_impl(
                teacher.sam,
                teacher.gnn if teacher.use_gnn else None,
                img_weak,
                masks_t,
                teacher.pixel_mean,
                teacher.pixel_std,
                teacher.mask_size,
                meta,
                prompt_policy="train_online",
                signal_type=weak_sig,
                use_gnn=teacher.use_gnn,
                use_sam_cache=True,
                threshold_policy=teacher.threshold_policy,
            )

            valid_pce, target_pce = build_pce_valid_mask(weak_signal, weak_sig)
            l_pce = partial_bce_loss(refined, target_pce, valid_pce)
            l_sym = symmetric_sam_triplet_loss(sam3)
            t_pce_sum = l_pce if t_pce_sum is None else t_pce_sum + l_pce
            t_sym_sum = l_sym if t_sym_sum is None else t_sym_sum + l_sym
            n_teacher += 1

            sam_probs = refined_probs_from_logits(sam3)
            pseudo_vote, _ = voting_pseudo_mask(sam_probs, threshold=thresh, vote_min=vote_min)

            pseudo_np = []
            for j in range(pseudo_vote.shape[0]):
                pm = (pseudo_vote[j, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                if pm.shape != (h_s, w_s):
                    pm = apply_geom_transform_to_mask(pm, dual.geom)
                pseudo_np.append(pm)

            rec["instances"] = _masks_to_instances(pseudo_np, cats, (h_s, w_s), device)
            rec["wssis_weak_signal"] = weak_signal
            rec["wssis_weak_signal_type"] = weak_sig
            rec["wssis_refined_logits"] = refined

        teacher_losses: Dict[str, torch.Tensor] = {}
        if n_teacher > 0 and t_pce_sum is not None and t_sym_sum is not None:
            teacher_losses["loss_teacher_pce"] = (t_pce_sum / n_teacher) * weights["lambda_t_pce"]
            teacher_losses["loss_teacher_sym"] = (t_sym_sum / n_teacher) * weights["lambda_t_sym"]
        return data, teacher_losses

    def _wssis_joint_aux_losses(
        self,
        data: List[dict],
        head_outputs: dict,
    ) -> Dict[str, torch.Tensor]:
        schedule = self._wssis_loss_schedule()
        weights = schedule.weights(self.iter, self.max_iter)
        fb_tau = float(getattr(self.cfg.WSSIS, "FEEDBACK_THRESHOLD", 0.95))
        pred_masks = head_outputs.get("pred_masks")
        if pred_masks is None:
            return {}

        semi_sum = None
        fb_sum = None
        n = 0
        for i, rec in enumerate(data):
            if rec.get("wssis_is_labeled", True):
                continue
            weak_signal = rec.get("wssis_weak_signal")
            if weak_signal is None or not rec["instances"].has("gt_masks"):
                continue
            if len(rec["instances"]) == 0:
                continue
            tgt = rec["instances"].gt_masks.tensor[0].float()
            pm = _best_query_mask(pred_masks[i], tgt)
            probs = pm.sigmoid()
            weak_sig = rec.get("wssis_weak_signal_type", "points_only")
            valid_pce, target_pce = build_pce_valid_mask(weak_signal, weak_sig)
            vm = valid_pce.to(pm.device)
            if vm.shape[-2:] != pm.shape[-2:]:
                vm = F.interpolate(vm, size=pm.shape[-2:], mode="nearest")
                target_pce = F.interpolate(target_pce, size=pm.shape[-2:], mode="nearest")
            l_semi = partial_bce_loss(pm.unsqueeze(0).unsqueeze(0), target_pce, vm)
            l_semi_d = partial_dice_loss(pm.unsqueeze(0).unsqueeze(0), target_pce, vm)
            semi_sum = (l_semi + l_semi_d) if semi_sum is None else semi_sum + (l_semi + l_semi_d)

            refined = rec.get("wssis_refined_logits")
            if refined is not None and weights["lambda_t_feedback"] > 0:
                fb = student_feedback_loss(
                    refined[0:1],
                    probs.unsqueeze(0).unsqueeze(0),
                    tau=fb_tau,
                )
                fb_sum = fb if fb_sum is None else fb_sum + fb
            n += 1

        out: Dict[str, torch.Tensor] = {}
        if n > 0 and semi_sum is not None:
            out["loss_semi"] = (semi_sum / n) * weights["lambda_s_semi"]
        if n > 0 and fb_sum is not None:
            out["loss_teacher_feedback"] = (fb_sum / n) * weights["lambda_t_feedback"]
        return out
