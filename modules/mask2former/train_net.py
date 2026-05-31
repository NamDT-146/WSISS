# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.(autocast|GradScaler).*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*timm\.models\.layers.*",
)

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except Exception:
    pass

import copy
import itertools
import logging
import os
from pathlib import Path

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

try:
    from modules.wssis.mask2former_config import add_wssis_config
    from modules.wssis.mask2former_datasets import ensure_wssis_datasets_in_cfg
except ImportError:
    add_wssis_config = None
    ensure_wssis_datasets_in_cfg = None


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            use_semi = getattr(cfg.WSSIS, "USE_SEMI_WEAK", False) if getattr(cfg, "WSSIS", None) else False
            if use_semi:
                import torch

                from modules.wssis.mask2former_mapper import WssisSemiWeakMapper
                from modules.wssis.mask2former_teacher import WssisTeacherStack
                from modules.wssis.paths import gnn_checkpoint

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                gnn_path = getattr(cfg.WSSIS, "GNN_CHECKPOINT", "") or None
                teacher = WssisTeacherStack(
                    device,
                    gnn_ckpt_path=Path(gnn_path) if gnn_path else gnn_checkpoint(),
                    use_gnn=getattr(cfg.WSSIS, "USE_GNN", True),
                    freeze_gnn=getattr(cfg.WSSIS, "FREEZE_GNN", False),
                )
                cfg.defrost()
                cfg.DATALOADER.NUM_WORKERS = 0
                cfg.freeze()
                mapper = WssisSemiWeakMapper(cfg, True, teacher=teacher)
            else:
                mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        # Feature projector (semi-weak distillation)
        if hasattr(model, "wssis_projector"):
            params.append(
                {
                    "params": list(model.wssis_projector.parameters()),
                    "lr": cfg.SOLVER.BASE_LR,
                    "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
                }
            )
            memo.update(p for p in model.wssis_projector.parameters())

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_model(cls, cfg):
        from detectron2.modeling import build_model as d2_build_model

        model = d2_build_model(cfg)
        if getattr(cfg, "WSSIS", None) and getattr(cfg.WSSIS, "USE_DISTILL", False):
            from modules.wssis.wssis_maskformer_distill import attach_wssis_distillation

            model = attach_wssis_distillation(model, cfg)
        return model

    def run_step(self):
        """Log WSSIS component losses when semi-weak training is enabled."""
        super().run_step()
        if not getattr(self.cfg, "WSSIS", None):
            return
        if not getattr(self.cfg.WSSIS, "USE_SEMI_WEAK", False):
            return
        storage = self.storage
        if storage is None:
            return

        def _latest(name: str):
            buf = storage.histories().get(name)
            if buf is None or len(buf) == 0:
                return None
            return buf.latest()

        loss_ce = _latest("loss_ce")
        loss_mask = _latest("loss_mask")
        loss_dice = _latest("loss_dice")
        sup = 0.0
        for v in (loss_ce, loss_mask, loss_dice):
            if v is not None:
                sup += float(v)
        storage.put_scalar("wssis/sup_loss", sup, smoothing_hint=False)
        ratio = float(getattr(self.cfg.WSSIS, "LABELED_BATCH_RATIO", 0.5))
        storage.put_scalar(
            "wssis/semi_loss",
            sup * max(0.0, 1.0 - ratio) if sup else 0.0,
            smoothing_hint=False,
        )
        loss_distill = _latest("loss_distill")
        storage.put_scalar(
            "wssis/distill_loss",
            float(loss_distill) if loss_distill is not None else 0.0,
            smoothing_hint=False,
        )

    def build_hooks(self):
        hooks_list = super().build_hooks()
        cfg = self.cfg
        if add_wssis_config is None or not getattr(cfg, "WSSIS", None):
            return hooks_list
        if not cfg.WSSIS.EXPERIMENT_ID:
            return hooks_list

        from detectron2.engine import hooks as d2_hooks

        from modules.wssis.mask2former_datasets import wssis_val_full_name
        from modules.wssis.mask2former_hooks import WssisEarlyStoppingHook, WssisEvalHook

        eval_period = cfg.TEST.EVAL_PERIOD
        patience = int(getattr(cfg.WSSIS, "EARLY_STOPPING_PATIENCE", 0))
        monitor_suffix = getattr(cfg.WSSIS, "EARLY_STOPPING_MONITOR", "segm/AP")
        use_full_val_final = getattr(cfg.WSSIS, "USE_FULL_VAL_FINAL", False)
        val_name = cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else ""
        storage_metric = f"{val_name}/{monitor_suffix}" if val_name else monitor_suffix

        def test_subset():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        def test_full():
            cfg_full = self.cfg.clone()
            cfg_full.defrost()
            cfg_full.DATASETS.TEST = (wssis_val_full_name(cfg.WSSIS.EXPERIMENT_ID),)
            cfg_full.freeze()
            results = self.test(cfg_full, self.model)
            self._last_eval_results = results
            logging.getLogger("mask2former").info(
                "WSSIS final eval on full val_all (%s)", cfg_full.DATASETS.TEST[0]
            )
            return results

        def _append_wssis_train_hooks(out: List[Any]) -> None:
            if patience <= 0:
                return
            out.append(
                WssisEarlyStoppingHook(
                    eval_period,
                    patience=patience,
                    monitor_suffix=monitor_suffix,
                )
            )
            if storage_metric:
                out.append(
                    d2_hooks.BestCheckpointer(
                        eval_period,
                        self.checkpointer,
                        storage_metric,
                        mode="max",
                        file_prefix="model_best",
                    )
                )

        patched: List[Any] = []
        replaced = False
        for hook in hooks_list:
            if isinstance(hook, d2_hooks.EvalHook):
                if use_full_val_final:
                    patched.append(WssisEvalHook(eval_period, test_subset, test_full))
                else:
                    patched.append(d2_hooks.EvalHook(eval_period, test_subset))
                _append_wssis_train_hooks(patched)
                replaced = True
            else:
                patched.append(hook)

        if not replaced:
            if use_full_val_final:
                patched.append(WssisEvalHook(eval_period, test_subset, test_full))
            else:
                patched.append(d2_hooks.EvalHook(eval_period, test_subset))
            _append_wssis_train_hooks(patched)

        return patched

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    if add_wssis_config is not None:
        add_wssis_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if ensure_wssis_datasets_in_cfg is not None:
        ensure_wssis_datasets_in_cfg(cfg)
    try:
        from modules.wssis.mask2former_config import apply_smoke_to_cfg

        apply_smoke_to_cfg(cfg)
    except ImportError:
        pass
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    try:
        from modules.wssis.proc_utils import cleanup_distributed, install_worker_signal_handlers
    except ImportError:
        cleanup_distributed = None
        install_worker_signal_handlers = None

    if install_worker_signal_handlers is not None:
        install_worker_signal_handlers()

    try:
        cfg = setup(args)

        if args.eval_only:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
            if cfg.TEST.AUG.ENABLED:
                res.update(Trainer.test_with_TTA(cfg, model))
            if comm.is_main_process():
                verify_results(cfg, res)
            return res

        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        return trainer.train()
    finally:
        if cleanup_distributed is not None:
            cleanup_distributed()


if __name__ == "__main__":
    try:
        from modules.wssis.mask2former_launch import launch
    except ImportError:
        from detectron2.engine import launch

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
