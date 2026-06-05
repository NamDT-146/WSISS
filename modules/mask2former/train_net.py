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
from typing import Any, Dict, List, Optional, Set

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
from detectron2.utils.events import EventStorage
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
    from modules.wssis.training.stage2_trainer import WssisStage2TrainerMixin
except ImportError:
    add_wssis_config = None
    ensure_wssis_datasets_in_cfg = None

    class WssisStage2TrainerMixin:  # type: ignore[no-redef]
        pass


class Trainer(WssisStage2TrainerMixin, DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    _wssis_teacher_ref = None

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

                if torch.cuda.is_available():
                    device = torch.device("cuda", comm.get_local_rank())
                else:
                    device = torch.device("cpu")
                gnn_path = getattr(cfg.WSSIS, "GNN_CHECKPOINT", "") or None
                from modules.wssis.pseudo_label_confidence import build_threshold_policy

                pseudo_cfg = {
                    "pseudo_label": {
                        "threshold_mode": str(
                            getattr(cfg.WSSIS, "PSEUDO_THRESHOLD_MODE", "fixed")
                        ),
                        "confidence_threshold": float(
                            getattr(cfg.WSSIS, "PSEUDO_CONFIDENCE_THRESHOLD", 0.9)
                        ),
                    }
                }
                threshold_policy = build_threshold_policy(pseudo_cfg)
                teacher = WssisTeacherStack(
                    device,
                    gnn_ckpt_path=Path(gnn_path) if gnn_path else gnn_checkpoint(),
                    use_gnn=getattr(cfg.WSSIS, "USE_GNN", True),
                    freeze_gnn=getattr(cfg.WSSIS, "FREEZE_GNN", False),
                    threshold_policy=threshold_policy,
                )
                cls._wssis_teacher_ref = teacher
                mapper = WssisSemiWeakMapper(cfg, True, teacher=teacher)
                # Online SAM/GNN in mapper must run in main process; keep eval NUM_WORKERS.
                eval_workers = int(cfg.DATALOADER.NUM_WORKERS)
                if eval_workers > 0:
                    logging.getLogger(__name__).info(
                        "Semi-weak train loader: NUM_WORKERS=0 (GPU teacher in mapper); "
                        "eval/test loaders keep NUM_WORKERS=%d",
                        eval_workers,
                    )
                cfg.defrost()
                cfg.DATALOADER.NUM_WORKERS = 0
                cfg.freeze()
                loader = build_detection_train_loader(cfg, mapper=mapper)
                if eval_workers > 0:
                    cfg.defrost()
                    cfg.DATALOADER.NUM_WORKERS = eval_workers
                    cfg.freeze()
                return loader
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

        return d2_build_model(cfg)

    def __init__(self, cfg):
        super().__init__(cfg)
        self._wssis_teacher = type(self)._wssis_teacher_ref
        self._wssis_teacher_opt = None
        wssis = getattr(cfg, "WSSIS", None)
        if (
            self._wssis_teacher is not None
            and wssis is not None
            and getattr(wssis, "USE_STAGE2_JOINT_LOSS", False)
            and self._wssis_teacher.gnn is not None
            and not getattr(wssis, "FREEZE_GNN", False)
        ):
            gnn_lr = float(getattr(wssis, "GNN_LR", 1e-5))
            self._wssis_gnn_base_lr = gnn_lr
            self._wssis_gnn_warmup_iters = int(getattr(wssis, "GNN_WARMUP_ITERS", 200))
            # Start at lr=0 so the (noisy) GNN does not yank the student early in training.
            self._wssis_teacher_opt = torch.optim.AdamW(
                self._wssis_teacher.gnn.parameters(),
                lr=0.0 if self._wssis_gnn_warmup_iters > 0 else gnn_lr,
            )

    def _run_step_joint(self) -> None:
        import time

        from detectron2.structures import ImageList

        from modules.wssis.training.stage2_trainer import _student_head_outputs, _unwrap_model

        inner = self._trainer
        inner.iter = self.iter

        assert self.model.training
        model = _unwrap_model(self.model)
        start = time.perf_counter()
        data = next(inner._data_loader_iter)
        data_time = time.perf_counter() - start

        data, teacher_losses = self._wssis_prepare_joint_batch(data)

        images = [x["image"].to(model.device) for x in data]
        images_norm = [(x - model.pixel_mean) / model.pixel_std for x in images]
        image_list = ImageList.from_tensors(images_norm, model.size_divisibility)

        head_out = _student_head_outputs(model, data)
        gt_instances = [x["instances"].to(model.device) for x in data]
        targets = model.prepare_targets(gt_instances, image_list)
        loss_dict = model.criterion(head_out, targets)
        for k in list(loss_dict.keys()):
            if k in model.criterion.weight_dict:
                loss_dict[k] *= model.criterion.weight_dict[k]
            else:
                loss_dict.pop(k)

        aux = self._wssis_joint_aux_losses(data, head_out)
        loss_dict.update(teacher_losses)
        loss_dict.update(aux)

        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        if self._wssis_teacher_opt is not None:
            self._wssis_teacher_opt.zero_grad()
        losses.backward()
        inner.after_backward()
        inner._write_metrics(loss_dict, data_time)
        self.optimizer.step()
        if self._wssis_teacher_opt is not None:
            self._wssis_apply_gnn_warmup()
            self._wssis_teacher_opt.step()

    def _wssis_apply_gnn_warmup(self) -> None:
        """Linearly ramp the GNN optimizer LR from 0 to its base value over the first iters."""
        base_lr = getattr(self, "_wssis_gnn_base_lr", None)
        if base_lr is None:
            return
        warmup = getattr(self, "_wssis_gnn_warmup_iters", 0)
        if warmup > 0:
            lr = base_lr * min(1.0, (self.iter + 1) / float(warmup))
        else:
            lr = base_lr
        for group in self._wssis_teacher_opt.param_groups:
            group["lr"] = lr
        storage = getattr(self, "storage", None)
        if storage is not None:
            storage.put_scalar("wssis/gnn_lr", lr, smoothing_hint=False)

    def run_step(self):
        """Joint teacher-student step or default Mask2Former step."""
        if self._wssis_joint_enabled():
            self._run_step_joint()
        else:
            super().run_step()
        self._wssis_log_metrics()

    def _wssis_log_metrics(self) -> None:
        if not getattr(self.cfg, "WSSIS", None):
            return
        if not getattr(self.cfg.WSSIS, "USE_SEMI_WEAK", False):
            return
        storage = self.storage
        if storage is None:
            return

        def _latest(name: str):
            buf = storage.histories().get(name)
            if buf is None:
                return None
            try:
                return buf.latest()
            except (KeyError, IndexError, ValueError):
                return None

        loss_ce = _latest("loss_ce")
        loss_mask = _latest("loss_mask")
        loss_dice = _latest("loss_dice")
        sup = 0.0
        for v in (loss_ce, loss_mask, loss_dice):
            if v is not None:
                sup += float(v)
        storage.put_scalar("wssis/sup_loss", sup, smoothing_hint=False)
        for key in (
            "loss_teacher_pce",
            "loss_teacher_sym",
            "loss_teacher_feedback",
            "loss_semi",
        ):
            val = _latest(key)
            if val is not None:
                storage.put_scalar(f"wssis/{key}", float(val), smoothing_hint=False)

    def _wssis_training_enabled(self) -> bool:
        wssis = getattr(self.cfg, "WSSIS", None)
        return wssis is not None and bool(getattr(wssis, "EXPERIMENT_ID", ""))

    def train(self):
        """
        Run training.

        WSSIS experiments use a dynamic loop so early stopping can terminate
        training before SOLVER.MAX_ITER (Detectron2's default for-loop bound is fixed).
        """
        if not self._wssis_training_enabled():
            super().train()
            return self._finalize_train_results()

        logger = logging.getLogger(__name__)
        logger.info(
            "Starting WSSIS training from iteration %s (max_iter=%s)",
            self.start_iter,
            self.max_iter,
        )

        self._wssis_stop_training = False
        self.iter = self.start_iter
        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                while self.iter < self.max_iter and not self._wssis_stop_training:
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

        return self._finalize_train_results()

    def _finalize_train_results(self):
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results
        return None

    def build_hooks(self):
        hooks_list = super().build_hooks()
        cfg = self.cfg
        if add_wssis_config is None or not getattr(cfg, "WSSIS", None):
            return hooks_list
        if not cfg.WSSIS.EXPERIMENT_ID:
            return hooks_list

        from detectron2.engine import hooks as d2_hooks

        from modules.wssis.mask2former_datasets import wssis_val_full_name
        from modules.wssis.mask2former_hooks import (
            WssisBestCheckpointer,
            WssisEarlyStoppingHook,
            WssisEvalHook,
            WssisTrainProgressHook,
        )

        eval_period = cfg.TEST.EVAL_PERIOD
        patience = int(getattr(cfg.WSSIS, "EARLY_STOPPING_PATIENCE", 0))
        monitor_suffix = getattr(cfg.WSSIS, "EARLY_STOPPING_MONITOR", "segm/AP")
        use_full_val_final = getattr(cfg.WSSIS, "USE_FULL_VAL_FINAL", False)
        logging.getLogger("mask2former").info(
            "WSSIS early stopping config: monitor=%s patience=%d eval_period=%d",
            monitor_suffix,
            patience,
            eval_period,
        )

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

        early_stop_hook: Optional[WssisEarlyStoppingHook] = None
        if patience > 0:
            early_stop_hook = WssisEarlyStoppingHook(
                eval_period,
                patience=patience,
                monitor_suffix=monitor_suffix,
            )

        best_ckpt_hook: Optional[WssisBestCheckpointer] = None
        if comm.is_main_process():
            best_ckpt_hook = WssisBestCheckpointer(
                eval_period,
                self.checkpointer,
                monitor_suffix=monitor_suffix,
                mode="max",
                file_prefix="model_best",
            )

        def _wssis_eval_bundle():
            if use_full_val_final:
                return WssisEvalHook(
                    eval_period,
                    test_subset,
                    test_full,
                    early_stopping_hook=early_stop_hook,
                    best_checkpointer_hook=best_ckpt_hook,
                )
            return WssisEvalHook(
                eval_period,
                test_subset,
                test_subset,
                early_stopping_hook=early_stop_hook,
                best_checkpointer_hook=best_ckpt_hook,
            )

        def _append_wssis_train_hooks(out: List[Any]) -> None:
            if early_stop_hook is not None:
                out.append(early_stop_hook)
            if comm.is_main_process():
                out.append(WssisTrainProgressHook())

        def _insert_before_writer(out: List[Any], bundle: List[Any]) -> None:
            for i, hook in enumerate(out):
                if isinstance(hook, d2_hooks.PeriodicWriter):
                    out[i:i] = bundle
                    return
            out.extend(bundle)

        patched: List[Any] = []
        replaced = False
        for hook in hooks_list:
            if isinstance(hook, d2_hooks.EvalHook):
                patched.append(_wssis_eval_bundle())
                _append_wssis_train_hooks(patched)
                replaced = True
            else:
                patched.append(hook)

        if not replaced:
            bundle: List[Any] = [_wssis_eval_bundle()]
            _append_wssis_train_hooks(bundle)
            _insert_before_writer(patched, bundle)

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
        from modules.wssis.mask2former_config import apply_gpu_batch_alignment, apply_smoke_to_cfg

        apply_smoke_to_cfg(cfg)
        apply_gpu_batch_alignment(cfg, getattr(args, "num_gpus", None))
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
