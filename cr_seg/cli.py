#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : cli.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 08/04/2023
#
# This file is part of cr_seg
"""

"""
import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import nibabel as nib
from mmengine.config import Config, DictAction


class SubmititTrainer(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, master_port=29500):
        import os

        from mmengine.runner import Runner

        from cr_seg.utils import register_all_modules

        os.environ["MASTER_PORT"] = str(master_port)

        # register all modules in mmselfsup into the registries
        # do not init the default scope here because it will be
        # init in the runner
        register_all_modules(init_default_scope=False)

        # Runner must be initialized for each task
        # in a slurm job.
        runner = Runner.from_cfg(self.cfg)
        runner.train()


class SubmititInferenceLabeledRunner(object):
    def __init__(self, model_cfg, dataset_cfg, metric_cfg, save_data_list):
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg
        self.metric_cfg = metric_cfg

        self.save_data_list = save_data_list
        all_data_name = ["img", "pred_seg_map", "logits_map", "gt_seg_map"]
        assert all([d in all_data_name for d in self.save_data_list])

    def __call__(
        self,
        ckpt_path,
        save_dir,
        affine,
        header,
    ):
        from mmengine.evaluator import Evaluator
        from mmengine.model.utils import revert_sync_batchnorm
        from mmengine.registry import DATASETS, MODELS
        from mmengine.runner import load_checkpoint

        from cr_seg.datasets.placenta.utils import _filename2subj_name_frame_idx
        from cr_seg.utils import register_all_modules

        register_all_modules(init_default_scope=False)

        model = MODELS.build(self.model_cfg)
        model = revert_sync_batchnorm(model)
        load_checkpoint(model, ckpt_path, map_location="cpu")
        model.cuda()
        model.eval()

        dataset = DATASETS.build(self.dataset_cfg)
        if hasattr(dataset, "full_init"):
            dataset.full_init()

        evaluator = Evaluator(self.metric_cfg)

        pred_save_dir = osp.join(save_dir, "pred")
        os.makedirs(pred_save_dir, exist_ok=True)

        filename_list = []
        subj_name_list = []
        for _i, data in enumerate(tqdm(dataset)):
            filename = data["img_metas"]["filename"]
            filename = osp.basename(filename)
            subj_name, _frame_idx = _filename2subj_name_frame_idx(filename)
            filename_list.append(filename)
            subj_name_list.append(subj_name)

            has_gt_seg_map = (
                "gt_seg_map" in data.keys() and data["gt_seg_map"] is not None
            )

            data["img"] = data["img"][None, ...].cuda()
            if has_gt_seg_map:
                data["gt_seg_map"] = data["gt_seg_map"][None, ...].cuda()
            else:
                data["gt_seg_map"] = None

            with torch.no_grad():
                _data_input = {k: v for k, v in data.items() if k != "img_metas"}
                pred_seg_map, logits_map = model(
                    **_data_input, mode="predict_and_logits"
                )

            # save data
            img = data["img"][0, 0][..., None].cpu().numpy().astype(np.float32)
            pred_seg_map_np = pred_seg_map[0].cpu().numpy().astype(np.float32)
            logits_map_np = logits_map[0].cpu().numpy().astype(np.float32)
            gt_seg_map = data["gt_seg_map"][0].cpu().numpy().astype(np.float32)

            d_list = [img, pred_seg_map_np, logits_map_np, gt_seg_map]
            data_name_list = ["img", "pred_seg_map", "logits_map", "gt_seg_map"]

            for d, data_name in zip(d_list, data_name_list):
                if data_name not in self.save_data_list:
                    continue
                save_path = osp.join(
                    pred_save_dir,
                    f"{data_name}",
                    filename.replace(".nii.gz", f"_{data_name}.nii.gz"),
                )
                os.makedirs(osp.dirname(save_path), exist_ok=True)
                header["data_type"] = d.dtype
                nib.save(nib.Nifti1Image(d, affine, header=header), save_path)

            # evaluator
            data["img"] = data["img"].cpu()
            data["gt_seg_map"] = data["gt_seg_map"].cpu()
            evaluator.process(data_batch=data, data_samples=[pred_seg_map[0]])

        metric_save_dir = osp.join(save_dir, "metric")
        os.makedirs(metric_save_dir, exist_ok=True)
        for metric, m_cfg in zip(evaluator.metrics, self.metric_cfg):
            flat_split = [r["split_tag"] for r in metric.results]
            flat_score = [float(r["score"][0].numpy()) for r in metric.results]

            data_frame = pd.DataFrame(
                {
                    "split": flat_split,
                    "score": flat_score,
                    "filename": filename_list,
                    "subj_name": subj_name_list,
                }
            )

            metric_type = m_cfg["type"]
            data_frame.to_csv(
                osp.join(metric_save_dir, f"{metric_type}.csv"), index=False
            )

        ret = evaluator.evaluate(len(dataset))
        print("ret: {}".format(ret))


class SubmititInferenceTimeSeriesRunner(object):
    def __init__(self, model_cfg, dataloader_cfg, save_data_list):
        self.model_cfg = model_cfg
        self.dataloader_cfg = dataloader_cfg

        self.save_data_list = save_data_list
        all_data_name = ["img", "pred_seg_map", "logits_map", "gt_seg_map"]
        assert all([d in all_data_name for d in self.save_data_list])

    def __call__(
        self,
        ckpt_path,
        save_dir,
        affine,
        header,
        split,  # train, val, test
    ):
        from mmengine.model.utils import revert_sync_batchnorm
        from mmengine.registry import MODELS
        from mmengine.runner import Runner, load_checkpoint

        from cr_seg.datasets.placenta.utils import _filename2subj_name_frame_idx
        from cr_seg.utils import register_all_modules

        register_all_modules(init_default_scope=False)

        model = MODELS.build(self.model_cfg)
        model = revert_sync_batchnorm(model)
        load_checkpoint(model, ckpt_path, map_location="cpu")
        model.cuda()
        model.eval()

        data_loader = Runner.build_dataloader(self.dataloader_cfg)
        data_iterator = iter(data_loader)

        pred_save_dir = osp.join(save_dir, "pred")
        os.makedirs(pred_save_dir, exist_ok=True)

        # loop over data set
        for _i, data in enumerate(tqdm(data_iterator)):
            data["img"] = data["img"].cuda()
            data["gt_seg_map"] = None

            _data_input = {k: v for k, v in data.items() if k != "img_metas"}
            with torch.no_grad():
                pred_seg_map, logits_map = model(
                    **_data_input, mode="predict_and_logits"
                )

            img = data["img"].cpu().numpy().astype(np.float32)
            pred_seg_map = pred_seg_map.cpu().numpy().astype(np.float32)
            logits_map = logits_map.cpu().numpy().astype(np.float32)

            batch_size = img.shape[0]
            for idx_batch in range(batch_size):
                filename = data["img_metas"]["filename"][idx_batch]
                subj_name, _frame_idx = _filename2subj_name_frame_idx(filename)
                subj_save_dir = osp.join(save_dir, split, f"{subj_name}")
                os.makedirs(subj_save_dir, exist_ok=True)

                this_img = img[idx_batch, 0][..., None]
                this_pred_seg_map = pred_seg_map[idx_batch]
                this_logits_map = logits_map[idx_batch]

                data_list = [this_img, this_pred_seg_map, this_logits_map]
                data_name_list = ["img", "pred_seg_map", "logits_map"]

                for d, data_name in zip(data_list, data_name_list):
                    if data_name not in self.save_data_list:
                        continue
                    save_path = osp.join(
                        subj_save_dir,
                        f"{data_name}",
                        filename.replace(".nii.gz", f"_{data_name}.nii.gz"),
                    )
                    os.makedirs(osp.dirname(save_path), exist_ok=True)
                    header["data_type"] = d.dtype
                    nib.save(nib.Nifti1Image(d, affine, header=header), save_path)


def add_mmengine_args(parser: argparse.ArgumentParser):
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        help="If specify checkpint path, resume from it, while if not "
        "specify, try to auto resume from the latest checkpoint "
        "in the work directory.",
    )

    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )

    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="specify master port in arg to avoid conflict in PyTorch DDP",
    )


def load_cfg_with_args(args):
    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # update configs according to CLI args if args.work_dir is not None
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # resume training
    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    return cfg


def add_slurm_submitit_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--job-name",
        type=str,
        default="test",
        help="job name",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="2080ti",
        help="partition to submit the job",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="nodes to exclude",
    )
    parser.add_argument(
        "--timeout-min",
        type=int,
        default=24 * 60,
        help="time limit in minutes",
    )
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        default=1,
        help="number of GPUs per node",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=8,
        help="number of cpus per task",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="number of nodes",
    )
    parser.add_argument(
        "--array-parallelism",
        type=int,
        default=8,
        help="number of jobs to run in parallel",
    )
