#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 08/04/2023
#
# Distributed under terms of the MIT license.

"""

"""

import argparse
import os.path as osp
import pickle
import re
from copy import deepcopy

import numpy as np

import submitit
from mmengine.config import Config

from cr_seg.cli import (
    SubmititInferenceLabeledRunner,
    SubmititInferenceTimeSeriesRunner,
    SubmititTrainer,
    add_slurm_submitit_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model using mmengine and submitit "
        "(cross val and lambda sweep)"
    )

    parser.add_argument(
        "--task-mode",
        type=str,
        default="train",
        help="task mode: train, inference_labeled, or inference_time_series",
    )

    parser.add_argument("--exp-name", help="experiment name")
    parser.add_argument("--config-name", help="config name")

    parser.add_argument(
        "--val-fold-list",
        type=str,
        default="0,1,2,3,4",
        help="list of validation folds (coma separated)",
    )
    parser.add_argument(
        "--num-fold",
        type=int,
        default=5,
        help="number of folds",
    )
    parser.add_argument(
        "--lambda-list",
        type=str,
        default="0.01,0.001,0.0001",
        help="list of lambda values (coma separated)",
    )
    parser.add_argument(
        "--lambda-t-list",
        type=str,
        default="0.01,0.001,0.0001",
        help="list of t lambda values (coma separated)",
    )

    ##############
    #  training  #
    ##############

    parser.add_argument(
        "--convert-four-2080ti-cfg-to-one-a6000",
        action="store_true",
        help=(
            "convert cfg from four 2080ti to one A6000" "mainly just change batch size."
        ),
    )

    ###############
    #  inference  #
    ###############

    parser.add_argument(
        "--ckpt-name",
        type=str,
        default="epoch_100",
        help="checkpoint name (default: epoch_100)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="whether to use tta model",
    )
    parser.add_argument(
        "--tta-cfg-path", help="config file path for tta model", default=None
    )
    parser.add_argument(
        "--save-data-name-list",
        type=str,
        default="img,pred_seg_map",
        help="list of data to save (coma separated)",
    )

    add_slurm_submitit_args(parser)
    args = parser.parse_args()

    args.val_fold_list = [int(x) for x in args.val_fold_list.split(",")]
    args.lambda_list = [float(x) for x in args.lambda_list.split(",")]
    args.lambda_str_list = [str(x).replace(".", "_") for x in args.lambda_list]
    args.lambda_t_list = [float(x) for x in args.lambda_t_list.split(",")]
    args.lambda_t_str_list = [str(x).replace(".", "_") for x in args.lambda_t_list]

    args.save_data_name_list = args.save_data_name_list.split(",")

    if len(args.val_fold_list) != args.num_fold:
        print(
            f"WARNING: len(val-fold-list) ({len(args.val_fold_list)}) != "
            f"num-fold ({args.num_fold})"
        )

    return args


def _submit_train(
    executor, args, cfg, lambda_value, lambda_t_value, val_fold, num_fold
):
    cfg.train_dataset.num_fold = num_fold
    cfg.train_dataset.val_fold_idx = val_fold
    cfg.train_dataloader.dataset.num_fold = num_fold
    cfg.train_dataloader.dataset.val_fold_idx = val_fold
    cfg.val_dataset.num_fold = num_fold
    cfg.val_dataset.val_fold_idx = val_fold
    cfg.val_dataloader.dataset.num_fold = num_fold
    cfg.val_dataloader.dataset.val_fold_idx = val_fold

    cfg.model.lambda_schedule_cfg.value = lambda_value
    cfg.model.lambda_t_schedule_cfg.value = lambda_t_value

    if hasattr(cfg.model.registration_cfg, "init_cfg"):
        if cfg.model.registration_cfg["type"] == "Pretrained":
            ckpt_path = cfg.model.registration_cfg.init_cfg["checkpoint"]
            # if path contains 'exp_{num_fold}_{val_fold_idx}'
            # then replace it with 'exp_{num_fold}_{val_fold}'
            if re.search(r"exp_\d+_\d+", ckpt_path):
                ckpt_path = re.sub(
                    r"exp_\d+_\d+", f"exp_{num_fold}_{val_fold}", ckpt_path
                )
            cfg.model.registration_cfg.init_cfg["checkpoint"] = ckpt_path

    if args.convert_four_2080ti_cfg_to_one_a6000:
        new_batch_size = cfg.batch_size * 4
        cfg.batch_size = new_batch_size
        cfg.train_dataloader.batch_size = new_batch_size
        cfg.val_dataloader.batch_size = new_batch_size

        cfg.train_dataloader.num_workers = cfg.train_dataloader.num_workers * 4

    trainer = SubmititTrainer(cfg)
    job = executor.submit(trainer, str(29500 + val_fold))

    return job


def _submit_inference(executor, args, cfg):
    if args.tta:
        cfg_tta = Config.fromfile(args.tta_cfg_path)
        model_cfg = cfg_tta.tta_model
        tta_input_size = cfg_tta.tta_input_size
        assert all([i >= j for i, j in zip(tta_input_size, cfg.crop_size)]), (
            "model input size must be no larger than the data crop size to "
            "enable sliding window style inference"
        )
        input_crop_size = tta_input_size

        inference_method_name = cfg_tta.tta_model.inference_cfg["type"]
        inference_method_name = re.sub(
            r"(?<!^)(?=[A-Z])", "_", inference_method_name
        ).lower()
        inference_method_name = inference_method_name.replace("_crop", "")

        save_dir = osp.join(
            cfg.work_dir,
            "inference",
            "labelled",
            args.ckpt_name + "_" + inference_method_name,
        )

    else:
        model_cfg = cfg.model
        input_crop_size = cfg.crop_size
        save_dir = osp.join(cfg.work_dir, "inference", "labelled", args.ckpt_name)

    dataset_cfg = deepcopy(cfg.val_dataloader["dataset"])
    dataset_cfg["transform"] = [
        dict(
            type="CenterCrop3D", crop_size=input_crop_size, pad_zero_to_match_shape=True
        ),
        dict(type="DefaultFormatBundle3D"),
        dict(
            type="Collect",
            keys=["img", "gt_seg_map"],
            meta_keys=["filename", "split_tag"],
        ),
    ]

    metric_cfg = [
        dict(type="DiceMetric", eval_by_split=True),
        dict(type="HausdorffDistance", eval_by_split=True),
        dict(type="HausdorffDistance95", eval_by_split=True),
    ]

    runner = SubmititInferenceLabeledRunner(
        model_cfg, dataset_cfg, metric_cfg, args.save_data_name_list
    )

    with open("./data/first_data_header.pkl", "rb") as f:
        header = pickle.load(f)
        dim_list = [3] + list(input_crop_size) + [1, 1, 1, 1]
        header["dim"] = np.array(dim_list)
    affine = np.array(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    ckpt_path = osp.join(cfg.work_dir, args.ckpt_name + ".pth")
    if not osp.exists(ckpt_path):
        raise RuntimeError(f"ckpt path {ckpt_path} does not exist")

    job = executor.submit(
        runner,
        ckpt_path,
        save_dir,
        affine,
        header,
    )
    return job


def _submit_inference_time_series(executor, args, cfg):
    if args.tta:
        cfg_tta = Config.fromfile(args.tta_cfg_path)
        model_cfg = cfg_tta.tta_model
        tta_input_size = cfg_tta.tta_input_size
        assert all([i >= j for i, j in zip(tta_input_size, cfg.crop_size)]), (
            "model input size must be no larger than the data crop size to "
            "enable sliding window style inference"
        )
        input_crop_size = tta_input_size

        inference_method_name = cfg_tta.tta_model.inference_cfg["type"]
        inference_method_name = re.sub(
            r"(?<!^)(?=[A-Z])", "_", inference_method_name
        ).lower()
        inference_method_name = inference_method_name.replace("_crop", "")

        save_dir = osp.join(
            cfg.work_dir,
            "inference",
            "3d_time_series",
            args.ckpt_name + "_" + inference_method_name,
        )

    else:
        model_cfg = cfg.model
        input_crop_size = cfg.crop_size
        save_dir = osp.join(cfg.work_dir, "inference", "3d_time_series", args.ckpt_name)

    transform = [
        dict(
            type="CenterCrop3D", crop_size=input_crop_size, pad_zero_to_match_shape=True
        ),
        dict(type="DefaultFormatBundle3D"),
        dict(type="Collect", keys=["img"], meta_keys=["filename"]),
    ]

    split = "val"  # only inference val for cross validation setting
    dataset_cfg = dict(
        type="placenta_3d_trimed_normed_cross_validation_unlabeled",
        train=split == "train",
        norm_method=cfg.val_dataloader.dataset.norm_method,
        transform=transform,
        num_fold=cfg.val_dataloader.dataset.num_fold,
        val_fold_idx=cfg.val_dataloader.dataset.val_fold_idx,
    )

    batch_size = 4 if not args.tta else 1
    dataloader_cfg = dict(
        batch_size=batch_size,
        sampler=dict(type="DefaultSampler", shuffle=False),
        collate_fn=dict(type="default_collate"),
        dataset=dataset_cfg,
        persistent_workers=True,
        pin_memory=True,
        num_workers=8,
    )

    runner = SubmititInferenceTimeSeriesRunner(
        model_cfg, dataloader_cfg, args.save_data_name_list
    )

    with open("./data/first_data_header.pkl", "rb") as f:
        header = pickle.load(f)
        dim_list = [3] + list(input_crop_size) + [1, 1, 1, 1]
        header["dim"] = np.array(dim_list)
    affine = np.array(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    ckpt_path = osp.join(cfg.work_dir, args.ckpt_name + ".pth")
    if not osp.exists(ckpt_path):
        raise RuntimeError(f"ckpt path {ckpt_path} does not exist")

    job = executor.submit(
        runner,
        ckpt_path,
        save_dir,
        affine,
        header,
        split,
    )
    return job


def main():
    args = parse_args()

    # we assume this script is launched at the root of git repo
    cfg_file_path = osp.join("./configs", args.exp_name, args.config_name + ".py")
    cfg = Config.fromfile(cfg_file_path)
    cfg.launcher = "slurm"

    # we save the slurm log to
    # ./results/{exp_name}/{config_name}/slurm_log/{job_id}
    # it seems that executor folder path need to be fixed when initialized
    log_folder = osp.join(
        "./results", args.exp_name, args.config_name, "slurm_log", "%j"
    )
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        mem_gb=32 * args.num_gpus_per_node,
        gpus_per_node=args.num_gpus_per_node,
        tasks_per_node=args.num_gpus_per_node,
        cpus_per_task=args.cpus_per_task,
        nodes=args.num_nodes,
        timeout_min=args.timeout_min,
        slurm_job_name=args.job_name,
        slurm_exclude=args.exclude,
        slurm_partition=args.partition,
        slurm_array_parallelism=args.array_parallelism,
        slurm_signal_delay_s=60,
    )

    job_list = []
    with executor.batch():
        for lambda_value, lambda_str in zip(args.lambda_list, args.lambda_str_list):
            for lambda_t_value, lambda_t_str in zip(
                args.lambda_t_list, args.lambda_t_str_list
            ):
                for val_fold in args.val_fold_list:
                    # setting the master port here within executor context
                    # is wrong. Because all jobs are launched at the same time
                    # when the context exits, all jobs will have the same master port
                    # -> set the master port in Trainer

                    # there is something wrong if we share the save cfg object
                    # across different jobs. and cfg.copy() is not deep copy.
                    # -> lets just don't share any object

                    work_dir = osp.join(
                        "./results",
                        args.exp_name,
                        args.config_name,
                        f"lambda_{lambda_str}",
                        f"lambda_t_{lambda_t_str}",
                        f"exp_{args.num_fold}_{val_fold}",
                    )

                    if args.task_mode == "train":
                        cfg_file_path = osp.join(
                            "./configs", args.exp_name, args.config_name + ".py"
                        )
                        cfg = Config.fromfile(cfg_file_path)
                        cfg.launcher = "slurm"
                        cfg.work_dir = work_dir

                        job = _submit_train(
                            executor,
                            args,
                            cfg,
                            lambda_value,
                            lambda_t_value,
                            val_fold,
                            args.num_fold,
                        )

                    elif args.task_mode == "inference_labeled":
                        cfg_file_path = osp.join(work_dir, f"{args.config_name}.py")
                        cfg = Config.fromfile(cfg_file_path)
                        cfg.launcher = "slurm"
                        job = _submit_inference(executor, args, cfg)

                    elif args.task_mode == "inference_time_series":
                        cfg_file_path = osp.join(work_dir, f"{args.config_name}.py")
                        cfg = Config.fromfile(cfg_file_path)
                        cfg.launcher = "slurm"
                        job = _submit_inference_time_series(executor, args, cfg)

                    else:
                        raise ValueError(f"Unknown task: {args.task_mode}")

                    job_list.append(job)


if __name__ == "__main__":
    main()
