#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/04/2023
#
# Distributed under terms of the MIT license.

"""

"""

import argparse
import os.path as osp

import submitit
from mmengine.config import Config

from cr_seg.cli import SubmititTrainer, add_slurm_submitit_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model using mmengine and submitit (cross val)"
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

    add_slurm_submitit_args(parser)
    args = parser.parse_args()

    args.val_fold_list = [int(x) for x in args.val_fold_list.split(",")]

    if len(args.val_fold_list) != args.num_fold:
        print(
            f"WARNING: len(val-fold-list) ({len(args.val_fold_list)}) != "
            f"num-fold ({args.num_fold})"
        )

    return args


def _submit_train(executor, cfg, val_fold, num_fold):
    cfg.train_dataset.num_fold = num_fold
    cfg.train_dataset.val_fold_idx = val_fold
    cfg.train_dataloader.dataset.num_fold = num_fold
    cfg.train_dataloader.dataset.val_fold_idx = val_fold
    cfg.val_dataset.num_fold = num_fold
    cfg.val_dataset.val_fold_idx = val_fold
    cfg.val_dataloader.dataset.num_fold = num_fold
    cfg.val_dataloader.dataset.val_fold_idx = val_fold

    trainer = SubmititTrainer(cfg)
    job = executor.submit(trainer, str(29500 + val_fold))

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
                f"exp_{args.num_fold}_{val_fold}",
            )
            cfg_file_path = osp.join(
                "./configs", args.exp_name, args.config_name + ".py"
            )
            cfg = Config.fromfile(cfg_file_path)
            cfg.launcher = "slurm"
            cfg.work_dir = work_dir

            job = _submit_train(
                executor,
                cfg,
                val_fold,
                args.num_fold,
            )

            job_list.append(job)


if __name__ == "__main__":
    main()
