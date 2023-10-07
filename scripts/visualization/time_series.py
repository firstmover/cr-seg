#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 07/07/2023
#
# Distributed under terms of the MIT license.

"""

"""
import argparse
import glob
import os
from os import path as osp

from joblib import Parallel, delayed

import submitit

from cr_seg.cli import add_slurm_submitit_args
from cr_seg.visualization import compute_seg_map_metric_sequence_from_dir


def _get_subj_to_metric_list(inf_dir: str):
    subj_name_list = os.listdir(inf_dir)
    subj_name_list = sorted([osp.basename(subj_name) for subj_name in subj_name_list])

    pred_seg_map_dir_list = []
    for _i, subj_name in enumerate(subj_name_list):
        pred_seg_map_dir = osp.join(inf_dir, subj_name, "pred_seg_map")
        pred_seg_map_dir_list.append(pred_seg_map_dir)

    # use joblib to parallelize the above
    Parallel(n_jobs=12)(
        delayed(compute_seg_map_metric_sequence_from_dir)(pred_seg_map_dir)
        for pred_seg_map_dir in pred_seg_map_dir_list
    )


def main(args):
    inference_dir_list = glob.glob(
        osp.join(args.result_root, "**", "inference/3d_time_series/*/val/"),
        recursive=True,
    )
    inference_dir_list = sorted(inference_dir_list)
    if len(inference_dir_list) == 0:
        raise ValueError("No inference dir found.")

    log_folder = osp.join("./log", "slurm", "%j")
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        mem_gb=32 * args.num_gpus_per_node,
        gpus_per_node=args.num_gpus_per_node,
        tasks_per_node=args.num_gpus_per_node,
        cpus_per_task=args.cpus_per_task,
        nodes=args.num_nodes,
        timeout_min=args.timeout_min,
        slurm_job_name=args.job_name,
        exclude=args.exclude,
        slurm_partition=args.partition,
        slurm_array_parallelism=args.array_parallelism,
        slurm_signal_delay_s=60,
    )

    with executor.batch():
        for inf_dir in inference_dir_list:
            executor.submit(_get_subj_to_metric_list, inf_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--result_root", type=str, required=True)
    add_slurm_submitit_args(parser)
    args = parser.parse_args()

    main(args)
