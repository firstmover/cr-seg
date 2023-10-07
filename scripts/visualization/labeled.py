#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 07/01/2023
#
# Distributed under terms of the MIT license.

"""

"""

import argparse
import glob
import os
from os import path as osp

import pandas as pd
import streamlit as st

import plotly.express as px


def visualize_histogram_all(metric_df, debug):
    option_marginal = st.sidebar.selectbox(
        "select marginal plot type", ["box", "violin", "rug"]
    )

    fig = px.histogram(
        metric_df,
        x="score",
        color="exp_lambda",
        marginal=option_marginal,
        title="all",
        histnorm="probability density",
        barmode="overlay",
    )
    # fig.update_xaxes(range=[0, 1])
    st.plotly_chart(fig)

    # compute statistics for each exp_lambda
    exp_lambda_list = sorted(metric_df["exp_lambda"].unique())
    exp_lambda_stat_df_list = []
    for exp_lambda in exp_lambda_list:
        exp_lambda_df = metric_df[metric_df["exp_lambda"] == exp_lambda]
        score = exp_lambda_df["score"]
        exp_lambda_stat_df = pd.DataFrame(
            {
                "exp_lambda": [exp_lambda],
                "mean": [score.mean()],
                "std": [score.std()],
                "median": [score.median()],
                "iqr": [score.quantile(0.75) - score.quantile(0.25)],
                "quan_5": [score.quantile(0.05)],
                "quan_10": [score.quantile(0.1)],
                "quan_25": [score.quantile(0.25)],
                "quan_75": [score.quantile(0.75)],
                "quan_90": [score.quantile(0.9)],
                "quan_95": [score.quantile(0.95)],
            }
        )
        exp_lambda_stat_df_list.append(exp_lambda_stat_df)

    exp_lambda_stat_df = pd.concat(exp_lambda_stat_df_list)
    st.dataframe(exp_lambda_stat_df)


def _load_metric_df(result_root, model_name, sel_metric_file_name):
    metric_file_path_list = glob.glob(
        osp.join(
            result_root,
            "**",
            "inference",
            "labelled",
            model_name,
            "metric",
            sel_metric_file_name,
        ),
        recursive=True,
    )
    metric_file_path_list.sort()

    metric_df_list = []
    for metric_file_path in metric_file_path_list:
        metric_df = pd.read_csv(metric_file_path)

        # parse experiment name, lambda name, cross_val_split, ckpt name
        # from file path
        rel_path = metric_file_path[len(result_root) + 1 :]
        p = osp.normpath(rel_path)
        split_dir = p.split(os.sep)
        exp_name, lambda_name, cross_val_split, _, _, ckpt_name = split_dir[:6]

        if "exp" not in cross_val_split and "lambda" in cross_val_split:
            # this is a exp with different lambda for spatial and temporal cr
            exp_name, lambda_name_s, lambda_name_t = split_dir[:3]
            lambda_name = lambda_name_s + "-" + lambda_name_t
            cross_val_split, _, _, ckpt_name = split_dir[3:7]

        elif "num_few_shot" in lambda_name:
            # this is a exp from few shot
            exp_name, num_few_shot, few_shot_offset = split_dir[:3]
            lambda_name = num_few_shot + "-" + few_shot_offset
            cross_val_split, _, _, ckpt_name = split_dir[3:7]

        metric_df["experiment"] = exp_name
        metric_df["lambda"] = lambda_name
        metric_df["cross_val_split"] = cross_val_split
        metric_df["ckpt"] = ckpt_name
        metric_df["exp_lambda"] = exp_name + "-" + lambda_name

        metric_df_list.append(metric_df)

    metric_df = pd.concat(metric_df_list)

    return metric_file_path_list, metric_df


def main(result_root, model_name, debug):
    st.set_page_config(layout="wide")

    st.write("result root: {}".format(result_root))

    # in result_root, all results are saved in the following structure:
    # {result_root}/{exp_name}/{lambda}/exp_5_{}/inference/labelled/{ckpt}/metric/{metric}

    metric_file_name_list = [
        "DiceMetric.csv",
        "HausdorffDistance.csv",
        "HausdorffDistance95.csv",
    ]
    sel_metric_file_name = st.sidebar.selectbox(
        "select metric file", metric_file_name_list
    )

    metric_file_path_list, metric_df = st.cache(allow_output_mutation=True)(
        _load_metric_df
    )(result_root, model_name, sel_metric_file_name)

    st.dataframe(pd.DataFrame(metric_file_path_list))

    split_train_val = st.sidebar.selectbox("select split", ["val", "train"])
    metric_df = metric_df[metric_df["split"] == split_train_val]

    st.dataframe(metric_df)

    visualize_histogram_all(metric_df.copy(), debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--result_root",
        type=str,
        required=True,
        help="Path to experiment folder such as ./results/cr_3",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="A model name such as epoch_100, or epoch_100_sliding_window",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode",
    )
    args = parser.parse_args()

    main(args.result_root, args.model_name, args.debug)
