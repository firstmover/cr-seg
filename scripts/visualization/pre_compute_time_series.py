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
from collections import defaultdict
from os import path as osp

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px

from cr_seg.visualization import compute_seg_map_metric_sequence_from_dir


def _get_sub2metric2score_list(exp_path: str) -> dict[str, dict[str, list[float]]]:
    subj_name_list = os.listdir(exp_path)
    subj_name_list = sorted([osp.basename(subj_name) for subj_name in subj_name_list])

    subj2metric2score_list_dict = defaultdict(dict)
    for _i, subj_name in enumerate(subj_name_list):
        pred_seg_map_dir = osp.join(exp_path, subj_name, "pred_seg_map")
        dice_list, hd_list, hd95_list = compute_seg_map_metric_sequence_from_dir(
            pred_seg_map_dir
        )
        subj2metric2score_list_dict[subj_name]["dice"] = dice_list
        subj2metric2score_list_dict[subj_name]["hd"] = hd_list
        subj2metric2score_list_dict[subj_name]["hd95"] = hd95_list

    return subj2metric2score_list_dict


def _visualize_all_samples(exp_lambda2subj2dice_list_dict: dict[str, dict[str, list]]):
    # convert to dataframe
    all_df = []
    for exp_lambda, subj2dice_list_dict in exp_lambda2subj2dice_list_dict.items():
        st.write(exp_lambda)

        df = pd.DataFrame.from_dict(subj2dice_list_dict, orient="index")
        df = df.reset_index()
        df = df.rename(columns={"index": "subj_name"})
        df = df.melt(id_vars=["subj_name"], var_name="time", value_name="dice")
        df["exp_lambda"] = exp_lambda
        all_df.append(df)

    # plotly violin plot, each subject uses one columns
    all_df = pd.concat(all_df)
    fig = px.violin(
        all_df,
        x="subj_name",
        y="dice",
        color="exp_lambda",
    )
    fig.update_layout(
        title="Dice distribution of each subject",
        xaxis_title="subject name",
        yaxis_title="dice",
    )
    st.plotly_chart(fig, use_container_width=True)


def _visualize_historgram_of_all_samples(
    exp_lambda2subj2dice_list_dict: dict[str, dict[str, list]]
):
    for exp_lambda, subj2dice_list in exp_lambda2subj2dice_list_dict.items():
        st.write(exp_lambda)

        all_score_list = []
        for _, dice_list in subj2dice_list.items():
            all_score_list.extend(dice_list)

        # remove nan, inf, -inf etc
        all_score_list = np.array(all_score_list)
        all_score_list = all_score_list[~np.isnan(all_score_list)]
        all_score_list = all_score_list[~np.isinf(all_score_list)]

        fig = px.histogram(
            all_score_list,
        )
        fig.update_traces(
            xbins=dict(start=0.0, end=1.0, size=0.01)  # bins used for histogram
        )
        fig.update_layout(
            xaxis=dict(
                tickfont=dict(size=20),
            ),
            yaxis=dict(
                tickfont=dict(size=20),
            ),
        )
        fig.update_layout(showlegend=False)
        fig.update_layout(
            title={
                "text": "",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            xaxis_title="",
            yaxis_title="",
        )
        fig.update_layout(
            width=600,
            height=500,
        )
        fig.update_yaxes(range=[0, 3000])
        fig.update_xaxes(range=[0, 1.0])
        st.plotly_chart(fig)


def main(result_root, model_name: str, debug):
    st.set_page_config(layout="wide")

    st.write("result root: {}".format(result_root))

    glob_pattern = osp.join(
        result_root, "**", f"inference/3d_time_series/{model_name:}/val/"
    )
    inference_dir_list = glob.glob(glob_pattern, recursive=True)
    inference_dir_list = sorted(inference_dir_list)

    st.table(inference_dir_list)

    exp_lambda2subj2dice_list_dict: dict[str, dict[str, list]] = defaultdict(dict)
    exp_lambda2subj2hd_list_dict: dict[str, dict[str, list]] = defaultdict(dict)
    exp_lambda2subj2hd95_list_dict: dict[str, dict[str, list]] = defaultdict(dict)
    for inf_dir in inference_dir_list:
        rel_path = inf_dir[len(result_root) + 1 :]
        p = osp.normpath(rel_path)
        split_dir = p.split(osp.sep)

        exp_name = split_dir[0]
        lambda_name = split_dir[1]
        if "lambda" in split_dir[2]:
            lambda_name = lambda_name + "-" + split_dir[2]
        exp_lambda_name = exp_name + "-" + lambda_name

        # subj2dice_list_dict = _get_subj_to_dice_list(inf_dir)
        subj2metric2score_list_dict = st.cache(
            _get_sub2metric2score_list, allow_output_mutation=True
        )(inf_dir)

        for subj_name, metric2score_list_dict in subj2metric2score_list_dict.items():
            dice_list = metric2score_list_dict["dice"]
            hd_list = metric2score_list_dict["hd"]
            hd95_list = metric2score_list_dict["hd95"]

            exp_lambda2subj2dice_list_dict[exp_lambda_name][subj_name] = dice_list
            exp_lambda2subj2hd_list_dict[exp_lambda_name][subj_name] = hd_list
            exp_lambda2subj2hd95_list_dict[exp_lambda_name][subj_name] = hd95_list

    score_option_list = ["dice", "hd", "hd95"]
    score_option = st.sidebar.selectbox("Score option", score_option_list, index=0)
    if score_option == "dice":
        score_dict = exp_lambda2subj2dice_list_dict
    elif score_option == "hd":
        score_dict = exp_lambda2subj2hd_list_dict
    elif score_option == "hd95":
        score_dict = exp_lambda2subj2hd95_list_dict
    else:
        raise ValueError("Unknown score option: {}".format(score_option))

    vis_option_list = ["all_samples", "histogram"]
    vis_option = st.sidebar.selectbox("Visualization option", vis_option_list, index=1)

    if vis_option == "all_samples":
        _visualize_all_samples(score_dict)

    elif vis_option == "histogram":
        _visualize_historgram_of_all_samples(score_dict)

    else:
        raise ValueError("Unknown visualization option: {}".format(vis_option))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--result_root", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args.result_root, args.model_name, args.debug)
