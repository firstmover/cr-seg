#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : metric.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 02/13/2023
#
# This file is part of cr_seg

"""

"""

from functools import partial

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance
from torch.nn import functional as F

from mmengine import METRICS
from mmengine.evaluator import BaseMetric


def compute_dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    From https://gist.github.com/JDWarner/6730747
    """
    # im1 = im1 > 0.5
    # im2 = im2 > 0.5
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    if isinstance(im1, torch.Tensor) and isinstance(im2, torch.Tensor):
        intersection = torch.logical_and(im1, im2)
    elif isinstance(im1, np.ndarray) and isinstance(im2, np.ndarray):
        intersection = np.logical_and(im1, im2)
    else:
        raise TypeError("im1 and im2 must be either torch.Tensor or np.ndarray")

    return 2.0 * intersection.sum() / im_sum


def compute_mean_square_error(im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if isinstance(im1, torch.Tensor) and isinstance(im2, torch.Tensor):
        mse = torch.mean((im1 - im2) ** 2)
    elif isinstance(im1, np.ndarray) and isinstance(im2, np.ndarray):
        mse = np.mean((im1 - im2) ** 2)
    else:
        raise TypeError("im1 and im2 must be either torch.Tensor or np.ndarray")

    return mse


def sliding_window_3d(arr, window_size, stride):
    shape = arr.shape
    sub_shape = tuple(np.subtract(shape, window_size) // stride + 1) + window_size
    strides = tuple(np.multiply(arr.strides, stride)) + arr.strides
    subarrays = np.lib.stride_tricks.as_strided(arr, sub_shape, strides)
    return subarrays


def compute_local_normalized_cross_correlation(
    im1, im2, window_size=(5, 5, 5), zero_mean=False, remove_zero_norm=True, eps=1e-6
):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    if len(im1.shape) != 3:
        raise ValueError("im1 and im2 must be 3D arrays")
    img_shape = im1.shape
    subarray_shape = tuple([img_shape[i] - window_size[i] + 1 for i in range(3)])

    if isinstance(im1, torch.Tensor) and isinstance(im2, torch.Tensor):
        im1 = im1.unsqueeze(0).unsqueeze(0).double()
        im2 = im2.unsqueeze(0).unsqueeze(0).double()

        mean_corr = batch_compute_local_cosine_similarity(
            im1,
            im2,
            window_size=window_size,
            remove_zero_norm=remove_zero_norm,
            eps=eps,
        )
        return mean_corr

    elif isinstance(im1, np.ndarray) and isinstance(im2, np.ndarray):
        im1_subarrays = sliding_window_3d(im1, window_size, stride=(1, 1, 1))
        im1_subarrays = im1_subarrays.reshape(*subarray_shape, -1)
        im2_subarrays = sliding_window_3d(im2, window_size, stride=(1, 1, 1))
        im2_subarrays = im2_subarrays.reshape(*subarray_shape, -1)

        if zero_mean:
            im1_subarrays = im1_subarrays - np.mean(im1_subarrays, axis=3)
            im2_subarrays = im2_subarrays - np.mean(im2_subarrays, axis=3)

        im1_subarrays_norm = np.linalg.norm(im1_subarrays, axis=3)
        im2_subarrays_norm = np.linalg.norm(im2_subarrays, axis=3)

        corr = np.sum(im1_subarrays * im2_subarrays, axis=3) / (
            np.maximum(im1_subarrays_norm * im2_subarrays_norm, eps)
        )

        if remove_zero_norm:
            is_zero_norm = np.logical_or(
                im1_subarrays_norm == 0, im2_subarrays_norm == 0
            )
            corr = np.mean(corr[~is_zero_norm])

        else:
            corr = np.mean(corr)

        return corr

    else:
        raise TypeError("im1 and im2 must be either torch.Tensor or np.ndarray")


def batch_compute_mean_square_error(img_x, img_y):
    if img_x.shape != img_y.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if not isinstance(img_x, torch.Tensor) or not isinstance(img_y, torch.Tensor):
        raise TypeError("im1 and im2 must be torch.Tensor")

    if len(img_x.shape) == 5:
        if not img_x.shape[1] in [1, 3]:
            raise ValueError(
                "The second dimension of im1 and im2 must be 1 or 3 (channel)"
            )
    else:
        raise ValueError(f"im1 and im2 must be batch of 3D arrays. Got {img_x.shape}")

    mse = torch.mean((img_x - img_y) ** 2, dim=(1, 2, 3, 4))
    return mse


def batch_compute_local_cosine_similarity(
    img_x,
    img_y,
    window_size=(5, 5, 5),
    eps=1e-6,
    remove_zero_norm=True,
    return_mean=True,
    padding: str = "valid",
):
    if img_x.shape != img_y.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if not isinstance(img_x, torch.Tensor) or not isinstance(img_y, torch.Tensor):
        raise TypeError("im1 and im2 must be torch.Tensor")

    if len(img_x.shape) == 5:
        if not img_x.shape[1] in [1, 3]:
            raise ValueError(
                "The second dimension of im1 and im2 must be 1 or 3 (channel)"
            )
    else:
        raise ValueError(f"im1 and im2 must be batch of 3D arrays. Got {img_x.shape}")

    xy = img_x * img_y
    xx = img_x * img_x
    yy = img_y * img_y

    weight = torch.ones(1, 1, *window_size).to(img_x.device).to(img_x.dtype)
    func_conv = partial(F.conv3d, stride=1, padding=padding, weight=weight)

    local_xy = func_conv(xy)
    local_xx = func_conv(xx)
    local_yy = func_conv(yy)

    f_is_zero = partial(
        torch.isclose,
        other=torch.tensor(0.0).to(img_x.device).to(img_x.dtype),
        atol=eps,
        rtol=0,
    )
    is_zero_norm = torch.logical_or(f_is_zero(local_xx), f_is_zero(local_yy))
    is_valid = ~is_zero_norm

    local_consine_similarity = local_xy / (
        torch.clamp(torch.sqrt(local_xx) * torch.sqrt(local_yy), min=eps)
    )

    if not return_mean:
        return local_consine_similarity, is_valid

    if remove_zero_norm:
        denom = torch.sum(is_valid, dim=(1, 2, 3, 4))
        local_consine_similarity[~is_valid] = 0
        mean_similarity = torch.sum(local_consine_similarity, dim=(1, 2, 3, 4)) / denom
    else:
        mean_similarity = torch.mean(local_consine_similarity, dim=(1, 2, 3, 4))

    return mean_similarity


def batch_compute_local_zero_mean_cosine_similarity(
    img_x,
    img_y,
    window_size=(5, 5, 5),
    eps=1e-8,
    remove_zero_norm=True,
    return_mean=True,
    padding: str = "valid",
):
    if img_x.shape != img_y.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if not isinstance(img_x, torch.Tensor) or not isinstance(img_y, torch.Tensor):
        raise TypeError("im1 and im2 must be torch.Tensor")

    if len(img_x.shape) == 5:
        if not img_x.shape[1] in [1, 3]:
            raise ValueError(
                "The second dimension of im1 and im2 must be 1 or 3 (channel)"
            )
    else:
        raise ValueError(f"im1 and im2 must be batch of 3D arrays. Got {img_x.shape}")

    # using float16 will create too much error
    # ie. torch.tensor(1e-8, dtype=torch.float16) = 0.0000

    xy = img_x * img_y
    xx = img_x * img_x
    yy = img_y * img_y
    n = torch.prod(torch.tensor(window_size))

    weight = torch.ones(1, 1, *window_size).to(img_x.device).to(img_x.dtype)
    func_conv = partial(F.conv3d, stride=1, padding=padding, weight=weight)

    local_mean_x = func_conv(img_x) / n
    local_mean_y = func_conv(img_y) / n
    local_sum_xy = func_conv(xy)
    local_sum_xx = func_conv(xx)
    local_sum_yy = func_conv(yy)

    local_norm_zero_mean_x = local_sum_xx - n * local_mean_x**2
    local_norm_zero_mean_y = local_sum_yy - n * local_mean_y**2

    f_is_zero = partial(
        torch.isclose,
        other=torch.tensor(0.0).to(img_x.device).to(img_x.dtype),
        atol=eps,
        rtol=0,
    )
    is_zero_norm = torch.logical_or(
        f_is_zero(local_norm_zero_mean_x), f_is_zero(local_norm_zero_mean_y)
    )
    is_valid = ~is_zero_norm

    # for sanity check
    is_negative = torch.logical_or(
        local_norm_zero_mean_x < 0, local_norm_zero_mean_y < 0
    )
    neg_non_zero = torch.logical_and(is_negative, ~is_zero_norm)
    if torch.any(is_negative) and torch.any(neg_non_zero):
        raise ValueError(
            "Impossible: nagative norm but non-zero. " "use larger eps or disable amp."
        )

    denom = torch.clamp(
        torch.sqrt(local_norm_zero_mean_x) * torch.sqrt(local_norm_zero_mean_y),
        min=eps,
    )
    local_zero_mean_cosine_similarity = (
        local_sum_xy - n * local_mean_x * local_mean_y
    ) / denom

    if not return_mean:
        return local_zero_mean_cosine_similarity, is_valid

    if remove_zero_norm:
        denom = torch.sum(is_valid, dim=(1, 2, 3, 4))
        local_zero_mean_cosine_similarity[~is_valid] = 0
        mean_similarity = (
            torch.sum(local_zero_mean_cosine_similarity, dim=(1, 2, 3, 4)) / denom
        )

    else:
        mean_similarity = torch.mean(
            local_zero_mean_cosine_similarity, dim=(1, 2, 3, 4)
        )

    return mean_similarity


def batch_compute_local_zero_mean_cosine_similarity_v2(
    img_x,
    img_y,
    window_size=(5, 5, 5),
    eps=1e-8,
    remove_zero_norm=True,
    return_mean=True,
    padding: str = "valid",
):
    if img_x.shape != img_y.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if not isinstance(img_x, torch.Tensor) or not isinstance(img_y, torch.Tensor):
        raise TypeError("im1 and im2 must be torch.Tensor")

    if len(img_x.shape) == 5:
        if not img_x.shape[1] in [1, 3]:
            raise ValueError(
                "The second dimension of im1 and im2 must be 1 or 3 (channel)"
            )
    else:
        raise ValueError(f"im1 and im2 must be batch of 3D arrays. Got {img_x.shape}")

    # using float16 will create too much error
    # ie. torch.tensor(1e-8, dtype=torch.float16) = 0.0000

    xy = img_x * img_y
    xx = img_x * img_x
    yy = img_y * img_y
    n = torch.prod(torch.tensor(window_size))

    weight = torch.ones(1, 1, *window_size).to(img_x.device).to(img_x.dtype) / n
    func_conv = partial(F.conv3d, stride=1, padding=padding, weight=weight)

    local_mean_x = func_conv(img_x)
    local_mean_y = func_conv(img_y)
    local_mean_xy = func_conv(xy)
    local_mean_xx = func_conv(xx)
    local_mean_yy = func_conv(yy)

    local_norm_zero_mean_x = local_mean_xx - local_mean_x**2
    local_norm_zero_mean_y = local_mean_yy - local_mean_y**2

    f_is_zero = partial(
        torch.isclose,
        other=torch.tensor(0.0).to(img_x.device).to(img_x.dtype),
        atol=eps,
        rtol=0,
    )
    is_zero_norm = torch.logical_or(
        f_is_zero(local_norm_zero_mean_x), f_is_zero(local_norm_zero_mean_y)
    )
    is_valid = ~is_zero_norm

    # for sanity check
    is_negative = torch.logical_or(
        local_norm_zero_mean_x < 0, local_norm_zero_mean_y < 0
    )
    neg_non_zero = torch.logical_and(is_negative, ~is_zero_norm)
    if torch.any(is_negative) and torch.any(neg_non_zero):
        raise ValueError("Impossible: nagative norm but non-zero. use larger eps.")

    denom = torch.clamp(
        torch.sqrt(local_norm_zero_mean_x) * torch.sqrt(local_norm_zero_mean_y),
        min=eps,
    )
    local_zero_mean_cosine_similarity = (
        local_mean_xy - local_mean_x * local_mean_y
    ) / denom

    if not return_mean:
        return local_zero_mean_cosine_similarity, is_valid

    if remove_zero_norm:
        denom = torch.sum(is_valid, dim=(1, 2, 3, 4))
        local_zero_mean_cosine_similarity[~is_valid] = 0
        mean_similarity = (
            torch.sum(local_zero_mean_cosine_similarity, dim=(1, 2, 3, 4)) / denom
        )

    else:
        mean_similarity = torch.mean(
            local_zero_mean_cosine_similarity, dim=(1, 2, 3, 4)
        )

    return mean_similarity


def _gather_results_by_split(result_list: list[dict]) -> dict[str, list]:
    split2score = {}
    for result in result_list:
        for score, split in zip(result["score"], result["split_tag"]):
            if split not in split2score:
                split2score[split] = []
            split2score[split].append(score)
    return split2score


@METRICS.register_module()
class DiceMetric(BaseMetric):
    def __init__(self, eval_by_split: bool = False):
        super().__init__()
        self.eval_by_split = eval_by_split

    def process(self, data_batch, data_samples):
        gt_seg_map = data_batch["gt_seg_map"].squeeze(1)
        outputs = torch.stack(data_samples).cpu()

        batch_size = len(gt_seg_map)
        dice_score_list = [
            compute_dice(outputs[i], gt_seg_map[i]) for i in range(batch_size)
        ]
        this_results = {"score": dice_score_list}

        if self.eval_by_split:
            split_tag_list = data_batch["img_metas"]["split_tag"]
            this_results["split_tag"] = split_tag_list

        self.results.append(this_results)

    def compute_metrics(self, results: list) -> dict[str, float]:
        if self.eval_by_split:
            split2score = _gather_results_by_split(results)
            return {
                split_tag + "/dice": sum(score_list) / len(score_list)
                for split_tag, score_list in split2score.items()
            }

        else:
            total_num = sum([len(result["score"]) for result in results])
            total_dice_score = sum([sum(result["score"]) for result in results])
            return {"dice": total_dice_score / total_num}


@METRICS.register_module()
class L2ReconError(BaseMetric):
    def __init__(self, eval_by_split: bool = False):
        super().__init__()
        self.eval_by_split = eval_by_split

    def process(self, data_batch, data_samples):
        imgs = data_batch["img"]
        pred_imgs, masks = data_samples
        pred_imgs, masks = pred_imgs.cpu(), masks.cpu()

        batch_size = len(pred_imgs)
        mask_ratio = masks.mean()
        error = ((pred_imgs - imgs) ** 2 * masks).reshape(batch_size, -1)
        error = error.mean(1) / mask_ratio
        # error = torch.mean((pred_imgs - imgs) ** 2 * masks) / mask_ratio

        this_results = {"score": error}

        if self.eval_by_split:
            split_tag_list = data_batch["img_metas"]["split_tag"]
            this_results["split_tag"] = split_tag_list

        self.results.append(this_results)

    def compute_metrics(self, results):
        if self.eval_by_split:
            split2score = _gather_results_by_split(results)
            return {
                split_tag + "/l2_error": sum(l2_error_list) / len(l2_error_list)
                for split_tag, l2_error_list in split2score.items()
            }

        else:
            total_num = sum([len(result["score"]) for result in results])
            total_error = sum(item["score"] for item in results)
            return dict(l2_error=total_error / total_num)


@METRICS.register_module()
class HausdorffDistance(BaseMetric):
    def __init__(self, eval_by_split: bool = False):
        super().__init__()
        self.eval_by_split = eval_by_split

    def process(self, data_batch, data_samples):
        gt_seg_map = data_batch["gt_seg_map"]
        outputs = torch.stack(data_samples).unsqueeze(1).cpu()

        hausdorff_dist = compute_hausdorff_distance(outputs, gt_seg_map)

        this_results = {"score": hausdorff_dist}

        if self.eval_by_split:
            split_tag_list = data_batch["img_metas"]["split_tag"]
            this_results["split_tag"] = split_tag_list

        self.results.append(this_results)

    def compute_metrics(self, results: list) -> dict:
        if self.eval_by_split:
            split2score = _gather_results_by_split(results)
            return {
                split_tag + "/hausdorff_dist": sum(score_list) / len(score_list)
                for split_tag, score_list in split2score.items()
            }

        else:
            total_num = sum([len(result["score"]) for result in results])
            total_score = sum([result["score"] for result in results])
            return {"hausdorff_distance": total_score / total_num}


@METRICS.register_module()
class HausdorffDistance95(BaseMetric):
    def __init__(self, eval_by_split: bool = False):
        super().__init__()
        self.eval_by_split = eval_by_split

    def process(self, data_batch, data_samples):
        gt_seg_map = data_batch["gt_seg_map"]
        outputs = torch.stack(data_samples).unsqueeze(1).cpu()

        hausdorff_dist = compute_hausdorff_distance(outputs, gt_seg_map, percentile=95)

        this_results = {"score": hausdorff_dist}

        if self.eval_by_split:
            split_tag_list = data_batch["img_metas"]["split_tag"]
            this_results["split_tag"] = split_tag_list

        self.results.append(this_results)

    def compute_metrics(self, results: list) -> dict:
        if self.eval_by_split:
            split2score = _gather_results_by_split(results)
            return {
                split_tag + "/hausdorff_dist_95": sum(score_list) / len(score_list)
                for split_tag, score_list in split2score.items()
            }

        else:
            total_num = sum([len(result["score"]) for result in results])
            total_score = sum([result["score"] for result in results])
            return {"hausdorff_distance_95": total_score / total_num}


@METRICS.register_module()
class RegiMSE(BaseMetric):
    def __init__(self, use_roi: bool = False, eval_by_split: bool = False):
        super().__init__()
        self.use_roi = use_roi
        self.eval_by_split = eval_by_split

        self.score_prefix = "regi_mse"
        if use_roi:
            self.score_prefix += "_roi"

    def process(self, data_batch, data_samples):
        # gt_seg_map = data_batch["gt_seg_map"]
        img_pair = data_batch["img"].cpu()
        # moving_img = img_pair[:, 0]
        fixed_img = img_pair[:, 1].unsqueeze(1)

        # NOTE(YL 07/21):: 0 refers to the first batch
        # not the first one in this batch
        wraped_img = data_samples[0]["wraped_img"].cpu()
        # flow = data_samples['flow']
        bs = len(wraped_img)

        assert wraped_img.shape == fixed_img.shape

        this_mse_map = (wraped_img - fixed_img) ** 2

        if self.use_roi:
            roi_map = data_samples[0]["roi_map"].cpu()
            this_mse = this_mse_map * roi_map
            this_mse = this_mse.reshape(bs, -1).sum(1) / roi_map.reshape(bs, -1).sum(1)

        else:
            this_mse = this_mse_map.reshape(len(wraped_img), -1).mean(1)

        this_mse = this_mse.tolist()
        this_results = {"score": this_mse}

        if self.eval_by_split:
            split_tag_list = data_batch["img_metas"]["split_tag"]
            this_results["split_tag"] = split_tag_list

        self.results.append(this_results)

    def compute_metrics(self, results: list) -> dict:
        if self.eval_by_split:
            split2score = _gather_results_by_split(results)
            return {
                self.score_prefix + "/" + split_tag: sum(score_list) / len(score_list)
                for split_tag, score_list in split2score.items()
            }
        else:
            total_num = sum([len(result["score"]) for result in results])
            total_score = sum([sum(result["score"]) for result in results])
            return {self.score_prefix: total_score / total_num}


@METRICS.register_module()
class RegiLNCC(BaseMetric):
    def __init__(
        self, use_roi: bool = False, eval_by_split: bool = False, zero_mean: bool = True
    ):
        super().__init__()
        self.use_roi = use_roi
        self.eval_by_split = eval_by_split
        self.zero_mean = zero_mean

        if self.zero_mean:
            self.score_prefix = "regi_lncc_zm"
        else:
            self.score_prefix = "regi_lncc"

        if use_roi:
            self.score_prefix += "_roi"

    def process(self, data_batch, data_samples):
        # NOTE(YL 07/21):: 0 refers to the first batch
        # not the first one in this batch
        wraped_img = data_samples[0]["wraped_img"].double()
        # flow = data_samples['flow']
        bs = len(wraped_img)

        # gt_seg_map = data_batch["gt_seg_map"]
        img_pair = data_batch["img"].to(wraped_img.device).to(wraped_img.dtype)
        # moving_img = img_pair[:, 0]
        fixed_img = img_pair[:, 1].unsqueeze(1)

        assert wraped_img.shape == fixed_img.shape

        func_lncc = partial(
            batch_compute_local_zero_mean_cosine_similarity
            if self.zero_mean
            else batch_compute_local_cosine_similarity,
            return_mean=False,
            padding="same",
        )
        sim, is_valid = func_lncc(wraped_img, fixed_img)

        if self.use_roi:
            roi_map = data_samples[0]["roi_map"]
            is_valid = torch.logical_and(roi_map, is_valid)

        sim[~is_valid] = 0
        batch_lncc = sim.reshape(bs, -1).sum(1) / is_valid.reshape(bs, -1).sum(1)

        this_results = {"score": batch_lncc.cpu().tolist()}

        if self.eval_by_split:
            split_tag_list = data_batch["img_metas"]["split_tag"]
            this_results["split_tag"] = split_tag_list

        self.results.append(this_results)

    def compute_metrics(self, results: list) -> dict:
        if self.eval_by_split:
            split2score = _gather_results_by_split(results)
            return {
                self.score_prefix + "/" + split_tag: sum(score_list) / len(score_list)
                for split_tag, score_list in split2score.items()
            }
        else:
            total_num = sum([len(result["score"]) for result in results])
            total_score = sum([sum(result["score"]) for result in results])
            return {self.score_prefix: total_score / total_num}
