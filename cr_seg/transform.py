#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : transform.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 02/08/2023
#
# This file is part of cr_sg

"""

"""
import random
from collections.abc import Sequence

import numpy as np
import torch
from monai.transforms import (
    RandAdjustContrast,
    RandAffined,
    RandBiasField,
    RandBiasFieldd,
)

import einops
import mmcv
import torchio as tio
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size

    def transform(self, results: dict) -> dict:
        img = results["img"]
        h, w = img.shape[:2]
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        results["img"] = img[y : y + self.crop_size, x : x + self.crop_size]

        if "gt_seg_map" in results:
            gt_seg_map = results["gt_seg_map"]
            results["gt_seg_map"] = gt_seg_map[
                y : y + self.crop_size, x : x + self.crop_size
            ]

        if "gt_keypoints" in results or "gt_bboxes" in results:
            raise NotImplementedError

        return results


@TRANSFORMS.register_module()
class Crop3D(BaseTransform):
    """crop 3d: random or center crop"""

    def __init__(
        self,
        mode: str,
        crop_size: tuple[int, int, int],
        pad_zero_to_match_shape: bool = True,
    ):
        super().__init__()
        assert len(crop_size) == 3, "crop_size should be a tuple of length 3"
        self.mode = mode
        assert mode in ["center", "random"]
        self.crop_size = crop_size
        self.pad_zero_to_match_shape = pad_zero_to_match_shape

    def transform(self, results: dict) -> dict:
        img = results["img"]
        if "gt_seg_map" in results:
            assert img.shape[:3] == results["gt_seg_map"].shape[:3], (
                f"gt_seg_map shape {results['gt_seg_map'].shape} "
                f"does not match img shape {img.shape}"
            )
        h, w, d = img.shape[:3]

        need_padding = self.pad_zero_to_match_shape and (
            h < self.crop_size[0] or w < self.crop_size[1] or d < self.crop_size[2]
        )

        def _pad_data(data):
            padh = max(self.crop_size[0] - data.shape[0], 0)
            padw = max(self.crop_size[1] - data.shape[1], 0)
            padd = max(self.crop_size[2] - data.shape[2], 0)
            pad_shape_cfg = [
                (padh // 2, padh - padh // 2),
                (padw // 2, padw - padw // 2),
                (padd // 2, padd - padd // 2),
            ]
            if len(data.shape) == 4:
                # if there is a forth dim, we assume it is channel dim
                # and img/seg_map is in shape [H, W, D, C]
                assert data.shape[3] in [1, 2], f"data.shape: {data.shape}"
                pad_shape_cfg.append((0, 0))
            data = np.pad(
                data,
                pad_shape_cfg,
                "constant",
                constant_values=0,
            )
            return data

        if need_padding:
            img = _pad_data(img)

        h, w, d = img.shape[:3]
        if self.mode == "center":
            x = (h - self.crop_size[0]) // 2
            y = (w - self.crop_size[1]) // 2
            z = (d - self.crop_size[2]) // 2
        elif self.mode == "random":
            x = random.randint(0, h - self.crop_size[0])
            y = random.randint(0, w - self.crop_size[1])
            z = random.randint(0, d - self.crop_size[2])
        else:
            raise NotImplementedError

        results["img"] = img[
            x : x + self.crop_size[0],
            y : y + self.crop_size[1],
            z : z + self.crop_size[2],
        ]

        if "gt_seg_map" in results:
            gt_seg_map = results["gt_seg_map"]
            if need_padding:
                gt_seg_map = _pad_data(gt_seg_map)

            results["gt_seg_map"] = gt_seg_map[
                x : x + self.crop_size[0],
                y : y + self.crop_size[1],
                z : z + self.crop_size[2],
            ]

        if "gt_keypoints" in results or "gt_bboxes" in results:
            raise NotImplementedError

        return results


@TRANSFORMS.register_module()
class RandomCrop3D(BaseTransform):
    def __init__(self, crop_size: tuple[int], pad_zero_to_match_shape: bool = True):
        super().__init__()
        assert len(crop_size) == 3, "crop_size should be a tuple of length 3"
        self.crop_size = crop_size
        self.pad_zero_to_match_shape = pad_zero_to_match_shape

    def transform(self, results: dict) -> dict:
        img = results["img"]
        h, w, d = img.shape[:3]

        need_padding = self.pad_zero_to_match_shape and (
            h < self.crop_size[0] or w < self.crop_size[1] or d < self.crop_size[2]
        )
        if need_padding:
            pad_h = max(0, self.crop_size[0] - h)
            pad_w = max(0, self.crop_size[1] - w)
            pad_d = max(0, self.crop_size[2] - d)
            pad_cfg = [
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (pad_d // 2, pad_d - pad_d // 2),
                (0, 0),
            ]
            img = np.pad(img, pad_cfg, "constant", constant_values=0)

        h, w, d = img.shape[:3]
        x = random.randint(0, h - self.crop_size[0])
        y = random.randint(0, w - self.crop_size[1])
        z = random.randint(0, d - self.crop_size[2])

        results["img"] = img[
            x : x + self.crop_size[0],
            y : y + self.crop_size[1],
            z : z + self.crop_size[2],
        ]

        if "gt_seg_map" in results:
            gt_seg_map = results["gt_seg_map"]

            if need_padding:
                # NOTE(YL 07/23):: will raise error for paired data
                # use Crop3D instead.
                pad_cfg = pad_cfg[:3]
                gt_seg_map = np.pad(gt_seg_map, pad_cfg, "constant", constant_values=0)

            results["gt_seg_map"] = gt_seg_map[
                x : x + self.crop_size[0],
                y : y + self.crop_size[1],
                z : z + self.crop_size[2],
            ]

        if "gt_keypoints" in results or "gt_bboxes" in results:
            raise NotImplementedError

        return results


@TRANSFORMS.register_module()
class CenterCrop3D(BaseTransform):
    def __init__(self, crop_size: tuple[int, int, int], pad_zero_to_match_shape=False):
        super().__init__()
        assert len(crop_size) == 3, "crop_size should be a tuple of length 3"
        self.crop_size = crop_size
        self.pad_zero_to_match_shape = pad_zero_to_match_shape

    def transform(self, results: dict) -> dict:
        img = results["img"]
        if "gt_seg_map" in results:
            assert img.shape[:3] == results["gt_seg_map"].shape[:3], (
                f"gt_seg_map shape {results['gt_seg_map'].shape} "
                f"does not match img shape {img.shape}"
            )
        h, w, d = img.shape[:3]

        need_padding = self.pad_zero_to_match_shape and (
            h < self.crop_size[0] or w < self.crop_size[1] or d < self.crop_size[2]
        )

        def _pad_data(data):
            padh = max(self.crop_size[0] - data.shape[0], 0)
            padw = max(self.crop_size[1] - data.shape[1], 0)
            padd = max(self.crop_size[2] - data.shape[2], 0)
            pad_shape_cfg = [
                (padh // 2, padh - padh // 2),
                (padw // 2, padw - padw // 2),
                (padd // 2, padd - padd // 2),
            ]
            if len(data.shape) == 4:
                assert data.shape[3] == 1, f"data.shape: {data.shape}"
                pad_shape_cfg.append((0, 0))
            data = np.pad(
                data,
                pad_shape_cfg,
                "constant",
                constant_values=0,
            )
            return data

        if need_padding:
            img = _pad_data(img)

        h, w, d = img.shape[:3]
        x = (h - self.crop_size[0]) // 2
        y = (w - self.crop_size[1]) // 2
        z = (d - self.crop_size[2]) // 2

        results["img"] = img[
            x : x + self.crop_size[0],
            y : y + self.crop_size[1],
            z : z + self.crop_size[2],
        ]

        if "gt_seg_map" in results:
            gt_seg_map = results["gt_seg_map"]
            if need_padding:
                gt_seg_map = _pad_data(gt_seg_map)

            results["gt_seg_map"] = gt_seg_map[
                x : x + self.crop_size[0],
                y : y + self.crop_size[1],
                z : z + self.crop_size[2],
            ]

        if "gt_keypoints" in results or "gt_bboxes" in results:
            raise NotImplementedError

        return results


@TRANSFORMS.register_module()
class RandomContrastBrightness(BaseTransform):
    def __init__(
        self, sigma_contrast_perturb: float = 0.1, sigma_brightness_perturb: float = 0.1
    ):
        super().__init__()
        self.sigma_contrast_perturb = sigma_contrast_perturb
        self.sigma_brightness_perturb = sigma_brightness_perturb

    def transform(self, results: dict) -> dict:
        rand_contrast_factor = np.random.normal(
            1.0, self.sigma_contrast_perturb, size=1
        )[0]
        rand_brightness_diff = np.random.normal(
            0.0, self.sigma_brightness_perturb, size=1
        )[0]

        img = results["img"]
        img_mean = np.mean(img)

        results["img"] = (
            (img - img_mean) * rand_contrast_factor + img_mean + rand_brightness_diff
        )

        return results


@TRANSFORMS.register_module()
class RandomRotate3DSingleDim(BaseTransform):
    def __init__(self, angle: int = 180, dim: int = 2, prob: float = 1):
        super().__init__()
        self.angle = angle
        self.dim = dim
        if not self.dim == 2:
            raise NotImplementedError
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if self.prob == 1 or random.random() < self.prob:
            # random float value between -angle and angle
            angle = random.uniform(-self.angle, self.angle)

            # rotate along the dim axis
            img = results["img"]
            img_list = einops.rearrange(img, "h w z c -> z h w c")
            rot_img_list = np.ascontiguousarray(
                [mmcv.imrotate(img, angle) for img in img_list]
            )
            if len(rot_img_list.shape) == 3:
                rot_img_list = rot_img_list[..., np.newaxis]
            img = einops.rearrange(rot_img_list, "z h w c -> h w z c")
            results["img"] = img

            if "gt_seg_map" in results:
                gt_seg_map = results["gt_seg_map"]
                gt_seg_map_list = einops.rearrange(gt_seg_map, "h w z -> z h w")
                rot_gt_seg_map_list = np.ascontiguousarray(
                    [
                        mmcv.imrotate(gt_seg_map, angle, interpolation="nearest")
                        for gt_seg_map in gt_seg_map_list
                    ]
                )
                results["gt_seg_map"] = einops.rearrange(
                    rot_gt_seg_map_list, "z h w -> h w z"
                )

            if "gt_keypoints" in results or "gt_bboxes" in results:
                raise NotImplementedError

        return results


@TRANSFORMS.register_module()
class RandomGaussianNoise(BaseTransform):
    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def transform(self, results: dict) -> dict:
        img = results["img"]
        noise = np.random.normal(0, self.sigma, img.shape).astype(np.float32)
        results["img"] = img + noise

        return results


@TRANSFORMS.register_module()
class ScaleIntensity01(BaseTransform):
    def __init__(self, mode: str = "min_max"):
        super().__init__()
        self.mode = mode

    def transform(self, results: dict) -> dict:
        img = results["img"]
        if self.mode == "min_max":
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            raise NotImplementedError
        results["img"] = img
        return results


@TRANSFORMS.register_module()
class RandomGammaAdjustContrast(BaseTransform):
    def __init__(
        self,
        gamma: tuple[float, float] = (0.5, 4.5),
    ):
        super().__init__()
        self.gamma = gamma
        self.func_trans = RandAdjustContrast(gamma=gamma)

    def transform(self, results: dict) -> dict:
        # img: (H, W, Z), gt_seg_map: (H, W, Z)
        assert len(results["img"].shape) == 4 and results["img"].shape[-1] == 1
        img = results["img"].squeeze(-1)
        img = self.func_trans(img)
        results["img"] = img[..., None].get_array()
        return results


@TRANSFORMS.register_module()
class RandomGammaAdjustContrastMultiChan(BaseTransform):
    def __init__(
        self,
        gamma: tuple[float, float] = (0.5, 4.5),
        same_all_channels: bool = True,
    ):
        super().__init__()
        self.gamma = gamma
        self.func_trans = RandAdjustContrast(gamma=gamma)
        self.same_all_channels = same_all_channels

    def transform(self, results: dict) -> dict:
        # img: (H, W, Z, C), gt_seg_map: (H, W, Z)
        img = results["img"]
        num_chan = img.shape[-1]
        assert len(img.shape) == 4 and num_chan >= 1

        if not self.same_all_channels:
            img = self.func_trans(img)
            results["img"] = img[..., None].get_array()

        else:
            img_per_channel = [img[..., i] for i in range(num_chan)]
            img_transformed_per_channel = [
                self.func_trans(img_per_channel[0], randomize=True).get_array()
            ]
            for i in range(1, num_chan):
                img_transformed_per_channel.append(
                    self.func_trans(img_per_channel[i], randomize=False).get_array()
                )
            img_transformed = np.stack(img_transformed_per_channel, axis=-1)
            results["img"] = img_transformed

        return results


@TRANSFORMS.register_module()
class RandomFlip3d(BaseTransform):
    def __init__(
        self,
        dim: int = 2,
        prob: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.prob = prob

    def transform(self, results: dict) -> dict:
        img = results["img"]
        assert len(img.shape) == 4 and img.shape[-1] in [1, 2]
        gt_seg_map = results["gt_seg_map"]

        if random.random() < self.prob:
            img = np.flip(img, self.dim)
            gt_seg_map = np.flip(gt_seg_map, self.dim)

        results["img"] = img
        results["gt_seg_map"] = gt_seg_map
        return results


@TRANSFORMS.register_module()
class RandomRot903d(BaseTransform):
    def __init__(
        self,
        axis: tuple[int, int] = (0, 1),
        k: int = 3,
    ):
        super().__init__()
        self.axis = axis
        self.k = k

    def transform(self, results: dict) -> dict:
        img = results["img"]
        assert len(img.shape) == 4 and img.shape[-1] in [1, 2]
        gt_seg_map = results["gt_seg_map"]

        k = random.randint(0, self.k - 1)
        if k != 0:
            img = np.rot90(img, k, self.axis)
            gt_seg_map = np.rot90(gt_seg_map, k, self.axis)

        results["img"] = img
        results["gt_seg_map"] = gt_seg_map
        return results


@TRANSFORMS.register_module()
class RandomBiasField(BaseTransform):
    def __init__(
        self,
        degree: int = 3,
        coeff_range: tuple[float, float] = (0.0, 0.1),
    ):
        self.func_trans = RandBiasFieldd(
            keys=["img"], prob=1, degree=degree, coeff_range=coeff_range
        )

    def transform(self, results: dict) -> dict:
        # img: (H, W, Z), gt_seg_map: (H, W, Z)
        assert len(results["img"].shape) == 4 and results["img"].shape[-1] == 1
        results["img"] = results["img"].squeeze(-1)
        results = self.func_trans(results)
        results["img"] = results["img"][..., None].get_array()
        return results


@TRANSFORMS.register_module()
class RandomBiasFieldMultiChan(BaseTransform):
    def __init__(
        self,
        degree: int = 3,
        coeff_range: tuple[float, float] = (0.0, 0.1),
        same_all_channels: bool = True,
    ):
        self.func_trans = RandBiasField(prob=1, degree=degree, coeff_range=coeff_range)
        self.same_all_channels = same_all_channels

    def transform(self, results: dict) -> dict:
        # img: (H, W, Z, C), gt_seg_map: (H, W, Z)
        num_chan = results["img"].shape[-1]
        assert (
            len(results["img"].shape) == 4 and num_chan >= 1
        ), f'shape of img: {results["img"].shape} is not supported.'

        img = results["img"]
        if not self.same_all_channels:
            # it seems that monai will apply different bias
            # field to each channel.
            img_transformed = self.func_trans(img, randomize=True).get_array()

        else:
            img_per_channel = [img[..., i] for i in range(num_chan)]

            img_transformed_per_channel = [
                self.func_trans(img_per_channel[0], randomize=True).get_array()
            ]
            for i in range(1, num_chan):
                img_transformed_per_channel.append(
                    self.func_trans(img_per_channel[i], randomize=False).get_array()
                )
            img_transformed = np.stack(img_transformed_per_channel, axis=-1)

        results["img"] = img_transformed
        return results


@TRANSFORMS.register_module()
class Random3DElastic(BaseTransform):
    def __init__(
        self,
        num_control_points=(8, 8, 5),
        max_displacement=(2, 2, 1),
        prob=0.25,
    ):
        self.func_trans = tio.transforms.RandomElasticDeformation(
            num_control_points, max_displacement
        )
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if random.random() < self.prob:
            assert len(results["img"].shape) == 4 and results["img"].shape[-1] == 1
            sample = tio.Subject(
                {
                    "img": tio.ScalarImage(tensor=results["img"]),
                    "label": tio.LabelMap(tensor=results["gt_seg_map"][..., None]),
                }
            )
            sample = self.func_trans(sample)
            results["img"] = sample["img"].numpy()
            results["gt_seg_map"] = sample["label"].numpy().squeeze(-1)
        return results


@TRANSFORMS.register_module()
class RandomAffine(BaseTransform):
    def __init__(self, rotate_range, translate_range, scale_range, spatial_size):
        self.func_trans = RandAffined(
            keys=["img", "gt_seg_map"],
            prob=1,
            mode=("bilinear", "nearest"),
            rotate_range=rotate_range,
            translate_range=None,
            scale_range=None,
            spatial_size=None,
        )

    def transform(self, results: dict) -> dict:
        assert len(results["img"].shape) == 4 and results["img"].shape[-1] == 1
        results["img"] = results["img"].squeeze(-1)
        results = self.func_trans(results)
        results["img"] = results["img"][..., None].get_array()
        results["gt_seg_map"] = results["gt_seg_map"].get_array()
        return results


@TRANSFORMS.register_module()
class DefaultFormatBundle3D(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_seg_map". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if "img" in results:
            img = results["img"]
            if len(img.shape) < 4:
                img = np.expand_dims(img, -1)
            img = einops.rearrange(img, "h w d c -> c h w d")
            # move channel dimention to the first dimention
            img = np.ascontiguousarray(img)
            results["img"] = to_tensor(img)

        if "gt_seg_map" in results:
            gt_seg_map = results["gt_seg_map"]

            if len(gt_seg_map.shape) == 3:
                # single frame of 3d gt seg map
                gt_seg_map = einops.rearrange(gt_seg_map, "h w d -> 1 h w d")
            elif len(gt_seg_map.shape) == 4 and gt_seg_map.shape[-1] == 2:
                # a pair of 3d gt seg map for registration
                gt_seg_map = einops.rearrange(gt_seg_map, "h w d c -> c h w d")
            else:
                raise NotImplementedError

            results["gt_seg_map"] = to_tensor(gt_seg_map.astype(np.int64))
        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_seg_map".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: (``filename``, ``ori_filename``, ``ori_shape``,
            ``img_shape``, ``pad_shape``, ``scale_factor``, ``flip``,
            ``flip_direction``, ``img_norm_cfg``)
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data["img_metas"] = img_meta
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )


@TRANSFORMS.register_module()
class UpsampleByInteger(BaseTransform):
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def transform(self, results: dict) -> dict:
        img = results["img"]
        results["img"] = (
            img.repeat(self.scale, axis=0)
            .repeat(self.scale, axis=1)
            .repeat(self.scale, axis=2)
        )

        if "gt_seg_map" in results:
            gt_seg_map = results["gt_seg_map"]
            results["gt_seg_map"] = (
                gt_seg_map.repeat(self.scale, axis=0)
                .repeat(self.scale, axis=1)
                .repeat(self.scale, axis=2)
            )

        if "gt_keypoints" in results or "gt_bboxes" in results:
            raise NotImplementedError

        return results
