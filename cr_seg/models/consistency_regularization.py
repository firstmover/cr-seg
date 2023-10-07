#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 05/16/2023
#
# Distributed under terms of the MIT license.

"""

"""

import typing as tp
from functools import partial

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import einops
from mmengine import Registry
from mmengine.model import BaseModel, ExponentialMovingAverage
from mmengine.registry import MODELS

from cr_seg.criterion import build_loss
from cr_seg.models.voxelmorph import losses as vm_losses

from .unet_3d_nobatchnorm import UNet


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not.
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


@MODELS.register_module()
class UNet3dCRWithRegiv2(BaseModel):
    def __init__(
        self,
        model_size: str = "large",
        dropout: bool = True,
        loss_cfg: tp.Optional[dict] = None,
        lambda_schedule_cfg: tp.Optional[dict] = None,
        lambda_t_schedule_cfg: tp.Optional[dict] = None,
        cr_trans_cfg: tp.Optional[list[dict]] = None,
        teacher_momentum: tp.Optional[float] = None,
        cr_criterion_cfg: tp.Optional[dict] = None,
        registration_cfg: tp.Optional[dict] = None,
    ):
        super(UNet3dCRWithRegiv2, self).__init__()
        self.model = UNet(size=model_size, dropout=dropout)

        assert teacher_momentum is not None, "we use teacher model in cr with tc"
        assert teacher_momentum > 0 and teacher_momentum < 1
        self.teacher_momentum = teacher_momentum
        self.ema_model = ExponentialMovingAverage(self.model, teacher_momentum)

        if loss_cfg is not None:
            self.criterion = build_loss(loss_cfg)

        self.lambda_schedule_cfg = lambda_schedule_cfg
        if self.lambda_schedule_cfg is None:
            self.lambda_schedule_cfg = dict(type="ConstantSchedule", value=1.0)
        self.lambda_schedule = LAMBDA_SCHEDULE.build(self.lambda_schedule_cfg)

        self.lambda_t_schedule_cfg = lambda_t_schedule_cfg
        if self.lambda_t_schedule_cfg is None:
            self.lambda_t_schedule_cfg = self.lambda_schedule_cfg
        self.lambda_t_schedule = LAMBDA_SCHEDULE.build(self.lambda_t_schedule_cfg)

        self.cr_trans_cfg = cr_trans_cfg
        if self.cr_trans_cfg is None:
            self.cr_trans_cfg = [dict(type="RandomRot903d"), dict(type="RamdomFlip3d")]

        self.cr_criterion_cfg = cr_criterion_cfg
        if self.cr_criterion_cfg is None:
            self.cr_criterion_cfg = dict(type="L2Norm")
        self.cr_criterion = CR_CRITERION.build(self.cr_criterion_cfg)

        self.regi_model = MODELS.build(registration_cfg)
        self.regi_model.eval()
        set_requires_grad(self.regi_model, False)

    def forward(self, img, gt_seg_map, mode, **kwargs):
        if mode == "loss":
            B = img.shape[0]

            img_lab = img[:, 0:1]
            img_unlab_t = img[:, 1:2]
            img_unlab_t_plus_1 = img[:, 2:3]
            gt_seg_map_lab = gt_seg_map

            # spatial cr is applied to labeled and unlabeled t
            funcs = [_transform_generator(self.cr_trans_cfg) for _ in range(B * 2)]
            img_lab_unlab_t = torch.cat([img_lab, img_unlab_t], dim=0)
            trans_img_lab_unlab_t = torch.stack(
                [f._call_img(img_lab_unlab_t[i]) for i, f in enumerate(funcs)]
            )

            if self.teacher_momentum is None:
                raise NotImplementedError
            else:
                ori_outputs_lab_unlab_t = self.model(img_lab_unlab_t)
                ori_outputs_lab = ori_outputs_lab_unlab_t[:B]
                ori_outputs_unlab_t = ori_outputs_lab_unlab_t[B:]
                with torch.no_grad():
                    self.ema_model.update_parameters(self.model)

                    # trans_outputs_lab_unlab_t = self.ema_model(trans_img_lab_unlab_t)
                    # outputs_unlab_t_plus_1 = self.ema_model(img_unlab_t_plus_1)
                    ema_input = torch.cat(
                        [trans_img_lab_unlab_t, img_unlab_t_plus_1], dim=0
                    )
                    ema_outputs = self.ema_model(ema_input)
                    trans_outputs_lab_unlab_t = ema_outputs[: B * 2]
                    outputs_unlab_t_plus_1 = ema_outputs[B * 2 :]

            # supervised loss is applied to labeled
            loss_sup = self.criterion(ori_outputs_lab, gt_seg_map_lab)

            # spatial consistency is applied to labeled and unlabeled t
            ori_outputs_lab_unlab_t_trans = torch.stack(
                [f._call_seg(ori_outputs_lab_unlab_t[i]) for i, f in enumerate(funcs)]
            )
            loss_consistency = self.cr_criterion(
                trans_outputs_lab_unlab_t, ori_outputs_lab_unlab_t_trans
            )

            # temporal consistency is applied to unlabeled t and unlabeled t + 1
            loss_consistency_t = self.cr_criterion(
                ori_outputs_unlab_t, outputs_unlab_t_plus_1
            )

            max_iter = kwargs["max_iter"]
            current_iter = kwargs["current_iter"]
            this_lambda = self.lambda_schedule(max_iter, current_iter)
            this_lambda_t = self.lambda_t_schedule(max_iter, current_iter)

            # compute deformation field. regi model is
            # already set_requires_grad(False)
            moving_img, fixed_img = img_unlab_t_plus_1, img_unlab_t
            img = torch.cat([moving_img, fixed_img], dim=1)
            flow = self.regi_model.output2flow(self.regi_model.unet(img))

            # registration related loss (just for inspection and debugging)
            wraped_img = self.regi_model.spatial_transformer(moving_img, flow)

            vm_mse_w_regi = vm_losses.MSE().loss(wraped_img, fixed_img)
            vm_mse_wo_regi = vm_losses.MSE().loss(moving_img, fixed_img)
            vm_grad = vm_losses.Grad("l2").loss(None, flow)

            # a wraped output of moving image logits map
            wraped_outputs_unlab_t_plus_1 = self.regi_model.spatial_transformer(
                outputs_unlab_t_plus_1, flow
            )

            # temporal consistency is applied to unlabeled t and unlabeled t + 1
            loss_consistency_t = self.cr_criterion(
                ori_outputs_unlab_t, wraped_outputs_unlab_t_plus_1
            )

            # model train step will sum up all the losses with "loss" in name
            return {
                "loss_sup": loss_sup,
                "loss_consistency_lambda": loss_consistency * this_lambda,
                "loss_consistency_temp_lambda": loss_consistency_t * this_lambda_t,
                "lambda": torch.tensor(this_lambda),
                "lambda_t": torch.tensor(this_lambda_t),
                "L_sup": loss_sup,
                "L_consistency": loss_consistency,
                "L_consistency_t": loss_consistency_t,
                "L_vm_mse_w_regi": vm_mse_w_regi,
                "L_vm_mse_wo_regi": vm_mse_wo_regi,
                "L_vm_grad": vm_grad,
            }

        elif mode == "predict":
            outputs = self.model(img)
            preds = torch.argmax(outputs, dim=1)
            return preds

        elif mode == "predict_and_logits":
            outputs = self.model(img)
            preds = torch.argmax(outputs, dim=1)
            return preds, outputs


LAMBDA_SCHEDULE = Registry(
    "lambda_schedule",
    scope="mmengine",
    locations=["cr_seg.models.consistency_regularization"],
)


@LAMBDA_SCHEDULE.register_module()
class ConstantSchedule:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, max_iter, current_iter):
        return float(self.value)


@LAMBDA_SCHEDULE.register_module()
class LinearSchedule:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def __call__(self, max_iter, current_iter):
        return self.start + (self.end - self.start) * current_iter / max_iter


@LAMBDA_SCHEDULE.register_module()
class ExpRampUpConstantSchedule:
    def __init__(self, value: float, ramp_up_total_ratio: float):
        self.value = value
        self.ramp_up_total_ratio = ramp_up_total_ratio

    def __call__(self, max_iter, current_iter):
        ramp_up_iter = int(self.ramp_up_total_ratio * max_iter)
        # NOTE(YL 05/29):: the following 4 line is from copilot
        # I am a bit worried because I never specify the detail of
        # how the ramp up stage will look like but copilot imputed
        # exactly the same formula as specified in the paper.
        # for example, I don't know how it came up with the -5 and ** 2 ...
        if current_iter < ramp_up_iter:
            return self.value * np.exp(-5 * (1 - current_iter / ramp_up_iter) ** 2)
        else:
            return self.value


CR_CRITERION = Registry(
    "cr_criterion",
    scope="mmengine",
    locations=["cr_seg.models.consistency_regularization"],
)


@CR_CRITERION.register_module()
class L2Norm(torch.nn.Module):
    def forward(self, trans_outputs, ori_outputs_trans):
        return torch.mean((trans_outputs - ori_outputs_trans) ** 2)


@CR_CRITERION.register_module()
class SymmetricCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SymmetricCrossEntropy, self).__init__()
        self.log_soft_max = torch.nn.LogSoftmax(dim=-1)
        self.soft_max = torch.nn.Softmax(dim=-1)

    def forward(self, trans_outputs, ori_outputs_trans):
        l1 = -torch.mean(
            self.soft_max(trans_outputs) * self.log_soft_max(ori_outputs_trans)
        )
        l2 = -torch.mean(
            self.soft_max(ori_outputs_trans) * self.log_soft_max(trans_outputs)
        )
        return 0.5 * (l1 + l2)


# notably, the randomness of these transformations are
# created at the time of instantiation. And after that, they
# are fixed and deterministic. so that we can apply the
# same transformation to both the image and the label

DIFF_TRANS = Registry(
    "diff_trans",
    scope="mmengine",
    locations=["cr_seg.models.consistency_regularization"],
)


class _ComposedDiffTrans:
    def __init__(self, func_list: list[tp.Callable]):
        self.func_list = func_list

    def _call_img(self, x):
        for func in self.func_list:
            x = func._call_img(x)
        return x

    def _call_seg(self, x):
        for func in self.func_list:
            x = func._call_seg(x)
        return x


def _transform_generator(func_cfg_list: list[dict]):
    func_list = [DIFF_TRANS.build(cfg) for cfg in func_cfg_list]
    return _ComposedDiffTrans(func_list)


class _Transform:
    def __call__(self, x):
        raise NotImplementedError

    def _call_img(self, x):
        return self.__call__(x)

    def _call_seg(self, x):
        return self.__call__(x)


@DIFF_TRANS.register_module()
class Identity(_Transform):
    def __call__(self, x):
        return x


@DIFF_TRANS.register_module()
class RandomRot903d(_Transform):
    def __init__(self, axis: tuple[int, int] = (1, 2), k: int = 3):
        self.axis = axis
        self.k = k

        self._init_randomness()

    def _init_randomness(self):
        self.k_rot = int(torch.randint(0, self.k, (1,)).item())

    def __call__(self, x):
        # x: (C, H, W, D)
        return torch.rot90(x, self.k_rot, self.axis)


@DIFF_TRANS.register_module()
class RamdomFlip3d(_Transform):
    def __init__(self, axis_list: tuple = (1, 2, 3), p: float = 0.5):
        self.axis_list = axis_list
        self.p = p

        self._init_randomness(p=self.p)

    def _init_randomness(self, p: float):
        self.do_flip_list = [torch.rand(1).item() < p for _ in self.axis_list]

    def __call__(self, x):
        # x: (C, H, W, D)
        for i, do_flip in enumerate(self.do_flip_list):
            if do_flip:
                x = torch.flip(x, dims=[i])
        return x


@DIFF_TRANS.register_module()
class RandomRot3d(_Transform):
    def __init__(
        self, max_angle: int = 45, interpolation: str = "bilinear", p: float = 0.5
    ):
        self.max_angle = max_angle
        self.interpolation = interpolation

        self.p = p
        self._init_randomness(p=self.p)

        # NOTE(YL 05/17):: this rotation will not work for
        # integer segmentation maps (it works on logits maps)
        if interpolation == "bilinear":
            self._func = partial(
                TF.rotate,
                angle=self.rot_angle,
                interpolation=InterpolationMode.BILINEAR,
            )
        else:
            raise NotImplementedError

    def _init_randomness(self, p: float):
        self.do_rot = torch.rand(1).item() < p
        self.rot_angle = torch.randint(-self.max_angle, self.max_angle, (1,)).item()

    def __call__(self, x):
        # x: (C, H, W, D)
        if self.do_rot:
            x = einops.rearrange(x, "c h w d -> c d h w")
            x = self._func(x)
            x = einops.rearrange(x, "c d h w -> c h w d")
        return x


@DIFF_TRANS.register_module()
class RandomTranslation3d(_Transform):
    def __init__(self, max_shift_pix: tuple[int, int], p: float = 0.5):
        # TODO(YL 07/31):: 3d version
        # implement the translation along h w plane

        self.max_shift_pix = max_shift_pix
        self.p = p

        self._init_randomness(p=self.p)

        self._func = partial(
            TF.affine,
            angle=0,
            translate=self.shift_pix,
            scale=1,
            shear=0.0,
            interpolation=InterpolationMode.BILINEAR,
        )

    def _init_randomness(self, p: float):
        self.do_shift = torch.rand(1).item() < p

        self.shift_pix = [
            torch.randint(-self.max_shift_pix[0], self.max_shift_pix[0], (1,)).item(),
            torch.randint(-self.max_shift_pix[1], self.max_shift_pix[1], (1,)).item(),
        ]

    def __call__(self, x):
        # x: (C, H, W, D)
        if self.do_shift:
            x = einops.rearrange(x, "c h w d -> c d h w")
            x = self._func(x)
            x = einops.rearrange(x, "c d h w -> c h w d")
        return x


@DIFF_TRANS.register_module()
class RandomGamma(_Transform):
    def __init__(self, gamma_range: tuple = (0.5, 4.5), p: float = 0.5):
        self.gamma_range = gamma_range
        self.p = p

        self._init_randomness(p=self.p)

    def _init_randomness(self, p: float):
        self.do_gamma = torch.rand(1).item() < p
        self.gamma = (
            torch.rand(1).item() * (self.gamma_range[1] - self.gamma_range[0])
            + self.gamma_range[0]
        )

    def _call_img(self, x):
        if self.do_gamma:
            epsilon = 1e-7
            x_min = x.min()
            x_range = x.max() - x_min
            x = ((x - x_min) / float(x_range + epsilon)) ** self.gamma * x_range + x_min
        return x

    def _call_seg(self, x):
        return x


@DIFF_TRANS.register_module()
class RandomGaussianNoise(_Transform):
    def __init__(self, sigma_range: tuple = (0.0, 0.1), p: float = 0.1):
        self.sigma_range = sigma_range
        self.p = p

        self._init_randomness(p=self.p)

        self.noise_matrix = None

    def _init_randomness(self, p: float):
        self.do_gaussian_noise = torch.rand(1).item() < p
        self.sigma = torch.rand(1).item() * (self.sigma_range[1] - self.sigma_range[0])

    def _call_img(self, x):
        if self.do_gaussian_noise:
            if self.noise_matrix is not None:
                noise_matrix = self.noise_matrix
                assert (
                    noise_matrix.shape == x.shape
                ), f"noise_matrix.shape: {noise_matrix.shape}, x.shape: {x.shape}"
            else:
                noise_matrix = torch.normal(
                    mean=0.0, std=self.sigma, size=x.shape, device=x.device
                )
                self.noise_matrix = noise_matrix
            x = x + noise_matrix
        return x

    def _call_seg(self, x):
        return x
