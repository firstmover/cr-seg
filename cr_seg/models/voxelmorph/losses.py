import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from mmengine import Registry

from cr_seg.metric import (
    batch_compute_local_cosine_similarity,
    batch_compute_local_zero_mean_cosine_similarity_v2,
)

VOXELMORPH_LOSS = Registry(
    "voxelmorph_loss",
    scope="mmengine",
    locations=["cr_seg.models.voxelmorph.losses"],
)


@VOXELMORPH_LOSS.register_module()
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def loss(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], (
            "volumes should be 1 to 3 dimensions. found: %d" % ndims
        )

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = 1
            padding = pad_no
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        return -torch.mean(cc)


@VOXELMORPH_LOSS.register_module()
class LNCC:
    def __init__(
        self,
        window_size: tuple[int, int, int] = (5, 5, 5),
        eps: float = 1e-5,
        zero_mean: bool = True,
        return_loss_map: bool = False,
    ):
        self.window_size = window_size
        self.eps = eps
        self.zero_mean = zero_mean
        self.return_loss_map = return_loss_map

    def loss(self, y_true, y_pred):
        f_lncc = partial(
            batch_compute_local_zero_mean_cosine_similarity_v2
            if self.zero_mean
            else batch_compute_local_cosine_similarity,
            window_size=self.window_size,
            eps=self.eps,
            return_mean=not self.return_loss_map,
            padding="same",
        )

        if self.return_loss_map:
            # TODO(YL 09/26):: fix this
            with torch.cuda.amp.autocast(enabled=False):
                sim, is_valid = f_lncc(y_true, y_pred)
            return -sim, is_valid
        else:
            with torch.cuda.amp.autocast(enabled=False):
                mean_lncc = torch.mean(f_lncc(y_true, y_pred))
            return -mean_lncc


@VOXELMORPH_LOSS.register_module()
class MSE:
    """
    Mean squared error loss.
    """

    def __init__(self, return_loss_map=False):
        self.return_loss_map = return_loss_map

    def loss(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        if self.return_loss_map:
            loss_map = (y_true - y_pred) ** 2
            is_valid = ~torch.isnan(loss_map)
            return loss_map, is_valid
        else:
            return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty="l1", loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [
                *range(d - 1, d + 1),
                *reversed(range(1, d - 1)),
                0,
                *range(d + 1, ndims + 2),
            ]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == "l1":
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == "l2", (
                "penalty can only be l1 or l2. Got: %s" % self.penalty
            )
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
