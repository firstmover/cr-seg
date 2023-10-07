#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : hooks.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 05/29/2023
#
# This file is part of cr_seg

"""

"""

from typing import Optional

import torch

from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval: int = 50) -> None:
        self.interval = interval

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[dict] = None,
    ) -> None:
        """Regularly check whether the loss is valid every n iterations.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict, Optional): Data from dataloader.
                Defaults to None.
            outputs (dict, Optional): Outputs from model. Defaults to None.
        """
        if self.every_n_train_iters(runner, self.interval):
            assert torch.isfinite(outputs["loss"]), runner.logger.info(
                "loss become infinite or NaN!"
            )


@HOOKS.register_module()
class AddMaxCurIterToDataHook(Hook):
    priority = "NORMAL"

    def before_train_iter(
        self, runner, batch_idx: int, data_batch: DATA_BATCH = None
    ) -> None:
        """This hook will add the max / current iteration
        number to the data batch so that the model can adjust
        the lambda based on the iteration.

        All subclasses should override this method, if they need any
        operations before each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode="train"
        )

        data_batch["max_iter"] = runner.max_iters  # type: ignore
        data_batch["current_iter"] = runner.iter  # type: ignore
