#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : loop.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 08/24/2023
#
# This file is part of cr_seg
"""
Copied from https://github.com/open-mmlab/mmrazor/blob/
main/mmrazor/engine/runner/darts_loop.py
"""
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

from mmengine.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop


@LOOPS.register_module()
class LabeledUnlabeledEpochBasedTrainLoop(EpochBasedTrainLoop):
    """EpochBasedTrainLoop for `Darts <https://arxiv.org/abs/1806.09055>`_

    In Darts, Two dataloaders are needed in the training stage. One
    (`dataloader`) is used to train the supernet and update its weights,
    another(`mutator_dataloader`) is only used to train and update the
    parameters of the supernet's architecture setting. In
    `DartsEpochBasedTrainLoop`, these dataloaders will be combined as a
    special dataloader, whose `data_batch` will contain both of the
    dataloaders' `data_batch`.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or Dict):
            A dataloader object or a dict to build a dataloader for
            training the model.
        mutator_dataloader (Dataloader or Dict):
            A dataloader object or a dict to build a dataloader for
            training the parameters of model architecture.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
    """

    def __init__(
        self,
        runner,
        dataloader: Union[Dict, DataLoader],
        unlabeled_dataloader: Union[Dict, DataLoader],
        max_epochs: int,
        val_begin: int = 1,
        val_interval: int = 1,
    ) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin, val_interval)
        if isinstance(unlabeled_dataloader, dict):
            self.unlabeled_dataloader = runner.build_dataloader(
                unlabeled_dataloader, seed=runner.seed
            )
        else:
            self.unlabeled_dataloader = unlabeled_dataloader

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()

        for idx, (labeled_data_batch, unlabeled_data_batch) in enumerate(
            EpochMultiLoader([self.dataloader, self.unlabeled_dataloader])
        ):
            # we stack the unlabeled paired img with labeled img
            # along the channel dimension to keep the model
            # interface clean and consistent

            labeled_img = labeled_data_batch["img"]
            unlabeled_img = unlabeled_data_batch["img"]

            batch_size = labeled_img.shape[0]

            # drop_last = True will make sure all gpu have the
            # same batch size but the last batch may be smaller
            if batch_size != unlabeled_img.shape[0]:
                assert unlabeled_img.shape[0] > batch_size
                unlabeled_img = unlabeled_img[:batch_size]

            assert labeled_img.shape[1] == 1
            assert (
                unlabeled_img.shape[1] == 2
            ), f"unlabeled img should be paired, got shape {unlabeled_img.shape}"

            data_batch = labeled_data_batch
            data_batch["img"] = torch.cat([labeled_img, unlabeled_img], dim=1)

            self.run_iter(idx, data_batch)

        self.runner.call_hook("after_train_epoch")
        self._epoch += 1


class EpochMultiLoader:
    """Multi loaders based on epoch."""

    def __init__(self, dataloaders: List[DataLoader]):
        self._dataloaders = dataloaders
        self.iter_loaders = [iter(loader) for loader in self._dataloaders]

    @property
    def num_loaders(self):
        """The number of dataloaders."""
        return len(self._dataloaders)

    def __iter__(self):
        """Return self when executing __iter__."""
        return self

    def __next__(self):
        """Get the next iter's data of multiple loaders."""
        data = tuple([next(loader) for loader in self.iter_loaders])

        return data

    def __len__(self):
        """Get the length of loader."""
        return min([len(loader) for loader in self._dataloaders])
