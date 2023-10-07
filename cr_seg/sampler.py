#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : sampler.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 05/17/2023
#
# This file is part of cr_seg

"""

"""

import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class MaxNumSampler(Sampler):
    """The data sampler that limits the number of samples.

    The default data sampler for both distributed and non-distributed
    environment.

    It has several differences from the PyTorch ``DistributedSampler`` as
    below:

    1. This sampler supports non-distributed environment.

    2. The round up behaviors are a little different.

       - If ``round_up=True``, this sampler will add extra samples to make the
         number of samples is evenly divisible by the world size. And
         this behavior is the same as the ``DistributedSampler`` with
         ``drop_last=False``.
       - If ``round_up=False``, this sampler won't remove or add any samples
         while the ``DistributedSampler`` with ``drop_last=True`` will remove
         tail samples.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(
        self,
        dataset: Sized,
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
        max_num_samples: int = 1000000,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up
        if not self.round_up:
            raise NotImplementedError

        if max_num_samples > len(self.dataset):
            raise ValueError(
                f"max_num_samples ({max_num_samples}) is larger than the "
                f"dataset length ({len(self.dataset)})!"
            )

        self.num_samples = math.ceil(len(self.dataset) / world_size)
        self.total_size = self.num_samples * self.world_size

        self.num_samples_limited = min(
            math.ceil(max_num_samples / world_size), self.num_samples
        )

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices = (indices * int(self.total_size / len(indices) + 1))[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.world_size]
        indices = indices[: self.num_samples_limited]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        # return self.num_samples
        return self.num_samples_limited

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
