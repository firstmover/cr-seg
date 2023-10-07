#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : utils.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 02/15/2023
#
# This file is part of cr_seg

"""

"""


def register_all_modules(init_default_scope: bool = True):
    """Register all modules in mmselfsup into the registries.
    Args:
        init_default_scope (bool): Whether initialize the mmselfsup default
            scope. When `init_default_scope=True`, the global default scope
            will be set to `mmselfsup`, and all registries will build modules
            from mmselfsup's registry node. To understand more about the
            registry, please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa

    from cr_seg import hooks  # noqa: F401,F403
    from cr_seg import loop  # noqa: F401,F403
    from cr_seg import metric  # noqa: F401,F403
    from cr_seg import sampler  # noqa: F401,F403
    from cr_seg import transform  # noqa: F401,F403
    from cr_seg.datasets.placenta import consistency_regularization  # noqa: F401,F403
    from cr_seg.datasets.placenta import registration  # noqa: F401,F403
    from cr_seg.datasets.placenta import vanilla_2d_3d  # noqa: F401,F403
    from cr_seg.models import consistency_regularization as model_cr  # noqa: F401,F403
    from cr_seg.models import inference  # noqa: F401,F403
    from cr_seg.models import unet_3d_nobatchnorm  # noqa: F401,F403
    from cr_seg.models.voxelmorph import build  # noqa: F401,F403
