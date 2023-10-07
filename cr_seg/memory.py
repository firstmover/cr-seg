#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File   : memory.py
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 02/22/2023
#
# This file is part of cr_seg

"""

"""

import os

from joblib import Memory

joblib_cache = os.getenv("JOBLIB_CACHE")

memory = Memory(
    location=joblib_cache if joblib_cache is not None else "./.cache",
    verbose=0,
)
