# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import numpy as np


def autocorrelation(x: np.ndarray, p: int, normalized: bool = True) -> float:
    if p == 0:
        return 1.0
    mean = x.mean()
    if normalized:
        return ((x[:-p] - mean) * (x[p:] - mean)).mean() / x.var()
    return ((x[:-p] - mean) * (x[p:] - mean)).sum() / np.square(x - mean).sum()
