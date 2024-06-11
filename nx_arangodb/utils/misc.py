# Copied from nx-cugraph
from __future__ import annotations

import itertools
import operator as op
import sys
from random import Random
from typing import TYPE_CHECKING, SupportsIndex

# import cupy as cp
import numpy as np

if TYPE_CHECKING:
    # import nx_cugraph as nxcg

    from ..typing import Dtype, EdgeKey  # noqa

__all__ = [
    "index_dtype",
    "_dtype_param",
]

# This may switch to np.uint32 at some point
index_dtype = np.int32

# To add to `extra_params=` of `networkx_algorithm`
_dtype_param = {
    "dtype : dtype or None, optional": (
        "The data type (np.float32, np.float64, or None) to use for the edge weights "
        "in the algorithm. If None, then dtype is determined by the edge values."
    ),
}
