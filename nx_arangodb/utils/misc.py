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

try:
    from itertools import pairwise  # Python >=3.10
except ImportError:

    def pairwise(it):
        it = iter(it)
        for prev in it:
            for cur in it:
                yield (prev, cur)
                prev = cur


__all__ = [
    "index_dtype",
    "_seed_to_int",
    "_get_int_dtype",
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


def _seed_to_int(seed: int | Random | None) -> int:
    """Handle any valid seed argument and convert it to an int if necessary."""
    if seed is None:
        return
    if isinstance(seed, Random):
        return seed.randint(0, sys.maxsize)
    return op.index(seed)  # Ensure seed is integral


def _get_int_dtype(
    val: SupportsIndex, *, signed: bool | None = None, unsigned: bool | None = None
):
    """Determine the smallest integer dtype that can store the integer ``val``.

    If signed or unsigned are unspecified, then signed integers are preferred
    unless the value can be represented by a smaller unsigned integer.

    Raises
    ------
    ValueError : If the value cannot be represented with an int dtype.
    """
    # This is similar in spirit to `np.min_scalar_type`
    if signed is not None:
        if unsigned is not None and (not signed) is (not unsigned):
            raise TypeError(
                f"signed (={signed}) and unsigned (={unsigned}) keyword arguments "
                "are incompatible."
            )
        signed = bool(signed)
        unsigned = not signed
    elif unsigned is not None:
        unsigned = bool(unsigned)
        signed = not unsigned

    val = op.index(val)  # Ensure val is integral
    if val < 0:
        if unsigned:
            raise ValueError(f"Value is incompatible with unsigned int: {val}.")
        signed = True
        unsigned = False

    if signed is not False:
        # Number of bytes (and a power of two)
        signed_nbytes = (val + (val < 0)).bit_length() // 8 + 1
        signed_nbytes = next(
            filter(
                signed_nbytes.__le__,
                itertools.accumulate(itertools.repeat(2), op.mul, initial=1),
            )
        )
    if unsigned is not False:
        # Number of bytes (and a power of two)
        unsigned_nbytes = (val.bit_length() + 7) // 8
        unsigned_nbytes = next(
            filter(
                unsigned_nbytes.__le__,
                itertools.accumulate(itertools.repeat(2), op.mul, initial=1),
            )
        )
        if signed is None and unsigned is None:
            # Prefer signed int if same size
            signed = signed_nbytes <= unsigned_nbytes

    if signed:
        dtype_string = f"i{signed_nbytes}"
    else:
        dtype_string = f"u{unsigned_nbytes}"
    try:
        return np.dtype(dtype_string)
    except TypeError as exc:
        raise ValueError("Value is too large to store as integer: {val}") from exc
