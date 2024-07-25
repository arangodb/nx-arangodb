# Copied from nx-cugraph

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Dict, TypeVar

import cupy as cp
import numpy as np
import numpy.typing as npt

AttrKey = TypeVar("AttrKey", bound=Hashable)
EdgeKey = TypeVar("EdgeKey", bound=Hashable)
NodeKey = TypeVar("NodeKey", bound=Hashable)
EdgeTuple = tuple[NodeKey, NodeKey]
EdgeValue = TypeVar("EdgeValue")
NodeValue = TypeVar("NodeValue")
IndexValue = TypeVar("IndexValue")
Dtype = TypeVar("Dtype")
AdjDict = Dict[str, Dict[str, Dict[str, Any]]]


class any_ndarray:
    def __class_getitem__(cls, item):
        return cp.ndarray[item] | npt.NDArray[item]
