# Copied from nx-cugraph

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Dict, TypeVar

import numpy.typing as npt

from nx_arangodb.logger import logger

try:
    import cupy as cp
except ModuleNotFoundError as e:
    GPU_ENABLED = False
    logger.info(f"NXCG is disabled. {e}.")


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
