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

# AdjDict is a dictionary of dictionaries of dictionaries
# The outer dict is holding _from_id(s) as keys
#  - It may or may not hold valid ArangoDB document _id(s)
# The inner dict is holding _to_id(s) as keys
#  - It may or may not hold valid ArangoDB document _id(s)
# The next inner dict contains then the actual edges data (key, val)
# Example
# {
#    'person/1': {
#        'person/32': {
#            '_id': 'knows/16',
#            'extraValue': '16'
#        },
#        'person/33': {
#            '_id': 'knows/17',
#            'extraValue': '17'
#        }
#        ...
#    }
#    ...
# }
# The above example is a graph with 2 edges from person/1 to person/32 and person/33
AdjDictEdge = Dict[str, Any]
AdjDictInner = Dict[str, AdjDictEdge]
AdjDict = Dict[str, AdjDictInner]


class any_ndarray:
    def __class_getitem__(cls, item):
        return cp.ndarray[item] | npt.NDArray[item]
