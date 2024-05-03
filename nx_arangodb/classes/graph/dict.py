from __future__ import annotations

from collections import UserDict
from collections.abc import Iterator
from typing import Any, Callable

from arango.database import StandardDatabase
from arango.graph import Graph

import nx_arangodb as nxadb


def adjlist_outer_dict_factory(
    db: StandardDatabase,
    graph: Graph,
    default_node_type: str,
    edge_type_func: Callable[[str, str], str],
) -> Callable[..., AdjListOuterDict]:
    return lambda: AdjListOuterDict(
        db,
        graph,
        default_node_type,
        edge_type_func,
    )


def adjlist_inner_dict_factory(
    db: StandardDatabase,
    graph: Graph,
    default_node_type: str,
    edge_type_func: Callable[[str, str], str],
) -> Callable[..., AdjListInnerDict]:
    return lambda: AdjListInnerDict(db, graph, default_node_type, edge_type_func)


def edge_attr_dict_factory(
    db: StandardDatabase, graph: Graph
) -> Callable[..., EdgeAttrDict]:
    return lambda: EdgeAttrDict(db, graph)
