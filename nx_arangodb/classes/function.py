"""
A collection of CRUD functions for the ArangoDB graph database.
Used by the nx_arangodb Graph, DiGraph, MultiGraph, and MultiDiGraph classes.
"""

from __future__ import annotations

from collections import UserDict
from typing import Any, Callable, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
from arango.collection import StandardCollection
from arango.cursor import Cursor
from arango.database import StandardDatabase
from arango.graph import Graph
from phenolrs.networkx import NetworkXLoader
from phenolrs.networkx.typings import (
    ArangoIDtoIndex,
    DiGraphAdjDict,
    DstIndices,
    EdgeIndices,
    GraphAdjDict,
    MultiDiGraphAdjDict,
    MultiGraphAdjDict,
    NodeDict,
    SrcIndices,
)

import nx_arangodb as nxadb
from nx_arangodb.logger import logger

from ..exceptions import (
    AQLMultipleResultsFound,
    GraphDoesNotExist,
    InvalidTraversalDirection,
)


def get_arangodb_graph(
    adb_graph: Graph,
    load_node_dict: bool,
    load_adj_dict: bool,
    load_coo: bool,
    load_all_vertex_attributes: bool,
    load_all_edge_attributes: bool,
    is_directed: bool,
    is_multigraph: bool,
    symmetrize_edges_if_directed: bool,
) -> Tuple[
    NodeDict,
    GraphAdjDict | DiGraphAdjDict | MultiGraphAdjDict | MultiDiGraphAdjDict,
    SrcIndices,
    DstIndices,
    EdgeIndices,
    ArangoIDtoIndex,
]:
    """Pulls the graph from the database, assuming the graph exists.

    Returns the folowing representations:
    - Node dictionary (nx.Graph)
    - Adjacency dictionary (nx.Graph)
    - Source Indices (COO)
    - Destination Indices (COO)
    - Node-ID-to-index mapping (COO)
    """
    v_cols = adb_graph.vertex_collections()
    edge_definitions = adb_graph.edge_definitions()
    e_cols = {c["edge_collection"] for c in edge_definitions}

    metagraph: dict[str, dict[str, Any]] = {
        "vertexCollections": {col: set() for col in v_cols},
        "edgeCollections": {col: set() for col in e_cols},
    }

    if not any((load_node_dict, load_adj_dict, load_coo)):
        raise ValueError("At least one of the load flags must be True.")

    if not load_node_dict:
        metagraph["vertexCollections"] = {}

    if not load_adj_dict and not load_coo:
        metagraph["edgeCollections"] = {}

    config = nx.config.backends.arangodb

    kwargs = {}
    if parallelism := config.get("read_parallelism"):
        kwargs["parallelism"] = parallelism
    if batch_size := config.get("read_batch_size"):
        kwargs["batch_size"] = batch_size

    assert config.db_name
    assert config.host
    assert config.username
    assert config.password

    (
        node_dict,
        adj_dict,
        src_indices,
        dst_indices,
        edge_indices,
        vertex_ids_to_index,
        edge_values,
    ) = NetworkXLoader.load_into_networkx(
        config.db_name,
        metagraph=metagraph,
        hosts=[config.host],
        username=config.username,
        password=config.password,
        load_adj_dict=load_adj_dict,
        load_coo=load_coo,
        load_all_vertex_attributes=load_all_vertex_attributes,
        load_all_edge_attributes=load_all_edge_attributes,
        is_directed=is_directed,
        is_multigraph=is_multigraph,
        symmetrize_edges_if_directed=symmetrize_edges_if_directed,
        **kwargs,
    )

    return (
        node_dict,
        adj_dict,
        src_indices,
        dst_indices,
        edge_indices,
        vertex_ids_to_index,
    )


def json_serializable(cls):
    def to_dict(self):
        return {
            key: dict(value) if isinstance(value, cls) else value
            for key, value in self.data.items()
        }

    cls.to_dict = to_dict
    return cls


def key_is_string(func: Callable[..., Any]) -> Any:
    """Decorator to check if the key is a string."""

    def wrapper(self: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
        """"""
        if not isinstance(key, str):
            if not isinstance(key, (int, float)):
                raise TypeError(f"{key} cannot be casted to string.")

            key = str(key)

        return func(self, key, *args, **kwargs)

    return wrapper


def key_is_int(func: Callable[..., Any]) -> Any:
    """Decorator to check if the key is an integer."""

    def wrapper(self: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
        """"""
        if not isinstance(key, int):
            raise TypeError(f"{key} must be an integer.")

        return func(self, key, *args, **kwargs)

    return wrapper


def key_is_adb_id_or_int(func: Callable[..., Any]) -> Any:
    """Decorator to check if the key is an ArangoDB ID."""

    def wrapper(self: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
        """"""
        if isinstance(key, str):
            if key != "-1" and "/" not in key:
                raise ValueError(f"{key} is not an ArangoDB ID.")

        elif isinstance(key, int):
            m = "Edge order is not guaranteed when using int as an edge key. It may raise a KeyError. Use at your own risk."  # noqa
            logger.warning(m)

        else:
            raise TypeError(f"{key} is not an ArangoDB Edge _id or integer.")

        return func(self, key, *args, **kwargs)

    return wrapper


def logger_debug(func: Callable[..., Any]) -> Any:
    """Decorator to log debug messages."""

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger.debug(f"{type(self)}.{func.__name__} - {args} - {kwargs}")
        return func(self, *args, **kwargs)

    return wrapper


def keys_are_strings(func: Callable[..., Any]) -> Any:
    """Decorator to check if the keys are strings."""

    def wrapper(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        data_dict = {}

        items: Any
        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, zip):
            items = list(data)
        else:
            raise TypeError(f"Decorator found unsupported type: {type(data)}.")

        for key, value in items:
            if not isinstance(key, str):
                if not isinstance(key, (int, float)):
                    raise TypeError(f"{key} cannot be casted to string.")

                key = str(key)

            data_dict[key] = value

        return func(self, data_dict, *args, **kwargs)

    return wrapper


RESERVED_KEYS = {"_id", "_key", "_rev"}


def key_is_not_reserved(func: Callable[..., Any]) -> Any:
    """Decorator to check if the key is not reserved."""

    def wrapper(self: Any, key: str, *args: Any, **kwargs: Any) -> Any:
        if key in RESERVED_KEYS:
            raise KeyError(f"'{key}' is a reserved key.")

        return func(self, key, *args, **kwargs)

    return wrapper


def keys_are_not_reserved(func: Any) -> Any:
    """Decorator to check if the keys are not reserved."""

    def wrapper(
        self: Any, data: dict[Any, Any] | zip[Any], *args: Any, **kwargs: Any
    ) -> Any:
        keys: Any
        if isinstance(data, dict):
            keys = data.keys()
        elif isinstance(data, zip):
            keys = (key for key, _ in list(data))
        else:
            raise TypeError(f"Decorator found unsupported type: {type(data)}.")

        for key in keys:
            if key in RESERVED_KEYS:
                raise KeyError(f"'{key}' is a reserved key.")

        return func(self, data, *args, **kwargs)

    return wrapper


def create_collection(
    db: StandardDatabase, collection_name: str, edge: bool = False
) -> StandardCollection:
    """Creates a collection if it does not exist and returns it."""
    if not db.has_collection(collection_name):
        db.create_collection(collection_name, edge=edge)

    return db.collection(collection_name)


def aql(
    db: StandardDatabase, query: str, bind_vars: dict[str, Any], **kwargs: Any
) -> Cursor:
    """Executes an AQL query and returns the cursor."""
    return db.aql.execute(query, bind_vars=bind_vars, stream=True, **kwargs)


def aql_as_list(
    db: StandardDatabase, query: str, bind_vars: dict[str, Any], **kwargs: Any
) -> list[Any]:
    """Executes an AQL query and returns the results as a list."""
    return list(aql(db, query, bind_vars, **kwargs))


def aql_single(
    db: StandardDatabase, query: str, bind_vars: dict[str, Any]
) -> Any | None:
    """Executes an AQL query and returns the first result."""
    result = aql_as_list(db, query, bind_vars)
    if len(result) == 0:
        return None

    if len(result) > 1:
        raise AQLMultipleResultsFound(f"Multiple results found: {result}")

    return result[0]


def aql_doc_has_key(
    db: StandardDatabase, id: str, key: str, nested_keys: list[str] = []
) -> bool:
    """Checks if a document has a key."""
    nested_keys_str = "." + ".".join(nested_keys) if nested_keys else ""
    query = f"RETURN HAS(DOCUMENT(@id){nested_keys_str}, @key)"
    bind_vars = {"id": id, "key": key}
    result = aql_single(db, query, bind_vars)
    return bool(result) if result is not None else False


def aql_doc_get_key(
    db: StandardDatabase, id: str, key: str, nested_keys: list[str] = []
) -> Any:
    """Gets a key from a document."""
    nested_keys_str = "." + ".".join(nested_keys) if nested_keys else ""
    query = f"RETURN DOCUMENT(@id){nested_keys_str}.@key"
    bind_vars = {"id": id, "key": key}
    return aql_single(db, query, bind_vars)


def aql_doc_get_items(
    db: StandardDatabase, id: str, nested_key: list[str] = []
) -> dict[str, Any]:
    """Gets the items of a document."""
    nested_key_str = "." + ".".join(nested_key) if nested_key else ""
    query = f"RETURN DOCUMENT(@id){nested_key_str}"
    bind_vars = {"id": id}
    result = aql_single(db, query, bind_vars)
    return result or {}


def aql_doc_get_keys(
    db: StandardDatabase, id: str, nested_keys: list[str] = []
) -> list[str]:
    """Gets the keys of a document."""
    nested_keys_str = "." + ".".join(nested_keys) if nested_keys else ""
    query = f"RETURN ATTRIBUTES(DOCUMENT(@id){nested_keys_str})"
    bind_vars = {"id": id}
    result = aql_single(db, query, bind_vars)
    return list(result or [])


def aql_doc_get_length(
    db: StandardDatabase, id: str, nested_keys: list[str] = []
) -> int:
    """Gets the length of a document."""
    nested_keys_str = "." + ".".join(nested_keys) if nested_keys else ""
    query = f"RETURN LENGTH(DOCUMENT(@id){nested_keys_str})"
    bind_vars = {"id": id}
    result = aql_single(db, query, bind_vars)
    return int(result or 0)


def aql_edge_exists(
    db: StandardDatabase,
    src_node_id: str,
    dst_node_id: str,
    graph_name: str,
    direction: str,
) -> bool | None:
    return aql_edge(
        db,
        src_node_id,
        dst_node_id,
        graph_name,
        direction,
        return_clause="true",
        limit_one=True,
        can_return_multiple=False,
    )


def aql_edge_get(
    db: StandardDatabase,
    src_node_id: str,
    dst_node_id: str,
    graph_name: str,
    direction: str,
    can_return_multiple: bool = False,
) -> Any | None:
    return_clause = "DISTINCT e" if direction == "ANY" else "e"
    return aql_edge(
        db,
        src_node_id,
        dst_node_id,
        graph_name,
        direction,
        return_clause=return_clause,
        limit_one=False,
        can_return_multiple=can_return_multiple,
    )


def aql_edge_id(
    db: StandardDatabase,
    src_node_id: str,
    dst_node_id: str,
    graph_name: str,
    direction: str,
    can_return_multiple: bool = False,
) -> Any | None:
    return_clause = "DISTINCT e._id" if direction == "ANY" else "e._id"
    return aql_edge(
        db,
        src_node_id,
        dst_node_id,
        graph_name,
        direction,
        return_clause=return_clause,
        limit_one=False,
        can_return_multiple=can_return_multiple,
    )


def aql_edge_count_src(
    db: StandardDatabase,
    src_node_id: str,
    graph_name: str,
    direction: str,
) -> int:
    query = f"""
        RETURN LENGTH(
            FOR v, e IN 1..1 {direction} @src_node_id GRAPH @graph_name
                RETURN DISTINCT e._id
        )
    """

    bind_vars = {
        "src_node_id": src_node_id,
        "graph_name": graph_name,
    }

    result = aql_single(db, query, bind_vars)

    return int(result) if result is not None else 0


def aql_edge_count_src_dst(
    db: StandardDatabase,
    src_node_id: str,
    dst_node_id: str,
    graph_name: str,
    direction: str,
) -> int:
    filter_clause = aql_edge_direction_filter(direction)

    query = f"""
        FOR v, e IN 1..1 {direction} @src_node_id GRAPH @graph_name
            FILTER {filter_clause}
            COLLECT WITH COUNT INTO length
            RETURN length
    """

    bind_vars = {
        "src_node_id": src_node_id,
        "dst_node_id": dst_node_id,
        "graph_name": graph_name,
    }

    result = aql_single(db, query, bind_vars)

    return int(result) if result is not None else 0


def aql_edge_direction_filter(direction: str) -> str:
    if direction == "INBOUND":
        return "e._from == @dst_node_id"
    if direction == "OUTBOUND":
        return "e._to == @dst_node_id"
    if direction == "ANY":
        return """
            (e._from == @dst_node_id AND e._to == @src_node_id)
            OR (e._to == @dst_node_id AND e._from == @src_node_id)
        """
    raise InvalidTraversalDirection(f"Invalid direction: {direction}")


def aql_edge(
    db: StandardDatabase,
    src_node_id: str,
    dst_node_id: str,
    graph_name: str,
    direction: str,
    return_clause: str,
    limit_one: bool,
    can_return_multiple: bool,
) -> Any | None:
    if limit_one and can_return_multiple:
        raise ValueError("Cannot return multiple results limit_one=True.")

    filter_clause = aql_edge_direction_filter(direction)
    limit_one_clause = "LIMIT 1" if limit_one else ""
    # sort_by_id_clause = "SORT e._id" if can_return_multiple else ""
    query = f"""
        FOR v, e IN 1..1 {direction} @src_node_id GRAPH @graph_name
            FILTER {filter_clause}
            {limit_one_clause}
            RETURN {return_clause}
    """

    bind_vars = {
        "src_node_id": src_node_id,
        "dst_node_id": dst_node_id,
        "graph_name": graph_name,
    }

    return (
        aql_as_list(db, query, bind_vars)
        if can_return_multiple
        else aql_single(db, query, bind_vars)
    )


def aql_fetch_data(
    db: StandardDatabase,
    collections: list[str],
    data: str,
    default: Any,
) -> dict[str, Any]:
    items = {}
    for collection in collections:
        query = """
            LET result = (
                FOR doc IN @@collection
                    RETURN {[doc._id]: doc.@data or @default}
            )

            RETURN MERGE(result)
        """

        bind_vars = {"data": data, "default": default, "@collection": collection}
        result = aql_single(db, query, bind_vars)
        items.update(result if result is not None else {})

    return items


def aql_fetch_data_edge(
    db: StandardDatabase,
    collections: list[str],
    data: str,
    default: Any,
) -> list[tuple[str, str, Any]]:
    items = []
    for collection in collections:
        query = """
            LET result = (
                FOR doc IN @@collection
                    RETURN [doc._from, doc._to, doc.@data or @default]
            )

            RETURN result
        """

        bind_vars = {"data": data, "default": default, "@collection": collection}
        result = aql_single(db, query, bind_vars)
        items.extend(result if result is not None else [])

    return items


def doc_update(
    db: StandardDatabase, id: str, data: dict[str, Any], **kwargs: Any
) -> str:
    """Updates a document in the collection."""
    res = db.update_document({**data, "_id": id}, keep_none=False, **kwargs)
    return str(res["_rev"])


def doc_delete(db: StandardDatabase, id: str, **kwargs: Any) -> None:
    """Deletes a document from the collection."""
    db.delete_document(id, silent=True, **kwargs)


def doc_insert(
    db: StandardDatabase,
    collection: str,
    id: str,
    data: dict[str, Any] = {},
    **kwargs: Any,
) -> dict[str, Any]:
    """Inserts a document into a collection."""
    result: dict[str, Any] = db.insert_document(
        collection, {**data, "_id": id}, overwrite=True, **kwargs
    )

    return result


def doc_get_or_insert(
    db: StandardDatabase, collection: str, id: str, **kwargs: Any
) -> dict[str, Any]:
    """Loads a document if existing, otherwise inserts it & returns it."""
    if db.has_document(id):
        result: dict[str, Any] = db.document(id)
        return result

    return doc_insert(db, collection, id, **kwargs)


def get_node_id(key: str, default_node_type: str) -> str:
    """Gets the node ID."""
    return key if "/" in key else f"{default_node_type}/{key}"


def get_node_type_and_id(key: str, default_node_type: str) -> tuple[str, str]:
    """Gets the node type and ID."""
    if "/" in key:
        return key.split("/")[0], key

    return default_node_type, f"{default_node_type}/{key}"


def get_update_dict(
    parent_keys: list[str], update_dict: dict[str, Any]
) -> dict[str, Any]:
    if parent_keys:
        for key in reversed(parent_keys):
            update_dict = {key: update_dict}

    return update_dict
