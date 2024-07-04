"""
A collection of CRUD functions for the ArangoDB graph database.
Used by the nx_arangodb Graph, DiGraph, MultiGraph, and MultiDiGraph classes.
"""

from __future__ import annotations

from collections import UserDict
from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt
from arango.collection import StandardCollection
from arango.cursor import Cursor
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.logger import logger

from ..exceptions import (
    AQLMultipleResultsFound,
    GraphDoesNotExist,
    InvalidTraversalDirection,
)


def get_arangodb_graph(
    G: nxadb.Graph | nxadb.DiGraph,
    load_node_dict: bool,
    load_adj_dict: bool,
    load_adj_dict_as_directed: bool,
    load_coo: bool,
) -> Tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    dict[str, int],
]:
    """Pulls the graph from the database, assuming the graph exists.

    Returns the folowing representations:
    - Node dictionary (nx.Graph)
    - Adjacency dictionary (nx.Graph)
    - Source Indices (COO)
    - Destination Indices (COO)
    - Node-ID-to-index mapping (COO)
    """
    if not G.graph_exists_in_db:
        raise GraphDoesNotExist(
            "Graph does not exist in the database. Can't load graph."
        )

    adb_graph = G.db.graph(G.graph_name)
    v_cols = adb_graph.vertex_collections()
    edge_definitions = adb_graph.edge_definitions()
    e_cols = {c["edge_collection"] for c in edge_definitions}

    metagraph: dict[str, dict[str, Any]] = {
        "vertexCollections": {col: {} for col in v_cols},
        "edgeCollections": {col: {} for col in e_cols},
    }

    from phenolrs.networkx_loader import NetworkXLoader

    kwargs = {}
    if G.graph_loader_parallelism is not None:
        kwargs["parallelism"] = G.graph_loader_parallelism
    if G.graph_loader_batch_size is not None:
        kwargs["batch_size"] = G.graph_loader_batch_size

    # TODO: Remove ignore when phenolrs is published
    return NetworkXLoader.load_into_networkx(  # type: ignore
        G.db.name,
        metagraph,
        [G._host],
        username=G._username,
        password=G._password,
        load_node_dict=load_node_dict,
        load_adj_dict=load_adj_dict,
        load_adj_dict_as_directed=load_adj_dict_as_directed,
        load_coo=load_coo,
        **kwargs,
    )


def key_is_string(func: Callable[..., Any]) -> Any:
    """Decorator to check if the key is a string."""

    def wrapper(self: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(key, str):
            if not isinstance(key, (int, float)):
                raise TypeError(f"{key} cannot be casted to string.")

            key = str(key)

        return func(self, key, *args, **kwargs)

    return wrapper


def logger_debug(func: Callable[..., Any]) -> Any:
    """Decorator to log debug messages."""

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger.debug(f"{func.__name__} - {args} - {kwargs}")
        return func(self, *args, **kwargs)

    return wrapper


def keys_are_strings(func: Callable[..., Any]) -> Any:
    """Decorator to check if the keys are strings."""

    def wrapper(
        self: Any, data: dict[Any, Any] | zip[Any], *args: Any, **kwargs: Any
    ) -> Any:
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
    )


def aql_edge_get(
    db: StandardDatabase,
    src_node_id: str,
    dst_node_id: str,
    graph_name: str,
    direction: str,
) -> Any | None:
    return_clause = "DISTINCT e" if direction == "ANY" else "e"
    return aql_edge(
        db,
        src_node_id,
        dst_node_id,
        graph_name,
        direction,
        return_clause=return_clause,
    )


def aql_edge_id(
    db: StandardDatabase,
    src_node_id: str,
    dst_node_id: str,
    graph_name: str,
    direction: str,
) -> str | None:
    return_clause = "DISTINCT e._id" if direction == "ANY" else "e._id"
    result = aql_edge(
        db,
        src_node_id,
        dst_node_id,
        graph_name,
        direction,
        return_clause=return_clause,
    )

    return str(result) if result is not None else None


def aql_edge(
    db: StandardDatabase,
    src_node_id: str,
    dst_node_id: str,
    graph_name: str,
    direction: str,
    return_clause: str,
) -> Any | None:
    if direction == "INBOUND":
        filter_clause = "e._from == @dst_node_id"
    elif direction == "OUTBOUND":
        filter_clause = "e._to == @dst_node_id"
    elif direction == "ANY":
        filter_clause = """
            (e._from == @dst_node_id AND e._to == @src_node_id)
            OR (e._to == @dst_node_id AND e._from == @src_node_id)
        """
    else:
        raise InvalidTraversalDirection(f"Invalid direction: {direction}")

    query = f"""
        FOR v, e IN 1..1 {direction} @src_node_id GRAPH @graph_name
            FILTER {filter_clause}
            RETURN {return_clause}
    """

    bind_vars = {
        "src_node_id": src_node_id,
        "dst_node_id": dst_node_id,
        "graph_name": graph_name,
    }

    return aql_single(db, query, bind_vars)


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
