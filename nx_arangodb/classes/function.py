from __future__ import annotations

from typing import Any, Tuple

import networkx as nx
import numpy as np
from arango.collection import StandardCollection
from arango.cursor import Cursor
from arango.database import StandardDatabase

import nx_arangodb as nxadb


def get_arangodb_graph(
    G: nxadb.Graph | nxadb.DiGraph,
    load_node_dict: bool,
    load_adj_dict: bool,
    load_adj_dict_as_undirected: bool,
    load_coo: bool,
) -> Tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
    np.ndarray,
    np.ndarray,
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
    if not G.graph_exists:
        raise ValueError("Graph does not exist in the database")

    adb_graph = G.db.graph(G.graph_name)
    v_cols = adb_graph.vertex_collections()
    edge_definitions = adb_graph.edge_definitions()
    e_cols = {c["edge_collection"] for c in edge_definitions}

    metagraph = {
        "vertexCollections": {col: {} for col in v_cols},
        "edgeCollections": {col: {} for col in e_cols},
    }

    from phenolrs.graph_loader import GraphLoader

    kwargs = {}
    if G.graph_loader_parallelism is not None:
        kwargs["parallelism"] = G.graph_loader_parallelism
    if G.graph_loader_batch_size is not None:
        kwargs["batch_size"] = G.graph_loader_batch_size

    return GraphLoader.load(
        G.db.name,
        metagraph,
        [G._host],
        username=G._username,
        password=G._password,
        load_node_dict=load_node_dict,
        load_adj_dict=load_adj_dict,
        load_adj_dict_as_undirected=load_adj_dict_as_undirected,
        load_coo=load_coo,
        **kwargs,
    )


def key_is_string(func) -> Any:
    """Decorator to check if the key is a string."""

    def wrapper(self, key, *args, **kwargs) -> Any:
        if not isinstance(key, str):
            raise TypeError(f"'{key}' is not a string.")

        return func(self, key, *args, **kwargs)

    return wrapper


def keys_are_strings(func) -> Any:
    """Decorator to check if the keys are strings."""

    def wrapper(self, dict, *args, **kwargs) -> Any:
        if not all(isinstance(key, str) for key in dict):
            raise TypeError(f"All keys must be strings.")

        return func(self, dict, *args, **kwargs)

    return wrapper


RESERVED_KEYS = {"_id", "_key", "_rev"}


def key_is_not_reserved(func) -> Any:
    """Decorator to check if the key is not reserved."""

    def wrapper(self, key, *args, **kwargs) -> Any:
        if key in RESERVED_KEYS:
            raise KeyError(f"'{key}' is a reserved key.")

        return func(self, key, *args, **kwargs)

    return wrapper


def keys_are_not_reserved(func) -> Any:
    """Decorator to check if the keys are not reserved."""

    def wrapper(self, dict, *args, **kwargs) -> Any:
        if any(key in RESERVED_KEYS for key in dict):
            raise KeyError(f"All keys must not be reserved.")

        return func(self, dict, *args, **kwargs)

    return wrapper


def create_collection(
    db: StandardDatabase, collection_name: str, edge: bool = False
) -> StandardCollection:
    """Creates a collection if it does not exist and returns it."""
    if not db.has_collection(collection_name):
        db.create_collection(collection_name, edge=edge)

    return db.collection(collection_name)


def aql(
    db: StandardDatabase, query: str, bind_vars: dict[str, Any], **kwargs
) -> Cursor:
    """Executes an AQL query and returns the cursor."""
    return db.aql.execute(query, bind_vars=bind_vars, stream=True, **kwargs)


def aql_as_list(
    db: StandardDatabase, query: str, bind_vars: dict[str, Any], **kwargs
) -> list[Any]:
    """Executes an AQL query and returns the results as a list."""
    return list(aql(db, query, bind_vars, **kwargs))


def aql_single(db: StandardDatabase, query: str, bind_vars: dict[str, Any]) -> Any:
    """Executes an AQL query and returns the first result."""
    result = aql_as_list(db, query, bind_vars)
    if len(result) == 0:
        return None

    return result[0]


def aql_doc_has_key(db: StandardDatabase, id: str, key: str) -> bool:
    """Checks if a document has a key."""
    query = f"RETURN HAS(DOCUMENT(@id), @key)"
    bind_vars = {"id": id, "key": key}
    return aql_single(db, query, bind_vars)


def aql_doc_get_key(db: StandardDatabase, id: str, key: str) -> Any:
    """Gets a key from a document."""
    query = f"RETURN DOCUMENT(@id).@key"
    bind_vars = {"id": id, "key": key}
    return aql_single(db, query, bind_vars)


def aql_doc_get_keys(db: StandardDatabase, id: str) -> list[str]:
    """Gets the keys of a document."""
    query = f"RETURN ATTRIBUTES(DOCUMENT(@id))"
    bind_vars = {"id": id}
    return aql_single(db, query, bind_vars)


def aql_doc_get_length(db: StandardDatabase, id: str) -> int:
    """Gets the length of a document."""
    query = f"RETURN LENGTH(DOCUMENT(@id))"
    bind_vars = {"id": id}
    return aql_single(db, query, bind_vars)


def aql_edge_get(
    db: StandardDatabase,
    src_node_id: str,
    dst_node_id: str,
    graph_name: str,
    direction: str,
    return_bool: bool,
):
    if direction == "INBOUND":
        filter_clause = f"e._from == @dst_node_id"
    elif direction == "OUTBOUND":
        filter_clause = f"e._to == @dst_node_id"
    elif direction == "ANY":
        filter_clause = f"e._from == @dst_node_id OR e._to == @dst_node_id"
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return_clause = "true" if return_bool else "e"

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


def doc_update(db: StandardDatabase, id: str, data: dict[str, Any], **kwargs) -> None:
    """Updates a document in the collection."""
    db.update_document({**data, "_id": id}, keep_none=False, silent=True, **kwargs)


def doc_delete(db: StandardDatabase, id: str, **kwargs) -> None:
    """Deletes a document from the collection."""
    db.delete_document(id, silent=True, **kwargs)


def doc_insert(
    db: StandardDatabase, collection: str, id: str, data: dict[str, Any] = {}, **kwargs
) -> dict[str, Any] | bool:
    """Inserts a document into a collection."""
    return db.insert_document(collection, {**data, "_id": id}, overwrite=True, **kwargs)


def doc_get_or_insert(
    db: StandardDatabase, collection: str, id: str, **kwargs
) -> dict[str, Any]:
    """Loads a document if existing, otherwise inserts it & returns it."""
    if db.has_document(id):
        return db.document(id)

    return doc_insert(db, collection, id, **kwargs)


def get_node_id(key: str, default_node_type: str) -> str:
    """Gets the node ID."""
    return key if "/" in key else f"{default_node_type}/{key}"


def get_node_type_and_id(key: str, default_node_type: str) -> tuple[str, str]:
    """Gets the node type and ID."""
    if "/" in key:
        return key.split("/")[0], key

    return default_node_type, f"{default_node_type}/{key}"
