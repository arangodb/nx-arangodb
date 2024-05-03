from __future__ import annotations

from typing import Any

import networkx as nx
from arango.collection import StandardCollection
from arango.cursor import Cursor
from arango.database import StandardDatabase

import nx_arangodb as nxadb


def pull(
    G: nxadb.Graph | nxadb.DiGraph,
    load_node_and_adj_dict: bool,
    load_adj_dict_as_undirected: bool,
    load_coo: bool,
) -> None:
    """Pulls the graph from the database, assuming the graph exists.

    Modifies the graph object in place! Uses a rust-based graph loader <3

    :param G: The graph object.
    :type G: nxadb.Graph | nxadb.DiGraph
    :param load_node_and_adj_dict: Whether to load the node and adjacency dictionaries.
    :type load_node_and_adj_dict: bool
    :param load_adj_dict_as_undirected: Whether to load the adjacency dictionary as undirected.
    :type load_adj_dict_as_undirected: bool
    :param load_coo: Whether to load the COO representation.
    :type load_coo: bool
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

    result = GraphLoader.load(
        G.db.name,
        metagraph,
        [G._host],
        username=G._username,
        password=G._password,
        load_node_dict=load_node_and_adj_dict,
        load_adj_dict=load_node_and_adj_dict,
        load_adj_dict_as_undirected=load_adj_dict_as_undirected,
        load_coo=load_coo,
        **kwargs,
    )

    if load_node_and_adj_dict:
        # hacky, i don't like this
        # need to revisit...
        # consider using nx.convert.from_dict_of_dicts instead

        G._node.clear()
        for node_id, node_data in result[0].items():
            node_attr_dict = nxadb.classes.dict.NodeAttrDict(G.db, G.adb_graph)
            node_attr_dict.node_id = node_id
            node_attr_dict.data = node_data
            G._node.data[node_id] = node_attr_dict

        # G._node = result[0]
        # TODO: fix this hack
        G._adj = result[1]

    if load_coo:
        G.src_indices = result[2]
        G.dst_indices = result[3]
        G.vertex_ids_to_index = result[4]


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
    """Executes an AQL query and returns the cursor. Stream is enabled."""
    return db.aql.execute(query, bind_vars=bind_vars, stream=True, **kwargs)


def aql_as_list(
    db: StandardDatabase, query: str, bind_vars: dict[str, Any]
) -> list[Any]:
    """Executes an AQL query and returns the results as a list."""
    result = []

    cursor = aql(db, query, bind_vars)
    while not cursor.empty():
        result.extend(cursor.batch())

        cursor.batch().clear()
        if cursor.has_more():
            cursor.fetch()

    return result


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
