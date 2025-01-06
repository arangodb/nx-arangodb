"""
A collection of CRUD functions for ArangoDB Graphs.

Used across the nx_arangodb package to interact with ArangoDB.
"""

from __future__ import annotations

from typing import Any, Callable, Generator, Tuple

import networkx as nx
from arango import ArangoError, DocumentInsertError
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
    EdgeValuesDict,
    GraphAdjDict,
    MultiDiGraphAdjDict,
    MultiGraphAdjDict,
    NodeDict,
    SrcIndices,
)

from nx_arangodb.logger import logger

from ..exceptions import AQLMultipleResultsFound, InvalidTraversalDirection
from .enum import GraphType

RESERVED_KEYS = {"_id", "_key", "_rev", "_from", "_to"}


def get_arangodb_graph(
    adb_graph: Graph,
    load_node_dict: bool,
    load_adj_dict: bool,
    load_coo: bool,
    edge_collections_attributes: set[str],
    load_all_vertex_attributes: bool,
    load_all_edge_attributes: bool,
    is_directed: bool,
    is_multigraph: bool,
    symmetrize_edges_if_directed: bool,
    read_parallelism: int,
    read_batch_size: int,
) -> Tuple[
    NodeDict,
    GraphAdjDict | DiGraphAdjDict | MultiGraphAdjDict | MultiDiGraphAdjDict,
    SrcIndices,
    DstIndices,
    EdgeIndices,
    ArangoIDtoIndex,
    EdgeValuesDict,
]:
    """Pulls ArangoDB Graph Data from the database using
    `phenolrs.networkx.NetworkXLoader`.

    Parameters
    ----------
    adb_graph : Graph
        The ArangoDB Graph object from python-arango.

    load_node_dict : bool
        Whether to load the Node dictionary representation.

    load_adj_dict : bool
        Whether to load the Adjacency dictionary representation.

    load_coo : bool
        Whether to load the COO representation.

    edge_collections_attributes : set[str]
        The set of edge attributes to load. Can be empty.

    load_all_vertex_attributes : bool
        Whether to load all vertex attributes.

    load_all_edge_attributes : bool
        Whether to load all edge attributes. Cannot be True if
        **edge_collections_attributes** is not empty.

    is_directed : bool
        Whether to load the graph as directed or undirected.

    is_multigraph : bool
        Whether to load the graph as a MultiGraph or Graph.

    symmetrize_edges_if_directed : bool
        Whether to duplicate edges in the adjacency dictionary if the graph is directed.

    Returns
    -------
    Tuple[
        NodeDict,
        GraphAdjDict | DiGraphAdjDict | MultiGraphAdjDict | MultiDiGraphAdjDict,
        SrcIndices,
        DstIndices,
        EdgeIndices,
        ArangoIDtoIndex,
        EdgeValuesDict
    ]
        A tuple containing the different representations of the graph.

    Raises
    ------
    ValueError
        If **load_all_edge_attributes** is True and
        **edge_collections_attributes** is not empty.

    ValueError
        If none of the load flags are True.

    PhenolrsError
        If an error occurs while loading the graph.
    """
    if len(edge_collections_attributes) != 0 and load_all_edge_attributes:
        raise ValueError(
            "You have specified to load at least one specific edge attribute"
            " and at the same time set the parameter `load_all_edge_attributes`"
            " to true. This combination is not allowed."
        )

    v_cols = adb_graph.vertex_collections()
    edge_definitions = adb_graph.edge_definitions()
    e_cols = {c["edge_collection"] for c in edge_definitions}

    metagraph: dict[str, dict[str, Any]] = {
        "vertexCollections": {col: set() for col in v_cols},
        "edgeCollections": {col: edge_collections_attributes for col in e_cols},
    }

    if not any((load_node_dict, load_adj_dict, load_coo)):
        raise ValueError("At least one of the load flags must be True.")

    if not load_node_dict:
        metagraph["vertexCollections"] = {}

    if not load_adj_dict and not load_coo:
        metagraph["edgeCollections"] = {}

    hosts = adb_graph._conn._hosts
    hosts = hosts.split(",") if type(hosts) is str else hosts
    db_name = adb_graph._conn._db_name
    username, password = adb_graph._conn._auth

    (
        node_dict,
        adj_dict,
        src_indices,
        dst_indices,
        edge_indices,
        vertex_ids_to_index,
        edge_values,
    ) = NetworkXLoader.load_into_networkx(
        database=db_name,
        metagraph=metagraph,
        hosts=hosts,
        username=username,
        password=password,
        load_adj_dict=load_adj_dict,
        load_coo=load_coo,
        load_all_vertex_attributes=load_all_vertex_attributes,
        load_all_edge_attributes=load_all_edge_attributes,
        is_directed=is_directed,
        is_multigraph=is_multigraph,
        symmetrize_edges_if_directed=symmetrize_edges_if_directed,
        parallelism=read_parallelism,
        batch_size=read_batch_size,
    )

    return (
        node_dict,
        adj_dict,
        src_indices,
        dst_indices,
        edge_indices,
        vertex_ids_to_index,
        edge_values,
    )


def json_serializable(cls):
    """Decorator to make a class JSON serializable. Only used for
    the NodeAttrDict, EdgeAttrDict, and GraphAttrDict classes.
    """

    def to_dict(self):
        return {
            key: dict(value) if isinstance(value, cls) else value
            for key, value in self.data.items()
        }

    cls.to_dict = to_dict
    return cls


def cast_to_string(value: Any) -> str:
    """Casts a value to a string."""
    if isinstance(value, str):
        return value

    if isinstance(value, (int, float)):
        return str(value)

    raise TypeError(f"{value} cannot be casted to string.")


def key_is_string(func: Callable[..., Any]) -> Any:
    """Decorator to check if the key is a string.
    Will attempt to cast the key to a string if it is not.
    """

    def wrapper(self: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
        if key is None:
            raise ValueError("Key cannot be None.")

        key = cast_to_string(key)
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
                raise KeyError(f"{key} is not an ArangoDB ID.")

        elif isinstance(key, int):
            m = "Edge order is not guaranteed when using int as an edge key. It may raise a KeyError. Use at your own risk."  # noqa
            logger.debug(m)

        else:
            raise TypeError(f"{key} is not an ArangoDB Edge _id or integer.")

        return func(self, key, *args, **kwargs)

    return wrapper


def keys_are_strings(func: Callable[..., Any]) -> Any:
    """Decorator to check if the keys are strings.
    Will attempt to cast the keys to strings if they are not.
    """

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
            key = cast_to_string(key)
            data_dict[key] = value

        return func(self, data_dict, *args, **kwargs)

    return wrapper


def key_is_not_reserved(func: Callable[..., Any]) -> Any:
    """Decorator to check if the key is not reserved."""

    def wrapper(self: Any, key: str, *args: Any, **kwargs: Any) -> Any:
        if key in RESERVED_KEYS:
            raise KeyError(f"'{key}' is a reserved key.")

        return func(self, key, *args, **kwargs)

    return wrapper


def keys_are_not_reserved(func: Any) -> Any:
    """Decorator to check if the keys are not reserved."""

    def wrapper(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
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
) -> Any | None:
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
    """Checks if an edge exists between two nodes."""
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
    """Gets an edge between two nodes."""
    return_clause = "UNSET(e, '_rev')"
    if direction == "ANY":
        return_clause = f"DISTINCT {return_clause}"

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
    """Gets the edge ID between two nodes."""
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
    """Counts the number of edges from a source node."""
    query = f"""
        FOR v, e IN 1..1 {direction} @src_node_id GRAPH @graph_name
            COLLECT id = e._id
            COLLECT WITH COUNT INTO num
            RETURN num
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
    """Counts the number of edges between two nodes."""
    filter_clause = aql_edge_direction_filter(direction)

    query = f"""
        FOR v, e IN 1..1 {direction} @src_node_id GRAPH @graph_name
            FILTER {filter_clause}
            COLLECT id = e._id
            COLLECT WITH COUNT INTO num
            RETURN num
    """

    bind_vars = {
        "src_node_id": src_node_id,
        "dst_node_id": dst_node_id,
        "graph_name": graph_name,
    }

    result = aql_single(db, query, bind_vars)

    return int(result) if result is not None else 0


def aql_edge_direction_filter(direction: str) -> str:
    """Returns the AQL filter clause for the edge direction."""
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
    """Fetches an edge between two nodes."""
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
) -> Generator[dict[str, Any], None, None]:
    """Fetches data from a collection (assumed to be vertex)."""
    bind_vars = {"data": data, "default": default}
    query = """
        FOR doc IN @@collection
            RETURN [doc._id, doc.@data or @default]
    """

    for collection in collections:
        bind_vars["@collection"] = collection
        yield from aql(db, query, bind_vars)


def aql_fetch_data_edge(
    db: StandardDatabase,
    collections: list[str],
    data: str,
    default: Any,
) -> Generator[tuple[str, str, Any], None, None]:
    """Fetches data from an edge collection."""
    bind_vars = {"data": data, "default": default}
    query = """
        FOR doc IN @@collection
            RETURN [doc._from, doc._to, doc.@data or @default]
    """

    for collection in collections:
        bind_vars["@collection"] = collection
        for item in aql(db, query, bind_vars):
            yield tuple(item)


def doc_update(
    db: StandardDatabase, id: str, data: dict[str, Any], **kwargs: Any
) -> None:
    """Updates a document in the collection."""
    db.update_document({**data, "_id": id}, keep_none=False, silent=True, **kwargs)


def doc_delete(db: StandardDatabase, id: str, **kwargs: Any) -> None:
    """Deletes a document from the collection."""
    db.delete_document(id, silent=True, **kwargs)


def edges_delete(
    db: StandardDatabase, graph: Graph, src_node_id: str, **kwargs: Any
) -> None:
    """Deletes all edges from a source node."""
    remove_statements = "\n".join(
        f"REMOVE e IN `{edge_def['edge_collection']}` OPTIONS {{ignoreErrors: true}}"  # noqa
        for edge_def in graph.edge_definitions()
    )

    query = f"""
        FOR v, e IN 1..1 ANY @src_node_id GRAPH @graph_name
            {remove_statements}
    """

    bind_vars = {"src_node_id": src_node_id, "graph_name": graph.name}

    aql(db, query, bind_vars)


def doc_insert(
    db: StandardDatabase,
    collection: str,
    id: str,
    data: dict[str, Any] = {},
    **kwargs: Any,
) -> dict[str, Any]:
    """Inserts a document into a collection. Returns document metadata."""
    result: dict[str, Any] = db.insert_document(
        collection, {**data, "_id": id}, overwrite=True, **kwargs
    )

    del result["_rev"]

    return result


def doc_get_or_insert(
    db: StandardDatabase, collection: str, id: str, **kwargs: Any
) -> dict[str, Any]:
    """Loads a document if existing, otherwise inserts it & returns it."""
    if db.has_document(id):
        result: dict[str, Any] = db.document(id)
        del result["_rev"]
        return result

    return doc_insert(db, collection, id, **kwargs)


def vertex_get(graph: Graph, id: str) -> dict[str, Any] | None:
    """Gets a vertex from the graph."""
    vertex: dict[str, Any] | None = graph.vertex(id)
    if vertex is None:
        return None

    del vertex["_rev"]
    return vertex


def edge_get(graph: Graph, id: str) -> dict[str, Any] | None:
    """Gets an edge from the graph."""
    edge: dict[str, Any] | None = graph.edge(id)
    if edge is None:
        return None

    del edge["_rev"]

    return edge


def edge_link(
    graph: Graph, collection: str, src_id: str, dst_id: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Links two vertices via an edge."""
    edge: dict[str, Any] = graph.link(collection, src_id, dst_id, data)
    del edge["_rev"]
    return edge


def is_arangodb_id(key):
    """Checks if the key is an ArangoDB ID."""
    return "/" in key


def get_node_type(key: str, default_node_type: str) -> str:
    """Gets the collection of a node."""
    return key.split("/")[0] if is_arangodb_id(key) else default_node_type


def get_node_id(key: str, default_node_type: str) -> str:
    """Gets the node ID."""
    return key if is_arangodb_id(key) else f"{default_node_type}/{key}"


def get_node_type_and_id(key: str, default_node_type: str) -> tuple[str, str]:
    """Gets the node collection (i.e type) and ID."""
    return (
        (key.split("/")[0], key)
        if is_arangodb_id(key)
        else (default_node_type, f"{default_node_type}/{key}")
    )


def get_node_type_and_key(key: str, default_node_type: str) -> tuple[str, str]:
    """Gets the node type and key."""
    if is_arangodb_id(key):
        col, key = key.split("/", 1)
        return col, key

    return default_node_type, key


def get_update_dict(
    parent_keys: list[str], update_dict: dict[str, Any]
) -> dict[str, Any]:
    """Builds the update dictionary for nested documents.
    Useful for updating nested documents in ArangoDB.
    """
    if parent_keys:
        for key in reversed(parent_keys):
            update_dict = {key: update_dict}

    return update_dict


class ArangoDBBatchError(ArangoError):
    """Custom exception for batch errors."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__(self._format_errors())

    def _format_errors(self):
        return "\n".join(str(error) for error in self.errors)


def check_update_list_for_errors(lst):
    """Checks if a list contains any errors."""
    for element in lst:
        if element is False:
            return False

        elif isinstance(element, list):
            for sub_element in element:
                if isinstance(sub_element, DocumentInsertError):
                    return False

    return True


def separate_nodes_by_collections(
    nodes: dict[str, Any], default_collection: str
) -> dict[str, dict[str, Any]]:
    """Separate the dictionary into collections based on whether IDs contain '/'.
    Returns dictionary where the keys are collection names and the values are
    dictionaries of key-value pairs belonging to those collections.
    """
    separated: dict[str, dict[str, Any]] = {}

    for key, value in nodes.items():
        collection, doc_key = get_node_type_and_key(key, default_collection)

        if collection not in separated:
            separated[collection] = {}

        separated[collection][doc_key] = value

    return separated


def transform_local_documents_for_adb(
    original_documents: dict[str, Any]
) -> list[dict[str, Any]]:
    """Transform original documents into a format suitable for UPSERT
    operations in ArangoDB. Returns a list of documents with '_key' attribute
    and additional attributes.
    """
    transformed_documents: list[dict[str, Any]] = []

    for key, values in original_documents.items():
        transformed_doc = {"_key": key}
        transformed_doc.update(values)
        transformed_documents.append(transformed_doc)

    return transformed_documents


def upsert_collection_documents(
    db: StandardDatabase, separated: dict[str, dict[str, Any]]
) -> list[Any]:
    """Process each collection in the separated dictionary.
    If inserting a document fails, the exception is not raised but
    returned as an object in the result list.
    """
    results = []

    for collection_name, documents in separated.items():
        collection = db.collection(collection_name)
        transformed_documents = transform_local_documents_for_adb(documents)
        results.append(
            collection.insert_many(
                transformed_documents, silent=False, overwrite_mode="update"
            )
        )

    return results


def separate_edges_by_collections_graph(
    edges: GraphAdjDict, default_node_type: str
) -> dict[str, list[dict[str, Any]]]:
    """Separate the dictionary into collections for Graph and DiGraph types.
    Returns a dictionary where the keys are collection names and the
    values are dictionaries of key-value pairs belonging to those collections.
    """
    separated: dict[str, list[dict[str, Any]]] = {}

    for from_doc_id, target_dict in edges.items():
        for to_doc_id, edge_doc in target_dict.items():
            assert edge_doc is not None and "_id" in edge_doc
            edge_collection_name = get_node_type_and_id(
                edge_doc["_id"], default_node_type
            )[0]

            if edge_collection_name not in separated:
                separated[edge_collection_name] = []

            edge_doc["_from"] = from_doc_id
            edge_doc["_to"] = to_doc_id

            separated[edge_collection_name].append(edge_doc)

    return separated


def separate_edges_by_collections_multigraph(
    edges: MultiGraphAdjDict, default_node_type: str
) -> Any:
    """
    Separate the dictionary into collections for MultiGraph and MultiDiGraph types.
    Returns a dictionary where the keys are collection names and the
    values are dictionaries of key-value pairs belonging to those collections.
    """
    separated: dict[str, list[dict[str, Any]]] = {}

    for from_doc_id, target_dict in edges.items():
        for to_doc_id, edge_doc in target_dict.items():
            # edge_doc is expected to be a list of edges in Multi(Di)Graph
            for m_edge_id, m_edge_doc in edge_doc.items():
                assert m_edge_doc is not None and "_id" in m_edge_doc
                edge_collection_name = get_node_type_and_id(
                    m_edge_doc["_id"], default_node_type
                )[0]

                if edge_collection_name not in separated:
                    separated[edge_collection_name] = []

                m_edge_doc["_from"] = from_doc_id
                m_edge_doc["_to"] = to_doc_id

                separated[edge_collection_name].append(m_edge_doc)

    return separated


def separate_edges_by_collections(
    edges: GraphAdjDict | MultiGraphAdjDict, graph_type: str, default_node_type: str
) -> Any:
    """
    Wrapper function to separate the dictionary into collections based on graph type.
    Returns a dictionary where the keys are collection names and the
    values are dictionaries of key-value pairs belonging to those collections.
    """
    if graph_type in [GraphType.Graph.name, GraphType.DiGraph.name]:
        return separate_edges_by_collections_graph(edges, default_node_type)
    elif graph_type in [GraphType.MultiGraph.name, GraphType.MultiDiGraph.name]:
        return separate_edges_by_collections_multigraph(edges, default_node_type)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")


def upsert_collection_edges(
    db: StandardDatabase, separated: dict[str, list[dict[str, Any]]]
) -> Any:
    """Process each collection in the separated dictionary.
    Returns a list of results from the insert_many operation.
    If inserting a document fails, the exception is not raised but
    returned as an object in the result list.
    """

    results = []

    for collection_name, documents_list in separated.items():
        collection = db.collection(collection_name)
        results.append(
            collection.insert_many(
                documents_list,
                silent=False,
                overwrite_mode="update",
            )
        )

    return results
