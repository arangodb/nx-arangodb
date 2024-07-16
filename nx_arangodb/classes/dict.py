"""
A collection of dictionary-like objects for interacting with ArangoDB.
Used as the underlying data structure for NetworkX-ArangoDB graphs.
"""

from __future__ import annotations

import warnings
from collections import UserDict, defaultdict
from collections.abc import Iterator
from typing import Any, Callable, Generator

from arango.database import StandardDatabase
from arango.exceptions import DocumentInsertError, ArangoError
from arango.graph import Graph

from nx_arangodb.logger import logger

from .function import (
    aql,
    aql_as_list,
    aql_doc_get_key,
    aql_doc_get_keys,
    aql_doc_get_length,
    aql_doc_has_key,
    aql_edge_exists,
    aql_edge_get,
    aql_edge_id,
    aql_fetch_data,
    aql_fetch_data_edge,
    aql_single,
    create_collection,
    doc_delete,
    doc_get_or_insert,
    doc_insert,
    doc_update,
    get_arangodb_graph,
    get_node_id,
    get_node_type_and_id,
    key_is_not_reserved,
    key_is_string,
    keys_are_not_reserved,
    keys_are_strings,
    logger_debug,
)
from ..utils.arangodb import (
    separate_nodes_by_collections,
    upsert_collection_documents,
    ArangoDBBatchError,
    check_list_for_errors,
)


#############
# Factories #
#############


def graph_dict_factory(
    db: StandardDatabase, graph_name: str
) -> Callable[..., GraphDict]:
    return lambda: GraphDict(db, graph_name)


def node_dict_factory(
    db: StandardDatabase, graph: Graph, default_node_type: str
) -> Callable[..., NodeDict]:
    return lambda: NodeDict(db, graph, default_node_type)


def node_attr_dict_factory(
    db: StandardDatabase, graph: Graph
) -> Callable[..., NodeAttrDict]:
    return lambda: NodeAttrDict(db, graph)


def adjlist_outer_dict_factory(
    db: StandardDatabase,
    graph: Graph,
    default_node_type: str,
    edge_type_func: Callable[[str, str], str],
) -> Callable[..., AdjListOuterDict]:
    return lambda: AdjListOuterDict(db, graph, default_node_type, edge_type_func)


def adjlist_inner_dict_factory(
    db: StandardDatabase,
    graph: Graph,
    default_node_type: str,
    edge_type_func: Callable[[str, str], str],
    adjlist_outer_dict: AdjListOuterDict | None = None,
) -> Callable[..., AdjListInnerDict]:
    return lambda: AdjListInnerDict(
        db, graph, default_node_type, edge_type_func, adjlist_outer_dict
    )


def edge_attr_dict_factory(
    db: StandardDatabase, graph: Graph
) -> Callable[..., EdgeAttrDict]:
    return lambda: EdgeAttrDict(db, graph)


#########
# Graph #
#########


class GraphDict(UserDict[str, Any]):
    """A dictionary-like object for storing graph attributes.

    Given that ArangoDB does not have a concept of graph attributes, this class
    stores the attributes in a collection with the graph name as the document key.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph_name: The graph name.
    :type graph_name: str
    """

    COLLECTION_NAME = "nxadb_graphs"

    @logger_debug
    def __init__(
        self, db: StandardDatabase, graph_name: str, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.data: dict[str, Any] = {}

        self.db = db
        self.graph_name = graph_name
        self.graph_id = f"{self.COLLECTION_NAME}/{graph_name}"

        self.adb_graph = db.graph(graph_name)
        self.collection = create_collection(db, self.COLLECTION_NAME)

        data = doc_get_or_insert(self.db, self.COLLECTION_NAME, self.graph_id)
        self.data.update(data)

    @key_is_string
    @logger_debug
    def __contains__(self, key: str) -> bool:
        """'foo' in G.graph"""
        if key in self.data:
            return True

        return aql_doc_has_key(self.db, self.graph_id, key)

    @key_is_string
    @logger_debug
    def __getitem__(self, key: str) -> Any:
        """G.graph['foo']"""

        if value := self.data.get(key):
            return value

        result = aql_doc_get_key(self.db, self.graph_id, key)

        if not result:
            raise KeyError(key)

        self.data[key] = result

        return result

    @key_is_string
    @key_is_not_reserved
    @logger_debug
    # @value_is_json_serializable # TODO?
    def __setitem__(self, key: str, value: Any) -> None:
        """G.graph['foo'] = 'bar'"""
        self.data[key] = value
        self.data["_rev"] = doc_update(self.db, self.graph_id, {key: value})

    @key_is_string
    @key_is_not_reserved
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """del G.graph['foo']"""
        self.data.pop(key, None)
        self.data["_rev"] = doc_update(self.db, self.graph_id, {key: None})

    @keys_are_strings
    @keys_are_not_reserved
    # @values_are_json_serializable # TODO?
    @logger_debug
    def update(self, attrs: Any) -> None:
        """G.graph.update({'foo': 'bar'})"""
        if not attrs:
            return

        self.data.update(attrs)
        self.data["_rev"] = doc_update(self.db, self.graph_id, attrs)

    # @logger_debug
    # def clear(self) -> None:
    #     """G.graph.clear()"""
    #     self.data.clear()

    #     # if clear_remote:
    #     #     doc_insert(self.db, self.COLLECTION_NAME, self.graph_id, silent=True)


########
# Node #
########


def process_node_attr_dict_value(parent: NodeAttrDict, key: str, value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    node_attr_dict = parent.node_attr_dict_factory()
    node_attr_dict.root = parent.root or parent
    node_attr_dict.node_id = parent.node_id
    node_attr_dict.parent_keys = parent.parent_keys + [key]
    node_attr_dict.data = build_node_attr_dict_data(node_attr_dict, value)

    return node_attr_dict


def build_node_attr_dict_data(
    parent: NodeAttrDict, data: dict[str, Any]
) -> dict[str, Any | NodeAttrDict]:
    """Recursively build a NodeAttrDict from a dict.

    It's possible that **value** is a nested dict, so we need to
    recursively build a NodeAttrDict for each nested dict.

    Returns the parent NodeAttrDict.
    """
    node_attr_dict_data = {}
    for key, value in data.items():
        node_attr_dict_value = process_node_attr_dict_value(parent, key, value)
        node_attr_dict_data[key] = node_attr_dict_value

    return node_attr_dict_data


def get_update_dict(
    parent_keys: list[str], update_dict: dict[str, Any]
) -> dict[str, Any]:
    if parent_keys:
        for key in reversed(parent_keys):
            update_dict = {key: update_dict}

    return update_dict


class NodeAttrDict(UserDict[str, Any]):
    """The inner-level of the dict of dict structure
    representing the nodes (vertices) of a graph.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    """

    @logger_debug
    def __init__(self, db: StandardDatabase, graph: Graph, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data: dict[str, Any] = {}

        self.db = db
        self.graph = graph
        self.node_id: str | None = None

        # NodeAttrDict may be a child of another NodeAttrDict
        # e.g G._node['node/1']['object']['foo'] = 'bar'
        # In this case, **parent_keys** would be ['object']
        # and **root** would be G._node['node/1']
        self.root: NodeAttrDict | None = None
        self.parent_keys: list[str] = []
        self.node_attr_dict_factory = node_attr_dict_factory(self.db, self.graph)

    @key_is_string
    @logger_debug
    def __contains__(self, key: str) -> bool:
        """'foo' in G._node['node/1']"""
        if key in self.data:
            return True

        assert self.node_id
        return aql_doc_has_key(self.db, self.node_id, key, self.parent_keys)

    @key_is_string
    @logger_debug
    def __getitem__(self, key: str) -> Any:
        """G._node['node/1']['foo']"""
        if value := self.data.get(key):
            return value

        assert self.node_id
        result = aql_doc_get_key(self.db, self.node_id, key, self.parent_keys)

        if not result:
            raise KeyError(key)

        node_attr_dict_value = process_node_attr_dict_value(self, key, result)
        self.data[key] = node_attr_dict_value

        return node_attr_dict_value

    @key_is_string
    @key_is_not_reserved
    # @value_is_json_serializable # TODO?
    @logger_debug
    def __setitem__(self, key: str, value: Any) -> None:
        """
        G._node['node/1']['foo'] = 'bar'
        G._node['node/1']['object'] = {'foo': 'bar'}
        G._node['node/1']['object']['foo'] = 'baz'
        """
        assert self.node_id
        node_attr_dict_value = process_node_attr_dict_value(self, key, value)
        update_dict = get_update_dict(self.parent_keys, {key: value})
        self.data[key] = node_attr_dict_value
        root_data = self.root.data if self.root else self.data
        root_data["_rev"] = doc_update(self.db, self.node_id, update_dict)

    @key_is_string
    @key_is_not_reserved
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """del G._node['node/1']['foo']"""
        assert self.node_id
        self.data.pop(key, None)
        update_dict = get_update_dict(self.parent_keys, {key: None})
        root_data = self.root.data if self.root else self.data
        root_data["_rev"] = doc_update(self.db, self.node_id, update_dict)

    @keys_are_strings
    @keys_are_not_reserved
    # @values_are_json_serializable # TODO?
    @logger_debug
    def update(self, attrs: Any) -> None:
        """G._node['node/1'].update({'foo': 'bar'})"""
        if not attrs:
            return

        self.data.update(build_node_attr_dict_data(self, attrs))

        if not self.node_id:
            logger.debug("Node ID not set, skipping NodeAttrDict(?).update()")
            return

        update_dict = get_update_dict(self.parent_keys, attrs)
        root_data = self.root.data if self.root else self.data
        root_data["_rev"] = doc_update(self.db, self.node_id, update_dict)


class NodeDict(UserDict[str, NodeAttrDict]):
    """The outer-level of the dict of dict structure representing the
    nodes (vertices) of a graph.

    The outer dict is keyed by ArangoDB Vertex IDs and the inner dict
    is keyed by Vertex attributes.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    :param default_node_type: The default node type. Used if the node ID
        is not formatted as 'type/id'.
    :type default_node_type: str
    """

    @logger_debug
    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        default_node_type: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.data: dict[str, NodeAttrDict] = {}

        self.db = db
        self.graph = graph
        self.default_node_type = default_node_type
        self.node_attr_dict_factory = node_attr_dict_factory(self.db, self.graph)

        self.FETCHED_ALL_DATA = False

    @key_is_string
    @logger_debug
    def __contains__(self, key: str) -> bool:
        """'node/1' in G._node"""
        node_id = get_node_id(key, self.default_node_type)

        if node_id in self.data:
            return True

        if self.FETCHED_ALL_DATA:
            return False

        return bool(self.graph.has_vertex(node_id))

    @key_is_string
    @logger_debug
    def __getitem__(self, key: str) -> NodeAttrDict:
        """G._node['node/1']"""
        node_id = get_node_id(key, self.default_node_type)

        if vertex := self.data.get(node_id):
            return vertex

        if self.FETCHED_ALL_DATA:
            raise KeyError(key)

        if vertex := self.graph.vertex(node_id):
            node_attr_dict: NodeAttrDict = self.node_attr_dict_factory()
            node_attr_dict.node_id = node_id
            node_attr_dict.data = build_node_attr_dict_data(node_attr_dict, vertex)
            self.data[node_id] = node_attr_dict

            return node_attr_dict

        raise KeyError(key)

    @key_is_string
    @logger_debug
    def __setitem__(self, key: str, value: NodeAttrDict) -> None:
        """G._node['node/1'] = {'foo': 'bar'}

        Not to be confused with:
        - G.add_node('node/1', foo='bar')
        """
        assert isinstance(value, NodeAttrDict)

        node_type, node_id = get_node_type_and_id(key, self.default_node_type)

        result = doc_insert(self.db, node_type, node_id, value.data)

        node_attr_dict = self.node_attr_dict_factory()
        node_attr_dict.node_id = node_id
        node_attr_dict.data = build_node_attr_dict_data(node_attr_dict, result)

        self.data[node_id] = node_attr_dict

    @key_is_string
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """del g._node['node/1']"""
        node_id = get_node_id(key, self.default_node_type)

        if not self.graph.has_vertex(node_id):
            raise KeyError(key)

        # TODO: wrap in edges_delete() method
        remove_statements = "\n".join(
            f"REMOVE e IN `{edge_def['edge_collection']}` OPTIONS {{ignoreErrors: true}}"  # noqa
            for edge_def in self.graph.edge_definitions()
        )

        query = f"""
            FOR v, e IN 1..1 ANY @src_node_id GRAPH @graph_name
                {remove_statements}
        """

        bind_vars = {"src_node_id": node_id, "graph_name": self.graph.name}

        aql(self.db, query, bind_vars)
        #####

        doc_delete(self.db, node_id)

        self.data.pop(node_id, None)

    @logger_debug
    def __len__(self) -> int:
        """len(g._node)"""
        return sum(
            [
                self.graph.vertex_collection(c).count()
                for c in self.graph.vertex_collections()
            ]
        )

    @logger_debug
    def __iter__(self) -> Iterator[str]:
        """iter(g._node)"""
        if self.FETCHED_ALL_DATA:
            yield from self.data.keys()
        else:
            for collection in self.graph.vertex_collections():
                yield from self.graph.vertex_collection(collection).ids()

    @logger_debug
    def keys(self) -> Any:
        """g._node.keys()"""
        return self.__iter__()

    @logger_debug
    def clear(self) -> None:
        """g._node.clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False

        # if clear_remote:
        #     for collection in self.graph.vertex_collections():
        #         self.graph.vertex_collection(collection).truncate()

    @keys_are_strings
    @logger_debug
    def update_local_nodes(self, nodes: Any) -> None:
        for node_id, node_data in nodes.items():
            node_attr_dict = self.node_attr_dict_factory()
            node_attr_dict.node_id = node_id
            node_attr_dict.data = build_node_attr_dict_data(node_attr_dict, node_data)

            self.data[node_id] = node_attr_dict

    @keys_are_strings
    @logger_debug
    def update(self, nodes: Any) -> None:
        """g._node.update({'node/1': {'foo': 'bar'}, 'node/2': {'baz': 'qux'}})"""
        separated_by_collection = separate_nodes_by_collections(
            nodes, self.default_node_type
        )

        result = upsert_collection_documents(self.db, separated_by_collection)

        all_good = check_list_for_errors(result)
        if all_good:
            # Means no single operation failed, in this case we update the local cache
            self.update_local_nodes(nodes)
        else:
            # In this case some or all documents failed. Right now we will not update
            # the local cache, but raise an error instead.
            # Reason: We cannot set silent to True, because we need as it does not report errors then.
            # We need to update the driver to also pass the errors back to the user, then we can adjust
            # the behavior here. This will also save network traffic and local computation time.
            errors = []
            for collections_results in result:
                for collection_result in collections_results:
                    errors.append(collection_result)
            warnings.warn(
                "Failed to insert at least one node. Will not update local cache."
            )
            raise ArangoDBBatchError(errors)

    # TODO: Revisit typing of return value
    @logger_debug
    def values(self) -> Any:
        """g._node.values()"""
        if not self.FETCHED_ALL_DATA:
            self.__fetch_all()

        yield from self.data.values()

    # TODO: Revisit typing of return value
    @logger_debug
    def items(self, data: str | None = None, default: Any | None = None) -> Any:
        """g._node.items() or G._node.items(data='foo')"""
        if data is None:
            if not self.FETCHED_ALL_DATA:
                self.__fetch_all()

            yield from self.data.items()
        else:
            v_cols = list(self.graph.vertex_collections())
            result = aql_fetch_data(self.db, v_cols, data, default)
            yield from result.items()

    @logger_debug
    def __fetch_all(self):
        self.clear()

        node_dict, _, _, _, _ = get_arangodb_graph(
            self.graph,
            load_node_dict=True,
            load_adj_dict=False,
            load_adj_dict_as_directed=False,  # not used
            load_adj_dict_as_multigraph=False,  # not used
            load_coo=False,
        )

        for node_id, node_data in node_dict.items():
            node_attr_dict = self.node_attr_dict_factory()
            node_attr_dict.node_id = node_id
            node_attr_dict.data = build_node_attr_dict_data(node_attr_dict, node_data)

            self.data[node_id] = node_attr_dict

        self.FETCHED_ALL_DATA = True


#############
# Adjacency #
#############


def process_edge_attr_dict_value(parent: EdgeAttrDict, key: str, value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    edge_attr_dict = parent.edge_attr_dict_factory()
    edge_attr_dict.root = parent.root or parent
    edge_attr_dict.edge_id = parent.edge_id
    edge_attr_dict.parent_keys = parent.parent_keys + [key]
    edge_attr_dict.data = build_edge_attr_dict_data(edge_attr_dict, value)

    return edge_attr_dict


def build_edge_attr_dict_data(
    parent: EdgeAttrDict, data: dict[str, Any]
) -> dict[str, Any | EdgeAttrDict]:
    """Recursively build an EdgeAttrDict from a dict.

    It's possible that **value** is a nested dict, so we need to
    recursively build a EdgeAttrDict for each nested dict.

    :param parent: The parent EdgeAttrDict.
    :type parent: EdgeAttrDict
    :param data: The data to build the EdgeAttrDict from.
    :type data: dict[str, Any]
    """
    edge_attr_dict_data = {}
    for key, value in data.items():
        edge_attr_dict_value = process_edge_attr_dict_value(parent, key, value)
        edge_attr_dict_data[key] = edge_attr_dict_value

    return edge_attr_dict_data


class EdgeAttrDict(UserDict[str, Any]):
    """The innermost-level of the dict of dict of dict structure
    representing the Adjacency List of a graph.

    The innermost-dict is keyed by the edge attribute key.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    """

    @logger_debug
    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.data: dict[str, Any] = {}

        self.db = db
        self.graph = graph
        self.edge_id: str | None = None

        # NodeAttrDict may be a child of another NodeAttrDict
        # e.g G._adj['node/1']['node/2']['object']['foo'] = 'bar'
        # In this case, **parent_keys** would be ['object']
        # and **root** would be G._adj['node/1']['node/2']
        self.root: EdgeAttrDict | None = None
        self.parent_keys: list[str] = []
        self.edge_attr_dict_factory = edge_attr_dict_factory(self.db, self.graph)

    @key_is_string
    @logger_debug
    def __contains__(self, key: str) -> bool:
        """'foo' in G._adj['node/1']['node/2']"""
        if key in self.data:
            return True

        assert self.edge_id
        return aql_doc_has_key(self.db, self.edge_id, key, self.parent_keys)

    @key_is_string
    @logger_debug
    def __getitem__(self, key: str) -> Any:
        """G._adj['node/1']['node/2']['foo']"""
        if value := self.data.get(key):
            return value

        assert self.edge_id
        result = aql_doc_get_key(self.db, self.edge_id, key, self.parent_keys)

        if not result:
            raise KeyError(key)

        edge_attr_dict_value = process_edge_attr_dict_value(self, key, result)
        self.data[key] = edge_attr_dict_value

        return edge_attr_dict_value

    @key_is_string
    @key_is_not_reserved
    # @value_is_json_serializable # TODO?
    @logger_debug
    def __setitem__(self, key: str, value: Any) -> None:
        """G._adj['node/1']['node/2']['foo'] = 'bar'"""
        assert self.edge_id
        edge_attr_dict_value = process_edge_attr_dict_value(self, key, value)
        update_dict = get_update_dict(self.parent_keys, {key: value})
        self.data[key] = edge_attr_dict_value
        root_data = self.root.data if self.root else self.data
        root_data["_rev"] = doc_update(self.db, self.edge_id, update_dict)

    @key_is_string
    @key_is_not_reserved
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """del G._adj['node/1']['node/2']['foo']"""
        assert self.edge_id
        self.data.pop(key, None)
        update_dict = get_update_dict(self.parent_keys, {key: None})
        root_data = self.root.data if self.root else self.data
        root_data["_rev"] = doc_update(self.db, self.edge_id, update_dict)

    @keys_are_strings
    @keys_are_not_reserved
    @logger_debug
    def update(self, attrs: Any) -> None:
        """G._adj['node/1']['node/'2].update({'foo': 'bar'})"""
        if not attrs:
            return

        self.data.update(build_edge_attr_dict_data(self, attrs))

        if not self.edge_id:
            logger.debug("Edge ID not set, skipping EdgeAttrDict(?).update()")
            return

        update_dict = get_update_dict(self.parent_keys, attrs)
        root_data = self.root.data if self.root else self.data
        root_data["_rev"] = doc_update(self.db, self.edge_id, update_dict)


class AdjListInnerDict(UserDict[str, EdgeAttrDict]):
    """The inner-level of the dict of dict of dict structure
    representing the Adjacency List of a graph.

    The inner-dict is keyed by the node ID of the destination node.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    :param default_node_type: The default node type.
    :type default_node_type: str
    :param edge_type_func: The function to generate the edge type.
    :type edge_type_func: Callable[[str, str], str]
    """

    @logger_debug
    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        default_node_type: str,
        edge_type_func: Callable[[str, str], str],
        adjlist_outer_dict: AdjListOuterDict | None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.data: dict[str, EdgeAttrDict] = {}

        self.db = db
        self.graph = graph
        self.edge_type_func = edge_type_func
        self.default_node_type = default_node_type
        self.edge_attr_dict_factory = edge_attr_dict_factory(self.db, self.graph)

        self.src_node_id: str | None = None
        self.adjlist_outer_dict = adjlist_outer_dict

        self.FETCHED_ALL_DATA = False

    @logger_debug
    def __get_mirrored_edge_attr_dict(self, dst_node_id: str) -> EdgeAttrDict | None:
        if self.adjlist_outer_dict is None:
            return None

        if dst_node_id in self.adjlist_outer_dict.data:
            if self.src_node_id in self.adjlist_outer_dict.data[dst_node_id].data:
                return self.adjlist_outer_dict.data[dst_node_id].data[self.src_node_id]

        return None

    @logger_debug
    def __repr__(self) -> str:
        return f"'{self.src_node_id}'"

    @logger_debug
    def __str__(self) -> str:
        return f"'{self.src_node_id}'"

    @key_is_string
    @logger_debug
    def __contains__(self, key: str) -> bool:
        """'node/2' in G.adj['node/1']"""
        assert self.src_node_id
        dst_node_id = get_node_id(key, self.default_node_type)

        if dst_node_id in self.data:
            return True

        if self.FETCHED_ALL_DATA:
            return False

        result = aql_edge_exists(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction="ANY",
        )

        return result if result else False

    @key_is_string
    @logger_debug
    def __getitem__(self, key: str) -> EdgeAttrDict:
        """g._adj['node/1']['node/2']"""
        dst_node_id = get_node_id(key, self.default_node_type)

        if edge := self.data.get(dst_node_id):
            return edge

        if edge := self.__get_mirrored_edge_attr_dict(dst_node_id):
            self.data[dst_node_id] = edge
            return edge  # type: ignore # false positive

        if self.FETCHED_ALL_DATA:
            raise KeyError(key)

        assert self.src_node_id
        edge = aql_edge_get(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction="ANY",
        )

        if not edge:
            raise KeyError(key)

        edge_attr_dict = self.edge_attr_dict_factory()
        edge_attr_dict.edge_id = edge["_id"]
        edge_attr_dict.data = build_edge_attr_dict_data(edge_attr_dict, edge)
        self.data[dst_node_id] = edge_attr_dict

        return edge_attr_dict

    @key_is_string
    @logger_debug
    def __setitem__(self, key: str, value: dict[str, Any] | EdgeAttrDict) -> None:
        """g._adj['node/1']['node/2'] = {'foo': 'bar'}"""
        assert isinstance(value, EdgeAttrDict)
        assert self.src_node_id

        src_node_type = self.src_node_id.split("/")[0]
        dst_node_type, dst_node_id = get_node_type_and_id(key, self.default_node_type)

        if edge := self.__get_mirrored_edge_attr_dict(dst_node_id):
            self.data[dst_node_id] = edge
            return

        edge_type = value.data.get("_edge_type")
        if edge_type is None:
            edge_type = self.edge_type_func(src_node_type, dst_node_type)

        edge_id: str | None
        if value.edge_id:
            self.graph.delete_edge(value.edge_id)

        elif edge_id := aql_edge_id(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction="ANY",
        ):
            self.graph.delete_edge(edge_id)

        edge_data = value.data
        edge = self.graph.link(edge_type, self.src_node_id, dst_node_id, edge_data)

        edge_attr_dict = self.edge_attr_dict_factory()
        edge_attr_dict.edge_id = edge["_id"]
        edge_data = {**edge_data, **edge, "_from": self.src_node_id, "_to": dst_node_id}
        edge_attr_dict.data = build_edge_attr_dict_data(edge_attr_dict, edge_data)
        self.data[dst_node_id] = edge_attr_dict

    @key_is_string
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """del g._adj['node/1']['node/2']"""
        assert self.src_node_id
        dst_node_id = get_node_id(key, self.default_node_type)
        self.data.pop(dst_node_id, None)

        if self.__get_mirrored_edge_attr_dict(dst_node_id):
            return

        edge_id = aql_edge_id(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction="ANY",
        )

        if not edge_id:
            return

        self.graph.delete_edge(edge_id)

    @logger_debug
    def __len__(self) -> int:
        """len(g._adj['node/1'])"""
        assert self.src_node_id

        if self.FETCHED_ALL_DATA:
            return len(self.data)

        # TODO: Create aql_edge_count() function
        query = """
            RETURN LENGTH(
                FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                    RETURN 1
            )
        """

        bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

        result = aql_single(self.db, query, bind_vars)
        #####

        if result is None:
            return 0

        return int(result)

    @logger_debug
    def __iter__(self) -> Iterator[str]:
        """for k in g._adj['node/1']"""
        if self.FETCHED_ALL_DATA:
            yield from self.data.keys()

        else:
            query = """
                FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                    RETURN e._to
            """

            bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

            yield from aql(self.db, query, bind_vars)

    # TODO: Revisit typing of return value
    @logger_debug
    def keys(self) -> Any:
        """g._adj['node/1'].keys()"""
        return self.__iter__()

    @logger_debug
    def clear(self) -> None:
        """G._adj['node/1'].clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False

    @keys_are_strings
    @logger_debug
    def update(self, edges: Any) -> None:
        """g._adj['node/1'].update({'node/2': {'foo': 'bar'}})"""
        raise NotImplementedError("AdjListInnerDict.update()")

    # TODO: Revisit typing of return value
    @logger_debug
    def values(self) -> Any:
        """g._adj['node/1'].values()"""
        if not self.FETCHED_ALL_DATA:
            self.__fetch_all()

        yield from self.data.values()

    # TODO: Revisit typing of return value
    @logger_debug
    def items(self) -> Any:
        """g._adj['node/1'].items()"""
        if not self.FETCHED_ALL_DATA:
            self.__fetch_all()

        yield from self.data.items()

    @logger_debug
    def __fetch_all(self) -> None:
        assert self.src_node_id

        self.clear()

        query = """
            FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                RETURN e
        """

        bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

        for edge in aql(self.db, query, bind_vars):
            edge_attr_dict = self.edge_attr_dict_factory()
            edge_attr_dict.edge_id = edge["_id"]
            edge_attr_dict.data = build_edge_attr_dict_data(edge_attr_dict, edge)
            self.data[edge["_to"]] = edge_attr_dict

        self.FETCHED_ALL_DATA = True


class AdjListOuterDict(UserDict[str, AdjListInnerDict]):
    """The outer-level of the dict of dict of dict structure
    representing the Adjacency List of a graph.

    The outer-dict is keyed by the node ID of the source node.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    :param default_node_type: The default node type.
    :type default_node_type: str
    :param edge_type_func: The function to generate the edge type.
    :type edge_type_func: Callable[[str, str], str]
    """

    @logger_debug
    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        default_node_type: str,
        edge_type_func: Callable[[str, str], str],
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.data: dict[str, AdjListInnerDict] = {}

        self.db = db
        self.graph = graph
        self.edge_type_func = edge_type_func
        self.default_node_type = default_node_type
        self.adjlist_inner_dict_factory = adjlist_inner_dict_factory(
            db, graph, default_node_type, edge_type_func, self
        )

        self.FETCHED_ALL_DATA = False

    @logger_debug
    def __repr__(self) -> str:
        return f"'{self.graph.name}'"

    @logger_debug
    def __str__(self) -> str:
        return f"'{self.graph.name}'"

    @key_is_string
    @logger_debug
    def __contains__(self, key: str) -> bool:
        """'node/1' in G.adj"""
        node_id = get_node_id(key, self.default_node_type)

        if node_id in self.data:
            return True

        if self.FETCHED_ALL_DATA:
            return False

        return bool(self.graph.has_vertex(node_id))

    @key_is_string
    @logger_debug
    def __getitem__(self, key: str) -> AdjListInnerDict:
        """G.adj["node/1"]"""
        node_id = get_node_id(key, self.default_node_type)

        if value := self.data.get(node_id):
            return value

        if self.graph.has_vertex(node_id):
            adjlist_inner_dict: AdjListInnerDict = self.adjlist_inner_dict_factory()
            adjlist_inner_dict.src_node_id = node_id
            self.data[node_id] = adjlist_inner_dict

            return adjlist_inner_dict

        raise KeyError(key)

    @key_is_string
    @logger_debug
    def __setitem__(self, src_key: str, adjlist_inner_dict: AdjListInnerDict) -> None:
        """
        g._adj['node/1'] = AdjListInnerDict()
        """
        assert isinstance(adjlist_inner_dict, AdjListInnerDict)
        assert len(adjlist_inner_dict.data) == 0  # See NOTE below

        src_node_type, src_node_id = get_node_type_and_id(
            src_key, self.default_node_type
        )

        # NOTE: this might not actually be needed...
        # results = {}
        # for dst_key, edge_dict in adjlist_inner_dict.data.items():
        #     dst_node_type, dst_node_id = get_node_type_and_id(
        #         dst_key, self.default_node_type
        #     )

        #     edge_type = edge_dict.get("_edge_type")
        #     if edge_type is None:
        #         edge_type = self.edge_type_func(src_node_type, dst_node_type)

        #     results[dst_key] = self.graph.link(
        #         edge_type, src_node_id, dst_node_id, edge_dict
        #     )

        adjlist_inner_dict.src_node_id = src_node_id
        # adjlist_inner_dict.data = results

        self.data[src_node_id] = adjlist_inner_dict

    @key_is_string
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """
        del G._adj['node/1']
        """
        # Nothing else to do here, as this delete is always invoked by
        # G.remove_node(), which already removes all edges via
        # del G._node['node/1']
        node_id = get_node_id(key, self.default_node_type)
        self.data.pop(node_id, None)

    @logger_debug
    def __len__(self) -> int:
        """len(g._adj)"""
        return sum(
            [
                self.graph.vertex_collection(c).count()
                for c in self.graph.vertex_collections()
            ]
        )

    @logger_debug
    def __iter__(self) -> Iterator[str]:
        """for k in g._adj"""
        if self.FETCHED_ALL_DATA:
            yield from self.data.keys()

        else:
            for collection in self.graph.vertex_collections():
                yield from self.graph.vertex_collection(collection).ids()

    # TODO: Revisit typing of return value
    @logger_debug
    def keys(self) -> Any:
        """g._adj.keys()"""
        return self.__iter__()

    @logger_debug
    def clear(self) -> None:
        """g._node.clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False

        # if clear_remote:
        #     for ed in self.graph.edge_definitions():
        #         self.graph.edge_collection(ed["edge_collection"]).truncate()

    @keys_are_strings
    @logger_debug
    def update(self, edges: Any) -> None:
        """g._adj.update({'node/1': {'node/2': {'foo': 'bar'}})"""
        raise NotImplementedError("AdjListOuterDict.update()")

    # TODO: Revisit typing of return value
    @logger_debug
    def values(self) -> Any:
        """g._adj.values()"""
        if not self.FETCHED_ALL_DATA:
            self.__fetch_all()

        yield from self.data.values()

    # TODO: Revisit typing of return value
    @logger_debug
    def items(self, data: str | None = None, default: Any | None = None) -> Any:
        # TODO: Revisit typing
        # -> (
        #     Generator[tuple[str, AdjListInnerDict], None, None]
        #     | Generator[tuple[str, str, Any], None, None]
        # ):
        """g._adj.items() or G._adj.items(data='foo')"""
        if data is None:
            if not self.FETCHED_ALL_DATA:
                self.__fetch_all()

            yield from self.data.items()

        else:
            e_cols = [ed["edge_collection"] for ed in self.graph.edge_definitions()]
            result = aql_fetch_data_edge(self.db, e_cols, data, default)
            yield from result

    @logger_debug
    def __fetch_all(self) -> None:
        self.clear()

        _, adj_dict, _, _, _ = get_arangodb_graph(
            self.graph,
            load_node_dict=False,
            load_adj_dict=True,
            load_adj_dict_as_directed=False,  # TODO: Abstract based on Graph type
            load_adj_dict_as_multigraph=False,  # TODO: Abstract based on Graph type
            load_coo=False,
        )

        for src_node_id, inner_dict in adj_dict.items():
            for dst_node_id, edge in inner_dict.items():

                if src_node_id in self.data:
                    if dst_node_id in self.data[src_node_id].data:
                        continue

                if src_node_id in self.data:
                    src_inner_dict = self.data[src_node_id]
                else:
                    src_inner_dict = self.adjlist_inner_dict_factory()
                    src_inner_dict.src_node_id = src_node_id
                    src_inner_dict.FETCHED_ALL_DATA = True
                    self.data[src_node_id] = src_inner_dict

                if dst_node_id in self.data:
                    dst_inner_dict = self.data[dst_node_id]
                else:
                    dst_inner_dict = self.adjlist_inner_dict_factory()
                    dst_inner_dict.src_node_id = dst_node_id
                    src_inner_dict.FETCHED_ALL_DATA = True
                    self.data[dst_node_id] = dst_inner_dict

                edge_attr_dict = src_inner_dict.edge_attr_dict_factory()
                edge_attr_dict.edge_id = edge["_id"]
                edge_attr_dict.data = build_edge_attr_dict_data(edge_attr_dict, edge)

                self.data[src_node_id].data[dst_node_id] = edge_attr_dict
                self.data[dst_node_id].data[src_node_id] = edge_attr_dict

        self.FETCHED_ALL_DATA = True
