from __future__ import annotations

from collections import UserDict
from collections.abc import Iterator
from typing import Any, Callable

from arango.database import StandardDatabase
from arango.graph import Graph

from nx_arangodb.logger import logger

from ..function import (
    ArangoDBBatchError,
    aql,
    aql_doc_get_key,
    aql_doc_has_key,
    aql_fetch_data,
    check_update_list_for_errors,
    doc_delete,
    doc_insert,
    doc_update,
    edges_delete,
    get_arangodb_graph,
    get_node_id,
    get_node_type_and_id,
    get_update_dict,
    json_serializable,
    key_is_not_reserved,
    key_is_string,
    keys_are_not_reserved,
    keys_are_strings,
    separate_nodes_by_collections,
    upsert_collection_documents,
    vertex_get,
)

#############
# Factories #
#############


def node_dict_factory(
    db: StandardDatabase, graph: Graph, default_node_type: str
) -> Callable[..., NodeDict]:
    """Factory function for creating a NodeDict."""
    return lambda: NodeDict(db, graph, default_node_type)


def node_attr_dict_factory(
    db: StandardDatabase, graph: Graph
) -> Callable[..., NodeAttrDict]:
    """Factory function for creating a NodeAttrDict."""
    return lambda: NodeAttrDict(db, graph)


########
# Node #
########


def build_node_attr_dict_data(
    parent: NodeAttrDict, data: dict[str, Any]
) -> dict[str, Any | NodeAttrDict]:
    """Recursively build a NodeAttrDict from a dict.

    It's possible that **value** is a nested dict, so we need to
    recursively build a NodeAttrDict for each nested dict.

    Parameters
    ----------
    parent : NodeAttrDict
        The parent NodeAttrDict.
    data : dict[str, Any]
        The data to build the NodeAttrDict from.

    Returns
    -------
    dict[str, Any | NodeAttrDict]
        The data for the new NodeAttrDict.
    """
    node_attr_dict_data = {}
    for key, value in data.items():
        node_attr_dict_value = process_node_attr_dict_value(parent, key, value)
        node_attr_dict_data[key] = node_attr_dict_value

    return node_attr_dict_data


def process_node_attr_dict_value(parent: NodeAttrDict, key: str, value: Any) -> Any:
    """Process the value of a particular key in a NodeAttrDict.

    If the value is a dict, then we need to recursively build an NodeAttrDict.
    Otherwise, we return the value as is.

    Parameters
    ----------
    parent : NodeAttrDict
        The parent NodeAttrDict.
    key : str
        The key of the value.
    value : Any
        The value to process.

    Returns
    -------
    Any
        The processed value.
    """
    if not isinstance(value, dict):
        return value

    node_attr_dict = parent.node_attr_dict_factory()
    node_attr_dict.node_id = parent.node_id
    node_attr_dict.parent_keys = parent.parent_keys + [key]
    node_attr_dict.data = build_node_attr_dict_data(node_attr_dict, value)

    return node_attr_dict


@json_serializable
class NodeAttrDict(UserDict[str, Any]):
    """The inner-level of the dict of dict structure
    representing the nodes (vertices) of a graph.

    Parameters
    ----------
    db : arango.database.StandardDatabase
        The ArangoDB database.

    graph : arango.graph.Graph
        The ArangoDB graph object.

    Example
    -------
    >>> G = nxadb.Graph("MyGraph")
    >>> G.add_node('node/1', foo='bar')
    >>> G.nodes['node/1']['foo']
    'bar'
    """

    def __init__(self, db: StandardDatabase, graph: Graph, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data: dict[str, Any] = {}

        self.db = db
        self.graph = graph
        self.node_id: str | None = None

        # NodeAttrDict may be a child of another NodeAttrDict
        # e.g G._node['node/1']['object']['foo'] = 'bar'
        # In this case, **parent_keys** would be ['object']
        self.parent_keys: list[str] = []
        self.node_attr_dict_factory = node_attr_dict_factory(self.db, self.graph)

    def clear(self) -> None:
        raise NotImplementedError("Cannot clear NodeAttrDict")

    def copy(self) -> Any:
        return self.data.copy()

    @key_is_string
    def __contains__(self, key: str) -> bool:
        """'foo' in G._node['node/1']"""
        if key in self.data:
            return True

        assert self.node_id
        result: bool = aql_doc_has_key(self.db, self.node_id, key, self.parent_keys)
        return result

    @key_is_string
    def __getitem__(self, key: str) -> Any:
        """G._node['node/1']['foo']"""
        if key in self.data:
            return self.data[key]

        assert self.node_id
        result = aql_doc_get_key(self.db, self.node_id, key, self.parent_keys)

        if result is None:
            raise KeyError(key)

        node_attr_dict_value = process_node_attr_dict_value(self, key, result)
        self.data[key] = node_attr_dict_value

        return node_attr_dict_value

    @key_is_string
    @key_is_not_reserved
    # @value_is_json_serializable # TODO?
    def __setitem__(self, key: str, value: Any) -> None:
        """
        G._node['node/1']['foo'] = 'bar'
        G._node['node/1']['object'] = {'foo': 'bar'}
        G._node['node/1']['object']['foo'] = 'baz'
        """
        if value is None:
            self.__delitem__(key)
            return

        assert self.node_id
        node_attr_dict_value = process_node_attr_dict_value(self, key, value)
        update_dict = get_update_dict(self.parent_keys, {key: value})
        self.data[key] = node_attr_dict_value
        doc_update(self.db, self.node_id, update_dict)

    @key_is_string
    @key_is_not_reserved
    def __delitem__(self, key: str) -> None:
        """del G._node['node/1']['foo']"""
        assert self.node_id
        self.data.pop(key, None)
        update_dict = get_update_dict(self.parent_keys, {key: None})
        doc_update(self.db, self.node_id, update_dict)

    @keys_are_strings
    @keys_are_not_reserved
    # @values_are_json_serializable # TODO?
    def update(self, attrs: Any) -> None:
        """G._node['node/1'].update({'foo': 'bar'})"""
        if not attrs:
            return

        self.data.update(build_node_attr_dict_data(self, attrs))

        if not self.node_id:
            logger.debug("Node ID not set, skipping NodeAttrDict(?).update()")
            return

        update_dict = get_update_dict(self.parent_keys, attrs)
        doc_update(self.db, self.node_id, update_dict)


class NodeDict(UserDict[str, NodeAttrDict]):
    """The outer-level of the dict of dict structure representing the
    nodes (vertices) of a graph.

    The outer dict is keyed by ArangoDB Vertex IDs and the inner dict
    is keyed by Vertex attributes.

    Parameters
    ----------
    db : arango.database.StandardDatabase
        The ArangoDB database.

    graph : arango.graph.Graph
        The ArangoDB graph object.

    default_node_type : str
        The default node type for the graph.

    Example
    -------
    >>> G = nxadb.Graph("MyGraph")
    >>> G.add_node('node/1', foo='bar')
    >>> G.nodes
    """

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
        self.FETCHED_ALL_IDS = False

    def _create_node_attr_dict(self, vertex: dict[str, Any]) -> NodeAttrDict:
        node_attr_dict = self.node_attr_dict_factory()
        node_attr_dict.node_id = vertex["_id"]
        node_attr_dict.data = build_node_attr_dict_data(node_attr_dict, vertex)

        return node_attr_dict

    def __repr__(self) -> str:
        if self.FETCHED_ALL_IDS:
            return self.data.keys().__repr__()

        return f"NodeDict('{self.graph.name}')"

    def __str__(self) -> str:
        return self.__repr__()

    @key_is_string
    def __contains__(self, key: str) -> bool:
        """'node/1' in G._node"""
        node_id = get_node_id(key, self.default_node_type)

        if node_id in self.data:
            return True

        if self.FETCHED_ALL_IDS:
            return False

        if self.graph.has_vertex(node_id):
            empty_node_attr_dict = self.node_attr_dict_factory()
            empty_node_attr_dict.node_id = node_id
            self.data[node_id] = empty_node_attr_dict
            return True

        return False

    @key_is_string
    def __getitem__(self, key: str) -> NodeAttrDict:
        """G._node['node/1']"""
        node_id = get_node_id(key, self.default_node_type)

        if vertex_cache := self.data.get(node_id):
            return vertex_cache

        if node_id not in self.data and self.FETCHED_ALL_IDS:
            raise KeyError(key)

        if vertex_db := vertex_get(self.graph, node_id):
            node_attr_dict = self._create_node_attr_dict(vertex_db)
            self.data[node_id] = node_attr_dict

            return node_attr_dict

        raise KeyError(key)

    @key_is_string
    def __setitem__(self, key: str, value: NodeAttrDict) -> None:
        """G._node['node/1'] = {'foo': 'bar'}

        Not to be confused with:
        - G.add_node('node/1', foo='bar')
        """
        assert isinstance(value, NodeAttrDict)

        node_type, node_id = get_node_type_and_id(key, self.default_node_type)

        result = doc_insert(self.db, node_type, node_id, value.data)

        node_attr_dict = self._create_node_attr_dict(result)

        self.data[node_id] = node_attr_dict

    @key_is_string
    def __delitem__(self, key: str) -> None:
        """del g._node['node/1']"""
        node_id = get_node_id(key, self.default_node_type)

        if not self.graph.has_vertex(node_id):
            raise KeyError(key)

        edges_delete(self.db, self.graph, node_id)

        doc_delete(self.db, node_id)

        self.data.pop(node_id, None)

    def __len__(self) -> int:
        """len(g._node)"""
        return sum(
            [
                self.graph.vertex_collection(c).count()
                for c in self.graph.vertex_collections()
            ]
        )

    def __iter__(self) -> Iterator[str]:
        """for k in g._node"""
        if not (self.FETCHED_ALL_IDS or self.FETCHED_ALL_DATA):
            self._fetch_all()

        yield from self.data.keys()

    def keys(self) -> Any:
        """g._node.keys()"""
        if self.FETCHED_ALL_IDS:
            yield from self.data.keys()
        else:
            self.FETCHED_ALL_IDS = True
            for collection in self.graph.vertex_collections():
                for node_id in self.graph.vertex_collection(collection).ids():
                    empty_node_attr_dict = self.node_attr_dict_factory()
                    empty_node_attr_dict.node_id = node_id
                    self.data[node_id] = empty_node_attr_dict
                    yield node_id

    def clear(self) -> None:
        """g._node.clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False
        self.FETCHED_ALL_IDS = False

    def copy(self) -> Any:
        """g._node.copy()"""
        if not self.FETCHED_ALL_DATA:
            self._fetch_all()

        return {key: value.copy() for key, value in self.data.items()}

    @keys_are_strings
    def __update_local_nodes(self, nodes: Any) -> None:
        for node_id, node_data in nodes.items():
            node_attr_dict = self.node_attr_dict_factory()
            node_attr_dict.node_id = node_id
            node_attr_dict.data = build_node_attr_dict_data(node_attr_dict, node_data)

            self.data[node_id] = node_attr_dict

    @keys_are_strings
    def update(self, nodes: Any) -> None:
        """g._node.update({'node/1': {'foo': 'bar'}, 'node/2': {'baz': 'qux'}})"""
        separated_by_collection = separate_nodes_by_collections(
            nodes, self.default_node_type
        )

        result = upsert_collection_documents(self.db, separated_by_collection)

        all_good = check_update_list_for_errors(result)
        if all_good:
            # Means no single operation failed, in this case we update the local cache
            self.__update_local_nodes(nodes)
        else:
            # In this case some or all documents failed. Right now we will not
            # update the local cache, but raise an error instead.
            # Reason: We cannot set silent to True, because we need as it does
            # not report errors then. We need to update the driver to also pass
            # the errors back to the user, then we can adjust the behavior here.
            # This will also save network traffic and local computation time.
            errors = []
            for collections_results in result:
                for collection_result in collections_results:
                    errors.append(collection_result)
            m = "Failed to insert at least one node. Will not update local cache."
            logger.warning(m)
            raise ArangoDBBatchError(errors)

    def values(self) -> Any:
        """g._node.values()"""
        if not self.FETCHED_ALL_DATA:
            self._fetch_all()

        yield from self.data.values()

    def items(self, data: str | None = None, default: Any | None = None) -> Any:
        """g._node.items() or G._node.items(data='foo')"""
        if data is None:
            if not self.FETCHED_ALL_DATA:
                self._fetch_all()

            yield from self.data.items()
        else:
            v_cols = list(self.graph.vertex_collections())
            yield from aql_fetch_data(self.db, v_cols, data, default)

    def _fetch_all(self):
        self.clear()

        (
            node_dict,
            *_,
        ) = get_arangodb_graph(
            self.graph,
            load_node_dict=True,
            load_adj_dict=False,
            load_coo=False,
            edge_collections_attributes=set(),  # not used
            load_all_vertex_attributes=True,
            load_all_edge_attributes=False,  # not used
            is_directed=False,  # not used
            is_multigraph=False,  # not used
            symmetrize_edges_if_directed=False,  # not used
        )

        for node_id, node_data in node_dict.items():
            del node_data["_rev"]  # TODO: Optimize away via phenolrs
            node_attr_dict = self._create_node_attr_dict(node_data)
            self.data[node_id] = node_attr_dict

        self.FETCHED_ALL_DATA = True
        self.FETCHED_ALL_IDS = True
