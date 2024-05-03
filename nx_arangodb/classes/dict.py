from __future__ import annotations

from collections import UserDict
from collections.abc import Iterator
from typing import Any, Callable

from arango.database import StandardDatabase
from arango.graph import Graph

from .function import (
    aql,
    aql_doc_get_key,
    aql_doc_get_keys,
    aql_doc_get_length,
    aql_doc_has_key,
    aql_single,
    create_collection,
    doc_delete,
    doc_get_or_insert,
    doc_insert,
    doc_update,
    get_node_id,
    get_node_type_and_id,
    key_is_not_reserved,
    key_is_string,
    keys_are_not_reserved,
    keys_are_strings,
)


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


class GraphDict(UserDict):
    """A dictionary-like object for storing graph attributes.

    Given that ArangoDB does not have a concept of graph attributes, this class
    stores the attributes in a collection with the graph name as the document key.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph_name: The graph name.
    :type graph_name: str
    """

    COLLECTION_NAME = "NXADB_GRAPH_ATTRIBUTES"

    def __init__(self, db: StandardDatabase, graph_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.db = db
        self.graph_name = graph_name
        self.graph_id = f"{self.COLLECTION_NAME}/{graph_name}"

        self.adb_graph = db.graph(graph_name)
        self.collection = create_collection(db, self.COLLECTION_NAME)

        data = doc_get_or_insert(self.db, self.COLLECTION_NAME, self.graph_id)
        self.data.update(data)

    @key_is_string
    def __contains__(self, key: str) -> bool:
        """'foo' in G.graph"""
        if key in self.data:
            return True

        return aql_doc_has_key(self.db, self.graph_id, key)

    @key_is_string
    def __getitem__(self, key: Any) -> Any:
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
    # @value_is_json_serializable # TODO?
    def __setitem__(self, key: str, value: Any):
        """G.graph['foo'] = 'bar'"""
        self.data[key] = value
        doc_update(self.db, self.graph_id, {key: value})

    @key_is_string
    @key_is_not_reserved
    def __delitem__(self, key):
        """del G.graph['foo']"""
        del self.data[key]
        doc_update(self.db, self.graph_id, {key: None})

    @keys_are_strings
    @keys_are_not_reserved
    # @values_are_json_serializable # TODO?
    def update(self, attrs, *args, **kwargs):
        """G.graph.update({'foo': 'bar'})"""
        self.data.update(attrs, *args, **kwargs)
        doc_update(self.db, self.graph_id, attrs)

    def clear(self, clear_remote: bool = False):
        """G.graph.clear()"""
        self.data.clear()

        if clear_remote:
            doc_insert(self.db, self.COLLECTION_NAME, self.graph_id, silent=True)


class NodeDict(UserDict):
    """The outer-level of the dict of dict structure representing the nodes (vertices) of a graph.

    The outer dict is keyed by ArangoDB Vertex IDs and the inner dict is keyed by Vertex attributes.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    :param default_node_type: The default node type. Used if the node ID is not formatted as 'type/id'.
    :type default_node_type: str
    """

    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        default_node_type: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.db = db
        self.graph = graph
        self.default_node_type = default_node_type
        self.node_attr_dict_factory = node_attr_dict_factory(self.db, self.graph)

    @key_is_string
    def __contains__(self, key: str) -> bool:
        """'node/1' in G._node"""
        node_id = get_node_id(key, self.default_node_type)

        if node_id in self.data:
            return True

        return self.graph.has_vertex(node_id)

    @key_is_string
    def __getitem__(self, key: str) -> NodeAttrDict:
        """G._node['node/1']"""
        node_id = get_node_id(key, self.default_node_type)

        if value := self.data.get(node_id):
            return value

        if value := self.graph.vertex(node_id):
            node_attr_dict: NodeAttrDict = self.node_attr_dict_factory()
            node_attr_dict.node_id = node_id
            node_attr_dict.data = value

            self.data[node_id] = node_attr_dict

            return node_attr_dict

        raise KeyError(key)

    @key_is_string
    def __setitem__(self, key: str, value: NodeAttrDict):
        """G._node['node/1'] = {'foo': 'bar'}

        Not to be confused with:
        - G.add_node('node/1', foo='bar')
        """
        assert isinstance(value, NodeAttrDict)

        node_type, node_id = get_node_type_and_id(key, self.default_node_type)

        result = doc_insert(self.db, node_type, node_id, value.data)

        node_attr_dict = self.node_attr_dict_factory()
        node_attr_dict.node_id = node_id
        node_attr_dict.data = result

        self.data[node_id] = node_attr_dict

    @key_is_string
    def __delitem__(self, key: Any) -> None:
        """del g._node['node/1']"""
        node_id = get_node_id(key, self.default_node_type)

        del self.data[node_id]
        doc_delete(self.db, node_id)

    def __len__(self) -> int:
        """len(g._node)"""
        return sum(
            [
                self.graph.vertex_collection(c).count()
                for c in self.graph.vertex_collections()
            ]
        )

    def __iter__(self) -> Iterator[str]:
        """iter(g._node)"""
        for collection in self.graph.vertex_collections():
            for node_id in self.graph.vertex_collection(collection).ids():
                yield node_id

    def clear(self, clear_remote: bool = False):
        """g._node.clear()"""
        self.data.clear()

        if clear_remote:
            for collection in self.graph.vertex_collections():
                self.graph.vertex_collection(collection).truncate()

    @keys_are_strings
    def update(self, nodes: dict[str, dict[str, Any]]):
        """g._node.update({'node/1': {'foo': 'bar'}, 'node/2': {'baz': 'qux'}})"""
        raise NotImplementedError("NodeDict.update()")
        # for node_id, attrs in nodes.items():
        #     node_id = get_node_id(node_id, self.default_node_type)

        #     result = doc_update(self.db, node_id, attrs)

        #     node_attr_dict = self.node_attr_dict_factory()
        #     node_attr_dict.node_id = node_id
        #     node_attr_dict.data = result

        #     self.data[node_id] = node_attr_dict

    def keys(self):
        """g._node.keys()"""
        return self.__iter__()

    def values(self, cache: bool = True):
        """g._node.values()"""
        for collection in self.graph.vertex_collections():
            for doc in self.graph.vertex_collection(collection).all():
                node_id = doc["_id"]

                node_attr_dict = self.node_attr_dict_factory()
                node_attr_dict.node_id = node_id
                node_attr_dict.data = doc

                if cache:
                    self.data[node_id] = node_attr_dict

                yield node_attr_dict

    def items(self, cache: bool = True):
        """g._node.items()"""
        for collection in self.graph.vertex_collections():
            for doc in self.graph.vertex_collection(collection).all():
                node_id = doc["_id"]

                node_attr_dict = self.node_attr_dict_factory()
                node_attr_dict.node_id = node_id
                node_attr_dict.data = doc

                if cache:
                    self.data[node_id] = node_attr_dict

                yield node_id, node_attr_dict


class NodeAttrDict(UserDict):
    """The inner-level of the dict of dict structure representing the nodes (vertices) of a graph.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    """

    def __init__(self, db: StandardDatabase, graph: Graph, *args, **kwargs):
        self.db = db
        self.graph = graph
        self.node_id: str | None = None

        super().__init__(*args, **kwargs)

    @key_is_string
    def __contains__(self, key: str) -> bool:
        """'foo' in G._node['node/1']"""
        if key in self.data:
            return True

        return aql_doc_has_key(self.db, self.node_id, key)

    @key_is_string
    def __getitem__(self, key: str) -> Any:
        """G._node['node/1']['foo']"""
        if value := self.data.get(key):
            return value

        result = aql_doc_get_key(self.db, self.node_id, key)

        if not result:
            raise KeyError(key)

        self.data[key] = result

        return result

    @key_is_string
    @key_is_not_reserved
    # @value_is_json_serializable # TODO?
    def __setitem__(self, key: str, value: Any):
        """G._node['node/1']['foo'] = 'bar'"""
        self.data[key] = value
        doc_update(self.db, self.node_id, {key: value})

    @key_is_string
    @key_is_not_reserved
    def __delitem__(self, key: str):
        """del G._node['node/1']['foo']"""
        del self.data[key]
        doc_update(self.db, self.node_id, {key: None})

    def __iter__(self) -> Iterator[str]:
        """for key in G._node['node/1']"""
        for key in aql_doc_get_keys(self.db, self.node_id):
            yield key

    def __len__(self) -> int:
        """len(G._node['node/1'])"""
        return aql_doc_get_length(self.db, self.node_id)

    def keys(self):
        """G._node['node/1'].keys()"""
        return self.__iter__()

    def values(self, cache: bool = True):
        """G._node['node/1'].values()"""
        doc = self.db.document(self.node_id)

        if cache:
            self.data = doc

        yield from doc.values()

    def items(self, cache: bool = True):
        """G._node['node/1'].items()"""
        doc = self.db.document(self.node_id)

        if cache:
            self.data = doc

        yield from doc.items()

    def clear(self, clear_remote: bool = False):
        """G._node['node/1'].clear()"""
        self.data.clear()

        if clear_remote:
            doc_insert(self.db, self.node_id, silent=True, overwrite=True)

    def update(self, attrs: dict[str, Any]):
        """G._node['node/1'].update({'foo': 'bar'})"""
        self.data.update(attrs)
        doc_update(self.db, self.node_id, attrs)
