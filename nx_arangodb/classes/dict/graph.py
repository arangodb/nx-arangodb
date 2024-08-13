from __future__ import annotations

from collections import UserDict
from typing import Any, Callable

from arango.database import StandardDatabase
from arango.graph import Graph

from ..function import (
    aql_doc_get_key,
    aql_doc_has_key,
    create_collection,
    doc_get_or_insert,
    doc_update,
    get_update_dict,
    json_serializable,
    key_is_not_reserved,
    key_is_string,
    logger_debug,
)

#############
# Factories #
#############


def graph_dict_factory(db: StandardDatabase, graph: Graph) -> Callable[..., GraphDict]:
    return lambda: GraphDict(db, graph)


def graph_attr_dict_factory(
    db: StandardDatabase, graph: Graph, graph_id: str
) -> Callable[..., GraphAttrDict]:
    return lambda: GraphAttrDict(db, graph, graph_id)


#########
# Graph #
#########


def build_graph_attr_dict_data(
    parent: GraphAttrDict, data: dict[str, Any]
) -> dict[str, Any | GraphAttrDict]:
    """Recursively build a GraphAttrDict from a dict.

    It's possible that **value** is a nested dict, so we need to
    recursively build a GraphAttrDict for each nested dict.

    Returns the parent GraphAttrDict.
    """
    graph_attr_dict_data = {}
    for key, value in data.items():
        graph_attr_dict_value = process_graph_attr_dict_value(parent, key, value)
        graph_attr_dict_data[key] = graph_attr_dict_value

    return graph_attr_dict_data


def process_graph_attr_dict_value(parent: GraphAttrDict, key: str, value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    graph_attr_dict = parent.graph_attr_dict_factory()
    graph_attr_dict.root = parent.root or parent
    graph_attr_dict.parent_keys = parent.parent_keys + [key]
    graph_attr_dict.data = build_graph_attr_dict_data(graph_attr_dict, value)

    return graph_attr_dict


class GraphDict(UserDict[str, Any]):
    """A dictionary-like object for storing graph attributes.

    Given that ArangoDB does not have a concept of graph attributes, this class
    stores the attributes in a collection with the graph name as the document key.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph_name: The graph name.
    :type graph_name: str
    """

    @logger_debug
    def __init__(self, db: StandardDatabase, graph: Graph, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data: dict[str, Any] = {}

        self.db = db
        self.adb_graph = graph
        self.graph_name = graph.name
        self.COLLECTION_NAME = "nxadb_graphs"
        self.graph_id = f"{self.COLLECTION_NAME}/{self.graph_name}"

        self.collection = create_collection(db, self.COLLECTION_NAME)
        self.graph_attr_dict_factory = graph_attr_dict_factory(
            self.db, self.adb_graph, self.graph_id
        )

        result = doc_get_or_insert(self.db, self.COLLECTION_NAME, self.graph_id)
        for k, v in result.items():
            self.data[k] = self.__process_graph_dict_value(k, v)

    def __process_graph_dict_value(self, key: str, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        graph_attr_dict = self.graph_attr_dict_factory()
        graph_attr_dict.parent_keys = [key]
        graph_attr_dict.data = build_graph_attr_dict_data(graph_attr_dict, value)

        return graph_attr_dict

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

        if result is None:
            raise KeyError(key)

        graph_dict_value = self.__process_graph_dict_value(key, result)
        self.data[key] = graph_dict_value

        return graph_dict_value

    @key_is_string
    @key_is_not_reserved
    @logger_debug
    def __setitem__(self, key: str, value: Any) -> None:
        """G.graph['foo'] = 'bar'"""
        if value is None:
            self.__delitem__(key)
            return

        graph_dict_value = self.__process_graph_dict_value(key, value)
        self.data[key] = graph_dict_value
        self.data["_rev"] = doc_update(self.db, self.graph_id, {key: value})

    @key_is_string
    @key_is_not_reserved
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """del G.graph['foo']"""
        self.data.pop(key, None)
        self.data["_rev"] = doc_update(self.db, self.graph_id, {key: None})

    # @values_are_json_serializable # TODO?
    @logger_debug
    def update(self, attrs: Any) -> None:
        """G.graph.update({'foo': 'bar'})"""

        if not attrs:
            return

        graph_attr_dict = self.graph_attr_dict_factory()
        graph_attr_dict_data = build_graph_attr_dict_data(graph_attr_dict, attrs)
        graph_attr_dict.data = graph_attr_dict_data

        self.data.update(graph_attr_dict_data)
        self.data["_rev"] = doc_update(self.db, self.graph_id, attrs)

    @logger_debug
    def clear(self) -> None:
        """G.graph.clear()"""
        self.data.clear()


@json_serializable
class GraphAttrDict(UserDict[str, Any]):
    """The inner-level of the dict of dict structure
    representing the attributes of a graph stored in the database.

    Only used if the value associated with a GraphDict key is a dict.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    :param graph_id: The ArangoDB graph ID.
    :type graph_id: str
    """

    @logger_debug
    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        graph_id: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.data: dict[str, Any] = {}

        self.db = db
        self.graph = graph
        self.graph_id: str = graph_id

        self.root: GraphAttrDict | None = None
        self.parent_keys: list[str] = []
        self.graph_attr_dict_factory = graph_attr_dict_factory(
            self.db, self.graph, self.graph_id
        )

    def clear(self) -> None:
        raise NotImplementedError("Cannot clear GraphAttrDict")

    @key_is_string
    @logger_debug
    def __contains__(self, key: str) -> bool:
        """'bar' in G.graph['foo']"""
        if key in self.data:
            return True

        return aql_doc_has_key(self.db, self.graph.name, key, self.parent_keys)

    @key_is_string
    @logger_debug
    def __getitem__(self, key: str) -> Any:
        """G.graph['foo']['bar']"""

        if value := self.data.get(key):
            return value

        result = aql_doc_get_key(self.db, self.graph_id, key, self.parent_keys)

        if result is None:
            raise KeyError(key)

        graph_attr_dict_value = process_graph_attr_dict_value(self, key, result)
        self.data[key] = graph_attr_dict_value

        return graph_attr_dict_value

    @key_is_string
    @logger_debug
    def __setitem__(self, key, value):
        """
        G.graph['foo'] = 'bar'
        G.graph['object'] = {'foo': 'bar'}
        G._node['object']['foo'] = 'baz'
        """
        if value is None:
            self.__delitem__(key)
            return

        graph_attr_dict_value = process_graph_attr_dict_value(self, key, value)
        update_dict = get_update_dict(self.parent_keys, {key: value})
        self.data[key] = graph_attr_dict_value
        root_data = self.root.data if self.root else self.data
        root_data["_rev"] = doc_update(self.db, self.graph_id, update_dict)

    @key_is_string
    @logger_debug
    def __delitem__(self, key):
        """del G.graph['foo']['bar']"""
        self.data.pop(key, None)
        update_dict = get_update_dict(self.parent_keys, {key: None})
        root_data = self.root.data if self.root else self.data
        root_data["_rev"] = doc_update(self.db, self.graph_id, update_dict)

    @logger_debug
    def update(self, attrs: Any) -> None:
        """G.graph['foo'].update({'bar': 'baz'})"""
        if not attrs:
            return

        self.data.update(build_graph_attr_dict_data(self, attrs))
        updated_dict = get_update_dict(self.parent_keys, attrs)
        root_data = self.root.data if self.root else self.data
        root_data["_rev"] = doc_update(self.db, self.graph_id, updated_dict)
