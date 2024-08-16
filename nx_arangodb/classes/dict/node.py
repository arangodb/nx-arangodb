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
    check_list_for_errors,
    doc_delete,
    doc_insert,
    doc_update,
    get_arangodb_graph,
    get_node_id,
    get_node_type_and_id,
    get_update_dict,
    json_serializable,
    key_is_not_reserved,
    key_is_string,
    keys_are_not_reserved,
    keys_are_strings,
    logger_debug,
    separate_nodes_by_collections,
    upsert_collection_documents,
)

#############
# Factories #
#############


def node_dict_factory(
    db: StandardDatabase, graph: Graph, default_node_type: str
) -> Callable[..., NodeDict]:
    return lambda: NodeDict(db, graph, default_node_type)


def node_attr_dict_factory(
    db: StandardDatabase, graph: Graph
) -> Callable[..., NodeAttrDict]:
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

    Returns the parent NodeAttrDict.
    """
    node_attr_dict_data = {}
    for key, value in data.items():
        node_attr_dict_value = process_node_attr_dict_value(parent, key, value)
        node_attr_dict_data[key] = node_attr_dict_value

    return node_attr_dict_data


def process_node_attr_dict_value(parent: NodeAttrDict, key: str, value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    node_attr_dict = parent.node_attr_dict_factory()
    node_attr_dict.root = parent.root or parent
    node_attr_dict.node_id = parent.node_id
    node_attr_dict.parent_keys = parent.parent_keys + [key]
    node_attr_dict.data = build_node_attr_dict_data(node_attr_dict, value)

    return node_attr_dict


@json_serializable
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

    def clear(self) -> None:
        raise NotImplementedError("Cannot clear NodeAttrDict")

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
        if value is None:
            self.__delitem__(key)
            return

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
    @logger_debug
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
    @logger_debug
    def __getitem__(self, key: str) -> NodeAttrDict:
        """G._node['node/1']"""
        node_id = get_node_id(key, self.default_node_type)

        if vertex := self.data.get(node_id):
            return vertex

        if node_id not in self.data and self.FETCHED_ALL_IDS:
            raise KeyError(key)

        if vertex := self.graph.vertex(node_id):
            node_attr_dict = self._create_node_attr_dict(vertex)
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

        node_attr_dict = self._create_node_attr_dict(result)

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

    @logger_debug
    def keys(self) -> Any:
        """g._node.keys()"""
        return self.__iter__()

    @logger_debug
    def clear(self) -> None:
        """g._node.clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False
        self.FETCHED_ALL_IDS = False

    @keys_are_strings
    @logger_debug
    def __update_local_nodes(self, nodes: Any) -> None:
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

    @logger_debug
    def values(self) -> Any:
        """g._node.values()"""
        if not self.FETCHED_ALL_DATA:
            self._fetch_all()

        yield from self.data.values()

    @logger_debug
    def items(self, data: str | None = None, default: Any | None = None) -> Any:
        """g._node.items() or G._node.items(data='foo')"""
        if data is None:
            if not self.FETCHED_ALL_DATA:
                self._fetch_all()

            yield from self.data.items()
        else:
            v_cols = list(self.graph.vertex_collections())
            result = aql_fetch_data(self.db, v_cols, data, default)
            yield from result.items()

    @logger_debug
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
            node_attr_dict = self._create_node_attr_dict(node_data)
            self.data[node_id] = node_attr_dict

        self.FETCHED_ALL_DATA = True
        self.FETCHED_ALL_IDS = True
