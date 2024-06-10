from __future__ import annotations

from collections import UserDict, defaultdict
from collections.abc import Iterator
from typing import Any, Callable

from arango.database import StandardDatabase
from arango.exceptions import DocumentInsertError
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


class GraphDict(UserDict):
    """A dictionary-like object for storing graph attributes.

    Given that ArangoDB does not have a concept of graph attributes, this class
    stores the attributes in a collection with the graph name as the document key.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph_name: The graph name.
    :type graph_name: str
    """

    COLLECTION_NAME = "nxadb_graphs"

    def __init__(self, db: StandardDatabase, graph_name: str, *args, **kwargs):
        logger.debug("GraphDict.__init__")
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
            logger.debug(f"cached in GraphDict.__contains__({key})")
            return True

        logger.debug("aql_doc_has_key in GraphDict.__contains__")
        return aql_doc_has_key(self.db, self.graph_id, key)

    @key_is_string
    def __getitem__(self, key: Any) -> Any:
        """G.graph['foo']"""
        if value := self.data.get(key):
            return value

        logger.debug("aql_doc_get_key in GraphDict.__getitem__")
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
        logger.debug(f"doc_update in GraphDict.__setitem__({key})")
        doc_update(self.db, self.graph_id, {key: value})

    @key_is_string
    @key_is_not_reserved
    def __delitem__(self, key):
        """del G.graph['foo']"""
        self.data.pop(key, None)
        logger.debug(f"doc_update in GraphDict.__delitem__({key})")
        doc_update(self.db, self.graph_id, {key: None})

    @keys_are_strings
    @keys_are_not_reserved
    # @values_are_json_serializable # TODO?
    def update(self, attrs):
        """G.graph.update({'foo': 'bar'})"""
        if attrs:
            self.data.update(attrs)
            logger.debug(f"doc_update in GraphDict.update({attrs})")
            doc_update(self.db, self.graph_id, attrs)

    def clear(self):
        """G.graph.clear()"""
        self.data.clear()
        logger.debug("cleared GraphDict")

        # if clear_remote:
        #     doc_insert(self.db, self.COLLECTION_NAME, self.graph_id, silent=True)


class NodeDict(UserDict):
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

    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        default_node_type: str,
        *args,
        **kwargs,
    ):
        logger.debug("NodeDict.__init__")
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
            logger.debug(f"cached in NodeDict.__contains__({node_id})")
            return True

        logger.debug(f"graph.has_vertex in NodeDict.__contains__({node_id})")
        return self.graph.has_vertex(node_id)

    @key_is_string
    def __getitem__(self, key: str) -> NodeAttrDict:
        """G._node['node/1']"""
        node_id = get_node_id(key, self.default_node_type)

        if value := self.data.get(node_id):
            logger.debug(f"cached in NodeDict.__getitem__({node_id})")
            return value

        if value := self.graph.vertex(node_id):
            logger.debug(f"graph.vertex in NodeDict.__getitem__({node_id})")
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

        logger.debug(f"doc_insert in NodeDict.__setitem__({key})")
        result = doc_insert(self.db, node_type, node_id, value.data)

        node_attr_dict = self.node_attr_dict_factory()
        node_attr_dict.node_id = node_id
        node_attr_dict.data = result

        self.data[node_id] = node_attr_dict

    @key_is_string
    def __delitem__(self, key: Any) -> None:
        """del g._node['node/1']"""
        node_id = get_node_id(key, self.default_node_type)

        if not self.graph.has_vertex(node_id):
            raise KeyError(key)

        remove_statements = "\n".join(
            f"REMOVE e IN `{edge_def['edge_collection']}` OPTIONS {{ignoreErrors: true}}"  # noqa
            for edge_def in self.graph.edge_definitions()
        )

        query = f"""
            FOR v, e IN 1..1 ANY @src_node_id GRAPH @graph_name
                {remove_statements}
        """

        bind_vars = {"src_node_id": node_id, "graph_name": self.graph.name}

        logger.debug(f"remove_edges in NodeDict.__delitem__({node_id})")
        aql(self.db, query, bind_vars)

        logger.debug(f"doc_delete in NodeDict.__delitem__({node_id})")
        doc_delete(self.db, node_id)

        self.data.pop(node_id, None)

    def __len__(self) -> int:
        """len(g._node)"""
        logger.debug("NodeDict.__len__")
        return sum(
            [
                self.graph.vertex_collection(c).count()
                for c in self.graph.vertex_collections()
            ]
        )

    def __iter__(self) -> Iterator[str]:
        """iter(g._node)"""
        logger.debug("NodeDict.__iter__")
        for collection in self.graph.vertex_collections():
            for node_id in self.graph.vertex_collection(collection).ids():
                yield node_id

    def clear(self):
        """g._node.clear()"""
        self.data.clear()
        logger.debug("cleared NodeDict")

        # if clear_remote:
        #     for collection in self.graph.vertex_collections():
        #         self.graph.vertex_collection(collection).truncate()

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
        logger.debug("NodeDict.keys()")
        return self.__iter__()

    def values(self):
        """g._node.values()"""
        logger.debug("NodeDict.values()")
        self.__fetch_all()
        return self.data.values()

    def items(self, data: str | None = None, default: Any | None = None):
        """g._node.items() or G._node.items(data='foo')"""
        if data is None:
            logger.debug("NodeDict.items(data=None)")
            self.__fetch_all()
            return self.data.items()

        logger.debug(f"NodeDict.items(data={data})")
        v_cols = list(self.graph.vertex_collections())
        return aql_fetch_data(self.db, v_cols, data, default, is_edge=False)

    def __fetch_all(self):
        logger.debug("NodeDict.__fetch_all()")

        self.data.clear()
        for collection in self.graph.vertex_collections():
            for doc in self.graph.vertex_collection(collection).all():
                node_id = doc["_id"]

                node_attr_dict = self.node_attr_dict_factory()
                node_attr_dict.node_id = node_id
                node_attr_dict.data = doc

                self.data[node_id] = node_attr_dict


class NodeAttrDict(UserDict):
    """The inner-level of the dict of dict structure
    representing the nodes (vertices) of a graph.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    """

    def __init__(self, db: StandardDatabase, graph: Graph, *args, **kwargs):
        logger.debug("NodeAttrDict.__init__")

        self.db = db
        self.graph = graph
        self.node_id: str | None = None

        super().__init__(*args, **kwargs)

    @key_is_string
    def __contains__(self, key: str) -> bool:
        """'foo' in G._node['node/1']"""
        if key in self.data:
            logger.debug(f"cached in NodeAttrDict.__contains__({key})")
            return True

        logger.debug("aql_doc_has_key in NodeAttrDict.__contains__")
        return aql_doc_has_key(self.db, self.node_id, key)

    @key_is_string
    def __getitem__(self, key: str) -> Any:
        """G._node['node/1']['foo']"""
        if value := self.data.get(key):
            logger.debug(f"cached in NodeAttrDict.__getitem__({key})")
            return value

        logger.debug(f"aql_doc_get_key in NodeAttrDict.__getitem__({key})")
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
        logger.debug(f"doc_update in NodeAttrDict.__setitem__({key})")
        doc_update(self.db, self.node_id, {key: value})

    @key_is_string
    @key_is_not_reserved
    def __delitem__(self, key: str):
        """del G._node['node/1']['foo']"""
        self.data.pop(key, None)
        logger.debug(f"doc_update in NodeAttrDict({self.node_id}).__delitem__({key})")
        doc_update(self.db, self.node_id, {key: None})

    def __iter__(self) -> Iterator[str]:
        """for key in G._node['node/1']"""
        logger.debug(f"NodeAttrDict({self.node_id}).__iter__")
        for key in aql_doc_get_keys(self.db, self.node_id):
            yield key

    def __len__(self) -> int:
        """len(G._node['node/1'])"""
        logger.debug(f"NodeAttrDict({self.node_id}).__len__")
        return aql_doc_get_length(self.db, self.node_id)

    def keys(self):
        """G._node['node/1'].keys()"""
        logger.debug(f"NodeAttrDict({self.node_id}).keys()")
        return self.__iter__()

    def values(self):
        """G._node['node/1'].values()"""
        logger.debug(f"NodeAttrDict({self.node_id}).values()")
        self.data = self.db.document(self.node_id)
        return self.data.values()

    def items(self):
        """G._node['node/1'].items()"""
        logger.debug(f"NodeAttrDict({self.node_id}).items()")
        self.data = self.db.document(self.node_id)
        return self.data.items()

    def clear(self):
        """G._node['node/1'].clear()"""
        self.data.clear()
        logger.debug(f"cleared NodeAttrDict({self.node_id})")

        # if clear_remote:
        #     doc_insert(self.db, self.node_id, silent=True, overwrite=True)

    def update(self, attrs: dict[str, Any]):
        """G._node['node/1'].update({'foo': 'bar'})"""
        if attrs:
            self.data.update(attrs)

            if not self.node_id:
                logger.debug("Node ID not set, skipping NodeAttrDict(?).update()")
                return

            logger.debug(f"NodeAttrDict({self.node_id}).update({attrs})")
            doc_update(self.db, self.node_id, attrs)


class AdjListOuterDict(UserDict):
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

    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        default_node_type: str,
        edge_type_func: Callable[[str, str], str],
        *args,
        **kwargs,
    ):
        logger.debug("AdjListOuterDict.__init__")

        super().__init__(*args, **kwargs)

        self.db = db
        self.graph = graph
        self.default_node_type = default_node_type
        self.edge_type_func = edge_type_func
        self.adjlist_inner_dict_factory = adjlist_inner_dict_factory(
            db, graph, default_node_type, edge_type_func, self
        )

        self.FETCHED_ALL_DATA = False

    # def __repr__(self) -> str:
    #     return f"'{self.graph.name}'"

    # def __str__(self) -> str:
    #     return f"'{self.graph.name}'"

    @key_is_string
    def __contains__(self, key) -> bool:
        """'node/1' in G.adj"""
        node_id = get_node_id(key, self.default_node_type)

        if node_id in self.data:
            logger.debug(f"cached in AdjListOuterDict.__contains__({node_id})")
            return True

        logger.debug("graph.has_vertex in AdjListOuterDict.__contains__")
        return self.graph.has_vertex(node_id)

    @key_is_string
    def __getitem__(self, key: str) -> AdjListInnerDict:
        """G.adj["node/1"]"""
        node_type, node_id = get_node_type_and_id(key, self.default_node_type)

        if value := self.data.get(node_id):
            logger.debug(f"cached in AdjListOuterDict.__getitem__({node_id})")
            return value

        if self.graph.has_vertex(node_id):
            logger.debug(f"graph.vertex in AdjListOuterDict.__getitem__({node_id})")
            adjlist_inner_dict: AdjListInnerDict = self.adjlist_inner_dict_factory()
            adjlist_inner_dict.src_node_id = node_id

            self.data[node_id] = adjlist_inner_dict

            return adjlist_inner_dict

        raise KeyError(key)

    @key_is_string
    def __setitem__(self, src_key: str, adjlist_inner_dict: AdjListInnerDict):
        """
        g._adj['node/1'] = AdjListInnerDict()
        """
        assert isinstance(adjlist_inner_dict, AdjListInnerDict)
        assert not adjlist_inner_dict.src_node_id

        logger.debug(f"AdjListOuterDict.__setitem__({src_key})")

        src_node_type, src_node_id = get_node_type_and_id(
            src_key, self.default_node_type
        )

        # NOTE: this might not actually be needed...
        results = {}
        edge_dict: dict[str, Any]
        for dst_key, edge_dict in adjlist_inner_dict.data.items():
            dst_node_type, dst_node_id = get_node_type_and_id(
                dst_key, self.default_node_type
            )

            edge_type = edge_dict.get("_edge_type")
            if edge_type is None:
                edge_type = self.edge_type_func(src_node_type, dst_node_type)

            logger.debug(f"graph.link({src_key}, {dst_key})")
            results[dst_key] = self.graph.link(
                edge_type, src_node_id, dst_node_id, edge_dict
            )

        adjlist_inner_dict.src_node_id = src_node_id
        adjlist_inner_dict.data = results

        self.data[src_node_id] = adjlist_inner_dict

    @key_is_string
    def __delitem__(self, key: Any) -> None:
        """
        del G._adj['node/1']
        """
        # Nothing else to do here, as this delete is always invoked by
        # G.remove_node(), which already removes all edges via
        # del G._node['node/1']
        logger.debug(f"AdjListOuterDict.__delitem__({key}) (just cache)")
        node_id = get_node_id(key, self.default_node_type)
        self.data.pop(node_id, None)

    def __len__(self) -> int:
        """len(g._adj)"""
        logger.debug("AdjListOuterDict.__len__")
        return sum(
            [
                self.graph.vertex_collection(c).count()
                for c in self.graph.vertex_collections()
            ]
        )

    def __iter__(self) -> Iterator[str]:
        """for k in g._adj"""
        logger.debug("AdjListOuterDict.__iter__")

        if self.FETCHED_ALL_DATA:
            yield from self.data.keys()

        else:
            for collection in self.graph.vertex_collections():
                for id in self.graph.vertex_collection(collection).ids():
                    yield id

    def keys(self):
        """g._adj.keys()"""
        logger.debug("AdjListOuterDict.keys()")
        return self.__iter__()

    def clear(self):
        """g._node.clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False
        logger.debug("cleared AdjListOuterDict")

        # if clear_remote:
        #     for ed in self.graph.edge_definitions():
        #         self.graph.edge_collection(ed["edge_collection"]).truncate()

    @keys_are_strings
    def update(self, edges: dict[str, dict[str, dict[str, Any]]]):
        """g._adj.update({'node/1': {'node/2': {'foo': 'bar'}})"""
        raise NotImplementedError("AdjListOuterDict.update()")

    def values(self):
        """g._adj.values()"""
        logger.debug("AdjListOuterDict.values()")
        self.__fetch_all()
        return self.data.values()

    def items(self, data: str | None = None, default: Any | None = None):
        """g._adj.items() or G._adj.items(data='foo')"""
        if data is None:
            logger.debug("AdjListOuterDict.items(data=None)")
            self.__fetch_all()
            return self.data.items()

        logger.debug(f"AdjListOuterDict.items(data={data})")
        e_cols = [ed["edge_collection"] for ed in self.graph.edge_definitions()]
        result = aql_fetch_data(self.db, e_cols, data, default, is_edge=True)
        yield from result

    # TODO: Revisit
    def __fetch_all(self) -> None:
        logger.debug("AdjListOuterDict.__fetch_all()")

        if self.FETCHED_ALL_DATA:
            logger.debug("Already fetched data, skipping fetch")
            return

        self.clear()
        # items = defaultdict(dict)
        for ed in self.graph.edge_definitions():
            collection = ed["edge_collection"]

            for edge in self.graph.edge_collection(collection):
                src_node_id = edge["_from"]
                dst_node_id = edge["_to"]

                # items[src_node_id][dst_node_id] = edge
                # items[dst_node_id][src_node_id] = edge

                if src_node_id in self.data:
                    src_inner_dict = self.data[src_node_id]
                else:
                    src_inner_dict = self.adjlist_inner_dict_factory()
                    src_inner_dict.src_node_id = src_node_id
                    self.data[src_node_id] = src_inner_dict

                if dst_node_id in self.data:
                    dst_inner_dict = self.data[dst_node_id]
                else:
                    dst_inner_dict = self.adjlist_inner_dict_factory()
                    dst_inner_dict.src_node_id = dst_node_id
                    self.data[dst_node_id] = dst_inner_dict

                edge_attr_dict = src_inner_dict.edge_attr_dict_factory()
                edge_attr_dict.edge_id = edge["_id"]
                edge_attr_dict.data = edge

                self.data[src_node_id].data[dst_node_id] = edge_attr_dict
                self.data[dst_node_id].data[src_node_id] = edge_attr_dict

        self.FETCHED_ALL_DATA = True


class AdjListInnerDict(UserDict):
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

    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        default_node_type: str,
        edge_type_func: Callable[[str, str], str],
        adjlist_outer_dict: AdjListOuterDict,
        *args,
        **kwargs,
    ):
        logger.debug("AdjListInnerDict.__init__")

        super().__init__(*args, **kwargs)

        self.db = db
        self.graph = graph
        self.default_node_type = default_node_type
        self.edge_type_func = edge_type_func
        self.adjlist_outer_dict = adjlist_outer_dict

        self.src_node_id = None

        self.edge_attr_dict_factory = edge_attr_dict_factory(self.db, self.graph)

        self.FETCHED_ALL_DATA = False

    def __get_mirrored_edge_attr_dict(self, dst_node_id: str) -> bool:
        logger.debug(f"checking for mirrored edge ({self.src_node_id}, {dst_node_id})")
        if dst_node_id in self.adjlist_outer_dict.data:
            if self.src_node_id in self.adjlist_outer_dict.data[dst_node_id].data:
                return self.adjlist_outer_dict.data[dst_node_id].data[self.src_node_id]

        return None

    # def __repr__(self) -> str:
    #     return f"'{self.src_node_id}'"

    # def __str__(self) -> str:
    #     return f"'{self.src_node_id}'"

    @key_is_string
    def __contains__(self, key) -> bool:
        """'node/2' in G.adj['node/1']"""
        dst_node_id = get_node_id(key, self.default_node_type)

        if dst_node_id in self.data:
            logger.debug(f"cached in AdjListInnerDict.__contains__({dst_node_id})")
            return True

        logger.debug(f"aql_edge_exists in AdjListInnerDict.__contains__({dst_node_id})")
        return aql_edge_exists(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction="ANY",
        )

    @key_is_string
    def __getitem__(self, key) -> EdgeAttrDict:
        """g._adj['node/1']['node/2']"""
        dst_node_id = get_node_id(key, self.default_node_type)

        if dst_node_id in self.data:
            m = f"cached in AdjListInnerDict({self.src_node_id}).__getitem__({dst_node_id})"  # noqa
            logger.debug(m)
            return self.data[dst_node_id]

        if mirrored_edge_attr_dict := self.__get_mirrored_edge_attr_dict(dst_node_id):
            logger.debug("No need to fetch the edge, as it is already cached")
            self.data[dst_node_id] = mirrored_edge_attr_dict
            return mirrored_edge_attr_dict

        m = f"aql_edge_get in AdjListInnerDict({self.src_node_id}).__getitem__({dst_node_id})"  # noqa
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
        edge_attr_dict.data = edge

        self.data[dst_node_id] = edge_attr_dict

        return edge_attr_dict

    @key_is_string
    def __setitem__(self, key: str, value: dict | EdgeAttrDict):
        """g._adj['node/1']['node/2'] = {'foo': 'bar'}"""
        assert isinstance(value, EdgeAttrDict)
        logger.debug(f"AdjListInnerDict({self.src_node_id}).__setitem__({key})")

        src_node_type = self.src_node_id.split("/")[0]
        dst_node_type, dst_node_id = get_node_type_and_id(key, self.default_node_type)

        if mirrored_edge_attr_dict := self.__get_mirrored_edge_attr_dict(dst_node_id):
            logger.debug("No need to create a new edge, as it already exists")
            self.data[dst_node_id] = mirrored_edge_attr_dict
            return

        edge_type = value.data.get("_edge_type")
        if edge_type is None:
            edge_type = self.edge_type_func(src_node_type, dst_node_type)
            logger.debug(f"No edge type specified, so generated: {edge_type})")

        if edge_id := value.edge_id:
            m = f"edge id found, deleting ({self.src_node_id, dst_node_id})"
            logger.debug(m)
            self.graph.delete_edge(edge_id)

        elif edge_id := aql_edge_id(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction="ANY",
        ):
            m = f"existing edge found, deleting ({self.src_node_id, dst_node_id})"
            logger.debug(m)
            self.graph.delete_edge(edge_id)

        edge_data = value.data
        logger.debug(f"graph.link({self.src_node_id}, {dst_node_id})")
        edge = self.graph.link(edge_type, self.src_node_id, dst_node_id, edge_data)

        edge_attr_dict = self.edge_attr_dict_factory()
        edge_attr_dict.edge_id = edge["_id"]
        edge_attr_dict.data = {
            **edge_data,
            **edge,
            "_from": self.src_node_id,
            "_to": dst_node_id,
        }

        self.data[dst_node_id] = edge_attr_dict

    @key_is_string
    def __delitem__(self, key: Any) -> None:
        """del g._adj['node/1']['node/2']"""
        dst_node_id = get_node_id(key, self.default_node_type)
        self.data.pop(dst_node_id, None)

        if self.__get_mirrored_edge_attr_dict(dst_node_id):
            m = "No need to delete the edge, as the next del will take care of it"
            logger.debug(m)
            return

        logger.debug(f"fetching edge ({self.src_node_id, dst_node_id})")
        edge_id = aql_edge_id(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction="ANY",
        )

        if not edge_id:
            m = f"edge not found, AdjListInnerDict({self.src_node_id}).__delitem__({dst_node_id})"  # noqa
            logger.debug(m)
            return

        logger.debug(f"graph.delete_edge({edge_id})")
        self.graph.delete_edge(edge_id)

    def __len__(self) -> int:
        """len(g._adj['node/1'])"""
        assert self.src_node_id

        if self.FETCHED_ALL_DATA:
            m = f"Already fetched data, skipping AdjListInnerDict({self.src_node_id}).__len__"  # noqa
            logger.debug(m)
            return len(self.data)

        query = """
            RETURN LENGTH(
                FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                    RETURN 1
            )
        """

        bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

        logger.debug(f"aql_single in AdjListInnerDict({self.src_node_id}).__len__")
        count = aql_single(self.db, query, bind_vars)

        return count if count is not None else 0

    def __iter__(self) -> Iterator[str]:
        """for k in g._adj['node/1']"""
        if self.FETCHED_ALL_DATA:
            m = f"Already fetched data, skipping AdjListInnerDict({self.src_node_id}).__iter__"  # noqa
            logger.debug(m)
            yield from self.data.keys()

        else:
            query = """
                FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                    RETURN e._to
            """

            bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

            logger.debug(f"aql in AdjListInnerDict({self.src_node_id}).__iter__")
            yield from aql(self.db, query, bind_vars)

    def keys(self):
        """g._adj['node/1'].keys()"""
        logger.debug(f"AdjListInnerDict({self.src_node_id}).keys()")
        return self.__iter__()

    def clear(self):
        """G._adj['node/1'].clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False
        logger.debug(f"cleared AdjListInnerDict({self.src_node_id})")

    def update(self, edges: dict[str, dict[str, Any]]):
        """g._adj['node/1'].update({'node/2': {'foo': 'bar'}})"""
        raise NotImplementedError("AdjListInnerDict.update()")

    def values(self):
        """g._adj['node/1'].values()"""
        logger.debug(f"AdjListInnerDict({self.src_node_id}).values()")
        self.__fetch_all()
        return self.data.values()

    def items(self):
        """g._adj['node/1'].items()"""
        logger.debug(f"AdjListInnerDict({self.src_node_id}).items()")
        self.__fetch_all()
        return self.data.items()

    def __fetch_all(self):
        logger.debug(f"AdjListInnerDict({self.src_node_id}).__fetch_all()")

        if self.FETCHED_ALL_DATA:
            logger.debug("Already fetched data, skipping fetch")
            return

        self.clear()

        query = """
            FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                RETURN e
        """

        bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

        for edge in aql(self.db, query, bind_vars):
            edge_attr_dict = self.edge_attr_dict_factory()
            edge_attr_dict.edge_id = edge["_id"]
            edge_attr_dict.data = edge

            self.data[edge["_to"]] = edge_attr_dict

        self.FETCHED_ALL_DATA = True


class EdgeAttrDict(UserDict):
    """The innermost-level of the dict of dict of dict structure
    representing the Adjacency List of a graph.

    The innermost-dict is keyed by the edge attribute key.

    :param db: The ArangoDB database.
    :type db: StandardDatabase
    :param graph: The ArangoDB graph.
    :type graph: Graph
    """

    def __init__(
        self,
        db: StandardDatabase,
        graph: Graph,
        *args,
        **kwargs,
    ):
        logger.debug("EdgeAttrDict.__init__")

        super().__init__(*args, **kwargs)

        self.db = db
        self.graph = graph
        self.edge_id: str | None = None

    @key_is_string
    def __contains__(self, key: str) -> bool:
        """'foo' in G._adj['node/1']['node/2']"""
        if key in self.data:
            logger.debug(f"cached in EdgeAttrDict({self.edge_id}).__contains__({key})")
            return True

        logger.debug(f"aql_doc_has_key in EdgeAttrDict({self.edge_id}).__contains__")
        return aql_doc_has_key(self.db, self.edge_id, key)

    @key_is_string
    def __getitem__(self, key: str) -> Any:
        """G._adj['node/1']['node/2']['foo']"""
        if value := self.data.get(key):
            logger.debug(f"cached in EdgeAttrDict({self.edge_id}).__getitem__({key})")
            return value

        logger.debug(
            f"aql_doc_get_key in EdgeAttrDict({self.edge_id}).__getitem__({key})"
        )
        result = aql_doc_get_key(self.db, self.edge_id, key)

        if not result:
            raise KeyError(key)

        self.data[key] = result

        return result

    @key_is_string
    @key_is_not_reserved
    # @value_is_json_serializable # TODO?
    def __setitem__(self, key: str, value: Any):
        """G._adj['node/1']['node/2']['foo'] = 'bar'"""
        self.data[key] = value
        logger.debug(f"doc_update in EdgeAttrDict({self.edge_id}).__setitem__({key})")
        doc_update(self.db, self.edge_id, {key: value})

    @key_is_string
    @key_is_not_reserved
    def __delitem__(self, key: str):
        """del G._adj['node/1']['node/2']['foo']"""
        self.data.pop(key, None)
        logger.debug(f"doc_update in EdgeAttrDict({self.edge_id}).__delitem__({key})")
        doc_update(self.db, self.edge_id, {key: None})

    def __iter__(self) -> Iterator[str]:
        """for key in G._adj['node/1']['node/2']"""
        logger.debug(f"EEdgeAttrDict({self.edge_id}).__iter__")
        for key in aql_doc_get_keys(self.db, self.edge_id):
            yield key

    def __len__(self) -> int:
        """len(G._adj['node/1']['node/'2])"""
        logger.debug(f"EdgeAttrDict({self.edge_id}).__len__")
        return aql_doc_get_length(self.db, self.edge_id)

    def keys(self):
        """G._adj['node/1']['node/'2].keys()"""
        logger.debug(f"EdgeAttrDict({self.edge_id}).keys()")
        return self.__iter__()

    def values(self):
        """G._adj['node/1']['node/'2].values()"""
        logger.debug(f"EdgeAttrDict({self.edge_id}).values()")
        self.data = self.db.document(self.edge_id)
        return self.data.values()

    def items(self):
        """G._adj['node/1']['node/'2].items()"""
        logger.debug(f"EdgeAttrDict({self.edge_id}).items()")
        self.data = self.db.document(self.edge_id)
        return self.data.items()

    def clear(self):
        """G._adj['node/1']['node/'2].clear()"""
        self.data.clear()
        logger.debug(f"cleared EdgeAttrDict({self.edge_id})")

    def update(self, attrs: dict[str, Any]):
        """G._adj['node/1']['node/'2].update({'foo': 'bar'})"""
        if attrs:
            self.data.update(attrs)

            if not self.edge_id:
                logger.debug("Edge ID not set, skipping EdgeAttrDict(?).update()")
                return

            logger.debug(f"EdgeAttrDict({self.edge_id}).update({attrs})")
            doc_update(self.db, self.edge_id, attrs)
