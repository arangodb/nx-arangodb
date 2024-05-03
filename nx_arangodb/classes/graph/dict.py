from __future__ import annotations

from collections import UserDict
from collections.abc import Iterator
from typing import Any, Callable

from arango.database import StandardDatabase
from arango.graph import Graph

import nx_arangodb as nxadb
from nx_arangodb.classes.function import (
    aql,
    aql_doc_get_key,
    aql_doc_get_keys,
    aql_doc_get_length,
    aql_doc_has_key,
    aql_edge_get,
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
) -> Callable[..., AdjListInnerDict]:
    return lambda: AdjListInnerDict(db, graph, default_node_type, edge_type_func)


def edge_attr_dict_factory(
    db: StandardDatabase, graph: Graph
) -> Callable[..., EdgeAttrDict]:
    return lambda: EdgeAttrDict(db, graph)


class AdjListOuterDict(UserDict):
    """The outer-level of the dict of dict of dict structure representing the Adjacency List of a graph.

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
        super().__init__(*args, **kwargs)

        self.db = db
        self.graph = graph
        self.default_node_type = default_node_type
        self.edge_type_func = edge_type_func
        self.adjlist_inner_dict_factory = adjlist_inner_dict_factory(
            db, graph, default_node_type, edge_type_func
        )

    @key_is_string
    def __contains__(self, key) -> bool:
        """'node/1' in G.adj"""
        node_id = get_node_id(key, self.default_node_type)

        if node_id in self.data:
            return True

        return self.graph.has_vertex(node_id)

    @key_is_string
    def __getitem__(self, key) -> AdjListInnerDict:
        """G.adj["node/1"]"""
        node_type, node_id = get_node_type_and_id(key, self.default_node_type)

        if value := self.data.get(node_id):
            return value

        if self.graph.has_vertex(node_id):
            adjlist_inner_dict: AdjListInnerDict = self.adjlist_inner_dict_factory()
            adjlist_inner_dict.src_node_id = node_id
            adjlist_inner_dict.src_node_type = node_type

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
        assert not adjlist_inner_dict.src_node_type

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

            edge_type = edge_dict.get("_edge_type")  # pop?
            if edge_type is None:
                edge_type = self.edge_type_func(src_node_type, dst_node_type)

            results[dst_key] = self.graph.link(
                edge_type, src_node_id, dst_node_id, edge_dict, silent=True
            )

        adjlist_inner_dict.src_node_id = src_node_id
        adjlist_inner_dict.src_node_type = src_node_type
        adjlist_inner_dict.data = results

        self.data[src_node_id] = adjlist_inner_dict

    @key_is_string
    def __delitem__(self, key: Any) -> None:
        """
        del G._adj['node/1']
        """
        raise NotImplementedError("AdjListOuterDict.__delitem__()")

    def __len__(self) -> int:
        """len(g._adj)"""
        return sum(
            [
                self.graph.edge_collection(ed["edge_collection"]).count()
                for ed in self.graph.edge_definitions()
            ]
        )

    def __iter__(self) -> Iterator[str]:
        """for k in g._adj"""
        for collection in self.graph.vertex_collections():
            for id in self.graph.vertex_collection(collection).ids():
                yield id

    def keys(self):
        """g._adj.keys()"""
        return self.__iter__()

    def clear(self, clear_remote: bool = False):
        """g._node.clear()"""
        self.data.clear()

        if clear_remote:
            for ed in self.graph.edge_definitions():
                self.graph.edge_collection(ed["edge_collection"]).truncate()

    @keys_are_strings
    def update(self, edges: dict[str, dict[str, dict[str, Any]]]):
        """g._adj.update({'node/1': {'node/2': {'foo': 'bar'}})"""
        raise NotImplementedError("AdjListOuterDict.update()")

        for src_key, dst_dict in edges.items():
            src_node_type, src_node_id = get_node_type_and_id(src_key)

            adjlist_inner_dict = self.adjlist_inner_dict_factory()
            adjlist_inner_dict.src_node_id = src_node_id
            adjlist_inner_dict.src_node_type = src_node_type

            results = {}
            for dst_key, edge_dict in dst_dict.items():
                dst_node_type, dst_node_id = get_node_type_and_id(dst_key)

                edge_type = edge_dict.get("_edge_type")  # pop?
                if edge_type is None:
                    edge_type = self.edge_type_func(src_node_type, dst_node_type)

                results[dst_key] = self.graph.link(
                    edge_type, src_node_id, dst_node_id, edge_dict, silent=True
                )

            adjlist_inner_dict.data = results

            self.data[src_node_id] = adjlist_inner_dict

    def get(self, key: str, default: Any = None) -> AdjListInnerDict:
        """g._adj.get('node/1')"""
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def values(self):
        """g._adj.values()"""
        raise NotImplementedError("AdjListOuterDict.values()")

    # def items(self, cache: bool = True):
    #     """g._adj.items()"""
    #     # raise NotImplementedError("AdjListOuterDict.items()")
    #     for ed in self.graph.edge_definitions():
    #         collection = ed["edge_collection"]

    #         for edge in self.graph.edge_collection(collection):
    #             src_node_id = edge["_from"]
    #             dst_node_id = edge["_to"]

    #             adjlist_inner_dict = self.adjlist_inner_dict_factory()
    #             adjlist_inner_dict.src_node_id = src_node_id
    #             adjlist_inner_dict.src_node_type = src_node_id.split("/")[0]

    #             edge_attr_dict = adjlist_inner_dict.edge_attr_dict_factory()
    #             edge_attr_dict.edge_id = edge["_id"]
    #             edge_attr_dict.data = edge

    #             adjlist_inner_dict.data[dst_node_id] = edge_attr_dict

    #             breakpoint()

    #             if src_node_id not in self.data:
    #                 self.data[src_node_id] = adjlist_inner_dict
    #             else:
    #                 self.data[src_node_id].update(adjlist_inner_dict)

    #             yield src_node_id, adjlist_inner_dict


class AdjListInnerDict(UserDict):
    """The inner-level of the dict of dict of dict structure representing the Adjacency List of a graph.

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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.db = db
        self.graph = graph
        self.default_node_type = default_node_type
        self.edge_type_func = edge_type_func

        self.src_node_id = None
        self.src_node_type = None

        self.edge_attr_dict_factory = edge_attr_dict_factory(self.db, self.graph)

    @key_is_string
    def __contains__(self, key) -> bool:
        """'node/2' in G.adj['node/1']"""
        dst_node_id = get_node_id(key, self.default_node_type)

        if dst_node_id in self.data:
            return True

        return aql_edge_get(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction="OUTBOUND",
            return_bool=True,
        )

    # CHECKPOINT...

    @key_is_string
    def __getitem__(self, key) -> dict[str, Any]:
        """g._adj['node/1']['node/2']"""
        dst_node_id = get_node_id(key, self.default_node_type)

        if value := self.data.get(dst_node_id):
            return value

        edge = aql_edge_get(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction="OUTBOUND",
            return_bool=False,
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

        dst_node_type, dst_node_id = get_node_type_and_id(key, self.default_node_type)

        edge_type = value.data.get("_edge_type")  # pop?
        if edge_type is None:
            edge_type = self.edge_type_func(self.src_node_type, dst_node_type)

        data = value.data
        edge = self.graph.link(edge_type, self.src_node_id, dst_node_id, data)

        edge_attr_dict = self.edge_attr_dict_factory()
        edge_attr_dict.edge_id = edge["_id"]
        edge_attr_dict.data = {**data, **edge}

        self.data[dst_node_id] = edge_attr_dict

    @key_is_string
    def __delitem__(self, key: Any) -> None:
        """del g._adj['node/1']['node/2']"""
        raise NotImplementedError("AdjListInnerDict.__delitem__()")

    def __len__(self) -> int:
        """len(g._adj['node/1'])"""
        query = """
            RETURN LENGTH(
                FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                    RETURN e
            )
        """

        bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

        count = aql_single(self.db, query, bind_vars)

        return count if count is not None else 0

    def __iter__(self) -> Iterator[str]:
        """for k in g._adj['node/1']"""

        query = """
            FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                RETURN e._to
        """

        bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

        yield from aql(self.db, query, bind_vars)

    def keys(self):
        """g._adj['node/1'].keys()"""
        return self.__iter__()

    def values(self, cache: bool = True):
        """g._adj['node/1'].values()"""

        query = """
            FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                RETURN e
        """

        bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

        for edge in aql(self.db, query, bind_vars):
            dst_node_id = edge["_to"]
            edge_attr_dict = self.edge_attr_dict_factory()
            edge_attr_dict.edge_id = edge["_id"]
            edge_attr_dict.data = edge

            if cache:
                self.data[dst_node_id] = edge_attr_dict

            yield edge_attr_dict

    def items(self, cache: bool = True):
        """g._adj['node/1'].items()"""

        query = """
            FOR v, e IN 1..1 OUTBOUND @src_node_id GRAPH @graph_name
                RETURN e
        """

        bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

        for edge in aql(self.db, query, bind_vars):
            dst_node_id = edge["_to"]
            edge_attr_dict = self.edge_attr_dict_factory()
            edge_attr_dict.edge_id = edge["_id"]
            edge_attr_dict.data = edge

            if cache:
                self.data[dst_node_id] = edge_attr_dict

            yield dst_node_id, edge

    # def update(self, edges: dict[str, dict[str, Any]]):
    #     """g._adj['node/1'].update({'node/2': {'foo': 'bar'}})"""
    #     if isinstance(edges, AdjListInnerDict):
    #         self.data.update(edges.data)
    #     else:
    #         for key, value in edges.items():
    #             self[key] = value


class EdgeAttrDict(UserDict):
    """The innermost-level of the dict of dict of dict structure representing the Adjacency List of a graph.

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
        super().__init__(*args, **kwargs)

        self.db = db
        self.graph = graph
        self.edge_id: str | None = None

    @key_is_string
    def __contains__(self, key: str) -> bool:
        """'foo' in G._adj['node/1']['node/2']"""
        if key in self.data:
            return True

        return aql_doc_has_key(self.db, self.edge_id, key)

    @key_is_string
    def __getitem__(self, key: str) -> Any:
        """G._adj['node/1']['node/2']['foo']"""
        if value := self.data.get(key):
            return value

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
        doc_update(self.db, self.edge_id, {key: value})

    @key_is_string
    @key_is_not_reserved
    def __delitem__(self, key: str):
        """del G._adj['node/1']['node/2']['foo']"""
        del self.data[key]
        doc_update(self.db, self.node_id, {key: None})

    def __iter__(self) -> Iterator[str]:
        """for key in G._adj['node/1']['node/2']"""
        for key in aql_doc_get_keys(self.db, self.edge_id):
            yield key

    def __len__(self) -> int:
        """len(G._adj['node/1']['node/'2])"""
        return aql_doc_get_length(self.db, self.edge_id)

    def keys(self):
        """G._adj['node/1']['node/'2].keys()"""
        return self.__iter__()

    def values(self, cache: bool = True):
        """G._adj['node/1']['node/'2].values()"""
        doc = self.db.document(self.edge_id)

        if cache:
            self.data = doc

        yield from doc.values()

    def items(self, cache: bool = True):
        """G._adj['node/1']['node/'2].items()"""
        doc = self.db.document(self.node_id)

        if cache:
            self.data = doc

        yield from doc.items()

    def clear(self, clear_remote: bool = False):
        """G._adj['node/1']['node/'2].clear()"""
        self.data.clear()

    def update(self, attrs: dict[str, Any]):
        """G._adj['node/1']['node/'2].update({'foo': 'bar'})"""
        self.data.update(attrs)
        if not self.edge_id:
            print("Silent Error: Edge ID not set, cannot invoke EdgeAttrDict.update()")
            return

        doc_update(self.db, self.edge_id, attrs)
