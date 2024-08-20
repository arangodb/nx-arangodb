from __future__ import annotations

from collections import UserDict
from collections.abc import Iterator
from itertools import islice
from typing import Any, Callable

from arango.database import StandardDatabase
from arango.exceptions import DocumentDeleteError
from arango.graph import Graph

from nx_arangodb.exceptions import EdgeTypeAmbiguity, MultipleEdgesFound
from nx_arangodb.logger import logger

from ..enum import DIRECTED_GRAPH_TYPES, MULTIGRAPH_TYPES, GraphType, TraversalDirection
from ..function import (
    aql,
    aql_doc_get_key,
    aql_doc_has_key,
    aql_edge_count_src,
    aql_edge_count_src_dst,
    aql_edge_exists,
    aql_edge_get,
    aql_edge_id,
    aql_fetch_data_edge,
    doc_insert,
    doc_update,
    get_arangodb_graph,
    get_node_id,
    get_node_type_and_id,
    get_update_dict,
    json_serializable,
    key_is_adb_id_or_int,
    key_is_not_reserved,
    key_is_string,
    keys_are_not_reserved,
    keys_are_strings,
    logger_debug,
)

#############
# Factories #
#############


def edge_attr_dict_factory(
    db: StandardDatabase, graph: Graph
) -> Callable[..., EdgeAttrDict]:
    return lambda: EdgeAttrDict(db, graph)


def edge_key_dict_factory(
    db: StandardDatabase,
    graph: Graph,
    edge_type_key: str,
    edge_type_func: Callable[[str, str], str],
    is_directed: bool,
    adjlist_inner_dict: AdjListInnerDict | None = None,
) -> Callable[..., EdgeKeyDict]:
    return lambda: EdgeKeyDict(
        db, graph, edge_type_key, edge_type_func, is_directed, adjlist_inner_dict
    )


def adjlist_inner_dict_factory(
    db: StandardDatabase,
    graph: Graph,
    default_node_type: str,
    edge_type_key: str,
    edge_type_func: Callable[[str, str], str],
    graph_type: str,
    adjlist_outer_dict: AdjListOuterDict | None = None,
) -> Callable[..., AdjListInnerDict]:
    return lambda: AdjListInnerDict(
        db,
        graph,
        default_node_type,
        edge_type_key,
        edge_type_func,
        graph_type,
        adjlist_outer_dict,
    )


def adjlist_outer_dict_factory(
    db: StandardDatabase,
    graph: Graph,
    default_node_type: str,
    edge_type_key: str,
    edge_type_func: Callable[[str, str], str],
    graph_type: str,
    symmetrize_edges_if_directed: bool,
) -> Callable[..., AdjListOuterDict]:
    return lambda: AdjListOuterDict(
        db,
        graph,
        default_node_type,
        edge_type_key,
        edge_type_func,
        graph_type,
        symmetrize_edges_if_directed,
    )


#############
# Adjacency #
#############


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


def process_edge_attr_dict_value(parent: EdgeAttrDict, key: str, value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    edge_attr_dict = parent.edge_attr_dict_factory()
    edge_attr_dict.root = parent.root or parent
    edge_attr_dict.edge_id = parent.edge_id
    edge_attr_dict.parent_keys = parent.parent_keys + [key]
    edge_attr_dict.data = build_edge_attr_dict_data(edge_attr_dict, value)

    return edge_attr_dict


@json_serializable
class EdgeAttrDict(UserDict[str, Any]):
    """The innermost-level of the dict of dict (of dict) of dict structure
    representing the Adjacency List of a graph.

    EdgeAttrDict is keyed by the edge attribute key.

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

    def clear(self) -> None:
        raise NotImplementedError("Cannot clear EdgeAttrDict")

    def copy(self) -> Any:
        # TODO: REVISIT THIS
        return self.data.copy()

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
        if key in self.data:
            return self.data[key]

        assert self.edge_id
        result = aql_doc_get_key(self.db, self.edge_id, key, self.parent_keys)

        if result is None:
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
        if value is None:
            self.__delitem__(key)
            return

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


class EdgeKeyDict(UserDict[str, EdgeAttrDict]):
    """The (optional) 3rd level of the dict of dict (*of dict*) of dict
    structure representing the Adjacency List of a MultiGraph.

    EdgeKeyDict is keyed by ArangoDB Edge IDs.

    Unique to MultiGraphs, edges are keyed by ArangoDB Edge IDs, allowing
    for multiple edges between the same nodes. Alternatively, if an Edge
    is already fetched, then it can also be keyed by a numerical index.
    However, this is not recommended because consistent ordering of edges
    is not guaranteed.

    ASSUMPTIONS (for now):
    - keys must be ArangoDB Edge IDs
    - key-to-edge mapping is 1-to-1

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
        edge_type_key: str,
        edge_type_func: Callable[[str, str], str],
        is_directed: bool,
        adjlist_inner_dict: AdjListInnerDict | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.data: dict[str, EdgeAttrDict] = {}

        self.is_directed = is_directed

        self.db = db
        self.graph = graph
        self.edge_type_key = edge_type_key
        self.edge_type_func = edge_type_func
        self._default_edge_type: str | None = None
        self.graph_name = graph.name
        self.edge_attr_dict_factory = edge_attr_dict_factory(self.db, self.graph)

        self.src_node_id: str | None = None
        self.dst_node_id: str | None = None
        self.adjlist_inner_dict = adjlist_inner_dict

        self.FETCHED_ALL_DATA = False
        self.FETCHED_ALL_IDS = False

        self.traversal_direction = (
            adjlist_inner_dict.traversal_direction
            if adjlist_inner_dict is not None
            else (
                TraversalDirection.OUTBOUND
                if self.is_directed
                else TraversalDirection.ANY
            )
        )

        edge_validation_methods = {
            TraversalDirection.OUTBOUND: self.__is_valid_edge_outbound,
            TraversalDirection.INBOUND: self.__is_valid_edge_inbound,
            TraversalDirection.ANY: self.__is_valid_edge_any,
        }

        self.__is_valid_edge = edge_validation_methods[self.traversal_direction]

    @property
    def default_edge_type(self) -> str:
        if self._default_edge_type is None:
            assert self.src_node_id
            assert self.dst_node_id
            src_node_type = self.src_node_id.split("/")[0]
            dst_node_type = self.dst_node_id.split("/")[0]
            self._default_edge_type = self.edge_type_func(src_node_type, dst_node_type)

        return self._default_edge_type

    def __process_int_edge_key(self, key: int) -> str:
        if key < 0:
            key = len(self.data) + key

        if key < 0 or key >= len(self.data):
            raise KeyError(key)

        return next(islice(self.data.keys(), key, key + 1))

    def __is_valid_edge_outbound(self, edge: dict[str, Any]) -> bool:
        return bool(
            edge["_from"] == self.src_node_id and edge["_to"] == self.dst_node_id
        )

    def __is_valid_edge_inbound(self, edge: dict[str, Any]) -> bool:
        return bool(
            edge["_from"] == self.dst_node_id and edge["_to"] == self.src_node_id
        )

    def __is_valid_edge_any(self, edge: dict[str, Any]) -> bool:
        return self.__is_valid_edge_outbound(edge) or self.__is_valid_edge_inbound(edge)

    @logger_debug
    def __get_mirrored_edge_attr(self, edge_id: str) -> EdgeAttrDict | None:
        """This method is used to get the EdgeAttrDict of the
        "mirrored" EdgeKeyDict.

        A "mirrored edge" is defined as a reference to an edge that
        represents both the forward and reverse edge between two nodes. This is useful
        because ArangoDB does not need to duplicate edges in both directions
        in the database.

        If the Graph is Undirected:
        - The "mirror" is the same adjlist_outer_dict because
            the adjacency list is the same in both directions (i.e _adj)

        If the Graph is Directed:
        - The "mirror" is the "reverse" adjlist_outer_dict because
            the adjacency list is different in both directions (i.e _pred and _succ)

        :param dst_node_id: The destination node ID.
        :type dst_node_id: str
        :return: The edge attribute dictionary if it exists.
        :rtype: EdgeAttrDict | None
        """
        if self.adjlist_inner_dict is None:
            return None

        if self.adjlist_inner_dict.adjlist_outer_dict is None:
            return None

        mirror = self.adjlist_inner_dict.adjlist_outer_dict  # fake mirror (i.e G._adj)
        if self.is_directed:
            mirror = mirror.mirror  # real mirror (i.e _pred or _succ)

        if self.dst_node_id in mirror.data:
            if self.src_node_id in mirror.data[self.dst_node_id].data:
                if edge_id in mirror.data[self.dst_node_id].data[self.src_node_id].data:
                    return (
                        mirror.data[self.dst_node_id]
                        .data[self.src_node_id]
                        .data[edge_id]
                    )

        return None

    @logger_debug
    def _create_edge_attr_dict(self, edge: dict[str, Any]) -> EdgeAttrDict:
        edge_attr_dict = self.edge_attr_dict_factory()
        edge_attr_dict.edge_id = edge["_id"]
        edge_attr_dict.data = build_edge_attr_dict_data(edge_attr_dict, edge)

        return edge_attr_dict

    def __repr__(self) -> str:
        if self.FETCHED_ALL_DATA:
            return self.data.__repr__()

        return f"EdgeKeyDict('{self.src_node_id}', '{self.dst_node_id}')"

    def __str__(self) -> str:
        return self.__repr__()

    @key_is_adb_id_or_int
    @logger_debug
    def __contains__(self, key: str | int) -> bool:
        """
        'edge/1' in G._adj['node/1']['node/2']
        0 in G._adj['node/1']['node/2']
        """
        # HACK: This is a workaround for the fact that
        # nxadb.MultiGraph does not yet support custom edge keys
        if key == "-1":
            return False

        if isinstance(key, int):
            key = self.__process_int_edge_key(key)

        if key in self.data:
            return True

        if self.FETCHED_ALL_IDS:
            return False

        edge = self.graph.edge(key)

        if edge is None:
            logger.warning(f"Edge '{key}' does not exist in Graph.")
            return False

        if not self.__is_valid_edge(edge):
            m = f"Edge '{key}' exists, but does not match the source & destination nodes."  # noqa
            logger.warning(m)
            return False

        # Contrary to other __contains__ methods, we immediately
        # populate the Dict Data because we had to retrieve
        # the entire edge from the database to check if it is valid.
        edge_attr_dict = self._create_edge_attr_dict(edge)
        self.data[key] = edge_attr_dict
        return True

    @key_is_adb_id_or_int
    @logger_debug
    def __getitem__(self, key: str | int) -> EdgeAttrDict:
        """G._adj['node/1']['node/2']['edge/1']"""
        # HACK: This is a workaround for the fact that
        # nxadb.MultiGraph does not yet support custom edge keys
        if key == "-1":
            raise KeyError(key)

        if isinstance(key, int):
            key = self.__process_int_edge_key(key)

        # Notice the use of walrus operator here,
        # because we can return the value immediately
        # given that __contains__ builds EdgeAttrDict.data
        if value := self.data.get(key):
            return value

        if result := self.__get_mirrored_edge_attr(key):
            self.data[key] = result
            return result  # type: ignore # false positive

        if key not in self.data and self.FETCHED_ALL_IDS:
            raise KeyError(key)

        edge = self.graph.edge(key)

        if edge is None:
            raise KeyError(key)

        if not self.__is_valid_edge(edge):
            m = f"Edge '{key}' exists, but does not match the source & destination nodes."  # noqa
            raise KeyError(m)

        edge_attr_dict: EdgeAttrDict = self._create_edge_attr_dict(edge)
        self.data[key] = edge_attr_dict
        return edge_attr_dict

    @logger_debug
    def __setitem__(self, key: int, edge_attr_dict: EdgeAttrDict) -> None:
        """G._adj['node/1']['node/2'][0] = {'foo': 'bar'}"""

        self.data[str(key)] = edge_attr_dict

        if edge_attr_dict.edge_id:
            # NOTE: We can get here from L514 in networkx/multigraph.py
            # Assuming that keydict.get(key) did not return None (L513)

            # If the edge_id is already set, it means that the
            # EdgeAttrDict.update() that was just called was
            # able to update the edge in the database.
            # Therefore, we don't need to insert anything.

            if self.edge_type_key in edge_attr_dict.data:
                m = f"Cannot set '{self.edge_type_key}' if edge already exists in DB."
                raise EdgeTypeAmbiguity(m)

            return

        if not self.src_node_id or not self.dst_node_id:
            # We can get here from L521 in networkx/multigraph.py
            logger.debug("Node IDs not set, skipping EdgeKeyDict(?).__setitem__()")
            return

        # NOTE: We can get here from L514 in networkx/multigraph.py
        # Assuming that keydict.get(key) returned None (L513)

        edge_type = edge_attr_dict.data.pop(self.edge_type_key, None)
        if not edge_type:
            edge_type = self.default_edge_type

        edge = self.graph.link(
            edge_type, self.src_node_id, self.dst_node_id, edge_attr_dict.data
        )

        edge_data: dict[str, Any] = {
            **edge_attr_dict.data,
            **edge,
            "_from": self.src_node_id,
            "_to": self.dst_node_id,
        }

        # We have to re-create the EdgeAttrDict because the
        # previous one was created without any **edge_id**
        # TODO: Could we somehow update the existing EdgeAttrDict?
        # i.e edge_attr_dict.data = edge_data
        # + some extra code to set the **edge_id** attribute
        # for any nested EdgeAttrDicts within edge_attr_dict
        edge_id = edge["_id"]
        edge_attr_dict = self._create_edge_attr_dict(edge_data)
        self.data[edge_id] = edge_attr_dict
        del self.data[str(key)]

    @key_is_string
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """del G._adj['node/1']['node/2']['edge/1']"""
        self.data.pop(key, None)

        if self.__get_mirrored_edge_attr(key):
            # We're skipping the DB deletion because the
            # edge deletion for mirrored edges is handled
            # twice (once for each direction).
            # i.e the DB edge will be deleted in via the
            # delitem() call on the mirrored edge
            return

        try:
            self.graph.delete_edge(key)
        except DocumentDeleteError:
            # TODO: Should we just return here?
            raise KeyError(key)

    @logger_debug
    def clear(self) -> None:
        """G._adj['node/1']['node/2'].clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False
        self.FETCHED_ALL_IDS = False

    @keys_are_strings
    @logger_debug
    def update(self, edges: Any) -> None:
        """g._adj['node/1']['node/2'].update(
            {'edge/1': {'foo': 'bar'}, 'edge/2': {'baz': 'qux'}}
        )
        """
        raise NotImplementedError("EdgeKeyDict.update()")

    def popitem(self) -> tuple[str, dict[str, Any]]:  # type: ignore
        """G._adj['node/1']['node/2'].popitem()"""
        last_key = list(self.data.keys())[-1]
        edge_attr_dict = self.data[last_key]

        assert hasattr(edge_attr_dict, "to_dict")
        dict = edge_attr_dict.to_dict()

        self.__delitem__(last_key)
        return (last_key, dict)

    @logger_debug
    def __len__(self) -> int:
        """len(g._adj['node/1']['node/2'])"""
        assert self.src_node_id
        assert self.dst_node_id

        if self.FETCHED_ALL_IDS:
            return len(self.data)

        return aql_edge_count_src_dst(
            self.db,
            self.src_node_id,
            self.dst_node_id,
            self.graph.name,
            self.traversal_direction.name,
        )

    @logger_debug
    def __iter__(self) -> Iterator[str]:
        """for k in g._adj['node/1']['node/2']"""
        if not (self.FETCHED_ALL_DATA or self.FETCHED_ALL_IDS):
            self._fetch_all()

        yield from self.data.keys()

    @logger_debug
    def keys(self) -> Any:
        """g._adj['node/1']['node/2'].keys()"""
        if self.FETCHED_ALL_IDS:
            yield from self.data.keys()

        else:
            assert self.src_node_id
            assert self.dst_node_id

            edge_ids: list[str] | None = aql_edge_id(
                self.db,
                self.src_node_id,
                self.dst_node_id,
                self.graph.name,
                self.traversal_direction.name,
                can_return_multiple=True,
            )

            if edge_ids is None:
                raise ValueError("Failed to fetch Edge IDs")

            self.FETCHED_ALL_IDS = True
            for edge_id in edge_ids:
                self.data[edge_id] = self.edge_attr_dict_factory()
                yield edge_id

    @logger_debug
    def values(self) -> Any:
        """g._adj['node/1']['node/2'].values()"""
        if not self.FETCHED_ALL_DATA:
            self._fetch_all()

        yield from self.data.values()

    @logger_debug
    def items(self) -> Any:
        """g._adj['node/1']['node/2'].items()"""
        if not self.FETCHED_ALL_DATA:
            self._fetch_all()

        yield from self.data.items()

    @logger_debug
    def _fetch_all(self) -> None:
        assert self.src_node_id
        assert self.dst_node_id

        self.clear()

        edges: list[dict[str, Any]] | None = aql_edge_get(
            self.db,
            self.src_node_id,
            self.dst_node_id,
            self.graph.name,
            direction=self.traversal_direction.name,
            can_return_multiple=True,
        )

        if edges is None:
            raise ValueError("Failed to fetch edges")

        for edge in edges:
            edge_attr_dict = self._create_edge_attr_dict(edge)
            self.data[edge["_id"]] = edge_attr_dict

        self.FETCHED_ALL_DATA = True
        self.FETCHED_ALL_IDS = True


class AdjListInnerDict(UserDict[str, EdgeAttrDict | EdgeKeyDict]):
    """The 2nd level of the dict of dict (of dict) of dict structure
    representing the Adjacency List of a graph.

    AdjListInnerDict is keyed by the node ID of the destination node.

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
        edge_type_key: str,
        edge_type_func: Callable[[str, str], str],
        graph_type: str,
        adjlist_outer_dict: AdjListOuterDict | None,
        *args: Any,
        **kwargs: Any,
    ):
        if graph_type not in GraphType.__members__:
            raise ValueError(f"**graph_type** not supported: {graph_type}")

        super().__init__(*args, **kwargs)
        self.data: dict[str, EdgeAttrDict | EdgeKeyDict] = {}

        self.graph_type = graph_type
        self.is_directed = graph_type in DIRECTED_GRAPH_TYPES
        self.is_multigraph = graph_type in MULTIGRAPH_TYPES

        self.db = db
        self.graph = graph
        self.edge_type_key = edge_type_key
        self.edge_type_func = edge_type_func
        self.default_node_type = default_node_type
        self.edge_attr_dict_factory = edge_attr_dict_factory(self.db, self.graph)
        self.edge_key_dict_factory = edge_key_dict_factory(
            self.db, self.graph, edge_type_key, edge_type_func, self.is_directed, self
        )

        self.src_node_id: str | None = None
        self.__src_node_type: str | None = None
        self.adjlist_outer_dict = adjlist_outer_dict

        self.FETCHED_ALL_DATA = False
        self.FETCHED_ALL_IDS = False

        self.traversal_direction: TraversalDirection = (
            adjlist_outer_dict.traversal_direction
            if adjlist_outer_dict is not None
            else (
                TraversalDirection.OUTBOUND
                if self.is_directed
                else TraversalDirection.ANY
            )
        )

        direction_mappings = {
            TraversalDirection.OUTBOUND: ("e._to", "_to"),
            TraversalDirection.INBOUND: ("e._from", "_from"),
            TraversalDirection.ANY: ("e._to == @src_node_id ? e._from : e._to", None),
        }

        k = self.traversal_direction
        self.__iter__return_str, self._fetch_all_dst_node_key = direction_mappings[k]

        if self.is_multigraph:
            self.__contains_helper = self.__contains__multigraph
            self.__getitem_helper_db = self.__getitem__multigraph_db
            self.__getitem_helper_cache = self.__getitem__multigraph_cache
            self.__setitem_helper = self.__setitem__multigraph
            self.__delitem_helper = self.__delitem__multigraph
            self.__fetch_all_helper = self.__fetch_all_multigraph
        else:
            self.__contains_helper = self.__contains__graph
            self.__getitem_helper_db = self.__getitem__graph_db
            self.__getitem_helper_cache = self.__getitem__graph_cache
            self.__setitem_helper = self.__setitem__graph
            self.__delitem_helper = self.__delitem__graph
            self.__fetch_all_helper = self.__fetch_all_graph

    @property
    def src_node_type(self) -> str:
        if self.__src_node_type is None:
            assert self.src_node_id
            self.__src_node_type = self.src_node_id.split("/")[0]

        return self.__src_node_type

    @logger_debug
    def _create_edge_attr_dict(self, edge: dict[str, Any]) -> EdgeAttrDict:
        edge_attr_dict = self.edge_attr_dict_factory()
        edge_attr_dict.edge_id = edge["_id"]
        edge_attr_dict.data = build_edge_attr_dict_data(edge_attr_dict, edge)

        return edge_attr_dict

    @logger_debug
    def __get_mirrored_edge_attr_or_key_dict(
        self, dst_node_id: str
    ) -> EdgeAttrDict | EdgeKeyDict | None:
        """This method is used to get the EdgeAttrDict or EdgeKeyDict of the
        "mirrored" AdJlistInnerDict.

        A "mirrored edge" is defined as a reference to an edge (or multiple edges) that
        represents both the forward and reverse edge between two nodes. This is useful
        because ArangoDB does not need to duplicate edges in both directions
        in the database.

        If the Graph is Undirected:
        - The "mirror" is the same adjlist_outer_dict because
            the adjacency list is the same in both directions (i.e _adj)

        If the Graph is Directed:
        - The "mirror" is the "reverse" adjlist_outer_dict because
            the adjacency list is different in both directions (i.e _pred and _succ)

        :param dst_node_id: The destination node ID.
        :type dst_node_id: str
        :return: The edge attribute dictionary if it exists.
        :rtype: EdgeAttrDict | None
        """
        if self.adjlist_outer_dict is None:
            return None

        mirror = self.adjlist_outer_dict  # fake mirror (i.e G._adj)
        if self.is_directed:
            mirror = mirror.mirror  # real mirror (i.e _pred or _succ)

        if dst_node_id in mirror.data:
            if self.src_node_id in mirror.data[dst_node_id].data:
                return mirror.data[dst_node_id].data[self.src_node_id]

        return None

    def __repr__(self) -> str:
        if self.FETCHED_ALL_DATA:
            return self.data.__repr__()

        return f"AdjListInnerDict('{self.src_node_id}')"

    def __str__(self) -> str:
        return self.__repr__()

    @key_is_string
    @logger_debug
    def __contains__(self, key: str) -> bool:
        """'node/2' in G.adj['node/1']"""
        assert self.src_node_id
        dst_node_id = get_node_id(key, self.default_node_type)

        if dst_node_id in self.data:
            return True

        if self.FETCHED_ALL_IDS:
            return False

        result = aql_edge_exists(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction=self.traversal_direction.name,
        )

        if not result:
            return False

        self.__contains_helper(dst_node_id)

        return True

    @logger_debug
    def __contains__graph(self, dst_node_id: str) -> None:
        """Helper function for __contains__ in Graphs."""
        empty_edge_attr_dict = self.edge_attr_dict_factory()
        self.data[dst_node_id] = empty_edge_attr_dict

    @logger_debug
    def __contains__multigraph(self, dst_node_id: str) -> None:
        """Helper function for __contains__ in MultiGraphs."""
        lazy_edge_key_dict = self.edge_key_dict_factory()
        lazy_edge_key_dict.src_node_id = self.src_node_id
        lazy_edge_key_dict.dst_node_id = dst_node_id
        self.data[dst_node_id] = lazy_edge_key_dict

    @key_is_string
    @logger_debug
    def __getitem__(self, key: str) -> EdgeAttrDict | EdgeKeyDict:
        """g._adj['node/1']['node/2']"""
        dst_node_id = get_node_id(key, self.default_node_type)

        if self.__getitem_helper_cache(dst_node_id):
            return self.data[dst_node_id]

        if result := self.__get_mirrored_edge_attr_or_key_dict(dst_node_id):
            self.data[dst_node_id] = result
            return result  # type: ignore # false positive

        if key not in self.data and self.FETCHED_ALL_IDS:
            raise KeyError(key)

        return self.__getitem_helper_db(key, dst_node_id)  # type: ignore

    @logger_debug
    def __getitem__graph_cache(self, dst_node_id: str) -> bool:
        """Cache Helper function for __getitem__ in Graphs."""
        if _ := self.data.get(dst_node_id):
            return True

        return False

    @logger_debug
    def __getitem__graph_db(self, key: str, dst_node_id: str) -> EdgeAttrDict:
        """DB Helper function for __getitem__ in Graphs."""
        assert self.src_node_id
        edge: dict[str, Any] | None = aql_edge_get(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction=self.traversal_direction.name,
            can_return_multiple=self.is_multigraph,
        )

        if not edge:
            raise KeyError(key)

        edge_attr_dict: EdgeAttrDict = self._create_edge_attr_dict(edge)

        self.data[dst_node_id] = edge_attr_dict
        return edge_attr_dict

    @logger_debug
    def __getitem__multigraph_cache(self, dst_node_id: str) -> bool:
        """Cache Helper function for __getitem__ in Graphs."""
        # Notice that we're not using the walrus operator here
        # compared to other __getitem__ methods.
        # This is because EdgeKeyDict is lazily populated
        # when the second key is accessed (e.g G._adj["node/1"]["node/2"]['edge/1']).
        # Therefore, there is no actual data in EdgeKeyDict.data
        # when it is first created!
        return dst_node_id in self.data

    @logger_debug
    def __getitem__multigraph_db(self, key: str, dst_node_id: str) -> EdgeKeyDict:
        """Helper function for __getitem__ in MultiGraphs."""
        assert self.src_node_id
        result = aql_edge_exists(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction=self.traversal_direction.name,
        )

        if not result:
            raise KeyError(key)

        lazy_edge_key_dict = self.edge_key_dict_factory()
        lazy_edge_key_dict.src_node_id = self.src_node_id
        lazy_edge_key_dict.dst_node_id = dst_node_id

        self.data[dst_node_id] = lazy_edge_key_dict
        return lazy_edge_key_dict

    @key_is_string
    @logger_debug
    def __setitem__(self, key: str, value: EdgeAttrDict | EdgeKeyDict) -> None:
        """
        g._adj['node/1']['node/2'] = {'foo': 'bar'}
        g._adj['node/1']['node/2'] = {0: {'foo': 'bar'}}
        """
        assert self.src_node_id
        assert isinstance(value, EdgeKeyDict if self.is_multigraph else EdgeAttrDict)

        dst_node_type, dst_node_id = get_node_type_and_id(key, self.default_node_type)

        if result := self.__get_mirrored_edge_attr_or_key_dict(dst_node_id):
            self.data[dst_node_id] = result
            return

        self.__setitem_helper(value, dst_node_type, dst_node_id)

    @logger_debug
    def __setitem__graph(
        self, edge_attr_dict: EdgeAttrDict, dst_node_type: str, dst_node_id: str
    ) -> None:
        """Helper function for __setitem__ in Graphs."""
        if edge_attr_dict.edge_id:
            # If the edge_id is already set, it means that the
            # EdgeAttrDict.update() that was just called was
            # able to update the edge in the database.
            # Therefore, we don't need to insert anything.

            if self.edge_type_key in edge_attr_dict.data:
                m = f"Cannot set '{self.edge_type_key}' if edge already exists in DB."
                raise EdgeTypeAmbiguity(m)

            return

        edge_type = edge_attr_dict.data.pop(self.edge_type_key, None)
        if edge_type is None:
            edge_type = self.edge_type_func(self.src_node_type, dst_node_type)

        assert self.src_node_id

        edge_id = aql_edge_id(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction=self.traversal_direction.name,
            can_return_multiple=False,
        )

        if edge_id:
            edge = doc_insert(self.db, edge_type, edge_id, edge_attr_dict.data)
        else:
            edge = self.graph.link(
                edge_type, self.src_node_id, dst_node_id, edge_attr_dict.data
            )

        edge_data: dict[str, Any] = {
            **edge_attr_dict.data,
            **edge,
            "_from": self.src_node_id,
            "_to": dst_node_id,
        }

        # We have to re-create the EdgeAttrDict because the
        # previous one was created without any **edge_id**
        # TODO: Could we somehow update the existing EdgeAttrDict?
        # i.e edge_attr_dict.data = edge_data
        # + some extra code to set the **edge_id** attribute
        # for any nested EdgeAttrDicts within edge_attr_dict
        edge_attr_dict = self._create_edge_attr_dict(edge_data)
        self.data[dst_node_id] = edge_attr_dict

    @logger_debug
    def __setitem__multigraph(
        self, edge_key_dict: EdgeKeyDict, dst_node_type: str, dst_node_id: str
    ) -> None:
        """Helper function for __setitem__ in MultiGraphs."""
        assert len(edge_key_dict.data) == 1
        assert list(edge_key_dict.data.keys())[0] == "-1"
        assert edge_key_dict.src_node_id is None
        assert edge_key_dict.dst_node_id is None

        edge_attr_dict = edge_key_dict.data["-1"]

        edge_type = edge_attr_dict.data.pop(self.edge_type_key, None)
        if edge_type is None:
            edge_type = self.edge_type_func(self.src_node_type, dst_node_type)

        edge = self.graph.link(
            edge_type, self.src_node_id, dst_node_id, edge_attr_dict.data
        )

        edge_data: dict[str, Any] = {
            **edge_attr_dict.data,
            **edge,
            "_from": self.src_node_id,
            "_to": dst_node_id,
        }

        # We have to re-create the EdgeAttrDict because the
        # previous one was created without any **edge_id**
        # TODO: Could we somehow update the existing EdgeAttrDict?
        # i.e edge_attr_dict.data = edge_data
        # + some extra code to set the **edge_id** attribute
        # for any nested EdgeAttrDicts within edge_attr_dict
        edge_id = edge["_id"]
        edge_key_dict.data[edge_id] = self._create_edge_attr_dict(edge_data)
        edge_key_dict.src_node_id = self.src_node_id
        edge_key_dict.dst_node_id = dst_node_id
        edge_key_dict.traversal_direction = self.traversal_direction

        self.data[dst_node_id] = edge_key_dict
        del edge_key_dict.data["-1"]

    @key_is_string
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """del g._adj['node/1']['node/2']"""
        assert self.src_node_id
        dst_node_id = get_node_id(key, self.default_node_type)
        self.data.pop(dst_node_id, None)

        if self.__get_mirrored_edge_attr_or_key_dict(dst_node_id):
            # We're skipping the DB deletion because the
            # edge deletion for mirrored edges is handled
            # twice (once for each direction).
            # i.e the DB edge will be deleted in via the
            # delitem() call on the mirrored edge
            return

        result = aql_edge_id(
            self.db,
            self.src_node_id,
            dst_node_id,
            self.graph.name,
            direction=self.traversal_direction.name,
        )

        if not result:
            # TODO: Should we raise a KeyError instead?
            return

        self.__delitem_helper(result)

    @key_is_string
    @logger_debug
    def __delitem__graph(self, edge_id: str) -> None:
        """Helper function for __delitem__ in Graphs."""
        self.graph.delete_edge(edge_id)

    @key_is_string
    @logger_debug
    def __delitem__multigraph(self, edge_ids: list[str]) -> None:
        """Helper function for __delitem__ in MultiGraphs."""
        # TODO: Consider separating **edge_ids** by edge collection,
        # and invoking db.collection(...).delete_many() instead of this:
        for edge_id in edge_ids:
            self.__delitem__graph(edge_id)

    @logger_debug
    def __len__(self) -> int:
        """len(g._adj['node/1'])"""
        assert self.src_node_id

        if self.FETCHED_ALL_IDS:
            return len(self.data)

        return aql_edge_count_src(
            self.db, self.src_node_id, self.graph.name, self.traversal_direction.name
        )

    @logger_debug
    def __iter__(self) -> Iterator[str]:
        """for k in g._adj['node/1']"""
        if not (self.FETCHED_ALL_DATA or self.FETCHED_ALL_IDS):
            self._fetch_all()

        yield from self.data.keys()

    @logger_debug
    def keys(self) -> Any:
        """g._adj['node/1'].keys()"""
        if self.FETCHED_ALL_IDS:
            yield from self.data.keys()

        else:
            query = f"""
                FOR v, e IN 1..1 {self.traversal_direction.name} @src_node_id
                GRAPH @graph_name
                    RETURN {self.__iter__return_str}
            """

            bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

            self.FETCHED_ALL_IDS = True
            for edge_id in aql(self.db, query, bind_vars):
                self.__contains_helper(edge_id)
                yield edge_id

    @logger_debug
    def clear(self) -> None:
        """G._adj['node/1'].clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False
        self.FETCHED_ALL_IDS = False

    @keys_are_strings
    @logger_debug
    def update(self, edges: Any) -> None:
        """g._adj['node/1'].update({'node/2': {'foo': 'bar'}})"""
        raise NotImplementedError("AdjListInnerDict.update()")

    @logger_debug
    def values(self) -> Any:
        """g._adj['node/1'].values()"""
        if not self.FETCHED_ALL_DATA:
            self._fetch_all()

        yield from self.data.values()

    @logger_debug
    def items(self) -> Any:
        """g._adj['node/1'].items()"""
        if not self.FETCHED_ALL_DATA:
            self._fetch_all()

        yield from self.data.items()

    @logger_debug
    def _fetch_all(self) -> None:
        assert self.src_node_id

        self.clear()

        query = f"""
            FOR v, e IN 1..1 {self.traversal_direction.name} @src_node_id
            GRAPH @graph_name
                RETURN e
        """

        bind_vars = {"src_node_id": self.src_node_id, "graph_name": self.graph.name}

        for edge in aql(self.db, query, bind_vars):
            edge_attr_dict: EdgeAttrDict = self._create_edge_attr_dict(edge)

            dst_node_id: str = (
                edge[self._fetch_all_dst_node_key]
                if self._fetch_all_dst_node_key
                else edge["_to"] if self.src_node_id == edge["_from"] else edge["_from"]
            )

            self.__fetch_all_helper(edge_attr_dict, dst_node_id)

        self.FETCHED_ALL_DATA = True
        self.FETCHED_ALL_IDS = True

    @logger_debug
    def __fetch_all_graph(self, edge_attr_dict: EdgeAttrDict, dst_node_id: str) -> None:
        """Helper function for _fetch_all() in Graphs."""
        if dst_node_id in self.data:
            # Don't raise an error if it's a self-loop
            if self.data[dst_node_id] == edge_attr_dict:
                return

            m = "Multiple edges between the same nodes are not supported in Graphs."
            m += f" Found 2 edges between {self.src_node_id} & {dst_node_id}."
            m += " Consider using a MultiGraph."
            raise MultipleEdgesFound(m)

        self.data[dst_node_id] = edge_attr_dict

    @logger_debug
    def __fetch_all_multigraph(
        self, edge_attr_dict: EdgeAttrDict, dst_node_id: str
    ) -> None:
        """Helper function for _fetch_all() in MultiGraphs."""
        edge_key_dict = self.data.get(dst_node_id)
        if edge_key_dict is None:
            edge_key_dict = self.edge_key_dict_factory()
            edge_key_dict.src_node_id = self.src_node_id
            edge_key_dict.dst_node_id = dst_node_id
            edge_key_dict.FETCHED_ALL_DATA = True
            edge_key_dict.FETCHED_ALL_IDS = True

        assert isinstance(edge_key_dict, EdgeKeyDict)
        edge_key_dict.data[edge_attr_dict.edge_id] = edge_attr_dict
        self.data[dst_node_id] = edge_key_dict


class AdjListOuterDict(UserDict[str, AdjListInnerDict]):
    """The 1st level of the dict of dict (of dict) of dict
    representing the Adjacency List of a graph.

    AdjListOuterDict is keyed by the node ID of the source node.

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
        edge_type_key: str,
        edge_type_func: Callable[[str, str], str],
        graph_type: str,
        symmetrize_edges_if_directed: bool,
        *args: Any,
        **kwargs: Any,
    ):
        if graph_type not in GraphType.__members__:
            raise ValueError(f"**graph_type** not supported: {graph_type}")

        super().__init__(*args, **kwargs)
        self.data: dict[str, AdjListInnerDict] = {}

        self.graph_type = graph_type
        self.is_directed = graph_type in DIRECTED_GRAPH_TYPES
        self.is_multigraph = graph_type in MULTIGRAPH_TYPES

        self.db = db
        self.graph = graph
        self.edge_type_key = edge_type_key
        self.edge_type_func = edge_type_func
        self.default_node_type = default_node_type
        self.adjlist_inner_dict_factory = adjlist_inner_dict_factory(
            db,
            graph,
            default_node_type,
            edge_type_key,
            edge_type_func,
            graph_type,
            self,
        )

        self.FETCHED_ALL_DATA = False
        self.FETCHED_ALL_IDS = False

        self.traversal_direction = (
            TraversalDirection.OUTBOUND if self.is_directed else TraversalDirection.ANY
        )
        self.symmetrize_edges_if_directed = (
            symmetrize_edges_if_directed and self.is_directed
        )

        self.mirror: AdjListOuterDict

    def __get_mirrored_adjlist_inner_dict(
        self, node_id: str
    ) -> AdjListInnerDict | None:
        """This method is used to get the AdjListInnerDict of the
        "mirrored" AdjListOuterDict.

        A "mirrored edge" is defined as a reference to an edge that
        represents both the forward and reverse edge between two nodes. This is useful
        because ArangoDB does not need to duplicate edges in both directions
        in the database.

        If the Graph is Undirected:
        - The "mirror" is the same AdjListOuterDict because
            the adjacency list is the same in both directions (i.e _adj)

        If the Graph is Directed:
        - The "mirror" is the "reverse" AdjListOuterDict because
            the adjacency list is different in both directions (i.e _pred and _succ)

        :param node_id: The source node ID.
        :type node_id: str
        :return: The adjacency list inner dictionary if it exists.
        :rtype: AdjListInnerDict | None
        """
        if not self.is_directed:
            return None

        if node_id in self.mirror.data:
            return self.mirror.data[node_id]

        return None

    def __repr__(self) -> str:
        if self.FETCHED_ALL_DATA:
            return self.data.__repr__()

        return f"AdjListOuterDict('{self.graph.name}')"

    def __str__(self) -> str:
        return self.__repr__()

    @key_is_string
    @logger_debug
    def __contains__(self, key: str) -> bool:
        """'node/1' in G.adj"""
        node_id = get_node_id(key, self.default_node_type)

        if node_id in self.data:
            return True

        if self.FETCHED_ALL_IDS:
            return False

        if self.graph.has_vertex(node_id):
            lazy_adjlist_inner_dict = self.adjlist_inner_dict_factory()
            lazy_adjlist_inner_dict.src_node_id = node_id
            self.data[node_id] = lazy_adjlist_inner_dict
            return True

        return False

    @key_is_string
    @logger_debug
    def __getitem__(self, key: str) -> AdjListInnerDict:
        """G._adj["node/1"]"""
        node_id = get_node_id(key, self.default_node_type)

        if node_id in self.data:
            # Notice that we're not using the walrus operator here
            # compared to other __getitem__ methods.
            # This is because AdjListInnerDict is lazily populated
            # when the second key is accessed (e.g G._adj["node/1"]["node/2"]).
            # Therefore, there is no actual data in AdjListInnerDict.data
            # when it is first created!
            return self.data[node_id]

        if self.__get_mirrored_adjlist_inner_dict(node_id):
            lazy_adjlist_inner_dict = self.adjlist_inner_dict_factory()
            lazy_adjlist_inner_dict.src_node_id = node_id
            self.data[node_id] = lazy_adjlist_inner_dict

            return lazy_adjlist_inner_dict

        if self.FETCHED_ALL_IDS:
            raise KeyError(key)

        if self.graph.has_vertex(node_id):
            lazy_adjlist_inner_dict = self.adjlist_inner_dict_factory()
            lazy_adjlist_inner_dict.src_node_id = node_id
            self.data[node_id] = lazy_adjlist_inner_dict

            return lazy_adjlist_inner_dict

        raise KeyError(key)

    @key_is_string
    @logger_debug
    def __setitem__(self, src_key: str, adjlist_inner_dict: AdjListInnerDict) -> None:
        """g._adj['node/1'] = AdjListInnerDict()"""
        assert isinstance(adjlist_inner_dict, AdjListInnerDict)
        assert len(adjlist_inner_dict.data) == 0

        src_node_id = get_node_id(src_key, self.default_node_type)
        adjlist_inner_dict.src_node_id = src_node_id
        adjlist_inner_dict.adjlist_outer_dict = self
        adjlist_inner_dict.traversal_direction = self.traversal_direction
        self.data[src_node_id] = adjlist_inner_dict

    @key_is_string
    @logger_debug
    def __delitem__(self, key: str) -> None:
        """del G._adj['node/1']"""
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
        if not (self.FETCHED_ALL_DATA or self.FETCHED_ALL_IDS):
            self._fetch_all()

        yield from self.data.keys()

    @logger_debug
    def keys(self) -> Any:
        """g._adj.keys()"""
        if self.FETCHED_ALL_IDS:
            yield from self.data.keys()

        else:
            self.FETCHED_ALL_IDS = True
            for collection in self.graph.vertex_collections():
                for node_id in self.graph.vertex_collection(collection).ids():
                    lazy_adjlist_inner_dict = self.adjlist_inner_dict_factory()
                    lazy_adjlist_inner_dict.src_node_id = node_id
                    self.data[node_id] = lazy_adjlist_inner_dict
                    yield node_id

    @logger_debug
    def clear(self) -> None:
        """g._node.clear()"""
        self.data.clear()
        self.FETCHED_ALL_DATA = False
        self.FETCHED_ALL_IDS = False

    @keys_are_strings
    @logger_debug
    def update(self, edges: Any) -> None:
        """g._adj.update({'node/1': {'node/2': {'foo': 'bar'}})"""
        raise NotImplementedError("AdjListOuterDict.update()")

    @logger_debug
    def values(self) -> Any:
        """g._adj.values()"""
        if not self.FETCHED_ALL_DATA:
            self._fetch_all()

        yield from self.data.values()

    @logger_debug
    def items(self, data: str | None = None, default: Any | None = None) -> Any:
        """g._adj.items() or G._adj.items(data='foo')"""
        if data is None:
            if not self.FETCHED_ALL_DATA:
                self._fetch_all()
            yield from self.data.items()

        else:
            e_cols = [ed["edge_collection"] for ed in self.graph.edge_definitions()]
            yield from aql_fetch_data_edge(self.db, e_cols, data, default)

    @logger_debug
    def _fetch_all(self) -> None:
        self.clear()

        def set_adj_inner_dict(
            adj_outer_dict: AdjListOuterDict, node_id: str
        ) -> AdjListInnerDict:
            if node_id in adj_outer_dict.data:
                return adj_outer_dict.data[node_id]

            adj_inner_dict = self.adjlist_inner_dict_factory()
            adj_inner_dict.src_node_id = node_id
            adj_inner_dict.FETCHED_ALL_DATA = True
            adj_inner_dict.FETCHED_ALL_IDS = True
            adj_outer_dict.data[node_id] = adj_inner_dict

            return adj_inner_dict

        def propagate_edge_undirected(
            src_node_id: str,
            dst_node_id: str,
            edge_key_or_attr_dict: EdgeKeyDict | EdgeAttrDict,
        ) -> None:
            self.data[dst_node_id].data[src_node_id] = edge_key_or_attr_dict

        def propagate_edge_directed(
            src_node_id: str,
            dst_node_id: str,
            edge_key_or_attr_dict: EdgeKeyDict | EdgeAttrDict,
        ) -> None:
            set_adj_inner_dict(self.mirror, dst_node_id)
            self.mirror.data[dst_node_id].data[src_node_id] = edge_key_or_attr_dict

        def propagate_edge_directed_symmetric(
            src_node_id: str,
            dst_node_id: str,
            edge_key_or_attr_dict: EdgeKeyDict | EdgeAttrDict,
        ) -> None:
            propagate_edge_directed(src_node_id, dst_node_id, edge_key_or_attr_dict)
            propagate_edge_undirected(src_node_id, dst_node_id, edge_key_or_attr_dict)
            set_adj_inner_dict(self.mirror, src_node_id)
            self.mirror.data[src_node_id].data[dst_node_id] = edge_key_or_attr_dict

        propagate_edge_func = (
            propagate_edge_directed_symmetric
            if self.is_directed and self.symmetrize_edges_if_directed
            else (
                propagate_edge_directed
                if self.is_directed
                else propagate_edge_undirected
            )
        )

        def set_edge_graph(
            src_node_id: str, dst_node_id: str, edge: dict[str, Any]
        ) -> EdgeAttrDict:
            adjlist_inner_dict = self.data[src_node_id]

            edge_attr_dict: EdgeAttrDict
            edge_attr_dict = adjlist_inner_dict._create_edge_attr_dict(edge)

            adjlist_inner_dict.data[dst_node_id] = edge_attr_dict

            return edge_attr_dict

        def set_edge_multigraph(
            src_node_id: str, dst_node_id: str, edges: dict[int, dict[str, Any]]
        ) -> EdgeKeyDict:
            adjlist_inner_dict = self.data[src_node_id]

            edge_key_dict = adjlist_inner_dict.edge_key_dict_factory()
            edge_key_dict.src_node_id = src_node_id
            edge_key_dict.dst_node_id = dst_node_id
            edge_key_dict.FETCHED_ALL_DATA = True
            edge_key_dict.FETCHED_ALL_IDS = True

            for edge in edges.values():
                edge_attr_dict = adjlist_inner_dict._create_edge_attr_dict(edge)
                edge_key_dict.data[edge["_id"]] = edge_attr_dict

            adjlist_inner_dict.data[dst_node_id] = edge_key_dict

            return edge_key_dict

        set_edge_func = set_edge_multigraph if self.is_multigraph else set_edge_graph

        (
            node_dict,
            adj_dict,
            *_,
        ) = get_arangodb_graph(
            self.graph,
            load_node_dict=True,
            load_adj_dict=True,
            load_coo=False,
            edge_collections_attributes=set(),  # not used
            load_all_vertex_attributes=False,
            load_all_edge_attributes=True,
            is_directed=self.is_directed,
            is_multigraph=self.is_multigraph,
            symmetrize_edges_if_directed=self.is_directed
            and self.symmetrize_edges_if_directed,
        )

        if self.is_directed:
            adj_dict = adj_dict["succ"]

        for node_id in node_dict.keys():
            set_adj_inner_dict(self, node_id)

        for src_node_id, inner_dict in adj_dict.items():
            for dst_node_id, edge_or_edges in inner_dict.items():

                if not self.is_directed:
                    if src_node_id in self.data:
                        if dst_node_id in self.data[src_node_id].data:
                            continue  # can skip due not directed

                edge_attr_or_key_dict = set_edge_func(  # type: ignore[operator]
                    src_node_id, dst_node_id, edge_or_edges
                )

                propagate_edge_func(src_node_id, dst_node_id, edge_attr_or_key_dict)

        self.FETCHED_ALL_DATA = True
        self.FETCHED_ALL_IDS = True
        if self.is_directed:
            self.mirror.FETCHED_ALL_DATA = True
            self.mirror.FETCHED_ALL_IDS = True
