from typing import Any, Callable, ClassVar

import networkx as nx
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.classes.graph import Graph as nxadb_Graph

from .dict import AdjListOuterDict
from .enum import TraversalDirection

networkx_api = nxadb.utils.decorators.networkx_class(nx.DiGraph)  # type: ignore

__all__ = ["DiGraph"]


class DiGraph(nxadb_Graph, nx.DiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph  # type: ignore[no-any-return]

    def __init__(
        self,
        graph_name: str | None = None,
        default_node_type: str = "node",
        edge_type_func: Callable[[str, str], str] = lambda u, v: f"{u}_to_{v}",
        db: StandardDatabase | None = None,
        symmetrize_edges: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            graph_name, default_node_type, edge_type_func, db, *args, **kwargs
        )

        self.symmetrize_edges = symmetrize_edges
        if self.graph_exists_in_db:
            assert isinstance(self._succ, AdjListOuterDict)
            assert isinstance(self._pred, AdjListOuterDict)
            self._succ.mirror = self._pred
            self._pred.mirror = self._succ
            self._succ.traversal_direction = TraversalDirection.OUTBOUND
            self._pred.traversal_direction = TraversalDirection.INBOUND

    #######################
    # nx.DiGraph Overides #
    #######################

    # TODO?
    # @cached_property
    # def in_edges(self):
    # pass

    # TODO?
    # @cached_property
    # def out_edges(self):
    # pass

    def add_node(self, node_for_adding, **attr):
        if node_for_adding not in self._succ:
            if node_for_adding is None:
                raise ValueError("None cannot be a node")

            self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
            self._pred[node_for_adding] = self.adjlist_inner_dict_factory()

            ######################
            # NOTE: monkey patch #
            ######################

            # Old:
            # attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            # attr_dict.update(attr)

            # New:
            self._node[node_for_adding] = self.node_attr_dict_factory()
            self._node[node_for_adding].update(attr)

            # Reason:
            # Invoking `update` on the `attr_dict` without `attr_dict.node_id` being set
            # i.e trying to update a node's attributes before we know _which_ node it is

            ###########################

        else:
            self._node[node_for_adding].update(attr)

        nx._clear_cache(self)

    def remove_node(self, n):
        try:

            ######################
            # NOTE: monkey patch #
            ######################

            # Old:
            # nbrs = self._succ[n]

            # New:
            nbrs_succ = list(self._succ[n])
            nbrs_pred = list(self._pred[n])

            # Reason:
            # We need to fetch the outbound/inbound edges _prior_ to deleting the node,
            # as node deletion will already take care of deleting edges

            ###########################

            del self._node[n]
        except KeyError as err:  # NetworkXError if n not in self
            raise nx.NetworkXError(f"The node {n} is not in the digraph.") from err
        for u in nbrs_succ:
            del self._pred[u][n]  # remove all edges n-u in digraph
        del self._succ[n]  # remove node from succ
        for u in nbrs_pred:
            del self._succ[u][n]  # remove all edges n-u in digraph
        del self._pred[n]  # remove node from pred
        nx._clear_cache(self)
