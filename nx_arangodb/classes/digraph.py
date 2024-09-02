from typing import Any, Callable, ClassVar

import networkx as nx
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.classes.graph import Graph
from nx_arangodb.logger import logger

from .dict.adj import AdjListOuterDict
from .enum import TraversalDirection
from .function import get_node_id

networkx_api = nxadb.utils.decorators.networkx_class(nx.DiGraph)  # type: ignore

__all__ = ["DiGraph"]


class DiGraph(Graph, nx.DiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph  # type: ignore[no-any-return]

    def __init__(
        self,
        incoming_graph_data: Any = None,
        name: str | None = None,
        default_node_type: str | None = None,
        edge_type_key: str = "_edge_type",
        edge_type_func: Callable[[str, str], str] | None = None,
        edge_collections_attributes: set[str] | None = None,
        db: StandardDatabase | None = None,
        read_parallelism: int = 10,
        read_batch_size: int = 100000,
        write_batch_size: int = 50000,
        write_async: bool = True,
        symmetrize_edges: bool = False,
        use_experimental_views: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            incoming_graph_data,
            name,
            default_node_type,
            edge_type_key,
            edge_type_func,
            edge_collections_attributes,
            db,
            read_parallelism,
            read_batch_size,
            write_batch_size,
            write_async,
            symmetrize_edges,
            use_experimental_views,
            *args,
            **kwargs,
        )

        if self.graph_exists_in_db:
            self.clear_edges = self.clear_edges_override
            self.add_node = self.add_node_override
            self.remove_node = self.remove_node_override
            self.reverse = self.reverse_override

            assert isinstance(self._succ, AdjListOuterDict)
            assert isinstance(self._pred, AdjListOuterDict)
            self._succ.mirror = self._pred
            self._pred.mirror = self._succ
            self._succ.traversal_direction = TraversalDirection.OUTBOUND
            self._pred.traversal_direction = TraversalDirection.INBOUND

        if (
            not self.is_multigraph()
            and incoming_graph_data is not None
            and not self._loaded_incoming_graph_data
        ):
            nx.convert.to_networkx_graph(incoming_graph_data, create_using=self)

    #######################
    # nx.DiGraph Overides #
    #######################

    # TODO?
    # If we want to continue with "Experimental Views" we need to implement the
    # InEdgeView and OutEdgeView classes.
    # @cached_property
    # def in_edges(self):
    # pass

    # TODO?
    # @cached_property
    # def out_edges(self):
    # pass

    def reverse_override(self, copy: bool = True) -> Any:
        if copy is False:
            raise NotImplementedError("In-place reverse is not supported yet.")

        return super().reverse(copy=True)

    def clear_edges_override(self):
        logger.info("Note that clearing edges ony erases the edges in the local cache")
        for predecessor_dict in self._pred.data.values():
            predecessor_dict.clear()

        super().clear_edges()

    def add_node_override(self, node_for_adding, **attr):
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

    def remove_node_override(self, n):
        if isinstance(n, (str, int)):
            n = get_node_id(str(n), self.default_node_type)

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
            ######################
            # NOTE: Monkey patch #
            ######################

            # Old: Nothing

            # New:
            if u == n:
                continue  # skip self loops

            # Reason: We need to skip self loops, as they are
            # already taken care of in the previous step. This
            # avoids getting a KeyError on the next line.

            ###########################

            del self._succ[u][n]  # remove all edges n-u in digraph
        del self._pred[n]  # remove node from pred
        nx._clear_cache(self)
