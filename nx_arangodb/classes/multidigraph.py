from copy import deepcopy
from typing import Any, Callable, ClassVar

import networkx as nx
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.classes.digraph import DiGraph
from nx_arangodb.classes.multigraph import MultiGraph

networkx_api = nxadb.utils.decorators.networkx_class(nx.MultiDiGraph)  # type: ignore

__all__ = ["MultiDiGraph"]


class MultiDiGraph(MultiGraph, DiGraph, nx.MultiDiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiDiGraph]:
        return nx.MultiDiGraph  # type: ignore[no-any-return]

    def __init__(
        self,
        incoming_graph_data: Any = None,
        multigraph_input: bool | None = None,
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
        use_arango_views: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            incoming_graph_data,
            multigraph_input,
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
            use_arango_views,
            *args,
            **kwargs,
        )

        if self.graph_exists_in_db:
            self.reverse = self.reverse_override
            self.to_undirected = self.to_undirected_override

    #######################
    # Init helper methods #
    #######################

    ##########################
    # nx.MultiGraph Overides #
    ##########################

    def reverse_override(self, copy: bool = True) -> Any:
        if copy is False:
            raise NotImplementedError("In-place reverse is not supported yet.")

        return super().reverse(copy=True)

    def to_undirected_override(self, reciprocal=False, as_view=False):
        if reciprocal is False:
            return super().to_undirected(reciprocal=False, as_view=as_view)

        graph_class = self.to_undirected_class()
        if as_view is True:
            return nx.graphviews.generic_graph_view(self, graph_class)

        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())

        ######################
        # NOTE: Monkey patch #
        ######################

        # Old
        # G.add_edges_from(
        #     (u, v, key, deepcopy(data))
        #     for u, nbrs in self._adj.items()
        #     for v, keydict in nbrs.items()
        #     for key, data in keydict.items()
        #     if v in self._pred[u] and key in self._pred[u][v]
        # )

        # New:
        G.add_edges_from(
            (u, v, key, deepcopy(data))
            for u, nbrs in self._adj.items()
            for v, keydict in nbrs.items()
            for key, data in keydict.items()
            if v in self._pred[u]  # and key in self._pred[u][v]
        )

        # Reason: MultiGraphs in `nxadb` don't use integer-based keys for edges.
        # They use ArangoDB Edge IDs. Therefore, the statement `key in self._pred[u][v]`
        # will always be False in the context of MultiDiGraphs. For more details on why
        # this adjustment is needed, see the `test_to_undirected_reciprocal`
        # in `test_multidigraph.py`.

        ###########################

        return G
