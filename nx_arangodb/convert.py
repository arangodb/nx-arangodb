from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any

import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.logger import logger

try:
    import cupy as cp
    import numpy as np
    import nx_cugraph as nxcg

    GPU_ENABLED = True
    logger.info("NXCG is enabled.")
except ModuleNotFoundError as e:
    GPU_ENABLED = False
    logger.info(f"NXCG is disabled. {e}.")


if TYPE_CHECKING:  # pragma: no cover
    from nx_arangodb.typing import AttrKey, Dtype, EdgeValue, NodeValue

__all__ = [
    "from_networkx",
    "to_networkx",
]

concat = itertools.chain.from_iterable
# A "required" attribute is one that all edges or nodes must have or KeyError is raised
REQUIRED = ...


def from_networkx(
    graph: nx.Graph,
    *args: Any,
    as_directed: bool = False,
    **kwargs: Any,
    # name: str | None = None,
    # graph_name: str | None = None,
) -> nxadb.Graph | nxadb.DiGraph:
    """Convert a networkx graph to nx_arangodb graph.

    Parameters
    ----------
    G : networkx.Graph

    See Also
    --------
    to_networkx : The opposite; convert nx_arangodb graph to networkx graph
    """
    logger.debug(f"from_networkx for {graph.__class__.__name__}")

    if not isinstance(graph, nx.Graph):
        if isinstance(graph, nx.classes.reportviews.NodeView):
            # Convert to a Graph with only nodes (no edges)
            G = nx.Graph()
            G.add_nodes_from(graph.items())
            graph = G
        else:
            raise TypeError(f"Expected networkx.Graph; got {type(graph)}")

    if graph.is_multigraph():
        if graph.is_directed() or as_directed:
            klass = nxadb.MultiDiGraph
        else:
            klass = nxadb.MultiGraph

    else:
        if graph.is_directed() or as_directed:
            klass = nxadb.DiGraph
        else:
            klass = nxadb.Graph

    # graph_name=kwargs.get("graph_name") ?
    return klass(incoming_graph_data=graph)


def to_networkx(G: nxadb.Graph, *args: Any, **kwargs: Any) -> nx.Graph:
    """Convert a nx_arangodb graph to networkx graph.

    All edge and node attributes and ``G.graph`` properties are converted.

    TEMPORARY ASSUMPTION: The nx_arangodb Graph is a subclass of networkx Graph.
    Therefore, I'm going to assume that we _should_ be able instantiate an
    nx Graph using the **incoming_graph_data** parameter.

    Parameters
    ----------
    G : nx_arangodb.Graph

    Returns
    -------
    networkx.Graph

    See Also
    --------
    from_networkx : The opposite; convert networkx graph to nx_cugraph graph
    """
    logger.debug(f"to_networkx for {G.__class__.__name__}")

    if not isinstance(G, nxadb.Graph):
        raise TypeError(f"Expected nx_arangodb.Graph; got {type(G)}")

    return G.to_networkx_class()(incoming_graph_data=G)


def from_networkx_arangodb(
    G: nxadb.Graph | nxadb.DiGraph, pull_graph: bool
) -> nx.Graph | nx.DiGraph:
    logger.debug(f"from_networkx_arangodb for {G.__class__.__name__}")

    if not isinstance(G, (nxadb.Graph, nxadb.DiGraph)):
        raise TypeError(f"Expected nx_arangodb.(Graph || DiGraph); got {type(G)}")

    if not G.graph_exists_in_db:
        logger.debug("graph does not exist, nothing to pull")
        return G

    if not pull_graph:
        if isinstance(G, nxadb.DiGraph):
            m = "nx_arangodb.DiGraph has no CRUD Support yet. Cannot rely on remote connection."  # noqa: E501
            raise NotImplementedError(m)

        logger.debug("graph exists, but not pulling. relying on remote connection...")
        return G

    # if G.use_nx_cache and G._node and G._adj:
    #     m = "**use_nx_cache** is enabled. using cached data. no pull required."
    #     logger.debug(m)
    #     return G

    logger.debug("pulling as NetworkX Graph...")
    print(f"Fetching {G.graph_name} as dictionaries...")
    start_time = time.time()
    _, adj_dict, _, _, _ = nxadb.classes.function.get_arangodb_graph(
        adb_graph=G.adb_graph,
        load_node_dict=False,  # TODO: Should we load node dict?
        load_adj_dict=True,
        is_directed=G.is_directed(),
        is_multigraph=G.is_multigraph(),
        load_coo=False,
    )
    end_time = time.time()
    logger.debug(f"load took {end_time - start_time} seconds")
    print(f"ADB -> Dictionaries load took {end_time - start_time} seconds")

    return G.to_networkx_class()(incoming_graph_data=adj_dict)
    # G._adj = adj_dict # TODO: Can I do this instead?


def _to_nx_graph(
    G: Any,
    pull_graph: bool = True,
) -> nx.Graph | nx.DiGraph:
    """Ensure that input type is an nx graph, and convert if necessary."""
    logger.debug(f"_to_nx_graph for {G.__class__.__name__}")

    if isinstance(G, (nxadb.Graph, nxadb.DiGraph)):
        return from_networkx_arangodb(G, pull_graph)

    if isinstance(G, nx.Graph):
        return G

    raise TypeError(f"Expected nx_arangodb.Graph or nx.Graph; got {type(G)}")


if GPU_ENABLED:

    def _to_nxcg_graph(G: Any, as_directed: bool = False) -> nxcg.Graph | nxcg.DiGraph:
        """Ensure that input type is a nx_cugraph graph, and convert if necessary."""
        logger.debug(f"_to_nxcg_graph for {G.__class__.__name__}")

        if isinstance(G, nxcg.Graph):
            logger.debug("already an nx_cugraph graph")
            return G

        if isinstance(G, (nxadb.Graph, nxadb.DiGraph)):
            # Assumption: G.adb_graph_name points to an existing graph in ArangoDB
            # Therefore, the user wants us to pull the graph from ArangoDB,
            # and convert it to an nx_cugraph graph.
            # We currently accomplish this by using the NetworkX adapter for ArangoDB,
            # which converts the ArangoDB graph to a NetworkX graph, and then we convert
            # the NetworkX graph to an nx_cugraph graph.
            # TODO: Implement a direct conversion from ArangoDB to nx_cugraph
            if G.graph_exists_in_db:
                logger.debug("converting nx_arangodb graph to nx_cugraph graph")
                return nxcg_from_networkx_arangodb(G, as_directed=as_directed)

        if isinstance(G, (nxadb.MultiGraph, nxadb.MultiDiGraph)):
            m = "nxadb.MultiGraph not yet supported for _to_nxcg_graph()"
            raise NotImplementedError(m)

        # TODO: handle cugraph.Graph
        raise TypeError(f"Expected nx_arangodb.Graph or nx.Graph; got {type(G)}")

    def nxcg_from_networkx_arangodb(
        G: nxadb.Graph | nxadb.DiGraph, as_directed: bool = False
    ) -> nxcg.Graph | nxcg.DiGraph:
        """Convert an nx_arangodb graph to nx_cugraph graph."""
        logger.debug(f"nxcg_from_networkx_arangodb for {G.__class__.__name__}")

        if G.is_multigraph():
            raise NotImplementedError("Multigraphs not yet supported")

        if (
            G.use_coo_cache
            and G.src_indices is not None
            and G.dst_indices is not None
            and G.vertex_ids_to_index is not None
        ):
            m = "**use_coo_cache** is enabled. using cached COO data. no pull required."
            logger.debug(m)

        else:
            logger.debug("pulling as NetworkX-CuGraph Graph...")
            print(f"Fetching {G.graph_name} as COO...")
            start_time = time.time()
            _, _, src_indices, dst_indices, vertex_ids_to_index = (
                nxadb.classes.function.get_arangodb_graph(
                    adb_graph=G.adb_graph,
                    load_node_dict=False,
                    load_adj_dict=False,
                    is_directed=G.is_directed(),  # not used
                    is_multigraph=G.is_multigraph(),  # not used
                    load_coo=True,
                )
            )
            end_time = time.time()
            logger.debug(f"load took {end_time - start_time} seconds")
            print(f"ADB -> COO load took {end_time - start_time} seconds")

            G.src_indices = src_indices
            G.dst_indices = dst_indices
            G.vertex_ids_to_index = vertex_ids_to_index

        N = len(G.vertex_ids_to_index)

        if G.is_directed() or as_directed:
            klass = nxcg.DiGraph
        else:
            klass = nxcg.Graph

        start_time = time.time()
        print("Building CuPy arrays...")
        src_indices_cp = cp.array(G.src_indices)
        dst_indices_cp = cp.array(G.dst_indices)
        end_time = time.time()
        print(f"COO (NumPy) -> COO (CuPy) took {end_time - start_time}")

        logger.debug("creating nx_cugraph graph from COO data...")
        print("creating nx_cugraph graph from COO data...")
        start_time = time.time()
        rv = klass.from_coo(
            N=N,
            src_indices=src_indices_cp,
            dst_indices=dst_indices_cp,
            key_to_id=G.vertex_ids_to_index,
        )
        end_time = time.time()
        print(f"COO -> NXCG took {end_time - start_time}")
        logger.debug(f"nxcg from_coo took {end_time - start_time}")

        return rv

else:

    def _to_nxcg_graph(G: Any, as_directed: bool = False) -> nxcg.Graph | nxcg.DiGraph:
        m = "nx-cugraph is not installed; cannot convert to nx-cugraph graph"
        raise NotImplementedError(m)
