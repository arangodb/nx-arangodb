"""Functions to convert between NetworkX, NetworkX-ArangoDB,
and NetworkX-cuGraph.

Examples
--------
>>> import networkx as nx
>>> import nx_arangodb as nxadb
>>> import nx_cugraph as nxcg
>>>
>>> G = nx.Graph()
>>> G.add_edge(1, 2, weight=3.0)
>>> G.add_edge(2, 3, weight=7.5)
>>>
>>> G_ADB = nxadb.convert._to_nxadb_graph(G)
>>> G_CG = nxadb.convert._to_nxcg_graph(G_ADB)
>>> G_NX = nxadb.convert._to_nx_graph(G_ADB)
"""

from __future__ import annotations

import time
from typing import Any

import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.classes.dict.adj import AdjListOuterDict
from nx_arangodb.classes.dict.node import NodeDict
from nx_arangodb.classes.function import do_load_all_edge_attributes
from nx_arangodb.logger import logger

try:
    import cupy as cp
    import nx_cugraph as nxcg

    GPU_AVAILABLE = True
    logger.info("NetworkX-cuGraph is available.")
except Exception as e:
    GPU_AVAILABLE = False
    logger.info(f"NetworkX-cuGraph is unavailable: {e}.")

__all__ = [
    "_to_nx_graph",
    "_to_nxadb_graph",
    "_to_nxcg_graph",
]


def _to_nx_graph(G: Any, *args: Any, **kwargs: Any) -> nx.Graph:
    """Convert a graph to a NetworkX graph.

    Parameters
    ----------
    G : Any
        The graph to convert.

        Currently supported types:
        - nx.Graph
        - nxadb.Graph

    Returns
    -------
    nx.Graph
        The converted graph.
    """
    logger.debug(f"_to_nx_graph for {G.__class__.__name__}")

    if isinstance(G, nxadb.Graph):
        return nxadb_to_nx(G)

    if isinstance(G, nx.Graph):
        return G

    raise TypeError(f"Expected nxadb.Graph or nx.Graph; got {type(G)}")


def _to_nxadb_graph(
    G: Any, *args: Any, as_directed: bool = False, **kwargs: Any
) -> nxadb.Graph:
    """Convert a graph to a NetworkX-ArangoDB graph.

    Parameters
    ----------
    G : Any
        The graph to convert.

        Currently supported types:
        - nx.Graph
        - nxadb.Graph

    as_directed : bool, optional
        Whether to convert the graph to a directed graph.
        Default is False.

    Returns
    -------
    nxadb.Graph
        The converted graph.
    """
    logger.debug(f"_to_nxadb_graph for {G.__class__.__name__}")

    if isinstance(G, nxadb.Graph):
        return G

    if isinstance(G, nx.Graph):
        return nx_to_nxadb(G, as_directed=as_directed)

    raise TypeError(f"Expected nxadb.Graph or nx.Graph; got {type(G)}")


if GPU_AVAILABLE:

    def _to_nxcg_graph(G: Any, as_directed: bool = False) -> nxcg.Graph:
        """Convert a graph to a NetworkX-cuGraph graph.

        NOTE: Only supported if NetworkX-cuGraph is installed.

        Parameters
        ----------
        G : Any
            The graph to convert.

            Currently supported types:
            - nxadb.Graph
            - nxcg.Graph

        as_directed : bool, optional
            Whether to convert the graph to a directed graph.
            Default is False.

        Returns
        -------
        nxcg.Graph
            The converted graph.
        """
        logger.debug(f"_to_nxcg_graph for {G.__class__.__name__}")

        if isinstance(G, nxcg.Graph):
            return G

        if isinstance(G, nxadb.Graph):
            if not G.graph_exists_in_db:
                m = "nx_arangodb.Graph does not exist in ArangoDB. Cannot pull graph."
                raise ValueError(m)

            logger.debug("converting nx_arangodb graph to nx_cugraph graph")
            return nxadb_to_nxcg(G, as_directed=as_directed)

        raise TypeError(f"Expected nx_arangodb.Graph or nxcg.Graph; got {type(G)}")

else:

    def _to_nxcg_graph(G: Any, as_directed: bool = False) -> nxcg.Graph:
        m = "nx-cugraph is not installed; cannot convert to nx-cugraph"
        raise NotImplementedError(m)


def nx_to_nxadb(
    graph: nx.Graph,
    *args: Any,
    as_directed: bool = False,
    **kwargs: Any,
) -> nxadb.Graph:
    """Convert a NetworkX graph to a NetworkX-ArangoDB graph.

    Parameters
    ----------
    graph : nx.Graph
        The NetworkX graph to convert.

    as_directed : bool, optional
        Whether to convert the graph to a directed graph.
        Default is False.

    Returns
    -------
    nxadb.Graph
        The converted graph.
    """
    logger.debug(f"from_networkx for {graph.__class__.__name__}")

    klass: type[nxadb.Graph]
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

    return klass(incoming_graph_data=graph)


def nxadb_to_nx(G: nxadb.Graph) -> nx.Graph:
    """Convert a NetworkX-ArangoDB graph to a NetworkX graph.

    This function will pull the graph from the database if it does
    not exist in the cache. A new NetworkX graph will be created
    using the node and adjacency dictionaries that are fetched.

    NOTE: The current downside of this approach is that we are not
    able to take advantage of the custom Dictionary classes that we
    have implemented in nx_arangodb.classes.dict. This is because
    the node and adjacency dictionaries are fetched as regular
    Python dictionaries. Furthermore, we don't cache the dictionaries
    themselves, so we have to fetch them every time we convert the
    graph, which is currently being invoked on *every* algorithm
    call. See the note below for a potential solution. As a temporary
    workaround, users can do the following:

    ```
        import networkx as nx
        import nx_arangodb as nxadb

        G_ADB = nxadb.Graph(name="MyGraph") # Connect to the graph
        G_NX = nxadb.convert._to_nx_graph(G_ADB) # Pull the graph

        nx.pagerank(G_NX)
        nx.betweenness_centrality(G_NX)
        ...
    ```

    Parameters
    ----------
    G : nxadb.Graph
        The NetworkX-ArangoDB graph to convert.

    Returns
    -------
    nx.Graph
        The converted graph.
    """
    if not G.graph_exists_in_db:
        # Since nxadb.Graph is a subclass of nx.Graph, we can return it as is.
        # This only applies if the graph does not exist in the database.
        return G

    assert isinstance(G._node, NodeDict)
    assert isinstance(G._adj, AdjListOuterDict)
    if G._node.FETCHED_ALL_DATA and G._adj.FETCHED_ALL_DATA:
        return G

    start_time = time.time()

    node_dict, adj_dict, *_ = nxadb.classes.function.get_arangodb_graph(
        adb_graph=G.adb_graph,
        load_node_dict=True,
        load_adj_dict=True,
        load_coo=False,
        edge_collections_attributes=G.edge_attributes,
        load_all_vertex_attributes=False,
        load_all_edge_attributes=do_load_all_edge_attributes(G.edge_attributes),
        is_directed=G.is_directed(),
        is_multigraph=G.is_multigraph(),
        symmetrize_edges_if_directed=G.symmetrize_edges if G.is_directed() else False,
    )

    logger.info(f"Graph '{G.adb_graph.name}' load took {time.time() - start_time}s")

    # NOTE: At this point, we _could_ choose to implement something similar to
    # NodeDict._fetch_all() and AdjListOuterDict._fetch_all() to iterate through
    # **node_dict** and **adj_dict**, and establish the "custom" Dictionary classes
    # that we've implemented in nx_arangodb.classes.dict.
    # However, this would involve adding additional for-loops and would likely be
    # slower than the current implementation.
    # Perhaps we should consider adding a feature flag to allow users to choose
    # between the two methods? e.g `build_remote_dicts=True/False`
    # If True, then we would return the (updated) nxadb.Graph that was passed in.
    # If False, then we would return the nx.Graph that is built below:

    G_NX: nx.Graph = G.to_networkx_class()()
    G_NX._node = node_dict

    if isinstance(G_NX, nx.DiGraph):
        G_NX._succ = G_NX._adj = adj_dict["succ"]
        G_NX._pred = adj_dict["pred"]

    else:
        G_NX._adj = adj_dict

    return G_NX


if GPU_AVAILABLE:

    def nxadb_to_nxcg(G: nxadb.Graph, as_directed: bool = False) -> nxcg.Graph:
        """Convert a NetworkX-ArangoDB graph to a NetworkX-cuGraph graph.

        This function will pull the graph from the database if it does
        not exist in the cache. A new NetworkX-cuGraph graph will be
        created using the COO format that is fetched. The created graph
        will be cached in the nxadb.Graph object for future use.

        Parameters
        ----------
        G : nxadb.Graph
            The NetworkX-ArangoDB graph to convert.

        as_directed : bool, optional
            Whether to convert the graph to a directed graph.
            Default is False.

        Returns
        -------
        nxcg.Graph
            The converted graph.
        """
        if G.use_nxcg_cache and G.nxcg_graph is not None:
            m = "**use_nxcg_cache** is enabled. using cached NXCG Graph. no pull required."  # noqa
            logger.debug(m)

            return G.nxcg_graph

        start_time = time.time()

        (
            _,
            _,
            src_indices,
            dst_indices,
            edge_indices,
            vertex_ids_to_index,
            edge_values,
        ) = nxadb.classes.function.get_arangodb_graph(
            adb_graph=G.adb_graph,
            load_node_dict=False,
            load_adj_dict=False,
            load_coo=True,
            edge_collections_attributes=G.edge_attributes,
            load_all_vertex_attributes=False,  # not used
            load_all_edge_attributes=do_load_all_edge_attributes(G.edge_attributes),
            is_directed=G.is_directed(),
            is_multigraph=G.is_multigraph(),
            symmetrize_edges_if_directed=(
                G.symmetrize_edges if G.is_directed() else False
            ),
        )

        logger.info(f"Graph '{G.adb_graph.name}' load took {time.time() - start_time}s")

        start_time = time.time()

        N = len(vertex_ids_to_index)
        src_indices_cp = cp.array(src_indices)
        dst_indices_cp = cp.array(dst_indices)
        edge_indices_cp = cp.array(edge_indices)

        if G.is_multigraph():
            if G.is_directed() or as_directed:
                klass = nxcg.MultiDiGraph
            else:
                klass = nxcg.MultiGraph

            G.nxcg_graph = klass.from_coo(
                N=N,
                src_indices=src_indices_cp,
                dst_indices=dst_indices_cp,
                edge_indices=edge_indices_cp,
                edge_values=edge_values,
                # edge_masks,
                # node_values,
                # node_masks,
                key_to_id=vertex_ids_to_index,
                # edge_keys=edge_keys,
            )

        else:
            if G.is_directed() or as_directed:
                klass = nxcg.DiGraph
            else:
                klass = nxcg.Graph

            G.nxcg_graph = klass.from_coo(
                N=N,
                src_indices=src_indices_cp,
                dst_indices=dst_indices_cp,
                edge_values=edge_values,
                # edge_masks,
                # node_values,
                # node_masks,
                key_to_id=vertex_ids_to_index,
            )

        logger.info(f"NXCG Graph construction took {time.time() - start_time}s")

        return G.nxcg_graph
