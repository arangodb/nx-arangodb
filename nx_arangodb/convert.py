from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING

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
    edge_attrs: AttrKey | dict[AttrKey, EdgeValue | None] | None = None,
    edge_dtypes: Dtype | dict[AttrKey, Dtype | None] | None = None,
    *,
    node_attrs: AttrKey | dict[AttrKey, NodeValue | None] | None = None,
    node_dtypes: Dtype | dict[AttrKey, Dtype | None] | None = None,
    preserve_all_attrs: bool = False,
    preserve_edge_attrs: bool = False,
    preserve_node_attrs: bool = False,
    preserve_graph_attrs: bool = False,
    as_directed: bool = False,
    name: str | None = None,
    graph_name: str | None = None,
) -> nxadb.Graph:
    """Convert a networkx graph to nx_arangodb graph; can convert all attributes.

    TEMPORARY ASSUMPTION: The nx_arangodb Graph is a subclass of networkx Graph.
    Therefore, I'm going to assume that we _should_ be able instantiate an
    nx_arangodb Graph using the **incoming_graph_data** parameter. Let's try it!

    Parameters
    ----------
    G : networkx.Graph
    edge_attrs : str or dict, optional
        Dict that maps edge attributes to default values if missing in ``G``.
        If None, then no edge attributes will be converted.
        If default value is None, then missing values are handled with a mask.
        A default value of ``nxcg.convert.REQUIRED`` or ``...`` indicates that
        all edges have data for this attribute, and raise `KeyError` if not.
        For convenience, `edge_attrs` may be a single attribute with default 1;
        for example ``edge_attrs="weight"``.
    edge_dtypes : dtype or dict, optional
    node_attrs : str or dict, optional
        Dict that maps node attributes to default values if missing in ``G``.
        If None, then no node attributes will be converted.
        If default value is None, then missing values are handled with a mask.
        A default value of ``nxcg.convert.REQUIRED`` or ``...`` indicates that
        all edges have data for this attribute, and raise `KeyError` if not.
        For convenience, `node_attrs` may be a single attribute with no default;
        for example ``node_attrs="weight"``.
    node_dtypes : dtype or dict, optional
    preserve_all_attrs : bool, default False
        If True, then equivalent to setting preserve_edge_attrs, preserve_node_attrs,
        and preserve_graph_attrs to True.
    preserve_edge_attrs : bool, default False
        Whether to preserve all edge attributes.
    preserve_node_attrs : bool, default False
        Whether to preserve all node attributes.
    preserve_graph_attrs : bool, default False
        Whether to preserve all graph attributes.
    as_directed : bool, default False
        If True, then the returned graph will be directed regardless of input.
        If False, then the returned graph type is determined by input graph.
    name : str, optional
        The name of the algorithm when dispatched from networkx.
    graph_name : str, optional
        The name of the graph argument geing converted when dispatched from networkx.

    Returns
    -------
    nx_arangodb.Graph

    Notes
    -----
    For optimal performance, be as specific as possible about what is being converted:

    1. Do you need edge values? Creating a graph with just the structure is the fastest.
    2. Do you know the edge attribute(s) you need? Specify with `edge_attrs`.
    3. Do you know the default values? Specify with ``edge_attrs={weight: default}``.
    4. Do you know if all edges have values? Specify with ``edge_attrs={weight: ...}``.
    5. Do you know the dtype of attributes? Specify with `edge_dtypes`.

    Conversely, using ``preserve_edge_attrs=True`` or ``preserve_all_attrs=True`` are
    the slowest, but are also the most flexible and generic.

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

    return klass(incoming_graph_data=graph)


def to_networkx(G: nxadb.Graph, *, sort_edges: bool = False) -> nx.Graph:
    """Convert a nx_arangodb graph to networkx graph.

    All edge and node attributes and ``G.graph`` properties are converted.

    TEMPORARY ASSUMPTION: The nx_arangodb Graph is a subclass of networkx Graph.
    Therefore, I'm going to assume that we _should_ be able instantiate an
    nx Graph using the **incoming_graph_data** parameter. Let's try it!

    Parameters
    ----------
    G : nx_arangodb.Graph
    sort_edges : bool, default False
        Whether to sort the edge data of the input graph by (src, dst) indices
        before converting. This can be useful to convert to networkx graphs
        that iterate over edges consistently since edges are stored in dicts
        in the order they were added.

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
) -> nxadb.Graph | nxadb.DiGraph:
    logger.debug(f"from_networkx_arangodb for {G.__class__.__name__}")

    if not isinstance(G, (nxadb.Graph, nxadb.DiGraph)):
        raise TypeError(f"Expected nx_arangodb.(Graph || DiGraph); got {type(G)}")

    if not G.graph_exists:
        logger.debug("graph does not exist, nothing to pull")
        return G

    if not pull_graph:
        if isinstance(G, nxadb.DiGraph):
            m = "nx_arangodb.DiGraph has no CRUD Support yet. Cannot rely on remote connection."
            raise NotImplementedError(m)

        logger.debug("graph exists, but not pulling. relying on remote connection...")
        return G

    # if G.use_nx_cache and G._node and G._adj:
    #     m = "**use_nx_cache** is enabled. using cached data. no pull required."
    #     logger.debug(m)
    #     return G

    logger.debug("pulling as NetworkX Graph...")
    start_time = time.time()
    node_dict, adj_dict, _, _, _ = nxadb.classes.function.get_arangodb_graph(
        G,
        load_node_dict=True,
        load_adj_dict=True,
        load_adj_dict_as_directed=G.is_directed(),
        load_coo=False,
    )
    end_time = time.time()
    logger.debug(f"load took {end_time - start_time} seconds")

    # Copied from nx.convert.to_networkx_graph
    try:
        logger.debug("creating nx graph from loaded ArangoDB data...")
        result = nx.convert.from_dict_of_dicts(
            adj_dict,
            create_using=G.__class__,
            multigraph_input=G.is_multigraph(),
        )

        for n, dd in node_dict.items():
            result._node[n].update(dd)

        return result

    except Exception as err:
        raise nx.NetworkXError("Input is not a correct NetworkX graph.") from err


def _to_nxadb_graph(
    G,
    edge_attr: AttrKey | None = None,
    edge_default: EdgeValue | None = 1,
    edge_dtype: Dtype | None = None,
    pull_graph: bool = True,
) -> nxadb.Graph | nxadb.DiGraph:
    """Ensure that input type is a nx_arangodb graph, and convert if necessary."""
    logger.debug(f"_to_nxadb_graph for {G.__class__.__name__}")

    if isinstance(G, (nxadb.Graph, nxadb.DiGraph)):
        return from_networkx_arangodb(G, pull_graph)

    if isinstance(G, nx.Graph):
        return from_networkx(
            G, {edge_attr: edge_default} if edge_attr is not None else None, edge_dtype
        )
    # TODO: handle cugraph.Graph
    raise TypeError


if GPU_ENABLED:

    def _to_nxcg_graph(
        G,
        edge_attr: AttrKey | None = None,
        edge_default: EdgeValue | None = 1,
        edge_dtype: Dtype | None = None,
        as_directed: bool = False,
    ) -> nxcg.Graph | nxcg.DiGraph:
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
            if G.graph_exists:
                logger.debug("converting nx_arangodb graph to nx_cugraph graph")
                return nxcg_from_networkx_arangodb(G, as_directed=as_directed)

        if isinstance(G, (nxadb.MultiGraph, nxadb.MultiDiGraph)):
            raise NotImplementedError(
                "nxadb.MultiGraph not yet supported for _to_nxcg_graph()"
            )

        # If G is a networkx graph, or is a nxadb graph that doesn't point to an "existing"
        # ArangoDB graph, then we just treat it as a normal networkx graph &
        # convert it to nx_cugraph.
        # TODO: Need to revisit the "existing" ArangoDB graph condition...
        if isinstance(G, nx.Graph):
            logger.debug("converting networkx graph to nx_cugraph graph")
            return nxcg.convert.from_networkx(
                G,
                {edge_attr: edge_default} if edge_attr is not None else None,
                edge_dtype,
                as_directed=as_directed,
            )

        # TODO: handle cugraph.Graph
        raise TypeError

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
            start_time = time.time()
            _, _, src_indices, dst_indices, vertex_ids_to_index = (
                nxadb.classes.function.get_arangodb_graph(
                    G,
                    load_node_dict=False,
                    load_adj_dict=False,
                    load_adj_dict_as_directed=G.is_directed(),
                    load_coo=True,
                )
            )
            end_time = time.time()
            logger.debug(f"load took {end_time - start_time} seconds")

            G.src_indices = src_indices
            G.dst_indices = dst_indices
            G.vertex_ids_to_index = vertex_ids_to_index

        N = len(G.vertex_ids_to_index)

        if G.is_directed() or as_directed:
            klass = nxcg.DiGraph
        else:
            klass = nxcg.Graph

        logger.debug("creating nx_cugraph graph from COO data...")
        start_time = time.time()
        rv = klass.from_coo(
            N,
            cp.array(G.src_indices),
            cp.array(G.dst_indices),
            key_to_id=G.vertex_ids_to_index,
        )
        end_time = time.time()
        logger.debug(f"nxcg from_coo took {end_time - start_time}")

        return rv

else:

    def _to_nxcg_graph(
        G,
        edge_attr: AttrKey | None = None,
        edge_default: EdgeValue | None = 1,
        edge_dtype: Dtype | None = None,
        as_directed: bool = False,
    ) -> nxcg.Graph | nxcg.DiGraph:
        m = "nx-cugraph is not installed; cannot convert to nx-cugraph graph"
        raise NotImplementedError(m)
