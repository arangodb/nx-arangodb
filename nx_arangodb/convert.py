from __future__ import annotations

import time
from typing import Any

import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.classes.function import do_load_all_edge_attributes
from nx_arangodb.logger import logger

try:
    import cupy as cp
    import numpy as np
    import nx_cugraph as nxcg

    GPU_ENABLED = True
    logger.info("NetworkX-cuGraph is enabled.")
except Exception as e:
    GPU_ENABLED = False
    logger.info(f"NetworkX-cuGraph is disabled: {e}.")

__all__ = [
    "_to_nx_graph",
    "_to_nxadb_graph",
    "_to_nxcg_graph",
]


def _to_nx_graph(G: Any, *args: Any, **kwargs: Any) -> nx.Graph:
    logger.debug(f"_to_nx_graph for {G.__class__.__name__}")

    if isinstance(G, nxadb.Graph | nxadb.DiGraph):
        return nxadb_to_nx(G)

    if isinstance(G, nx.Graph):
        return G

    raise TypeError(f"Expected nxadb.Graph or nx.Graph; got {type(G)}")


def _to_nxadb_graph(
    G: Any,
    *args: Any,
    as_directed: bool = False,
    **kwargs: Any,
) -> nxadb.Graph:
    logger.debug(f"_to_nxadb_graph for {G.__class__.__name__}")

    if isinstance(G, nxadb.Graph):
        return G

    if isinstance(G, nx.Graph):
        return nx_to_nxadb(G, as_directed=as_directed)

    raise TypeError(f"Expected nxadb.Graph or nx.Graph; got {type(G)}")


if GPU_ENABLED:

    def _to_nxcg_graph(G: Any, as_directed: bool = False) -> nxcg.Graph:
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
    # name: str | None = None,
    # graph_name: str | None = None,
) -> nxadb.Graph:
    logger.debug(f"from_networkx for {graph.__class__.__name__}")

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


def nxadb_to_nx(G: nxadb.Graph) -> nx.Graph:
    if not G.graph_exists_in_db:
        logger.debug("graph does not exist, nothing to pull")
        # TODO: Consider just returning G here?
        # Avoids the need to re-create the graph from scratch
        return G.to_networkx_class()(incoming_graph_data=G)

    # TODO: Re-enable this
    # if G.use_nx_cache and G._node and G._adj:
    #     m = "**use_nx_cache** is enabled. using cached data. no pull required."
    #     logger.debug(m)
    #     return G

    start_time = time.time()

    node_dict, adj_dict, *_ = nxadb.classes.function.get_arangodb_graph(
        adb_graph=G.adb_graph,
        load_node_dict=True,
        load_adj_dict=True,
        load_coo=False,
        edge_collections_attributes=G.get_edge_attributes,
        load_all_vertex_attributes=False,
        load_all_edge_attributes=do_load_all_edge_attributes(G.get_edge_attributes),
        is_directed=G.is_directed(),
        is_multigraph=G.is_multigraph(),
        symmetrize_edges_if_directed=G.symmetrize_edges if G.is_directed() else False,
    )

    print(f"ADB -> NX took {time.time() - start_time}s")

    G_NX: nx.Graph | nx.DiGraph = G.to_networkx_class()()
    G_NX._node = node_dict

    if isinstance(G_NX, nx.DiGraph):
        G_NX._succ = G._adj = adj_dict["succ"]
        G_NX._pred = adj_dict["pred"]

    else:
        G_NX._adj = adj_dict

    return G_NX


if GPU_ENABLED:

    def nxadb_to_nxcg(G: nxadb.Graph, as_directed: bool = False) -> nxcg.Graph:
        if G.use_nxcg_cache and G.nxcg_graph is not None:
            m = "**use_nxcg_cache** is enabled. using cached NXCG Graph. no pull required."
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
            edge_collections_attributes=G.get_edge_attributes,
            load_all_vertex_attributes=False,  # not used
            load_all_edge_attributes=do_load_all_edge_attributes(G.get_edge_attributes),
            is_directed=G.is_directed(),
            is_multigraph=G.is_multigraph(),
            symmetrize_edges_if_directed=(
                G.symmetrize_edges if G.is_directed() else False
            ),
        )

        print(f"ADB -> COO took {time.time() - start_time}s")

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

        print(f"COO -> NXCG Graph took {time.time() - start_time}s")

        return G.nxcg_graph
