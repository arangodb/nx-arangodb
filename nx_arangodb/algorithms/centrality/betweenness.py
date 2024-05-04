import networkx as nx

from nx_arangodb.convert import _to_nxadb_graph, _to_nxcg_graph
from nx_arangodb.logger import logger
from nx_arangodb.utils import networkx_algorithm

try:
    import nx_cugraph as nxcg

    GPU_ENABLED = True
except ModuleNotFoundError:
    GPU_ENABLED = False


__all__ = ["betweenness_centrality"]


@networkx_algorithm(
    is_incomplete=True,
    is_different=True,
    version_added="23.10",
    _plc="betweenness_centrality",
)
def betweenness_centrality(
    G,
    k=None,
    normalized=True,
    weight=None,
    endpoints=False,
    seed=None,
    run_on_gpu=True,
    pull_graph_on_cpu=True,
):
    logger.debug(f"nxadb.betweenness_centrality for {G.__class__.__name__}")

    if GPU_ENABLED and run_on_gpu:
        G = _to_nxcg_graph(G, weight)

        logger.debug("using nxcg.betweenness_centrality")
        return nxcg.betweenness_centrality(G, k=k, normalized=normalized, weight=weight)

    G = _to_nxadb_graph(G, pull_graph=pull_graph_on_cpu)

    logger.debug("using nx.betweenness_centrality")
    return nx.betweenness_centrality.orig_func(
        G, k=k, normalized=normalized, weight=weight, endpoints=endpoints
    )
