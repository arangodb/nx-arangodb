import networkx as nx

from nx_arangodb.convert import _to_nxadb_graph, _to_nxcg_graph
from nx_arangodb.utils import networkx_algorithm

try:
    import nx_cugraph as nxcg

    GPU_ENABLED = True
    print("ANTHONY: GPU is enabled")
except ModuleNotFoundError:
    GPU_ENABLED = False
    print("ANTHONY: GPU is disabled")


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
    print("ANTHONY: Calling betweenness_centrality from nx_arangodb")

    if GPU_ENABLED and run_on_gpu:
        print("ANTHONY: to_nxcg")
        G = _to_nxcg_graph(G, weight)

        print("ANTHONY: Using nxcg bc()")
        return nxcg.betweenness_centrality(G, k=k, normalized=normalized, weight=weight)

    print("ANTHONY: to_nxadb")
    G = _to_nxadb_graph(G, pull_graph=pull_graph_on_cpu)

    print("ANTHONY: Using nx bc()")
    return nx.betweenness_centrality.orig_func(
        G, k=k, normalized=normalized, weight=weight, endpoints=endpoints
    )
