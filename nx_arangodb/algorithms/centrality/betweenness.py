from networkx.algorithms.centrality import betweenness as nx_betweenness

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

    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    if k is None:
        nodes = G
    else:
        nodes = seed.sample(list(G.nodes()), k)
    for s in nodes:
        # single source shortest paths
        if weight is None:  # use BFS
            S, P, sigma, _ = nx_betweenness._single_source_shortest_path_basic(G, s)
        else:  # use Dijkstra's algorithm
            S, P, sigma, _ = nx_betweenness._single_source_dijkstra_path_basic(
                G, s, weight
            )
        # accumulation
        if endpoints:
            betweenness, _ = nx_betweenness._accumulate_endpoints(
                betweenness, S, P, sigma, s
            )
        else:
            betweenness, _ = nx_betweenness._accumulate_basic(
                betweenness, S, P, sigma, s
            )

    betweenness = nx_betweenness._rescale(
        betweenness,
        len(G),
        normalized=normalized,
        directed=G.is_directed(),
        k=k,
        endpoints=endpoints,
    )

    return betweenness
