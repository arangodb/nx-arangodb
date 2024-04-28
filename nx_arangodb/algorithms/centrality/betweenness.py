from networkx.algorithms.centrality import betweenness as nxbc

from nx_arangodb.convert import _to_graph as _to_nx_arangodb_graph
from nx_arangodb.utils import networkx_algorithm

try:
    import pylibcugraph as plc
    from nx_cugraph.convert import _to_graph as _to_nx_cugraph_graph
    from nx_cugraph.utils import _seed_to_int

    GPU_ENABLED = True
except ModuleNotFoundError:
    GPU_ENABLED = False


__all__ = ["betweenness_centrality"]

# 1. If GPU is enabled, call nx-cugraph bc() after converting to a nx_cugraph graph (in-memory graph)
# 2. If GPU is not enabled, call networkx bc() after converting to a networkx graph (in-memory graph)
# 3. If GPU is not enabled, call networkx bc() **without** converting to a networkx graph (remote graph)


@networkx_algorithm(
    is_incomplete=True,
    is_different=True,
    version_added="23.10",
    _plc="betweenness_centrality",
)
def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    print("ANTHONY: Calling betweenness_centrality from nx_arangodb")

    # 1.
    if GPU_ENABLED:
        print("ANTHONY: GPU is enabled. Using nx-cugraph bc()")

        if weight is not None:
            raise NotImplementedError(
                "Weighted implementation of betweenness centrality not currently supported"
            )

        seed = _seed_to_int(seed)
        G = _to_nx_cugraph_graph(G, weight)
        node_ids, values = plc.betweenness_centrality(
            resource_handle=plc.ResourceHandle(),
            graph=G._get_plc_graph(),
            k=k,
            random_state=seed,
            normalized=normalized,
            include_endpoints=endpoints,
            do_expensive_check=False,
        )

        return G._nodearrays_to_dict(node_ids, values)

    # 2.
    else:
        print("ANTHONY: GPU is disabled. Using nx bc()")

        G = _to_nx_arangodb_graph(G)

        betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
        if k is None:
            nodes = G
        else:
            nodes = seed.sample(list(G.nodes()), k)
        for s in nodes:
            # single source shortest paths
            if weight is None:  # use BFS
                S, P, sigma, _ = nxbc._single_source_shortest_path_basic(G, s)
            else:  # use Dijkstra's algorithm
                S, P, sigma, _ = nxbc._single_source_dijkstra_path_basic(G, s, weight)
            # accumulation
            if endpoints:
                betweenness, _ = nxbc._accumulate_endpoints(betweenness, S, P, sigma, s)
            else:
                betweenness, _ = nxbc._accumulate_basic(betweenness, S, P, sigma, s)

        betweenness = nxbc._rescale(
            betweenness,
            len(G),
            normalized=normalized,
            directed=G.is_directed(),
            k=k,
            endpoints=endpoints,
        )

        return betweenness

    # 3. TODO
