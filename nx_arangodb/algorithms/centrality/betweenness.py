from networkx.algorithms.centrality import betweenness as nxbc

from nx_arangodb.convert import _to_graph
from nx_arangodb.utils import networkx_algorithm

# import pylibcugraph as plc
# from nx_cugraph.utils import networkx_algorithm, _seed_to_int


__all__ = ["betweenness_centrality"]


@networkx_algorithm(
    is_incomplete=True,
    is_different=True,
    version_added="23.10",
    _plc="betweenness_centrality",
)
def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    # We're just calling the original function from networkx here
    # to test things out for now. i.e no nx-cugraph stuff here

    print("ANTHONY: Calling betweenness_centrality from nx_arangodb")
    G = _to_graph(G)

    ##############################
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
    # rescaling
    betweenness = nxbc._rescale(
        betweenness,
        len(G),
        normalized=normalized,
        directed=G.is_directed(),
        k=k,
        endpoints=endpoints,
    )

    return betweenness
    ##############################

    # if weight is not None:
    #     raise NotImplementedError(
    #         "Weighted implementation of betweenness centrality not currently supported"
    #     )

    # seed = _seed_to_int(seed)

    # G = _nx_arangodb_graph_to_nx_cugraph_graph(G, weight)

    # node_ids, values = plc.betweenness_centrality(
    #     resource_handle=plc.ResourceHandle(),
    #     graph=G._get_plc_graph(),
    #     k=k,
    #     random_state=seed,
    #     normalized=normalized,
    #     include_endpoints=endpoints,
    #     do_expensive_check=False,
    # )

    # return G._nodearrays_to_dict(node_ids, values)
