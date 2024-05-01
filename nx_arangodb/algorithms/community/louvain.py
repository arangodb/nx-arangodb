from collections import deque

import networkx as nx

from nx_arangodb.convert import _to_nxadb_graph, _to_nxcg_graph
from nx_arangodb.utils import _dtype_param, networkx_algorithm

try:
    import nx_cugraph as nxcg

    GPU_ENABLED = True
    print("ANTHONY: GPU is enabled")
except ModuleNotFoundError:
    GPU_ENABLED = False
    print("ANTHONY: GPU is disabled")


@networkx_algorithm(
    extra_params={
        **_dtype_param,
    },
    is_incomplete=True,  # seed not supported; self-loops not supported
    is_different=True,  # RNG different
    version_added="23.10",
    _plc="louvain",
    name="louvain_communities",
)
def louvain_communities(
    G,
    weight="weight",
    resolution=1,
    threshold=0.0000001,
    max_level=None,
    seed=None,
):
    if GPU_ENABLED:
        print("ANTHONY: to_nxcg")
        G = _to_nxcg_graph(G, weight)

        print("ANTHONY: Using nxcg louvain()")
        return nxcg.algorithms.community.louvain._louvain_communities(
            G,
            weight=weight,
            resolution=resolution,
            threshold=threshold,
            max_level=max_level,
            seed=seed,
        )

    else:
        print("ANTHONY: to_nxadb")
        G = _to_nxadb_graph(G)

        print("ANTHONY: Using nx pagerank()")
        import random

        d = louvain_partitions(G, weight, resolution, threshold, random.Random())
        q = deque(d, maxlen=1)
        return q.pop()


@networkx_algorithm(
    extra_params={
        **_dtype_param,
    },
    is_incomplete=True,  # seed not supported; self-loops not supported
    is_different=True,  # RNG different
    version_added="23.10",
    _plc="louvain",
    name="louvain_partitions",
)
def louvain_partitions(
    G, weight="weight", resolution=1, threshold=0.0000001, seed=None
):
    partition = [{u} for u in G.nodes()]
    if nx.is_empty(G):
        yield partition
        return
    mod = modularity(G, partition, resolution=resolution, weight=weight)
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = nx.community._convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    m = graph.size(weight="weight")
    partition, inner_partition, improvement = nx.community.louvain._one_level(
        graph, m, partition, resolution, is_directed, seed
    )
    improvement = True
    while improvement:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
        new_mod = modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        graph = nx.community.louvain._gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = nx.community.louvain._one_level(
            graph, m, partition, resolution, is_directed, seed
        )


@networkx_algorithm(
    extra_params={
        **_dtype_param,
    },
    is_incomplete=True,  # seed not supported; self-loops not supported
    is_different=True,  # RNG different
    version_added="23.10",
)
def modularity(G, communities, weight="weight", resolution=1):
    if not isinstance(communities, list):
        communities = list(communities)
    # if not is_partition(G, communities):
    #     raise NotAPartition(G, communities)

    directed = G.is_directed()
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        m = sum(out_degree.values())
        norm = 1 / m**2
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum**2

    def community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = sum(in_degree[u] for u in comm) if directed else out_degree_sum

        return L_c / m - resolution * out_degree_sum * in_degree_sum * norm

    return sum(map(community_contribution, communities))
