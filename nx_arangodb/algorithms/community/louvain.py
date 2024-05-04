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
    run_on_gpu=True,
    pull_graph_on_cpu=True,
):
    if GPU_ENABLED and run_on_gpu:
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

    print("ANTHONY: to_nxadb")
    G = _to_nxadb_graph(G, pull_graph=pull_graph_on_cpu)
    return nx.community.louvain_communities.orig_func(
        G,
        weight=weight,
        resolution=resolution,
        threshold=threshold,
        max_level=max_level,
        seed=seed,
    )


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
    return nx.community.louvain_partitions.orig_func(
        G, weight=weight, resolution=resolution, threshold=threshold, seed=seed
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
    return nx.community.modularity.orig_func(
        G, communities, weight=weight, resolution=resolution
    )


@networkx_algorithm(
    extra_params={
        **_dtype_param,
    },
    is_incomplete=True,  # seed not supported; self-loops not supported
    is_different=True,  # RNG different
    version_added="23.10",
)
def is_partition(G, communities):
    return nx.community.is_partition.orig_func(G, communities)
