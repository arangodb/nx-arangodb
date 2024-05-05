from collections import deque

import networkx as nx

from nx_arangodb.convert import _to_nxadb_graph, _to_nxcg_graph
from nx_arangodb.logger import logger
from nx_arangodb.utils import _dtype_param, networkx_algorithm

try:
    import nx_cugraph as nxcg

    GPU_ENABLED = True
except ModuleNotFoundError:
    GPU_ENABLED = False


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
    logger.debug(f"nxadb.louvain_communities for {G.__class__.__name__}")

    if GPU_ENABLED and run_on_gpu:
        G = _to_nxcg_graph(G, weight)

        logger.debug("using nxcg.louvain_communities")
        print("Running nxcg.louvain_communities()")
        return nxcg.algorithms.community.louvain._louvain_communities(
            G,
            weight=weight,
            resolution=resolution,
            threshold=threshold,
            max_level=max_level,
            seed=seed,
        )

    G = _to_nxadb_graph(G, pull_graph=pull_graph_on_cpu)

    logger.debug("using nx.louvain_communities")
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
