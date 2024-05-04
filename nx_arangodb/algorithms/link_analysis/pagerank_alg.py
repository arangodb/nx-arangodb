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
    extra_params=_dtype_param,
    is_incomplete=True,  # dangling not supported
    version_added="23.12",
    _plc={"pagerank", "personalized_pagerank"},
)
def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
    *,
    dtype=None,
    run_on_gpu=True,
    pull_graph_on_cpu=True,
):
    print("ANTHONY: Calling pagerank from nx_arangodb")

    if GPU_ENABLED and run_on_gpu:
        print("ANTHONY: to_nxcg")
        G = _to_nxcg_graph(G, weight)

        print("ANTHONY: Using nxcg pagerank()")
        return nxcg.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
            nstart=nstart,
            weight=weight,
            dangling=dangling,
            dtype=dtype,
        )

    print("ANTHONY: to_nxadb")
    G = _to_nxadb_graph(G, pull_graph=pull_graph_on_cpu)

    print("ANTHONY: Using nx pagerank()")
    return nx.algorithms.pagerank.orig_func(
        G,
        alpha=alpha,
        personalization=personalization,
        max_iter=max_iter,
        tol=tol,
        nstart=nstart,
        weight=weight,
        dangling=dangling,
    )


@networkx_algorithm(
    extra_params=_dtype_param,
    version_added="23.12",
)
def to_scipy_sparse_array(G, nodelist=None, dtype=None, weight="weight", format="csr"):
    return nx.to_scipy_sparse_array.orig_func(G, nodelist, dtype, weight, format)
