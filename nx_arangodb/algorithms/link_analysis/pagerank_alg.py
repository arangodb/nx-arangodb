from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_scipy

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
):
    print("ANTHONY: Calling pagerank from nx_arangodb")

    # 1.
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

    # 2.
    else:
        print("ANTHONY: to_nxadb")
        G = _to_nxadb_graph(G)

        print("ANTHONY: Using nx pagerank()")
        return _pagerank_scipy(
            G,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
            nstart=nstart,
            weight=weight,
            dangling=dangling,
        )
