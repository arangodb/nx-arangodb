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
    pull_graph_on_cpu=True,
    run_on_gpu=True,
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
    return nx.algorithms.link_analysis.pagerank_alg._pagerank_scipy(
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
    import scipy as sp

    if len(G) == 0:
        raise nx.NetworkXError("Graph has no nodes or edges")

    if nodelist is None:
        nodelist = list(G)
        nlen = len(G)
    else:
        nlen = len(nodelist)
        if nlen == 0:
            raise nx.NetworkXError("nodelist has no nodes")
        nodeset = set(G.nbunch_iter(nodelist))
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f"Node {n} in nodelist is not in G")
            raise nx.NetworkXError("nodelist contains duplicates.")
        if nlen < len(G):
            G = G.subgraph(nodelist)

    index = dict(zip(nodelist, range(nlen)))
    coefficients = zip(
        *((index[u], index[v], wt) for u, v, wt in G.edges(data=weight, default=1))
    )
    try:
        row, col, data = coefficients
    except ValueError:
        # there is no edge in the subgraph
        row, col, data = [], [], []

    if G.is_directed():
        A = sp.sparse.coo_array((data, (row, col)), shape=(nlen, nlen), dtype=dtype)
    else:
        # symmetrize matrix
        d = data + data
        r = row + col
        c = col + row
        # selfloop entries get double counted when symmetrizing
        # so we subtract the data on the diagonal
        selfloops = list(nx.selfloop_edges(G, data=weight, default=1))
        if selfloops:
            diag_index, diag_data = zip(*((index[u], -wt) for u, v, wt in selfloops))
            d += diag_data
            r += diag_index
            c += diag_index
        A = sp.sparse.coo_array((d, (r, c)), shape=(nlen, nlen), dtype=dtype)
    try:
        return A.asformat(format)
    except ValueError as err:
        raise nx.NetworkXError(f"Unknown sparse matrix format: {format}") from err
