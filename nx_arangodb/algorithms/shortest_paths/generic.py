import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.exceptions import ShortestPathError
from nx_arangodb.utils import _dtype_param, networkx_algorithm

__all__ = ["shortest_path"]


@networkx_algorithm(
    extra_params=_dtype_param, version_added="24.04", _plc={"bfs", "sssp"}
)
def shortest_path(
    G: nxadb.Graph | nxadb.DiGraph,
    source=None,
    target=None,
    weight=None,
    method="dijkstra",
    *,
    dtype=None,
):
    """limited version of nx.shortest_path"""

    if not G.graph_exists:
        return nx.shortest_path.orig_func(
            G, source=source, target=target, weight=weight, method=method
        )

    if target is None or source is None:
        raise ShortestPathError("Both source and target must be specified for now")

    if method != "dijkstra":
        raise ShortestPathError("Only dijkstra method is supported")

    query = """
        FOR vertex IN ANY SHORTEST_PATH @source TO @target GRAPH @graph
        OPTIONS {'weightAttribute': @weight}
            RETURN vertex._id
    """

    bind_vars = {
        "source": source,
        "target": target,
        "graph": G.graph_name,
        "weight": weight,
    }

    result = list(G.aql(query, bind_vars=bind_vars))

    if not result:
        raise nx.NodeNotFound(f"Either source {source} or target {target} is not in G")

    return result
