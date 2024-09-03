# type: ignore
# NOTE: NetworkX algorithms are not typed

import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.utils import _dtype_param, networkx_algorithm

__all__ = ["shortest_path"]


@networkx_algorithm(
    extra_params=_dtype_param, version_added="24.04", _plc={"bfs", "sssp"}
)
def shortest_path(
    G: nxadb.Graph,
    source=None,
    target=None,
    weight=None,
    method="dijkstra",
    *,
    dtype=None,
):
    """A server-side implementation of the nx.shortest_path algorithm.

    This algorithm will invoke the original NetworkX algorithm if one
    of the following conditions is met:
    - The graph is not stored in the database.
    - The method is not 'dijkstra'.
    - The target or source is not specified.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
        Starting node for path. If not specified, compute shortest
        paths for each possible starting node.

    target : node, optional
        Ending node for path. If not specified, compute shortest
        paths to all possible nodes.

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'dijkstra')
        The algorithm to use to compute the path.
        Supported options: 'dijkstra', 'bellman-ford'.
        Other inputs produce a ValueError.
        If `weight` is None, unweighted graph methods are used, and this
        suggestion is ignored.

    Returns
    -------
    path : list
        List of nodes in a shortest path.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    ValueError
        If `method` is not among the supported options.
    """

    graph_does_not_exist = not G.graph_exists_in_db
    target_or_source_not_specified = target is None or source is None
    method_not_dijkstra = method != "dijkstra"

    if any([graph_does_not_exist, target_or_source_not_specified, method_not_dijkstra]):
        return nx.shortest_path.orig_func(
            G, source=source, target=target, weight=weight, method=method
        )

    if isinstance(source, int):
        source = G.nodes[source]["_id"]

    if isinstance(target, int):
        target = G.nodes[target]["_id"]

    query = """
        FOR vertex IN ANY SHORTEST_PATH @source TO @target GRAPH @graph
        OPTIONS {'weightAttribute': @weight}
            RETURN vertex._id
    """

    bind_vars = {
        "source": source,
        "target": target,
        "graph": G.name,
        "weight": weight,
    }

    result = list(G.query(query, bind_vars=bind_vars))

    if not result:
        raise nx.NodeNotFound(f"Either source {source} or target {target} is not in G")

    return result
