from copy import deepcopy
from typing import Any, Callable, ClassVar

import networkx as nx
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.classes.digraph import DiGraph
from nx_arangodb.classes.multigraph import MultiGraph

networkx_api = nxadb.utils.decorators.networkx_class(nx.MultiDiGraph)  # type: ignore

__all__ = ["MultiDiGraph"]


class MultiDiGraph(MultiGraph, DiGraph, nx.MultiDiGraph):
    """
    A directed graph class that can store multiedges.

    Subclasses ``nxadb.MultiGraph``, ``nxadb.Digraph``, and ``nx.MultiGraph``.

    In order to connect to an ArangoDB instance, the following environment
    variables must be set:

    1. ``DATABASE_HOST``
    2. ``DATABASE_USERNAME``
    3. ``DATABASE_PASSWORD``
    4. ``DATABASE_NAME``

    Furthermore, the ``name`` parameter is required to create a new graph
    or to connect to an existing graph in the database.

    Example
    -------
    >>> import os
    >>> import networkx as nx
    >>> import nx_arangodb as nxadb
    >>>
    >>> os.environ["DATABASE_HOST"] = "http://localhost:8529"
    >>> os.environ["DATABASE_USERNAME"] = "root"
    >>> os.environ["DATABASE_PASSWORD"] = "openSesame"
    >>> os.environ["DATABASE_NAME"] = "_system"
    >>>
    >>> G = nxadb.DiGraph(name="MyGraph")
    >>> ...


    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created. Must be used in conjunction with **name** if
        the user wants to persist the graph in ArangoDB. NOTE: It is
        recommended for incoming_graph_data to be a NetworkX graph due
        to faster loading times.

    multigraph_input : bool or None (default None)
        Note: Only used when `incoming_graph_data` is a dict.
        If True, `incoming_graph_data` is assumed to be a
        dict-of-dict-of-dict-of-dict structure keyed by
        node to neighbor to edge keys to edge data for multi-edges.
        A NetworkXError is raised if this is not the case.
        If False, :func:`to_networkx_graph` is used to try to determine
        the dict's graph data structure as either a dict-of-dict-of-dict
        keyed by node to neighbor to edge data, or a dict-of-iterable
        keyed by node to neighbors.
        If None, the treatment for True is tried, but if it fails,
        the treatment for False is tried.

    name : str (optional, default: None)
        Name of the graph in the database. If the graph already exists,
        the user can pass the name of the graph to connect to it. If
        the graph does not exist, a General Graph will be created by
        passing the **name**. NOTE: Must be used in conjunction with
        **incoming_graph_data** if the user wants to persist the graph
        in ArangoDB.

    default_node_type : str (optional, default: None)
        Default node type for the graph. In ArangoDB terms, this is the
        default vertex collection. If the graph already exists, the user can
        omit this parameter and the default node type will be set to the
        first vertex collection in the graph. If the graph does not exist,
        the user can pass the default node type to create the default vertex
        collection.

    edge_type_key : str (optional, default: "_edge_type")
        Key used to store the edge type when inserting edges into the graph.
        Useful for working with Heterogeneous Graphs.

    edge_type_func : Callable[[str, str], str] (optional, default: None)
        Function to determine the edge type between two nodes. If the graph
        already exists, the user can omit this parameter and the edge type
        function will be set based on the existing edge definitions. If the
        graph does not exist, the user can pass a function that determines
        the edge type between two nodes.

    edge_collections_attributes : set[str] (optional, default: None)
        Set of edge attributes to fetch when executing a NetworkX algorithm.
        Useful if the user has edge weights or other edge attributes that
        they want to use in a NetworkX algorithm.

    db : arango.database.StandardDatabase (optional, default: None)
        ArangoDB database object. If the user has an existing python-arango
        connection to the database, they can pass the database object to the graph.
        If not provided, a database object will be created using the environment
        variables DATABASE_HOST, DATABASE_USERNAME, DATABASE_PASSWORD, and
        DATABASE_NAME.

    read_parallelism : int (optional, default: 10)
        Number of parallel threads to use when reading data from ArangoDB.
        Used for fetching node and edge data from the database.

    read_batch_size : int (optional, default: 100000)
        Number of documents to fetch in a single batch when reading data from ArangoDB.
        Used for fetching node and edge data from the database.

    write_batch_size : int (optional, default: 50000)
        Number of documents to insert in a single batch when writing data to ArangoDB.
        Used for inserting node and edge data into the database if and only if
        **incoming_graph_data** is a NetworkX graph.

    write_async : bool (optional, default: True)
        Whether to insert data into ArangoDB asynchronously. Used for inserting
        node and edge data into the database if and only if **incoming_graph_data**
        is a NetworkX graph.

    symmetrize_edges : bool (optional, default: False)
        Whether to symmetrize the edges in the graph when fetched from the database.
        Only applies to directed graphs, thereby converting them to undirected graphs.

    use_arango_views : bool (optional, default: False)
        Whether to use experimental work-in-progress ArangoDB Views for the
        nodes, adjacency list, and edges. These views are designed to improve
        data processing performance by delegating CRUD operations to the database
        whenever possible. NOTE: This feature is experimental and may not work
        as expected.

    overwrite_graph : bool (optional, default: False)
        Whether to overwrite the graph in the database if it already exists. If
        set to True, the graph collections will be dropped and recreated. Note that
        this operation is irreversible and will result in the loss of all data in
        the graph. NOTE: If set to True, Collection Indexes will also be lost.

    args: positional arguments for nx.Graph
        Additional arguments passed to nx.Graph.

    kwargs: keyword arguments for nx.Graph
        Additional arguments passed to nx.Graph.
    """

    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiDiGraph]:
        return nx.MultiDiGraph  # type: ignore[no-any-return]

    def __init__(
        self,
        incoming_graph_data: Any = None,
        multigraph_input: bool | None = None,
        name: str | None = None,
        default_node_type: str | None = None,
        edge_type_key: str = "_edge_type",
        edge_type_func: Callable[[str, str], str] | None = None,
        edge_collections_attributes: set[str] | None = None,
        db: StandardDatabase | None = None,
        read_parallelism: int = 10,
        read_batch_size: int = 100000,
        write_batch_size: int = 50000,
        write_async: bool = True,
        symmetrize_edges: bool = False,
        use_arango_views: bool = False,
        overwrite_graph: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            incoming_graph_data,
            multigraph_input,
            name,
            default_node_type,
            edge_type_key,
            edge_type_func,
            edge_collections_attributes,
            db,
            read_parallelism,
            read_batch_size,
            write_batch_size,
            write_async,
            symmetrize_edges,
            use_arango_views,
            overwrite_graph,
            *args,
            **kwargs,
        )

        if self.graph_exists_in_db:
            self.reverse = self.reverse_override
            self.to_undirected = self.to_undirected_override

    #######################
    # Init helper methods #
    #######################

    ##########################
    # nx.MultiGraph Overides #
    ##########################

    def reverse_override(self, copy: bool = True) -> Any:
        if copy is False:
            raise NotImplementedError("In-place reverse is not supported yet.")

        return super().reverse(copy=True)

    def to_undirected_override(self, reciprocal=False, as_view=False):
        if reciprocal is False:
            return super().to_undirected(reciprocal=False, as_view=as_view)

        graph_class = self.to_undirected_class()
        if as_view is True:
            return nx.graphviews.generic_graph_view(self, graph_class)

        # deepcopy when not a view
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())

        ######################
        # NOTE: Monkey patch #
        ######################

        # Old
        # G.add_edges_from(
        #     (u, v, key, deepcopy(data))
        #     for u, nbrs in self._adj.items()
        #     for v, keydict in nbrs.items()
        #     for key, data in keydict.items()
        #     if v in self._pred[u] and key in self._pred[u][v]
        # )

        # New:
        G.add_edges_from(
            (u, v, key, deepcopy(data))
            for u, nbrs in self._adj.items()
            for v, keydict in nbrs.items()
            for key, data in keydict.items()
            if v in self._pred[u]  # and key in self._pred[u][v]
        )

        # Reason: MultiGraphs in `nxadb` don't use integer-based keys for edges.
        # They use ArangoDB Edge IDs. Therefore, the statement `key in self._pred[u][v]`
        # will always be False in the context of MultiDiGraphs. For more details on why
        # this adjustment is needed, see the `test_to_undirected_reciprocal`
        # in `test_multidigraph.py`.

        ###########################

        return G
