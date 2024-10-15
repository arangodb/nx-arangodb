from typing import Any, Callable, ClassVar

import networkx as nx
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.classes.graph import Graph
from nx_arangodb.logger import logger

from .dict.adj import AdjListOuterDict
from .enum import TraversalDirection
from .function import get_node_id, mirror_to_nxcg

networkx_api = nxadb.utils.decorators.networkx_class(nx.DiGraph)  # type: ignore

__all__ = ["DiGraph"]


class DiGraph(Graph, nx.DiGraph):
    """
    Base class for directed graphs.

    Subclasses ``nxadb.Graph`` and ``nx.DiGraph``.

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
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph  # type: ignore[no-any-return]

    def __init__(
        self,
        incoming_graph_data: Any = None,
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
        mirror_crud_to_nxcg: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            incoming_graph_data,
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
            mirror_crud_to_nxcg,
            *args,
            **kwargs,
        )

        if self.graph_exists_in_db:
            self.clear_edges = self.clear_edges_override
            self.reverse = self.reverse_override

            self.add_node = self.add_node_override
            self.add_nodes_from = self.add_nodes_from_override
            self.remove_node = self.remove_node_override
            self.remove_nodes_from = self.remove_nodes_from_override
            self.add_edge = self.add_edge_override
            self.add_edges_from = self.add_edges_from_override
            self.remove_edge = self.remove_edge_override
            self.remove_edges_from = self.remove_edges_from_override

            assert isinstance(self._succ, AdjListOuterDict)
            assert isinstance(self._pred, AdjListOuterDict)
            self._succ.mirror = self._pred
            self._pred.mirror = self._succ
            self._succ.traversal_direction = TraversalDirection.OUTBOUND
            self._pred.traversal_direction = TraversalDirection.INBOUND

        if (
            not self.is_multigraph()
            and incoming_graph_data is not None
            and not self._loaded_incoming_graph_data
        ):
            nx.convert.to_networkx_graph(incoming_graph_data, create_using=self)
            self._loaded_incoming_graph_data = True

    #######################
    # nx.DiGraph Overides #
    #######################

    # TODO?
    # If we want to continue with "Experimental Views" we need to implement the
    # InEdgeView and OutEdgeView classes.
    # @cached_property
    # def in_edges(self):
    # pass

    # TODO?
    # @cached_property
    # def out_edges(self):
    # pass

    def reverse_override(self, copy: bool = True) -> Any:
        if copy is False:
            raise NotImplementedError("In-place reverse is not supported yet.")

        return super().reverse(copy=True)

    def clear_edges_override(self):
        logger.info("Note that clearing edges ony erases the edges in the local cache")
        for predecessor_dict in self._pred.data.values():
            predecessor_dict.clear()

        super().clear_edges()

    @mirror_to_nxcg
    def add_node_override(self, node_for_adding, **attr):
        if node_for_adding is None:
            raise ValueError("None cannot be a node")

        if node_for_adding not in self._succ:

            self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
            self._pred[node_for_adding] = self.adjlist_inner_dict_factory()

            ######################
            # NOTE: monkey patch #
            ######################

            # Old:
            # attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            # attr_dict.update(attr)

            # New:
            node_attr_dict = self.node_attr_dict_factory()
            node_attr_dict.data = attr
            self._node[node_for_adding] = node_attr_dict

            # Reason:
            # We can optimize the process of adding a node by creating avoiding
            # the creation of a new dictionary and updating it with the attributes.
            # Instead, we can create a new node_attr_dict object and set the attributes
            # directly. This only makes 1 network call to the database instead of 2.

            ###########################

        else:
            self._node[node_for_adding].update(attr)

        nx._clear_cache(self)

    @mirror_to_nxcg
    def add_nodes_from_override(self, nodes_for_adding, **attr):
        for n in nodes_for_adding:
            try:
                newnode = n not in self._node
                newdict = attr
            except TypeError:
                n, ndict = n
                newnode = n not in self._node
                newdict = attr.copy()
                newdict.update(ndict)
            if newnode:
                if n is None:
                    raise ValueError("None cannot be a node")
                self._succ[n] = self.adjlist_inner_dict_factory()
                self._pred[n] = self.adjlist_inner_dict_factory()

                ######################
                # NOTE: monkey patch #
                ######################

                # Old:
                #   self._node[n] = self.node_attr_dict_factory()
                #
                # self._node[n].update(newdict)

                # New:
                node_attr_dict = self.node_attr_dict_factory()
                node_attr_dict.data = newdict
                self._node[n] = node_attr_dict

            else:
                self._node[n].update(newdict)

                # Reason:
                # We can optimize the process of adding a node by creating avoiding
                # the creation of a new dictionary and updating it with the attributes.
                # Instead, we create a new node_attr_dict object and set the attributes
                # directly. This only makes 1 network call to the database instead of 2.

                ###########################

        nx._clear_cache(self)

    @mirror_to_nxcg
    def remove_node_override(self, n):
        if isinstance(n, (str, int)):
            n = get_node_id(str(n), self.default_node_type)

        try:

            ######################
            # NOTE: monkey patch #
            ######################

            # Old:
            # nbrs = self._succ[n]

            # New:
            nbrs_succ = list(self._succ[n])
            nbrs_pred = list(self._pred[n])

            # Reason:
            # We need to fetch the outbound/inbound edges _prior_ to deleting the node,
            # as node deletion will already take care of deleting edges

            ###########################

            del self._node[n]
        except KeyError as err:  # NetworkXError if n not in self
            raise nx.NetworkXError(f"The node {n} is not in the digraph.") from err
        for u in nbrs_succ:
            del self._pred[u][n]  # remove all edges n-u in digraph
        del self._succ[n]  # remove node from succ
        for u in nbrs_pred:
            ######################
            # NOTE: Monkey patch #
            ######################

            # Old: Nothing

            # New:
            if u == n:
                continue  # skip self loops

            # Reason: We need to skip self loops, as they are
            # already taken care of in the previous step. This
            # avoids getting a KeyError on the next line.

            ###########################

            del self._succ[u][n]  # remove all edges n-u in digraph
        del self._pred[n]  # remove node from pred
        nx._clear_cache(self)

    @mirror_to_nxcg
    def remove_nodes_from_override(self, nodes):
        super().remove_nodes_from(nodes)

    @mirror_to_nxcg
    def add_edge_override(self, u, v, **attr):
        super().add_edge(u, v, **attr)

    @mirror_to_nxcg
    def add_edges_from_override(self, ebunch_to_add, **attr):
        super().add_edges_from(ebunch_to_add, **attr)

    @mirror_to_nxcg
    def remove_edge_override(self, u, v):
        super().remove_edge(u, v)

    @mirror_to_nxcg
    def remove_edges_from_override(self, ebunch):
        super().remove_edges_from(ebunch)
