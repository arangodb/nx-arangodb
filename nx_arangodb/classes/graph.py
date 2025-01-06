import os
from functools import cached_property
from typing import Any, Callable, ClassVar

import networkx as nx
from adbnx_adapter import ADBNX_Adapter, ADBNX_Controller
from adbnx_adapter.typings import NxData, NxId
from arango import ArangoClient
from arango.cursor import Cursor
from arango.database import StandardDatabase
from networkx.exception import NetworkXError

import nx_arangodb as nxadb
from nx_arangodb.exceptions import (
    DatabaseNotSet,
    EdgeTypeAmbiguity,
    GraphNameNotSet,
    GraphNotEmpty,
    InvalidDefaultNodeType,
)
from nx_arangodb.logger import logger

from .coreviews import ArangoAdjacencyView
from .dict import (
    adjlist_inner_dict_factory,
    adjlist_outer_dict_factory,
    edge_attr_dict_factory,
    graph_dict_factory,
    node_attr_dict_factory,
    node_dict_factory,
)
from .function import get_node_id
from .reportviews import ArangoEdgeView, ArangoNodeView

networkx_api = nxadb.utils.decorators.networkx_class(nx.Graph)  # type: ignore

__all__ = ["Graph"]

try:
    from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
    from langchain_community.graphs import ArangoGraph
    from langchain_core.language_models import BaseLanguageModel
    from langchain_openai import ChatOpenAI

    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

    class BaseLanguageModel:  # type: ignore[no-redef]
        pass


class Graph(nx.Graph):
    """
    Base class for undirected graphs. Designed to work with ArangoDB graphs.

    Subclasses ``nx.Graph``.

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
    >>> G = nxadb.Graph(name="MyGraph")
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
    def to_networkx_class(cls) -> type[nx.Graph]:
        return nx.Graph  # type: ignore[no-any-return]

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
        *args: Any,
        **kwargs: Any,
    ):
        self.__db = None
        self.__use_arango_views = use_arango_views
        self.__graph_exists_in_db = False

        self.__set_db(db)
        if all([self.__db, name]):
            self.__set_graph(name, default_node_type, edge_type_func)
            self.__set_edge_collections_attributes(edge_collections_attributes)

        # NOTE: Need to revisit these...
        # self.maintain_node_dict_cache = False
        # self.maintain_adj_dict_cache = False
        # self.use_nx_cache = True
        self.use_nxcg_cache = True
        self.nxcg_graph = None

        self.edge_type_key = edge_type_key
        self.read_parallelism = read_parallelism
        self.read_batch_size = read_batch_size

        # Does not apply to undirected graphs
        self.symmetrize_edges = symmetrize_edges

        # TODO: Consider this
        # if not self.__graph_name:
        #     if incoming_graph_data is not None:
        #         m = "Must set **graph_name** if passing **incoming_graph_data**"
        #         raise ValueError(m)

        self._loaded_incoming_graph_data = False
        if self.graph_exists_in_db:
            self._set_factory_methods(read_parallelism, read_batch_size)
            self.__set_arangodb_backend_config()

            if overwrite_graph:
                logger.info("Overwriting graph...")

                properties = self.adb_graph.properties()
                self.db.delete_graph(name, drop_collections=True)
                self.db.create_graph(
                    name=name,
                    edge_definitions=properties["edge_definitions"],
                    orphan_collections=properties["orphan_collections"],
                    smart=properties.get("smart"),
                    disjoint=properties.get("disjoint"),
                    smart_field=properties.get("smart_field"),
                    shard_count=properties.get("shard_count"),
                    replication_factor=properties.get("replication_factor"),
                    write_concern=properties.get("write_concern"),
                )

            if isinstance(incoming_graph_data, nx.Graph):
                self._load_nx_graph(incoming_graph_data, write_batch_size, write_async)
                self._loaded_incoming_graph_data = True

        if name is not None:
            kwargs["name"] = name

        super().__init__(*args, **kwargs)

        if self.graph_exists_in_db:
            self.copy = self.copy_override
            self.subgraph = self.subgraph_override
            self.clear = self.clear_override
            self.clear_edges = self.clear_edges_override
            self.add_node = self.add_node_override
            self.add_nodes_from = self.add_nodes_from_override
            self.number_of_edges = self.number_of_edges_override
            self.nbunch_iter = self.nbunch_iter_override

        # If incoming_graph_data wasn't loaded by the NetworkX Adapter,
        # then we can rely on the CRUD operations of the modified dictionaries
        # to load the data into the graph. However, if the graph is directed
        # or multigraph, then we leave that responsibility to the child classes
        # due to the possibility of additional CRUD-based method overrides.
        if (
            not self.is_directed()
            and not self.is_multigraph()
            and incoming_graph_data is not None
            and not self._loaded_incoming_graph_data
        ):
            nx.convert.to_networkx_graph(incoming_graph_data, create_using=self)
            self._loaded_incoming_graph_data = True

    #######################
    # Init helper methods #
    #######################

    def _set_factory_methods(self, read_parallelism: int, read_batch_size: int) -> None:
        """Set the factory methods for the graph, _node, and _adj dictionaries.

        The ArangoDB CRUD operations are handled by the modified dictionaries.

        Handles the creation of the following dictionaries:
        - graph_attr_dict_factory (graph-level attributes)
        - node_dict_factory (nodes in the graph)
        - node_attr_dict_factory (attributes of the nodes in the graph)
        - adjlist_outer_dict_factory (outer dictionary for the adjacency list)
        - adjlist_inner_dict_factory (inner dictionary for the adjacency list)
        - edge_attr_dict_factory (attributes of the edges in the graph)
        """

        base_args = (self.db, self.adb_graph)

        node_args = (*base_args, self.default_node_type)
        node_args_with_read = (*node_args, read_parallelism, read_batch_size)

        adj_args = (self.edge_type_key, self.edge_type_func, self.__class__.__name__)
        adj_inner_args = (*node_args, *adj_args)
        adj_outer_args = (
            *node_args_with_read,
            *adj_args,
            self.symmetrize_edges,
        )

        self.graph_attr_dict_factory = graph_dict_factory(*base_args)

        self.node_dict_factory = node_dict_factory(*node_args_with_read)
        self.node_attr_dict_factory = node_attr_dict_factory(*base_args)

        self.edge_attr_dict_factory = edge_attr_dict_factory(*base_args)
        self.adjlist_inner_dict_factory = adjlist_inner_dict_factory(*adj_inner_args)
        self.adjlist_outer_dict_factory = adjlist_outer_dict_factory(*adj_outer_args)

    def __set_arangodb_backend_config(self) -> None:
        config = nx.config.backends.arangodb
        config.use_gpu = True  # Only used by default if nx-cugraph is available

    def __set_edge_collections_attributes(self, attributes: set[str] | None) -> None:
        if not attributes:
            self._edge_collections_attributes = set()
            return

        self._edge_collections_attributes = attributes

        if "_id" not in attributes:
            self._edge_collections_attributes.add("_id")

    def __set_db(self, db: Any = None) -> None:
        self._hosts = os.getenv("DATABASE_HOST", "").split(",")
        self._username = os.getenv("DATABASE_USERNAME")
        self._password = os.getenv("DATABASE_PASSWORD")
        self._db_name = os.getenv("DATABASE_NAME")

        if db is not None:
            if not isinstance(db, StandardDatabase):
                m = "arango.database.StandardDatabase"
                raise TypeError(m)

            db.version()  # make sure the connection is valid
            self.__db = db
            self._db_name = db.name
            self._hosts = db._conn._hosts
            self._username, self._password = db._conn._auth
            return

        if not all([self._hosts, self._username, self._password, self._db_name]):
            m = "Database environment variables not set. Can't connect to the database"
            logger.warning(m)
            self.__db = None
            return

        self.__db = ArangoClient(hosts=self._hosts, request_timeout=None).db(
            self._db_name, self._username, self._password, verify=True
        )

    def __set_graph(
        self,
        name: Any,
        default_node_type: str | None = None,
        edge_type_func: Callable[[str, str], str] | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("**name** must be a string")

        if self.db.has_graph(name):
            logger.info(f"Graph '{name}' exists.")

            if edge_type_func is not None:
                m = "Cannot pass **edge_type_func** if the graph already exists"
                raise NotImplementedError(m)

            self.adb_graph = self.db.graph(name)
            vertex_collections = self.adb_graph.vertex_collections()
            edge_definitions = self.adb_graph.edge_definitions()

            if default_node_type is None:
                default_node_type = list(vertex_collections)[0]
                logger.info(f"Default node type set to '{default_node_type}'")

            elif default_node_type not in vertex_collections:
                m = f"Default node type '{default_node_type}' not found in graph '{name}'"  # noqa: E501
                raise InvalidDefaultNodeType(m)

            node_types_to_edge_type_map: dict[tuple[str, str], str] = {}
            for e_d in edge_definitions:
                for f in e_d["from_vertex_collections"]:
                    for t in e_d["to_vertex_collections"]:
                        if (f, t) in node_types_to_edge_type_map:
                            # TODO: Should we log a warning at least?
                            continue

                        node_types_to_edge_type_map[(f, t)] = e_d["edge_collection"]

            def edge_type_func(u: str, v: str) -> str:
                try:
                    return node_types_to_edge_type_map[(u, v)]
                except KeyError:
                    m = f"Edge type ambiguity between '{u}' and '{v}'"
                    raise EdgeTypeAmbiguity(m)

        else:
            prefix = f"{name}_" if name else ""

            if default_node_type is None:
                default_node_type = f"{prefix}node"

            if edge_type_func is None:
                edge_type_func = lambda u, v: f"{u}_to_{v}"  # noqa: E731

            # TODO: Parameterize the edge definitions
            # How can we work with a heterogenous **incoming_graph_data**?
            default_edge_type = edge_type_func(default_node_type, default_node_type)
            edge_definitions = [
                {
                    "edge_collection": default_edge_type,
                    "from_vertex_collections": [default_node_type],
                    "to_vertex_collections": [default_node_type],
                }
            ]

            # Create a general graph if it doesn't exist
            self.adb_graph = self.db.create_graph(
                name=name,
                edge_definitions=edge_definitions,
            )

            logger.info(f"Graph '{name}' created.")

        self.__name = name
        self.__graph_exists_in_db = True
        self.edge_type_func = edge_type_func
        self.default_node_type = default_node_type

        properties = self.adb_graph.properties()
        self.__is_smart: bool = properties.get("smart", False)
        self.__smart_field: str | None = properties.get("smart_field")

    def _load_nx_graph(
        self, nx_graph: nx.Graph, write_batch_size: int, write_async: bool
    ) -> None:
        collections = list(self.adb_graph.vertex_collections())
        collections += [e["edge_collection"] for e in self.adb_graph.edge_definitions()]

        for col in collections:
            cursor = self.db.aql.execute(
                "FOR doc IN @@collection LIMIT 1 RETURN 1",
                bind_vars={"@collection": col},
            )

            if not cursor.empty():
                m = f"Graph '{self.adb_graph.name}' already has data (in '{col}'). Use **overwrite_graph=True** to clear it."  # noqa: E501
                raise GraphNotEmpty(m)

        controller = ADBNX_Controller

        if all([self.is_smart, self.smart_field]):
            smart_field = self.__smart_field

            class SmartController(ADBNX_Controller):
                def _keyify_networkx_node(
                    self, i: int, nx_node_id: NxId, nx_node: NxData, col: str
                ) -> str:
                    if smart_field not in nx_node:
                        m = f"Node {nx_node_id} missing smart field '{smart_field}'"  # noqa: E501
                        raise KeyError(m)

                    return f"{nx_node[smart_field]}:{str(i)}"

                def _prepare_networkx_edge(self, nx_edge: NxData, col: str) -> None:
                    del nx_edge["_key"]

            controller = SmartController
            logger.info(f"Using smart field '{smart_field}' for node keys")

        ADBNX_Adapter(self.db, controller()).networkx_to_arangodb(
            self.adb_graph.name,
            nx_graph,
            batch_size=write_batch_size,
            use_async=write_async,
        )

    ###########
    # Getters #
    ###########

    @property
    def db(self) -> StandardDatabase:
        if self.__db is None:
            raise DatabaseNotSet("Database not set")

        return self.__db

    @property
    def name(self) -> str:
        if self.__name is None:
            raise GraphNameNotSet("Graph name not set")

        return self.__name

    @name.setter
    def name(self, s):
        if self.graph_exists_in_db:
            raise ValueError("Existing graph cannot be renamed")

        m = "Note that setting the graph name does not create the graph in the database"  # noqa: E501
        logger.warning(m)

        self.__name = s
        self.graph["name"] = s
        nx._clear_cache(self)

    @property
    def graph_exists_in_db(self) -> bool:
        return self.__graph_exists_in_db

    @property
    def edge_attributes(self) -> set[str]:
        return self._edge_collections_attributes

    @property
    def is_smart(self) -> bool:
        return self.__is_smart

    @property
    def smart_field(self) -> str | None:
        return self.__smart_field

    ###########
    # Setters #
    ###########

    ####################
    # ArangoDB Methods #
    ####################

    def clear_nxcg_cache(self):
        self.nxcg_graph = None

    def query(
        self, query: str, bind_vars: dict[str, Any] = {}, **kwargs: Any
    ) -> Cursor:
        """Execute an AQL query on the graph.

        Read more about AQL here:
        https://www.arangodb.com/docs/stable/aql/

        Parameters
        ----------
        query : str
            AQL query to execute.

        bind_vars : dict[str, Any] (optional, default: {})
            Bind variables to pass to the query.

        kwargs : dict[str, Any]
            Additional keyword arguments to pass to the query.

        Returns
        -------
        arango.cursor.Cursor
            Cursor object containing the results of the query.
        """
        return nxadb.classes.function.aql(self.db, query, bind_vars, **kwargs)

    # def pull(self) -> None:
    #     TODO: what would this look like?

    # def push(self) -> None:
    #     TODO: what would this look like?

    def chat(
        self, prompt: str, verbose: bool = False, llm: BaseLanguageModel | None = None
    ) -> str:
        """Chat with the graph using an LLM. Use at your own risk.

        Parameters
        ----------
        prompt : str
            Prompt to chat with the graph.

        verbose : bool (optional, default: False)
            Whether to print the intermediate steps of the conversation.

        llm : langchain_core.language_models.BaseLanguageModel (optional, default: None)
            Language model to use for the conversation. If None, the default
            language model is ChatOpenAI with the GPT-4 model, which expects the
            OpenAI API key to be set in the environment variable OPENAI_API_KEY.

        Returns
        -------
        str
            Response from the Language Model.
        """
        if not LLM_AVAILABLE:
            m = "LLM dependencies not installed. Install with **pip install nx-arangodb[llm]**"  # noqa: E501
            raise ModuleNotFoundError(m)

        if not self.__graph_exists_in_db:
            m = "Cannot chat without a graph in the database"
            raise GraphNameNotSet(m)

        if llm is None:
            llm = ChatOpenAI(temperature=0, model_name="gpt-4")

        chain = ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=ArangoGraph(self.db),
            verbose=verbose,
        )

        response = chain.invoke(prompt)

        return str(response["result"])

    #####################
    # nx.Graph Overides #
    #####################

    @cached_property
    def nodes(self):
        if self.__use_arango_views and self.graph_exists_in_db:
            logger.warning("nxadb.ArangoNodeView is currently EXPERIMENTAL")
            return ArangoNodeView(self)

        return super().nodes

    @cached_property
    def adj(self):
        if self.__use_arango_views and self.graph_exists_in_db:
            logger.warning("nxadb.ArangoAdjacencyView is currently EXPERIMENTAL")
            return ArangoAdjacencyView(self._adj)

        return super().adj

    @cached_property
    def edges(self):
        if self.__use_arango_views and self.graph_exists_in_db:
            if self.is_directed():
                logger.warning("ArangoEdgeView for DiGraphs not yet implemented")
                return super().edges

            if self.is_multigraph():
                logger.warning("ArangoEdgeView for MultiGraphs not yet implemented")
                return super().edges

            logger.warning("nxadb.ArangoEdgeView is currently EXPERIMENTAL")
            return ArangoEdgeView(self)

        return super().edges

    def copy_override(self, *args, **kwargs):
        logger.warning("Note that copying a graph loses the connection to the database")
        G = super().copy(*args, **kwargs)
        G.node_dict_factory = nx.Graph.node_dict_factory
        G.node_attr_dict_factory = nx.Graph.node_attr_dict_factory
        G.edge_attr_dict_factory = nx.Graph.edge_attr_dict_factory
        G.adjlist_inner_dict_factory = nx.Graph.adjlist_inner_dict_factory
        G.adjlist_outer_dict_factory = nx.Graph.adjlist_outer_dict_factory
        return G

    def subgraph_override(self, nbunch):
        if self.graph_exists_in_db:
            m = "Subgraphing an ArangoDB Graph is not yet implemented"
            raise NotImplementedError(m)

        return super().subgraph(nbunch)

    def clear_override(self):
        logger.info("Note that clearing only erases the local cache")
        super().clear()

    def clear_edges_override(self):
        logger.info("Note that clearing edges ony erases the edges in the local cache")
        for nbr_dict in self._adj.data.values():
            nbr_dict.clear()
        nx._clear_cache(self)

    def add_node_override(self, node_for_adding, **attr):
        if node_for_adding is None:
            raise ValueError("None cannot be a node")

        if node_for_adding not in self._node:
            self._adj[node_for_adding] = self.adjlist_inner_dict_factory()

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
                self._adj[n] = self.adjlist_inner_dict_factory()

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

    def number_of_edges_override(self, u=None, v=None):
        if u is not None:
            return super().number_of_edges(u, v)

        ######################
        # NOTE: monkey patch #
        ######################

        # Old:
        # return int(self.size())

        # New:
        edge_collections = {
            e_d["edge_collection"] for e_d in self.adb_graph.edge_definitions()
        }
        num = sum(self.adb_graph.edge_collection(e).count() for e in edge_collections)
        num *= 2 if self.is_directed() and self.symmetrize_edges else 1

        return num

        # Reason:
        # It is more efficient to count the number of edges in the edge collections
        # compared to relying on the DegreeView.

    def nbunch_iter_override(self, nbunch=None):
        if nbunch is None:
            bunch = iter(self._adj)
        elif nbunch in self:
            ######################
            # NOTE: monkey patch #
            ######################

            # Old: Nothing

            # New:
            if isinstance(nbunch, (str, int)):
                nbunch = get_node_id(str(nbunch), self.default_node_type)

            # Reason:
            # ArangoDB only uses strings as node IDs. Therefore, we need to convert
            # the non-prefixed node ID to an ArangoDB ID before
            # using it in an iterator.

            bunch = iter([nbunch])
        else:

            def bunch_iter(nlist, adj):
                try:
                    for n in nlist:
                        ######################
                        # NOTE: monkey patch #
                        ######################

                        # Old: Nothing

                        # New:
                        if isinstance(n, (str, int)):
                            n = get_node_id(str(n), self.default_node_type)

                        # Reason:
                        # ArangoDB only uses strings as node IDs. Therefore,
                        # we need to convert non-prefixed node IDs to an
                        # ArangoDB ID before using it in an iterator.

                        ######################

                        if n in adj:
                            yield n

                except TypeError as err:
                    exc, message = err, err.args[0]
                    if "iter" in message:
                        m = "nbunch is not a node or a sequence of nodes."
                        exc = NetworkXError(m)
                    if "hashable" in message:
                        m = f"Node {n} in sequence nbunch is not a valid node."
                        exc = NetworkXError(m)
                    raise exc

            bunch = bunch_iter(nbunch, self._adj)
        return bunch
