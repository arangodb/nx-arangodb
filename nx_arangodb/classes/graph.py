import os
from functools import cached_property
from typing import Any, Callable, ClassVar

import networkx as nx
from adbnx_adapter import ADBNX_Adapter
from arango import ArangoClient
from arango.cursor import Cursor
from arango.database import StandardDatabase
from networkx.exception import NetworkXError

import nx_arangodb as nxadb
from nx_arangodb.exceptions import (
    DatabaseNotSet,
    EdgeTypeAmbiguity,
    GraphNameNotSet,
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
        *args: Any,
        **kwargs: Any,
    ):
        self.__db = None
        self.__name = None
        self.__use_arango_views = use_arango_views
        self.__graph_exists_in_db = False

        self.__set_db(db)
        if self.__db is not None:
            self.__set_graph_name(name)

        self.__set_edge_collections_attributes(edge_collections_attributes)

        # NOTE: Need to revisit these...
        # self.maintain_node_dict_cache = False
        # self.maintain_adj_dict_cache = False
        # self.use_nx_cache = True
        self.use_nxcg_cache = True
        self.nxcg_graph = None

        # Does not apply to undirected graphs
        self.symmetrize_edges = symmetrize_edges

        self.edge_type_key = edge_type_key

        # TODO: Consider this
        # if not self.__graph_name:
        #     if incoming_graph_data is not None:
        #         m = "Must set **graph_name** if passing **incoming_graph_data**"
        #         raise ValueError(m)

        self._loaded_incoming_graph_data = False

        if self.__graph_exists_in_db:
            if incoming_graph_data is not None:
                m = "Cannot pass both **incoming_graph_data** and **name** yet if the already graph exists"  # noqa: E501
                raise NotImplementedError(m)

            if edge_type_func is not None:
                m = "Cannot pass **edge_type_func** if the graph already exists"
                raise NotImplementedError(m)

            self.adb_graph = self.db.graph(self.__name)
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

            self.edge_type_func = edge_type_func
            self.default_node_type = default_node_type

            self._set_factory_methods()
            self.__set_arangodb_backend_config(read_parallelism, read_batch_size)

        elif self.__name:

            prefix = f"{name}_" if name else ""
            if default_node_type is None:
                default_node_type = f"{prefix}node"
            if edge_type_func is None:
                edge_type_func = lambda u, v: f"{u}_to_{v}"  # noqa: E731

            self.edge_type_func = edge_type_func
            self.default_node_type = default_node_type

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

            if isinstance(incoming_graph_data, nx.Graph):
                self.adb_graph = ADBNX_Adapter(self.db).networkx_to_arangodb(
                    self.__name,
                    incoming_graph_data,
                    edge_definitions=edge_definitions,
                    batch_size=write_batch_size,
                    use_async=write_async,
                )

                self._loaded_incoming_graph_data = True

            else:
                self.adb_graph = self.db.create_graph(
                    self.__name,
                    edge_definitions=edge_definitions,
                )

            self._set_factory_methods()
            self.__set_arangodb_backend_config(read_parallelism, read_batch_size)
            logger.info(f"Graph '{name}' created.")
            self.__graph_exists_in_db = True

        if self.__name is not None:
            kwargs["name"] = self.__name

        super().__init__(*args, **kwargs)

        if self.graph_exists_in_db:
            self.copy = self.copy_override
            self.subgraph = self.subgraph_override
            self.clear = self.clear_override
            self.clear_edges = self.clear_edges_override
            self.add_node = self.add_node_override
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

    #######################
    # Init helper methods #
    #######################

    def _set_factory_methods(self) -> None:
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
        adj_args = (
            *node_args,
            self.edge_type_key,
            self.edge_type_func,
            self.__class__.__name__,
        )

        self.graph_attr_dict_factory = graph_dict_factory(*base_args)

        self.node_dict_factory = node_dict_factory(*node_args)
        self.node_attr_dict_factory = node_attr_dict_factory(*base_args)

        self.edge_attr_dict_factory = edge_attr_dict_factory(*base_args)
        self.adjlist_inner_dict_factory = adjlist_inner_dict_factory(*adj_args)
        self.adjlist_outer_dict_factory = adjlist_outer_dict_factory(
            *adj_args, self.symmetrize_edges
        )

    def __set_arangodb_backend_config(
        self, read_parallelism: int, read_batch_size: int
    ) -> None:
        if not all([self._host, self._username, self._password, self._db_name]):
            m = "Must set all environment variables to use the ArangoDB Backend with an existing graph"  # noqa: E501
            raise OSError(m)

        config = nx.config.backends.arangodb
        config.host = self._host
        config.username = self._username
        config.password = self._password
        config.db_name = self._db_name
        config.read_parallelism = read_parallelism
        config.read_batch_size = read_batch_size
        config.use_gpu = True  # Only used by default if nx-cugraph is available

    def __set_edge_collections_attributes(self, attributes: set[str] | None) -> None:
        if not attributes:
            self._edge_collections_attributes = set()
            return

        self._edge_collections_attributes = attributes

        if "_id" not in attributes:
            self._edge_collections_attributes.add("_id")

    def __set_db(self, db: Any = None) -> None:
        self._host = os.getenv("DATABASE_HOST")
        self._username = os.getenv("DATABASE_USERNAME")
        self._password = os.getenv("DATABASE_PASSWORD")
        self._db_name = os.getenv("DATABASE_NAME")

        if db is not None:
            if not isinstance(db, StandardDatabase):
                m = "arango.database.StandardDatabase"
                raise TypeError(m)

            db.version()
            self.__db = db
            return

        if not all([self._host, self._username, self._password, self._db_name]):
            m = "Database environment variables not set. Can't connect to the database"
            logger.warning(m)
            self.__db = None
            return

        self.__db = ArangoClient(hosts=self._host, request_timeout=None).db(
            self._db_name, self._username, self._password, verify=True
        )

    def __set_graph_name(self, name: Any = None) -> None:
        if self.__db is None:
            m = "Cannot set graph name without setting the database first"
            raise DatabaseNotSet(m)

        if not name:
            self.__graph_exists_in_db = False
            logger.warning(f"**name** not set for {self.__class__.__name__}")
            return

        if not isinstance(name, str):
            raise TypeError("**name** must be a string")

        self.__name = name
        self.__graph_exists_in_db = self.db.has_graph(name)

        logger.info(f"Graph '{name}' exists: {self.__graph_exists_in_db}")

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
        return nxadb.classes.function.aql(self.db, query, bind_vars, **kwargs)

    # def pull(self) -> None:
    #     TODO: what would this look like?

    # def push(self) -> None:
    #     TODO: what would this look like?

    def chat(
        self, prompt: str, verbose: bool = False, llm: BaseLanguageModel | None = None
    ) -> str:
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
        if node_for_adding not in self._node:
            if node_for_adding is None:
                raise ValueError("None cannot be a node")

            self._adj[node_for_adding] = self.adjlist_inner_dict_factory()

            ######################
            # NOTE: monkey patch #
            ######################

            # Old:
            # attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            # attr_dict.update(attr)

            # New:
            self._node[node_for_adding] = self.node_attr_dict_factory()
            self._node[node_for_adding].update(attr)

            # Reason:
            # Invoking `update` on the `attr_dict` without `attr_dict.node_id` being set
            # i.e trying to update a node's attributes before we know _which_ node it is

            ###########################

        else:
            self._node[node_for_adding].update(attr)

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
