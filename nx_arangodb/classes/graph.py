import os
from functools import cached_property
from typing import Callable, ClassVar

import networkx as nx
from adbnx_adapter import ADBNX_Adapter
from arango import ArangoClient
from arango.cursor import Cursor
from arango.database import StandardDatabase
from arango.exceptions import ServerConnectionError

import nx_arangodb as nxadb
from nx_arangodb.exceptions import *
from nx_arangodb.logger import logger

from .dict import (
    adjlist_inner_dict_factory,
    adjlist_outer_dict_factory,
    edge_attr_dict_factory,
    graph_dict_factory,
    node_attr_dict_factory,
    node_dict_factory,
)
from .reportviews import CustomEdgeView, CustomNodeView

networkx_api = nxadb.utils.decorators.networkx_class(nx.Graph)

__all__ = ["Graph"]


class Graph(nx.Graph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.Graph]:
        return nx.Graph

    def __init__(
        self,
        graph_name: str | None = None,
        default_node_type: str = "nxadb_node",
        edge_type_func: Callable[[str, str], str] = lambda u, v: f"{u}_to_{v}",
        *args,
        **kwargs,
    ):
        self.__db = None
        self.__graph_name = None
        self.__graph_exists = False

        self.__set_db()
        if self.__db is not None:
            self.__set_graph_name(graph_name)

        self.auto_sync = True

        self.graph_loader_parallelism = 10
        self.graph_loader_batch_size = 100000

        # NOTE: Need to revisit these...
        # self.maintain_node_dict_cache = False
        # self.maintain_adj_dict_cache = False
        self.use_nx_cache = True
        self.use_coo_cache = True

        self.src_indices = None
        self.dst_indices = None
        self.vertex_ids_to_index = None

        self.default_node_type = default_node_type
        self.edge_type_func = edge_type_func
        self.default_edge_type = edge_type_func(default_node_type, default_node_type)

        incoming_graph_data = kwargs.pop("incoming_graph_data", None)
        if self.__graph_exists:
            self.adb_graph = self.db.graph(graph_name)
            self.__create_default_collections()
            self.__set_factory_methods()

            if incoming_graph_data:
                m = "Cannot pass both **incoming_graph_data** and **graph_name** yet"
                raise NotImplementedError(m)

        elif self.__graph_name:
            if isinstance(incoming_graph_data, nx.Graph):
                adapter = ADBNX_Adapter(self.db)
                self.adb_graph = adapter.networkx_to_arangodb(
                    graph_name,
                    incoming_graph_data,
                    # TODO: Parameterize the edge definitions?
                    # How can we work with a heterogenous **incoming_graph_data**?
                    edge_definitions=[
                        {
                            "edge_collection": self.default_edge_type,
                            "from_vertex_collections": [self.default_node_type],
                            "to_vertex_collections": [self.default_node_type],
                        }
                    ],
                )

                self.__set_factory_methods()
                self.__graph_exists = True
            else:
                m = f"Type of **incoming_graph_data** not supported yet ({type(incoming_graph_data)})"  # noqa: E501
                raise NotImplementedError(m)

        # self.__qa_chain = None

        super().__init__(*args, **kwargs)

    #######################
    # Init helper methods #
    #######################

    def __set_factory_methods(self) -> None:
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
        self.graph_attr_dict_factory = graph_dict_factory(self.db, self.graph_name)

        self.node_dict_factory = node_dict_factory(
            self.db, self.adb_graph, self.default_node_type
        )

        self.node_attr_dict_factory = node_attr_dict_factory(self.db, self.adb_graph)

        self.adjlist_outer_dict_factory = adjlist_outer_dict_factory(
            self.db, self.adb_graph, self.default_node_type, self.edge_type_func
        )
        self.adjlist_inner_dict_factory = adjlist_inner_dict_factory(
            self.db, self.adb_graph, self.default_node_type, self.edge_type_func
        )
        self.edge_attr_dict_factory = edge_attr_dict_factory(self.db, self.adb_graph)

    def __create_default_collections(self) -> None:
        if self.default_node_type not in self.adb_graph.vertex_collections():
            self.adb_graph.create_vertex_collection(self.default_node_type)

        if not self.adb_graph.has_edge_definition(self.default_edge_type):
            self.adb_graph.create_edge_definition(
                edge_collection=self.default_edge_type,
                from_vertex_collections=[self.default_node_type],
                to_vertex_collections=[self.default_node_type],
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
    def graph_name(self) -> str:
        if self.__graph_name is None:
            raise GraphNameNotSet("Graph name not set")

        return self.__graph_name

    @property
    def graph_exists(self) -> bool:
        return self.__graph_exists

    ###########
    # Setters #
    ###########

    def __set_db(self, db: StandardDatabase | None = None):
        if db is not None:
            if not isinstance(db, StandardDatabase):
                m = "arango.database.StandardDatabase"
                raise TypeError(m)

            self.__db = db
            return

        self._host = os.getenv("DATABASE_HOST")
        self._username = os.getenv("DATABASE_USERNAME")
        self._password = os.getenv("DATABASE_PASSWORD")
        self._db_name = os.getenv("DATABASE_NAME")

        # TODO: Raise a custom exception if any of the environment
        # variables are missing. For now, we'll just set db to None.
        if not all([self._host, self._username, self._password, self._db_name]):
            self.__db = None
            logger.warning("Database environment variables not set")
            return

        try:
            self.__db = ArangoClient(hosts=self._host, request_timeout=None).db(
                self._db_name, self._username, self._password, verify=True
            )
        except ServerConnectionError as e:
            self.__db = None
            logger.warning(f"Could not connect to the database: {e}")

    def __set_graph_name(self, graph_name: str | None = None):
        if self.__db is None:
            raise DatabaseNotSet(
                "Cannot set graph name without setting the database first"
            )

        if graph_name is None:
            self.__graph_exists = False
            logger.warning(f"**graph_name** not set for {self.__class__.__name__}")
            return

        if not isinstance(graph_name, str):
            raise TypeError("**graph_name** must be a string")

        self.__graph_name = graph_name
        self.__graph_exists = self.db.has_graph(graph_name)

        logger.info(f"Graph '{graph_name}' exists: {self.__graph_exists}")

    ####################
    # ArangoDB Methods #
    ####################

    # TODO: proper subgraphing!
    def aql(self, query: str, bind_vars: dict | None = None, **kwargs) -> Cursor:
        return nxadb.classes.function.aql(self.db, query, bind_vars, **kwargs)

    # NOTE: Ignore this for now
    # def chat(self, prompt: str) -> str:
    #     if self.__qa_chain is None:
    #         if not self.__graph_exists:
    #             return "Could not initialize QA chain: Graph does not exist"

    #         # try:
    #         from langchain.chains import ArangoGraphQAChain
    #         from langchain_community.graphs import ArangoGraph
    #         from langchain_openai import ChatOpenAI

    #         model = ChatOpenAI(temperature=0, model_name="gpt-4")

    #         self.__qa_chain = ArangoGraphQAChain.from_llm(
    #             llm=model, graph=ArangoGraph(self.db), verbose=True
    #         )

    #         # except Exception as e:
    #         #     return f"Could not initialize QA chain: {e}"

    #     self.__qa_chain.graph.set_schema()
    #     result = self.__qa_chain.invoke(prompt)

    #     print(result["result"])

    def pull(self, load_node_dict=True, load_adj_dict=True, load_coo=True):
        """Load the graph from the ArangoDB database, and update existing graph object.

        :param load_node_dict: Load the node dictionary.
            Enabling this option will clear the existing node dictionary,
            and replace it with the node data from the database. Comes with
            a remote reference to the database. <--- TODO: Should we paramaterize this?
        :type load_node_dict: bool
        :param load_adj_dict: Load the adjacency dictionary.
            Enabling this option will clear the existing adjacency dictionary,
            and replace it with the edge data from the database. Comes with
            a remote reference to the database. <--- TODO: Should we paramaterize this?
        :type load_adj_dict: bool
        :param load_coo: Load the COO representation. If False, the src & dst indices will be empty,
            along with the node-ID-to-index mapping. Used for nx-cuGraph compatibility.
        :type load_coo: bool
        """
        node_dict, adj_dict, src_indices, dst_indices, vertex_ids_to_indices = (
            nxadb.classes.function.get_arangodb_graph(
                self,
                load_node_dict=load_node_dict,
                load_adj_dict=load_adj_dict,
                load_adj_dict_as_directed=False,
                load_coo=load_coo,
            )
        )

        if load_node_dict:
            self._node.clear()

            for node_id, node_data in node_dict.items():
                node_attr_dict = self.node_attr_dict_factory()
                node_attr_dict.node_id = node_id
                node_attr_dict.data = node_data
                self._node.data[node_id] = node_attr_dict

        if load_adj_dict:
            self._adj.clear()

            for src_node_id, dst_dict in adj_dict.items():
                adjlist_inner_dict = self.adjlist_inner_dict_factory()
                adjlist_inner_dict.src_node_id = src_node_id

                self._adj.data[src_node_id] = adjlist_inner_dict

                for dst_id, edge_data in dst_dict.items():
                    edge_attr_dict = self.edge_attr_dict_factory()
                    edge_attr_dict.edge_id = edge_data["_id"]
                    edge_attr_dict.data = edge_data

                    adjlist_inner_dict.data[dst_id] = edge_attr_dict

        if load_coo:
            self.src_indices = src_indices
            self.dst_indices = dst_indices
            self.vertex_ids_to_index = vertex_ids_to_indices

    def push(self):
        raise NotImplementedError("What would this look like?")

    #####################
    # nx.Graph Overides #
    #####################

    @cached_property
    def nodes(self):
        if self.graph_exists:
            logger.warning("nxadb.CustomNodeView is currently EXPERIMENTAL")
            return CustomNodeView(self)

        return nx.classes.reportviews.NodeView(self)

    @cached_property
    def edges(self):
        if self.graph_exists:
            logger.warning("nxadb.CustomEdgeView is currently EXPERIMENTAL")
            return CustomEdgeView(self)

        return nx.classes.reportviews.EdgeView(self)

    def add_node(self, node_for_adding, **attr):
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
