import os
from functools import cached_property
from typing import Any, Callable, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
from adbnx_adapter import ADBNX_Adapter
from arango import ArangoClient
from arango.cursor import Cursor
from arango.database import StandardDatabase
from arango.exceptions import ServerConnectionError
from networkx.utils import Config

import nx_arangodb as nxadb
from nx_arangodb.exceptions import DatabaseNotSet, GraphDoesNotExist, GraphNameNotSet
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

networkx_api = nxadb.utils.decorators.networkx_class(nx.Graph)  # type: ignore

__all__ = ["Graph"]


class Graph(nx.Graph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.Graph]:
        return nx.Graph  # type: ignore[no-any-return]

    def __init__(
        self,
        graph_name: str | None = None,
        default_node_type: str = "node",
        edge_type_func: Callable[[str, str], str] = lambda u, v: f"{u}_to_{v}",
        db: StandardDatabase | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.__db = None
        self.__graph_name = None
        self.__graph_exists_in_db = False

        self.__set_db(db)
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

        self.src_indices: npt.NDArray[np.int64] | None = None
        self.dst_indices: npt.NDArray[np.int64] | None = None
        self.vertex_ids_to_index: dict[str, int] | None = None

        self.default_node_type = default_node_type
        self.edge_type_func = edge_type_func
        self.default_edge_type = edge_type_func(default_node_type, default_node_type)

        # self.__qa_chain = None

        incoming_graph_data = kwargs.get("incoming_graph_data")
        if self.__graph_exists_in_db:
            if incoming_graph_data is not None:
                m = "Cannot pass both **incoming_graph_data** and **graph_name** yet if the already graph exists"  # noqa: E501
                raise NotImplementedError(m)

            self.adb_graph = self.db.graph(self.__graph_name)
            self.__create_default_collections()
            self.__set_factory_methods()
            self.__set_arangodb_backend_config()

        elif self.__graph_name and incoming_graph_data is not None:
            # TODO: Parameterize the edge definitions
            # How can we work with a heterogenous **incoming_graph_data**?
            edge_definitions = [
                {
                    "edge_collection": self.default_edge_type,
                    "from_vertex_collections": [self.default_node_type],
                    "to_vertex_collections": [self.default_node_type],
                }
            ]

            if isinstance(incoming_graph_data, nx.Graph):
                self.adb_graph = ADBNX_Adapter(self.db).networkx_to_arangodb(
                    self.__graph_name,
                    incoming_graph_data,
                    edge_definitions=edge_definitions,
                )

                # No longer need this (we've already populated the graph)
                del kwargs["incoming_graph_data"]

            else:
                self.adb_graph = self.db.create_graph(
                    self.__graph_name,
                    edge_definitions=edge_definitions,
                )

            self.__set_factory_methods()
            self.__set_arangodb_backend_config()
            self.__graph_exists_in_db = True

        super().__init__(*args, **kwargs)

    #######################
    # Init helper methods #
    #######################

    def __set_arangodb_backend_config(self) -> None:
        if not all([self._host, self._username, self._password, self._db_name]):
            m = "Must set all environment variables to use the ArangoDB Backend with an existing graph"  # noqa: E501
            raise OSError(m)

        config = nx.config.backends.arangodb
        config.host = self._host
        config.username = self._username
        config.password = self._password
        config.db_name = self._db_name
        config.load_parallelism = self.graph_loader_parallelism
        config.load_batch_size = self.graph_loader_batch_size

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
    def graph_exists_in_db(self) -> bool:
        return self.__graph_exists_in_db

    ###########
    # Setters #
    ###########

    def __set_db(self, db: StandardDatabase | None = None) -> None:
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

        # TODO: Raise a custom exception if any of the environment
        # variables are missing. For now, we'll just set db to None.
        if not all([self._host, self._username, self._password, self._db_name]):
            self.__db = None
            logger.warning("Database environment variables not set")
            return

        self.__db = ArangoClient(hosts=self._host, request_timeout=None).db(
            self._db_name, self._username, self._password, verify=True
        )

    def __set_graph_name(self, graph_name: str | None = None) -> None:
        if self.__db is None:
            m = "Cannot set graph name without setting the database first"
            raise DatabaseNotSet(m)

        if graph_name is None:
            self.__graph_exists_in_db = False
            logger.warning(f"**graph_name** not set for {self.__class__.__name__}")
            return

        if not isinstance(graph_name, str):
            raise TypeError("**graph_name** must be a string")

        self.__graph_name = graph_name
        self.__graph_exists_in_db = self.db.has_graph(graph_name)

        logger.info(f"Graph '{graph_name}' exists: {self.__graph_exists_in_db}")

    ####################
    # ArangoDB Methods #
    ####################

    def aql(self, query: str, bind_vars: dict[str, Any] = {}, **kwargs: Any) -> Cursor:
        return nxadb.classes.function.aql(self.db, query, bind_vars, **kwargs)

    # NOTE: OUT OF SERVICE
    # def chat(self, prompt: str) -> str:
    #     if self.__qa_chain is None:
    #         if not self.__graph_exists_in_db:
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

    #####################
    # nx.Graph Overides #
    #####################

    @cached_property
    def nodes(self):
        if self.graph_exists_in_db:
            logger.warning("nxadb.CustomNodeView is currently EXPERIMENTAL")
            return CustomNodeView(self)

        return nx.classes.reportviews.NodeView(self)

    @cached_property
    def edges(self):
        if self.graph_exists_in_db:
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
