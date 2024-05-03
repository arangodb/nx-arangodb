import os
from typing import Callable, ClassVar

import networkx as nx
from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import ServerConnectionError

import nx_arangodb as nxadb
from nx_arangodb.classes.dict import (
    graph_dict_factory,
    node_attr_dict_factory,
    node_dict_factory,
)

from .dict import (
    adjlist_inner_dict_factory,
    adjlist_outer_dict_factory,
    edge_attr_dict_factory,
)

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
        default_node_type: str = "NXADB_NODES",
        edge_type_func: Callable[[str, str], str] = lambda u, v: f"{u}_to_{v}",
        *args,
        **kwargs,
    ):
        self.__db = None
        self.__graph_name = None
        self.__graph_exists = False

        self.set_db()
        if self.__db is not None:
            self.set_graph_name(graph_name)

        self.auto_sync = True

        self.graph_loader_parallelism = None
        self.graph_loader_batch_size = None

        # NOTE: Need to revisit these...
        # self.maintain_node_dict_cache = False
        # self.maintain_adj_dict_cache = False
        self.use_node_and_adj_dict_cache_for_algorithms = False
        self.use_coo_cache_for_algorithms = False

        self.src_indices = None
        self.dst_indices = None
        self.vertex_ids_to_index = None

        self.default_node_type = default_node_type
        self.edge_type_func = edge_type_func
        self.default_edge_type = edge_type_func(default_node_type, default_node_type)

        if self.__graph_exists:
            self.adb_graph = self.db.graph(graph_name)
            self.__create_default_collections()
            self.__set_factory_methods()

        super().__init__(*args, **kwargs)

    def __set_factory_methods(self) -> None:
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

    @property
    def db(self) -> StandardDatabase:
        if self.__db is None:
            raise ValueError("Database not set")

        return self.__db

    @property
    def graph_name(self) -> str:
        if self.__graph_name is None:
            raise ValueError("Graph name not set")

        return self.__graph_name

    @property
    def graph_exists(self) -> bool:
        return self.__graph_exists

    # TODO: Revisit. Causing some issues with incoming_graph_data=...
    # def clear(self, clear_remote: bool | None = None):
    #     """Clears the _adj, _node, and graph dictionaries, as well as
    #     the COO representation of the graph (if any).

    #     :param clear_remote: If True, the graph will also be cleared from the database.
    #         Defaults to False.
    #     :type clear_remote: bool
    #     """
    #     # self._adj.clear(clear_remote)
    #     self._node.clear(clear_remote)
    #     self.graph.clear(clear_remote)

    #     self.src_indices = None
    #     self.dst_indices = None
    #     self.vertex_ids_to_index = None

    #     nx._clear_cache(self)

    def set_db(self, db: StandardDatabase | None = None):
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
            print("Database environment variables not set")
            return

        try:
            self.__db = ArangoClient(hosts=self._host, request_timeout=None).db(
                self._db_name, self._username, self._password, verify=True
            )
        except ServerConnectionError as e:
            self.__db = None
            print(f"Could not connect to the database: {e}")

    def set_graph_name(self, graph_name: str | None = None):
        if self.__db is None:
            raise ValueError("Cannot set graph name without setting the database first")

        if graph_name is None:
            self.__graph_exists = False
            print("**graph_name** attribute not set")
            return

        if not isinstance(graph_name, str):
            raise TypeError("**graph_name** must be a string")

        self.__graph_name = graph_name
        self.__graph_exists = self.db.has_graph(graph_name)
        print(f"Graph '{graph_name}' exists: {self.__graph_exists}")

    def pull(self, load_node_and_adj_dict=True, load_coo=True):
        nxadb.classes.function.pull(
            self,
            load_node_and_adj_dict=load_node_and_adj_dict,
            load_adj_dict_as_undirected=True,
            load_coo=load_coo,
        )

    def push(self):
        raise NotImplementedError("What would this look like?")

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

        else:  # update attr even if node already exists
            self._node[node_for_adding].update(attr)
        nx._clear_cache(self)
