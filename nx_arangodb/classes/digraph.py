import os
from typing import ClassVar

import networkx as nx
from arango import ArangoClient
from arango.database import StandardDatabase
from arango.exceptions import ServerConnectionError

import nx_arangodb as nxadb

networkx_api = nxadb.utils.decorators.networkx_class(nx.DiGraph)

__all__ = ["DiGraph"]


class DiGraph(nx.DiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph

    def __init__(
        self,
        graph_name: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.__db = None
        self.__graph_name = None
        self.__graph_exists = False

        self.graph_loader_parallelism = None
        self.graph_loader_batch_size = None

        self.use_node_and_adj_dict_cache = False
        self.use_coo_cache = False

        self.src_indices = None
        self.dst_indices = None
        self.vertex_ids_to_index = None

        self.set_db()
        if self.__db is not None:
            self.set_graph_name(graph_name)

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

    def clear_coo_cache(self):
        self.src_indices = None
        self.dst_indices = None
        self.vertex_ids_to_index = None

    def set_db(self, db: StandardDatabase | None = None):
        if db is not None:
            if not isinstance(db, StandardDatabase):
                m = "arango.database.StandardDatabase"
                raise TypeError(m)

            self.__db = db
            return

        self.__host = os.getenv("DATABASE_HOST")
        self.__username = os.getenv("DATABASE_USERNAME")
        self.__password = os.getenv("DATABASE_PASSWORD")
        self.__db_name = os.getenv("DATABASE_NAME")

        # TODO: Raise a custom exception if any of the environment
        # variables are missing. For now, we'll just set db to None.
        if not all([self.__host, self.__username, self.__password, self.__db_name]):
            self.__db = None
            print("Database environment variables not set")
            return

        try:
            self.__db = ArangoClient(hosts=self.__host, request_timeout=None).db(
                self.__db_name, self.__username, self.__password, verify=True
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

        if not isinstance(graph_name, str):
            raise TypeError("**graph_name** must be a string")

        self.__graph_name = graph_name
        self.__graph_exists = self.db.has_graph(graph_name)
        print(f"Graph '{graph_name}' exists: {self.__graph_exists}")

    def pull(self, load_node_and_adj_dict=True, load_coo=True):
        nxadb.classes.function.get_arangodb_graph(
            self,
            load_node_and_adj_dict=load_node_and_adj_dict,
            load_adj_dict_as_undirected=False,
            load_coo=load_coo,
        )

    def push(self):
        raise NotImplementedError("What would this look like?")
