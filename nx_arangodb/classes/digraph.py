import os

import networkx as nx
from arango import ArangoClient
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.classes.graph import Graph

networkx_api = nxadb.utils.decorators.networkx_class(nx.DiGraph)

__all__ = ["DiGraph"]


class DiGraph(nx.DiGraph, Graph):
    @classmethod
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph

    def __init__(
        self,
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
            self.set_graph_name()

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
                raise TypeError(
                    "**db** must be an instance of arango.database.StandardDatabase"
                )

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
            return

        self.__db = ArangoClient(hosts=self.__host, request_timeout=None).db(
            self.__db_name, self.__username, self.__password, verify=True
        )

    def set_graph_name(self, graph_name: str | None = None):
        if self.__db is None:
            raise ValueError("Cannot set graph name without setting the database first")

        self.__graph_name = os.getenv("DATABASE_GRAPH_NAME")
        if graph_name is not None:
            if not isinstance(graph_name, str):
                raise TypeError("**graph_name** must be a string")

            self.__graph_name = graph_name

        if self.__graph_name is None:
            self.__graph_exists = False
            print("DATABASE_GRAPH_NAME environment variable not set")

        elif not self.db.has_graph(self.__graph_name):
            self.__graph_exists = False
            print(f"Graph '{self.__graph_name}' does not exist in the database")

        else:
            self.__graph_exists = True
            print(f"Found graph '{self.__graph_name}' in the database")

    def pull(self, load_node_and_adj_dict=True, load_coo=True):
        if not self.graph_exists:
            raise ValueError("Graph does not exist in the database")

        adb_graph = self.db.graph(self.graph_name)

        v_cols = adb_graph.vertex_collections()
        edge_definitions = adb_graph.edge_definitions()
        e_cols = {c["edge_collection"] for c in edge_definitions}

        metagraph = {
            "vertexCollections": {col: {} for col in v_cols},
            "edgeCollections": {col: {} for col in e_cols},
        }

        from phenolrs.graph_loader import GraphLoader

        kwargs = {}
        if self.graph_loader_parallelism is not None:
            kwargs["parallelism"] = self.graph_loader_parallelism
        if self.graph_loader_batch_size is not None:
            kwargs["batch_size"] = self.graph_loader_batch_size

        result = GraphLoader.load(
            self.db.name,
            metagraph,
            [self.__host],
            username=self.__username,
            password=self.__password,
            load_node_dict=load_node_and_adj_dict,
            load_adj_dict=load_node_and_adj_dict,
            load_adj_dict_as_directed=True,
            load_coo=load_coo,
            **kwargs,
        )

        if load_node_and_adj_dict:
            # hacky, i don't like this
            # need to revisit...
            # consider using nx.convert.from_dict_of_dicts instead
            self._node = result[0]
            self._adj = result[1]

        if load_coo:
            self.src_indices = result[2]
            self.dst_indices = result[3]
            self.vertex_ids_to_index = result[4]

    def push(self):
        raise NotImplementedError("What would this look like?")
