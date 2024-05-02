import networkx as nx
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
