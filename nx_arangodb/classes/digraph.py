import networkx as nx

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

        self.coo_use_cache = False
        self.coo_load_parallelism = None
        self.coo_load_batch_size = None
        self.src_indices = None
        self.dst_indices = None
        self.vertex_ids_to_index = None

        self.set_db()
        if self.__db is not None:
            self.set_graph_name()

    @property
    def graph_exists(self) -> bool:
        return self.__graph_exists
