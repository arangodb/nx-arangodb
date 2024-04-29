import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.classes.digraph import DiGraph
from nx_arangodb.classes.multigraph import MultiGraph

networkx_api = nxadb.utils.decorators.networkx_class(nx.MultiDiGraph)

__all__ = ["MultiDiGraph"]


class MultiDiGraph(nx.MultiDiGraph, MultiGraph, DiGraph):
    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiDiGraph]:
        return nx.MultiDiGraph

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.set_db()

        self.__graph_exists = False
        if self.__db is not None:
            self.set_graph_name()
