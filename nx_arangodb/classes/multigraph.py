import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.classes.graph import Graph

networkx_api = nxadb.utils.decorators.networkx_class(nx.MultiGraph)

__all__ = ["MultiGraph"]


class MultiGraph(nx.MultiGraph, Graph):
    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiGraph]:
        return nx.MultiGraph
