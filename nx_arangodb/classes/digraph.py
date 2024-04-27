import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.classes.graph import Graph

networkx_api = nxadb.utils.decorators.networkx_class(nx.DiGraph)

__all__ = ["DiGraph"]


class DiGraph(nx.DiGraph, Graph):
    @classmethod
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph
