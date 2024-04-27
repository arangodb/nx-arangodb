from typing import ClassVar

import networkx as nx

import nx_arangodb as nxadb

networkx_api = nxadb.utils.decorators.networkx_class(nx.Graph)

__all__ = ["Graph"]


class Graph(nx.Graph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.Graph]:
        return nx.Graph
