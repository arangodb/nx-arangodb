from typing import ClassVar

import networkx as nx

import nx_arangodb as nxadb

networkx_api = nxadb.utils.decorators.networkx_class(nx.MultiGraph)

__all__ = ["MultiGraph"]


class MultiGraph(nx.MultiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2
