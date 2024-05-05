from typing import ClassVar

import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.logger import logger

networkx_api = nxadb.utils.decorators.networkx_class(nx.DiGraph)

__all__ = ["DiGraph"]


class DiGraph(nx.DiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_exists = False
        m = "nxadb.DiGraph has not been implemented yet. This is a pass-through subclass of nx.DiGraph for now."
        logger.warning(m)
