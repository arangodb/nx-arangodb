from typing import ClassVar

import networkx as nx

import nx_arangodb as nxadb
from nx_arangodb.logger import logger

networkx_api = nxadb.utils.decorators.networkx_class(nx.MultiGraph)  # type: ignore

__all__ = ["MultiGraph"]


class MultiGraph(nx.MultiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiGraph]:
        return nx.MultiGraph  # type: ignore[no-any-return]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m = "nxadb.MultiGraph has not been implemented yet. This is a pass-through subclass of nx.MultiGraph for now."  # noqa
        logger.warning(m)
