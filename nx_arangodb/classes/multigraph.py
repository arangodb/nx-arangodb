from typing import Any, Callable, ClassVar

import networkx as nx
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.classes.graph import Graph as nxadb_Graph
from nx_arangodb.logger import logger

from .dict import edge_key_dict_factory

networkx_api = nxadb.utils.decorators.networkx_class(nx.MultiGraph)  # type: ignore

__all__ = ["MultiGraph"]


class MultiGraph(nxadb_Graph, nx.MultiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiGraph]:
        return nx.MultiGraph  # type: ignore[no-any-return]

    def __init__(
        self,
        graph_name: str | None = None,
        default_node_type: str = "node",
        edge_type_func: Callable[[str, str], str] = lambda u, v: f"{u}_to_{v}",
        db: StandardDatabase | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            graph_name, default_node_type, edge_type_func, db, *args, **kwargs
        )

    #######################
    # Init helper methods #
    #######################

    def _set_factory_methods(self) -> None:
        super()._set_factory_methods()

        base_args = (self.db, self.adb_graph)
        self.edge_key_dict_factory = edge_key_dict_factory(*base_args)

    ##########################
    # nx.MultiGraph Overides #
    ##########################
