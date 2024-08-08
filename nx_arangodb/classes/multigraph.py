from typing import Any, Callable, ClassVar

import networkx as nx
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.classes.graph import Graph
from nx_arangodb.logger import logger

from .dict import edge_key_dict_factory

networkx_api = nxadb.utils.decorators.networkx_class(nx.MultiGraph)  # type: ignore

__all__ = ["MultiGraph"]


class MultiGraph(Graph, nx.MultiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiGraph]:
        return nx.MultiGraph  # type: ignore[no-any-return]

    def __init__(
        self,
        graph_name: str | None = None,
        default_node_type: str | None = None,
        edge_type_func: Callable[[str, str], str] | None = None,
        db: StandardDatabase | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            graph_name, default_node_type, edge_type_func, db, *args, **kwargs
        )

        if self._graph_exists_in_db:
            self.add_edge = self.add_edge_override

    #######################
    # Init helper methods #
    #######################

    def _set_factory_methods(self) -> None:
        super()._set_factory_methods()
        self.edge_key_dict_factory = edge_key_dict_factory(
            self.db, self.adb_graph, self.edge_type_func, self.is_directed()
        )

    ##########################
    # nx.MultiGraph Overides #
    ##########################

    def add_edge_override(self, u_for_edge, v_for_edge, key=None, **attr):
        if key is not None:
            m = "ArangoDB MultiGraph does not support custom edge keys yet."
            logger.warning(m)

        _ = super().add_edge(u_for_edge, v_for_edge, key="-1", **attr)

        ######################
        # NOTE: monkey patch #
        ######################

        # Old:
        # return key

        # New:
        keys = list(self._adj[u_for_edge][v_for_edge].data.keys())
        last_key = keys[-1]
        return last_key

        # Reason:
        # nxadb.MultiGraph does not yet support the ability to work
        # with custom edge keys. As a Database, we must rely on the official
        # ArangoDB Edge _id to uniquely identify edges. The EdgeKeyDict.__setitem__
        # method will be responsible for setting the edge key to the _id of the edge
        # document. This will allow us to use the edge key as a unique identifier

        ###########################
