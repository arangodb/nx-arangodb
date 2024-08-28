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
        incoming_graph_data: Any = None,
        multigraph_input: bool | None = None,
        name: str | None = None,
        default_node_type: str | None = None,
        edge_type_key: str = "_edge_type",
        edge_type_func: Callable[[str, str], str] | None = None,
        edge_collections_attributes: set[str] | None = None,
        db: StandardDatabase | None = None,
        read_parallelism: int = 10,
        read_batch_size: int = 100000,
        write_batch_size: int = 50000,
        write_async: bool = True,
        symmetrize_edges: bool = False,
        use_experimental_views: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            incoming_graph_data,
            name,
            default_node_type,
            edge_type_key,
            edge_type_func,
            edge_collections_attributes,
            db,
            read_parallelism,
            read_batch_size,
            write_batch_size,
            write_async,
            symmetrize_edges,
            use_experimental_views,
            *args,
            **kwargs,
        )

        if self.graph_exists_in_db:
            self.add_edge = self.add_edge_override
            self.has_edge = self.has_edge_override
            self.copy = self.copy_override

        if incoming_graph_data is not None and not self._loaded_incoming_graph_data:
            # Taken from networkx.MultiGraph.__init__
            if isinstance(incoming_graph_data, dict) and multigraph_input is not False:
                try:
                    nx.convert.from_dict_of_dicts(
                        incoming_graph_data, create_using=self, multigraph_input=True
                    )
                except Exception as err:
                    if multigraph_input is True:
                        m = f"converting multigraph_input raised:\n{type(err)}: {err}"
                        raise nx.NetworkXError(m)

                    nx.convert.to_networkx_graph(incoming_graph_data, create_using=self)
            else:
                nx.convert.to_networkx_graph(incoming_graph_data, create_using=self)

    #######################
    # Init helper methods #
    #######################

    def _set_factory_methods(self) -> None:
        super()._set_factory_methods()
        self.edge_key_dict_factory = edge_key_dict_factory(
            self.db,
            self.adb_graph,
            self.edge_type_key,
            self.edge_type_func,
            self.is_directed(),
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

    def has_edge_override(self, u, v, key=None):
        try:
            if key is None:
                return v in self._adj[u]
            else:
                ######################
                # NOTE: monkey patch #
                ######################

                # Old: Nothing

                # New:
                if isinstance(key, int):
                    return len(self._adj[u][v]) > key

                # Reason:
                # Integer keys in nxadb.MultiGraph are simply used
                # as syntactic sugar to access the edge data of a specific
                # edge that is **cached** in the adjacency dictionary.
                # So we simply just check if the integer key is within the
                # range of the number of edges between u and v.

                return key in self._adj[u][v]
        except KeyError:
            return False

    def copy_override(self, *args, **kwargs):
        logger.warning("Note that copying a graph loses the connection to the database")
        G = super().copy(*args, **kwargs)
        G.edge_key_dict_factory = nx.MultiGraph.edge_key_dict_factory
        return G
