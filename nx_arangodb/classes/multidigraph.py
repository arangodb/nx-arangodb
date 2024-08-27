from typing import Any, Callable, ClassVar

import networkx as nx
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.classes.digraph import DiGraph
from nx_arangodb.classes.multigraph import MultiGraph
from nx_arangodb.logger import logger

networkx_api = nxadb.utils.decorators.networkx_class(nx.MultiDiGraph)  # type: ignore

__all__ = ["MultiDiGraph"]


class MultiDiGraph(MultiGraph, DiGraph, nx.MultiDiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiDiGraph]:
        return nx.MultiDiGraph  # type: ignore[no-any-return]

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
        symmetrize_edges: bool = False,
        use_experimental_views: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            incoming_graph_data,
            multigraph_input,
            name,
            default_node_type,
            edge_type_key,
            edge_type_func,
            edge_collections_attributes,
            db,
            read_parallelism,
            read_batch_size,
            write_batch_size,
            symmetrize_edges,
            use_experimental_views,
            *args,
            **kwargs,
        )

    #######################
    # Init helper methods #
    #######################

    ##########################
    # nx.MultiGraph Overides #
    ##########################
