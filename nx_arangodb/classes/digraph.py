import os
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
from arango import ArangoClient
from arango.cursor import Cursor
from arango.database import StandardDatabase
from arango.exceptions import ServerConnectionError

import nx_arangodb as nxadb
from nx_arangodb.exceptions import DatabaseNotSet, GraphNameNotSet
from nx_arangodb.logger import logger

networkx_api = nxadb.utils.decorators.networkx_class(nx.DiGraph)  # type: ignore

__all__ = ["DiGraph"]


class DiGraph(nx.DiGraph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.DiGraph]:
        return nx.DiGraph  # type: ignore[no-any-return]

    def __init__(
        self,
        graph_name: str | None = None,
        # default_node_type: str = "node",
        # edge_type_func: Callable[[str, str], str] = lambda u, v: f"{u}_to_{v}",
        symmetrize_edges: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        m = "Please note that nxadb.DiGraph has no ArangoDB CRUD support yet."
        logger.warning(m)

        if kwargs.get("incoming_graph_data") is not None and graph_name is not None:
            m = "Cannot pass both **incoming_graph_data** and **graph_name** yet"
            raise NotImplementedError(m)

        self.__db: StandardDatabase | None = None
        self.__graph_name: str | None = None
        self.__graph_exists_in_db = False

        self.__set_db()
        if self.__db is not None:
            self.__set_graph_name(graph_name)

        self.auto_sync = True

        self.graph_loader_parallelism = 20
        self.graph_loader_batch_size = 5000000

        # NOTE: Need to revisit these...
        # self.maintain_node_dict_cache = False
        # self.maintain_adj_dict_cache = False
        self.use_nx_cache = True
        self.use_coo_cache = True

        self.src_indices: npt.NDArray[np.int64] | None = None
        self.dst_indices: npt.NDArray[np.int64] | None = None
        self.edge_indices: npt.NDArray[np.int64] | None = None
        self.vertex_ids_to_index: dict[str, int] | None = None

        # self.default_node_type = default_node_type
        # self.edge_type_func = edge_type_func
        # self.default_edge_type = edge_type_func(default_node_type, default_node_type)

        if self.__graph_exists_in_db:
            self.adb_graph = self.db.graph(graph_name)
            # self.__create_default_collections()
            # self.__set_factory_methods()

        super().__init__(*args, **kwargs)

        self.symmetrize_edges = symmetrize_edges

    ###########
    # Getters #
    ###########

    @property
    def db(self) -> StandardDatabase:
        if self.__db is None:
            raise DatabaseNotSet("Database not set")

        return self.__db

    @property
    def graph_name(self) -> str:
        if self.__graph_name is None:
            raise GraphNameNotSet("Graph name not set")

        return self.__graph_name

    @property
    def graph_exists_in_db(self) -> bool:
        return self.__graph_exists_in_db

    ###########
    # Setters #
    ###########

    def __set_db(self, db: StandardDatabase | None = None) -> None:
        if db is not None:
            if not isinstance(db, StandardDatabase):
                m = "arango.database.StandardDatabase"
                raise TypeError(m)

            self.__db = db
            return

        self._host = os.getenv("DATABASE_HOST")
        self._username = os.getenv("DATABASE_USERNAME")
        self._password = os.getenv("DATABASE_PASSWORD")
        self._db_name = os.getenv("DATABASE_NAME")

        # TODO: Raise a custom exception if any of the environment
        # variables are missing. For now, we'll just set db to None.
        if not all([self._host, self._username, self._password, self._db_name]):
            self.__db = None
            logger.warning("Database environment variables not set")
            return

        try:
            self.__db = ArangoClient(hosts=self._host, request_timeout=None).db(
                self._db_name, self._username, self._password, verify=True
            )
        except ServerConnectionError as e:
            self.__db = None
            logger.warning(f"Could not connect to the database: {e}")

    def __set_graph_name(self, graph_name: str | None = None) -> None:
        if self.__db is None:
            m = "Cannot set graph name without setting the database first"
            raise DatabaseNotSet(m)

        if graph_name is None:
            self.__graph_exists_in_db = False
            logger.warning(f"**graph_name** not set for {self.__class__.__name__}")
            return

        if not isinstance(graph_name, str):
            raise TypeError("**graph_name** must be a string")

        self.__graph_name = graph_name
        self.__graph_exists_in_db = self.db.has_graph(graph_name)

        logger.info(f"Graph '{graph_name}' exists: {self.__graph_exists_in_db}")

    ####################
    # ArangoDB Methods #
    ####################

    def aql(self, query: str, bind_vars: dict[str, Any] = {}, **kwargs: Any) -> Cursor:
        return nxadb.classes.function.aql(self.db, query, bind_vars, **kwargs)
