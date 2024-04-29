import os
from typing import ClassVar

import networkx as nx
from arango import ArangoClient
from arango.database import StandardDatabase

import nx_arangodb as nxadb

networkx_api = nxadb.utils.decorators.networkx_class(nx.Graph)

__all__ = ["Graph"]


class Graph(nx.Graph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.Graph]:
        return nx.Graph

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.set_db()

        self.__graph_exists = False
        if self.__db is not None:
            self.set_graph_name()

    @property
    def db(self) -> StandardDatabase:
        if self.__db is None:
            raise ValueError("Database not set")

        return self.__db

    @property
    def graph_name(self) -> str:
        if self.__graph_name is None:
            raise ValueError("Graph name not set")

        return self.__graph_name

    @property
    def graph_exists(self) -> bool:
        return self.__graph_exists

    def set_db(self, db: StandardDatabase | None = None):
        if db is not None:
            if not isinstance(db, StandardDatabase):
                raise TypeError(
                    "**db** must be an instance of arango.database.StandardDatabase"
                )

            self.__db = db
            return

        host = os.getenv("DATABASE_HOST")
        username = os.getenv("DATABASE_USERNAME")
        password = os.getenv("DATABASE_PASSWORD")
        db_name = os.getenv("DATABASE_NAME")

        # TODO: Raise a custom exception if any of the environment
        # variables are missing. For now, we'll just set db to None.
        if not all([host, username, password, db_name]):
            self.__db = None
            return

        self.__db = ArangoClient(hosts=host, request_timeout=None).db(
            db_name, username, password, verify=True
        )

    def set_graph_name(self, graph_name: str | None = None):
        if self.__db is None:
            raise ValueError("Cannot set graph name without setting the database first")

        self.__graph_name = os.getenv("DATABASE_GRAPH_NAME")
        if graph_name is not None:
            if not isinstance(graph_name, str):
                raise TypeError("**graph_name** must be a string")

            self.__graph_name = graph_name

        if self.__graph_name is None:
            self.graph_exists = False
            print("DATABASE_GRAPH_NAME environment variable not set")

        elif not self.db.has_graph(self.__graph_name):
            self.graph_exists = False
            print(f"Graph '{self.__graph_name}' does not exist in the database")

        else:
            self.graph_exists = True
            print(f"Found graph '{self.__graph_name}' in the database")
