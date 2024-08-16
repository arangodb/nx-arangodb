import os
from functools import cached_property
from typing import Any, Callable, ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt
from adbnx_adapter import ADBNX_Adapter
from arango import ArangoClient
from arango.cursor import Cursor
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.exceptions import (
    DatabaseNotSet,
    EdgeTypeAmbiguity,
    GraphNameNotSet,
    InvalidDefaultNodeType,
)
from nx_arangodb.logger import logger

from .dict import (
    adjlist_inner_dict_factory,
    adjlist_outer_dict_factory,
    edge_attr_dict_factory,
    graph_dict_factory,
    node_attr_dict_factory,
    node_dict_factory,
)
from .reportviews import CustomEdgeView, CustomNodeView

networkx_api = nxadb.utils.decorators.networkx_class(nx.Graph)  # type: ignore

__all__ = ["Graph"]


class Graph(nx.Graph):
    __networkx_backend__: ClassVar[str] = "arangodb"  # nx >=3.2
    __networkx_plugin__: ClassVar[str] = "arangodb"  # nx <3.2

    @classmethod
    def to_networkx_class(cls) -> type[nx.Graph]:
        return nx.Graph  # type: ignore[no-any-return]

    def __init__(
        self,
        graph_name: str | None = None,
        default_node_type: str | None = None,
        edge_type_key: str = "_edge_type",
        edge_type_func: Callable[[str, str], str] | None = None,
        edge_collections_attributes: set[str] | None = None,
        db: StandardDatabase | None = None,
        read_parallelism: int = 10,
        read_batch_size: int = 100000,
        write_batch_size: int = 50000,
        symmetrize_edges: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        self._db = None
        self._graph_name = None
        self._graph_exists_in_db = False

        self._set_db(db)
        if self._db is not None:
            self._set_graph_name(graph_name)

        # We need to store the data transfer properties as some functions will need them
        self.read_parallelism = read_parallelism
        self.read_batch_size = read_batch_size
        self.write_batch_size = write_batch_size

        self._set_edge_collections_attributes_to_fetch(edge_collections_attributes)

        # NOTE: Need to revisit these...
        # self.maintain_node_dict_cache = False
        # self.maintain_adj_dict_cache = False
        self.use_nx_cache = True
        self.use_nxcg_cache = True
        self.nxcg_graph = None

        # self.__qa_chain = None

        # Does not apply to undirected graphs
        self.symmetrize_edges = symmetrize_edges

        self.edge_type_key = edge_type_key

        incoming_graph_data = kwargs.get("incoming_graph_data")

        # TODO: Consider this
        # if not self.__graph_name:
        #     if incoming_graph_data is not None:
        #         m = "Must set **graph_name** if passing **incoming_graph_data**"
        #         raise ValueError(m)

        if self._graph_exists_in_db:
            if incoming_graph_data is not None:
                m = "Cannot pass both **incoming_graph_data** and **graph_name** yet if the already graph exists"  # noqa: E501
                raise NotImplementedError(m)

            if edge_type_func is not None:
                m = "Cannot pass **edge_type_func** if the graph already exists"
                raise NotImplementedError(m)

            self.adb_graph = self.db.graph(self._graph_name)
            vertex_collections = self.adb_graph.vertex_collections()
            edge_definitions = self.adb_graph.edge_definitions()

            if default_node_type is None:
                default_node_type = list(vertex_collections)[0]
                logger.info(f"Default node type set to '{default_node_type}'")
            elif default_node_type not in vertex_collections:
                m = f"Default node type '{default_node_type}' not found in graph '{graph_name}'"  # noqa: E501
                raise InvalidDefaultNodeType(m)

            node_types_to_edge_type_map: dict[tuple[str, str], str] = {}
            for e_d in edge_definitions:
                for f in e_d["from_vertex_collections"]:
                    for t in e_d["to_vertex_collections"]:
                        if (f, t) in node_types_to_edge_type_map:
                            # TODO: Should we log a warning at least?
                            continue

                        node_types_to_edge_type_map[(f, t)] = e_d["edge_collection"]

            def edge_type_func(u: str, v: str) -> str:
                try:
                    return node_types_to_edge_type_map[(u, v)]
                except KeyError:
                    m = f"Edge type ambiguity between '{u}' and '{v}'"
                    raise EdgeTypeAmbiguity(m)

            self.edge_type_func = edge_type_func
            self.default_node_type = default_node_type

            self._set_factory_methods()
            self._set_arangodb_backend_config()

        elif self._graph_name:

            prefix = f"{graph_name}_" if graph_name else ""
            if default_node_type is None:
                default_node_type = f"{prefix}node"
            if edge_type_func is None:
                edge_type_func = lambda u, v: f"{u}_to_{v}"  # noqa: E731

            self.edge_type_func = edge_type_func
            self.default_node_type = default_node_type

            # TODO: Parameterize the edge definitions
            # How can we work with a heterogenous **incoming_graph_data**?
            default_edge_type = edge_type_func(default_node_type, default_node_type)
            edge_definitions = [
                {
                    "edge_collection": default_edge_type,
                    "from_vertex_collections": [default_node_type],
                    "to_vertex_collections": [default_node_type],
                }
            ]

            if isinstance(incoming_graph_data, nx.Graph):
                self.adb_graph = ADBNX_Adapter(self.db).networkx_to_arangodb(
                    self._graph_name,
                    incoming_graph_data,
                    edge_definitions=edge_definitions,
                    batch_size=self.write_batch_size,
                    use_async=True,
                )

                # No longer need this (we've already populated the graph)
                del kwargs["incoming_graph_data"]

            else:
                self.adb_graph = self.db.create_graph(
                    self._graph_name,
                    edge_definitions=edge_definitions,
                )

            self._set_factory_methods()
            self._set_arangodb_backend_config()
            logger.info(f"Graph '{graph_name}' created.")
            self._graph_exists_in_db = True

        super().__init__(*args, **kwargs)

    #######################
    # Init helper methods #
    #######################

    def _set_arangodb_backend_config(self) -> None:
        if not all([self._host, self._username, self._password, self._db_name]):
            m = "Must set all environment variables to use the ArangoDB Backend with an existing graph"  # noqa: E501
            raise OSError(m)

        config = nx.config.backends.arangodb
        config.host = self._host
        config.username = self._username
        config.password = self._password
        config.db_name = self._db_name
        config.read_parallelism = self.read_parallelism
        config.read_batch_size = self.read_batch_size
        config.write_batch_size = self.write_batch_size

    def _set_factory_methods(self) -> None:
        """Set the factory methods for the graph, _node, and _adj dictionaries.

        The ArangoDB CRUD operations are handled by the modified dictionaries.

        Handles the creation of the following dictionaries:
        - graph_attr_dict_factory (graph-level attributes)
        - node_dict_factory (nodes in the graph)
        - node_attr_dict_factory (attributes of the nodes in the graph)
        - adjlist_outer_dict_factory (outer dictionary for the adjacency list)
        - adjlist_inner_dict_factory (inner dictionary for the adjacency list)
        - edge_attr_dict_factory (attributes of the edges in the graph)
        """

        base_args = (self.db, self.adb_graph)
        node_args = (*base_args, self.default_node_type)
        adj_args = (
            *node_args,
            self.edge_type_key,
            self.edge_type_func,
            self.__class__.__name__,
        )

        self.graph_attr_dict_factory = graph_dict_factory(*base_args)

        self.node_dict_factory = node_dict_factory(*node_args)
        self.node_attr_dict_factory = node_attr_dict_factory(*base_args)

        self.edge_attr_dict_factory = edge_attr_dict_factory(*base_args)
        self.adjlist_inner_dict_factory = adjlist_inner_dict_factory(*adj_args)
        self.adjlist_outer_dict_factory = adjlist_outer_dict_factory(
            *adj_args, self.symmetrize_edges
        )

    def _set_edge_collections_attributes_to_fetch(
        self, attributes: set[str] | None
    ) -> None:
        if attributes is None:
            self._edge_collections_attributes = set()
            return
        if len(attributes) > 0:
            self._edge_collections_attributes = attributes
            if "_id" not in attributes:
                self._edge_collections_attributes.add("_id")

    ###########
    # Getters #
    ###########

    @property
    def db(self) -> StandardDatabase:
        if self._db is None:
            raise DatabaseNotSet("Database not set")

        return self._db

    @property
    def graph_name(self) -> str:
        if self._graph_name is None:
            raise GraphNameNotSet("Graph name not set")

        return self._graph_name

    @property
    def graph_exists_in_db(self) -> bool:
        return self._graph_exists_in_db

    @property
    def get_edge_attributes(self) -> set[str]:
        return self._edge_collections_attributes

    ###########
    # Setters #
    ###########

    def _set_db(self, db: StandardDatabase | None = None) -> None:
        self._host = os.getenv("DATABASE_HOST")
        self._username = os.getenv("DATABASE_USERNAME")
        self._password = os.getenv("DATABASE_PASSWORD")
        self._db_name = os.getenv("DATABASE_NAME")

        if db is not None:
            if not isinstance(db, StandardDatabase):
                m = "arango.database.StandardDatabase"
                raise TypeError(m)

            db.version()
            self._db = db
            return

        # TODO: Raise a custom exception if any of the environment
        # variables are missing. For now, we'll just set db to None.
        if not all([self._host, self._username, self._password, self._db_name]):
            self._db = None
            logger.warning("Database environment variables not set")
            return

        self._db = ArangoClient(hosts=self._host, request_timeout=None).db(
            self._db_name, self._username, self._password, verify=True
        )

    def _set_graph_name(self, graph_name: str | None = None) -> None:
        if self._db is None:
            m = "Cannot set graph name without setting the database first"
            raise DatabaseNotSet(m)

        if graph_name is None:
            self._graph_exists_in_db = False
            logger.warning(f"**graph_name** not set for {self.__class__.__name__}")
            return

        if not isinstance(graph_name, str):
            raise TypeError("**graph_name** must be a string")

        self._graph_name = graph_name
        self._graph_exists_in_db = self.db.has_graph(graph_name)

        logger.info(f"Graph '{graph_name}' exists: {self._graph_exists_in_db}")

    ####################
    # ArangoDB Methods #
    ####################

    def aql(self, query: str, bind_vars: dict[str, Any] = {}, **kwargs: Any) -> Cursor:
        return nxadb.classes.function.aql(self.db, query, bind_vars, **kwargs)

    # def pull(self) -> None:
    #     self._node._fetch_all()
    #     self._adj._fetch_all()

    # NOTE: OUT OF SERVICE
    # def chat(self, prompt: str) -> str:
    #     if self.__qa_chain is None:
    #         if not self.__graph_exists_in_db:
    #             return "Could not initialize QA chain: Graph does not exist"

    #         # try:
    #         from langchain.chains import ArangoGraphQAChain
    #         from langchain_community.graphs import ArangoGraph
    #         from langchain_openai import ChatOpenAI

    #         model = ChatOpenAI(temperature=0, model_name="gpt-4")

    #         self.__qa_chain = ArangoGraphQAChain.from_llm(
    #             llm=model, graph=ArangoGraph(self.db), verbose=True
    #         )

    #         # except Exception as e:
    #         #     return f"Could not initialize QA chain: {e}"

    #     self.__qa_chain.graph.set_schema()
    #     result = self.__qa_chain.invoke(prompt)

    #     print(result["result"])

    #####################
    # nx.Graph Overides #
    #####################

    def copy(self, *args, **kwargs):
        raise NotImplementedError("Copying an ArangoDB Graph is not yet implemented")

    def subgraph(self, nbunch):
        raise NotImplementedError("Subgraphing is not yet implemented")

    def clear(self):
        logger.info("Note that clearing only erases the local cache")
        super().clear()

    def clear_edges(self):
        logger.info("Note that clearing edges ony erases the edges in the local cache")
        super().clear_edges()

    def clear_nxcg_cache(self):
        self.nxcg_graph = None

    @cached_property
    def nodes(self):
        if self.graph_exists_in_db:
            logger.warning("nxadb.CustomNodeView is currently EXPERIMENTAL")
            return CustomNodeView(self)

        return super().nodes

    @cached_property
    def edges(self):
        if self.graph_exists_in_db:
            if self.is_directed():
                logger.warning("CustomEdgeView for Directed Graphs not yet implemented")
                return super().edges

            if self.is_multigraph():
                logger.warning("CustomEdgeView for MultiGraphs not yet implemented")
                return super().edges

            logger.warning("nxadb.CustomEdgeView is currently EXPERIMENTAL")
            return CustomEdgeView(self)

        return super().edges

    def add_node(self, node_for_adding, **attr):
        if node_for_adding not in self._node:
            if node_for_adding is None:
                raise ValueError("None cannot be a node")
            self._adj[node_for_adding] = self.adjlist_inner_dict_factory()

            ######################
            # NOTE: monkey patch #
            ######################

            # Old:
            # attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            # attr_dict.update(attr)

            # New:
            self._node[node_for_adding] = self.node_attr_dict_factory()
            self._node[node_for_adding].update(attr)

            # Reason:
            # Invoking `update` on the `attr_dict` without `attr_dict.node_id` being set
            # i.e trying to update a node's attributes before we know _which_ node it is

            ###########################

        else:
            self._node[node_for_adding].update(attr)

        nx._clear_cache(self)

    def number_of_edges(self, u=None, v=None):
        if not self.graph_exists_in_db:
            return super().number_of_edges(u, v)

        if u is not None:
            return super().number_of_edges(u, v)

        ######################
        # NOTE: monkey patch #
        ######################

        # Old:
        # return int(self.size())

        # New:
        edge_collections = {
            e_d["edge_collection"] for e_d in self.adb_graph.edge_definitions()
        }
        num = sum(self.adb_graph.edge_collection(e).count() for e in edge_collections)
        num *= 2 if self.is_directed() and self.symmetrize_edges else 1

        return num

        # Reason:
        # It is more efficient to count the number of edges in the edge collections
        # compared to relying on the DegreeView.
