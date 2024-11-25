import logging
import os
import sys
from io import StringIO
from typing import Any

import networkx as nx
import pytest
from adbnx_adapter import ADBNX_Adapter
from arango import ArangoClient
from arango.database import StandardDatabase

import nx_arangodb as nxadb
from nx_arangodb.logger import logger

logger.setLevel(logging.INFO)

db: StandardDatabase
run_gpu_tests: bool


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--url", action="store", default="http://localhost:8529")
    parser.addoption("--dbName", action="store", default="_system")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="test")
    parser.addoption(
        "--run-gpu-tests", action="store_true", default=False, help="Run GPU tests"
    )


def pytest_configure(config: Any) -> None:
    con = {
        "url": config.getoption("url"),
        "username": config.getoption("username"),
        "password": config.getoption("password"),
        "dbName": config.getoption("dbName"),
    }

    print("----------------------------------------")
    print("URL: " + con["url"])
    print("Username: " + con["username"])
    print("Password: " + con["password"])
    print("Database: " + con["dbName"])

    global db
    db = ArangoClient(hosts=con["url"]).db(
        con["dbName"], con["username"], con["password"], verify=True
    )

    print("Version: " + db.version())
    print("----------------------------------------")

    os.environ["DATABASE_HOST"] = con["url"]
    os.environ["DATABASE_USERNAME"] = con["username"]
    os.environ["DATABASE_PASSWORD"] = con["password"]
    os.environ["DATABASE_NAME"] = con["dbName"]

    global run_gpu_tests
    run_gpu_tests = config.getoption("--run-gpu-tests")


@pytest.fixture(scope="function")
def load_karate_graph() -> None:
    global db
    db.delete_graph("KarateGraph", drop_collections=True, ignore_missing=True)
    adapter = ADBNX_Adapter(db)
    adapter.networkx_to_arangodb(
        "KarateGraph",
        nx.karate_club_graph(),
        edge_definitions=[
            {
                "edge_collection": "knows",
                "from_vertex_collections": ["person"],
                "to_vertex_collections": ["person"],
            }
        ],
    )


@pytest.fixture(scope="function")
def load_two_relation_graph() -> None:
    global db
    graph_name = "IntegrationTestTwoRelationGraph"
    v1 = graph_name + "_v1"
    v2 = graph_name + "_v2"
    e1 = graph_name + "_e1"
    e2 = graph_name + "_e2"

    if db.has_graph(graph_name):
        db.delete_graph(graph_name, drop_collections=True)

    g = db.create_graph(graph_name)
    g.create_edge_definition(
        e1, from_vertex_collections=[v1], to_vertex_collections=[v2]
    )
    g.create_edge_definition(
        e2, from_vertex_collections=[v2], to_vertex_collections=[v1]
    )


def create_line_graph(load_attributes: set[str]) -> nxadb.Graph:
    G = nx.Graph()
    G.add_edge(1, 2, my_custom_weight=1)
    G.add_edge(2, 3, my_custom_weight=1)
    G.add_edge(3, 4, my_custom_weight=1000)
    G.add_edge(4, 5, my_custom_weight=1000)

    return nxadb.Graph(
        incoming_graph_data=G,
        name="LineGraph",
        edge_collections_attributes=load_attributes,
    )


def create_grid_graph(graph_cls: type[nxadb.Graph]) -> nxadb.Graph:
    global db
    if db.has_graph("GridGraph"):
        return graph_cls(name="GridGraph")

    grid_graph = nx.grid_graph(dim=(500, 500))
    return graph_cls(
        incoming_graph_data=grid_graph, name="GridGraph", write_async=False
    )
