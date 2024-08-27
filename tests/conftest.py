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
from nx_arangodb.classes.dict.adj import AdjListOuterDict
from nx_arangodb.classes.dict.node import NodeDict
from nx_arangodb.logger import logger

logger.setLevel(logging.INFO)

db: StandardDatabase


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
    print("----------------------------------------")

    global db
    db = ArangoClient(hosts=con["url"]).db(
        con["dbName"], con["username"], con["password"], verify=True
    )

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
    return graph_cls(incoming_graph_data=grid_graph, name="GridGraph")


def assert_remote_dict(G: nxadb.Graph) -> None:
    assert isinstance(G._node, NodeDict)
    assert isinstance(G._adj, AdjListOuterDict)


def extract_arangodb_key(adb_id: str) -> str:
    return adb_id.split("/")[1]


def assert_same_dict_values(
    d1: dict[str | int, float], d2: dict[str | int, float], digit: int
) -> None:
    if type(next(iter(d1.keys()))) == int:
        d1 = {f"person/{k}": v for k, v in d1.items()}

    if type(next(iter(d2.keys()))) == int:
        d2 = {f"person/{k}": v for k, v in d2.items()}

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    difference = d1_keys ^ d2_keys
    assert difference == set(), "Dictionaries have different keys"

    for key in d1:
        m = f"Values for key '{key}' are not equal up to digit {digit}"
        assert round(d1[key], digit) == round(d2[key], digit), m


def assert_bc(d1: dict[str | int, float], d2: dict[str | int, float]) -> None:
    assert d1
    assert d2
    assert_same_dict_values(d1, d2, 14)


def assert_pagerank(
    d1: dict[str | int, float], d2: dict[str | int, float], digit: int = 15
) -> None:
    assert d1
    assert d2
    assert_same_dict_values(d1, d2, digit)


def assert_louvain(l1: list[set[Any]], l2: list[set[Any]]) -> None:
    # TODO: Implement some kind of comparison
    # Reason: Louvain returns different results on different runs
    assert l1
    assert l2
    pass


def assert_k_components(
    d1: dict[int, list[set[Any]]], d2: dict[int, list[set[Any]]]
) -> None:
    assert d1
    assert d2
    assert d1.keys() == d2.keys(), "Dictionaries have different keys"
    assert d1 == d2


# Taken from:
# https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class Capturing(list[str]):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout
