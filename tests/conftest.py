import os
from typing import Any

import networkx as nx
from adbnx_adapter import ADBNX_Adapter
from arango import ArangoClient


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--url", action="store", default="http://localhost:8529")
    parser.addoption("--dbName", action="store", default="_system")
    parser.addoption("--username", action="store", default="root")
    parser.addoption("--password", action="store", default="passwd")


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
