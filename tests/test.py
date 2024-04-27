import pytest
import networkx as nx
import nx_arangodb as nxadb

from .conftest import db


def test_db():
    assert db.version()


def test_bc():
    G_1 = nx.karate_club_graph()

    G_2 = nxadb.Graph(G_1)

    assert nx.betweenness_centrality(G_1) == nx.betweenness_centrality(G_2)
