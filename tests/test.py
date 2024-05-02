import pytest
import networkx as nx
import nx_arangodb as nxadb

from .conftest import db


def test_db():
    assert db.version()


def test_bc():
    G_1 = nx.karate_club_graph()

    G_2 = nxadb.Graph(G_1)

    r_1 = nx.betweenness_centrality(G_1)
    r_2 = nx.betweenness_centrality(G_2)
    r_3 = nx.betweenness_centrality(G_1, backend="arangodb")
    r_4 = nx.betweenness_centrality(G_2, backend="arangodb")

    assert r_1 and r_2 and r_3 and r_4

def test_pagerank():
    G_1 = nx.karate_club_graph()

    G_2 = nxadb.Graph(G_1)

    r_1 = nx.pagerank(G_1)
    r_2 = nx.pagerank(G_2)
    r_3 = nx.pagerank(G_1, backend="arangodb")
    r_4 = nx.pagerank(G_2, backend="arangodb")

    assert r_1 and r_2 and r_3 and r_4


def test_louvain():
    G_1 = nx.karate_club_graph()

    G_2 = nxadb.Graph(G_1)

    r_1 = nx.community.louvain_communities(G_1)
    r_2 = nx.community.louvain_communities(G_2)
    r_3 = nx.community.louvain_communities(G_1, backend="arangodb")
    r_4 = nx.community.louvain_communities(G_2, backend="arangodb")

    assert r_1 and r_2 and r_3 and r_4
