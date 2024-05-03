import networkx as nx
import pytest

import nx_arangodb as nxadb

from .conftest import db


def test_db():
    assert db.version()


def test_bc():
    G_1 = nx.karate_club_graph()

    G_2 = nxadb.Graph(incoming_graph_data=G_1)

    r_1 = nx.betweenness_centrality(G_1)
    r_2 = nx.betweenness_centrality(G_2)
    r_3 = nx.betweenness_centrality(G_1, backend="arangodb")
    r_4 = nx.betweenness_centrality(G_2, backend="arangodb")

    assert r_1 and r_2 and r_3 and r_4


def test_pagerank():
    G_1 = nx.karate_club_graph()

    G_2 = nxadb.Graph(incoming_graph_data=G_1)

    r_1 = nx.pagerank(G_1)
    r_2 = nx.pagerank(G_2)
    r_3 = nx.pagerank(G_1, backend="arangodb")
    r_4 = nx.pagerank(G_2, backend="arangodb")

    assert r_1 and r_2 and r_3 and r_4


def test_louvain():
    G_1 = nx.karate_club_graph()

    G_2 = nxadb.Graph(incoming_graph_data=G_1)

    r_1 = nx.community.louvain_communities(G_1)
    r_2 = nx.community.louvain_communities(G_2)
    r_3 = nx.community.louvain_communities(G_1, backend="arangodb")
    r_4 = nx.community.louvain_communities(G_2, backend="arangodb")

    assert r_1 and r_2 and r_3 and r_4


def test_crud():
    G_1 = nxadb.Graph(graph_name="KarateGraph", foo="bar")
    G_2 = nx.Graph(nx.karate_club_graph())

    try:
        import phenolrs
        nx.pagerank(G_1, backend="arangodb")
        # TODO: Experiment with algorithm, but turn off CPU pull()!
    except ModuleNotFoundError:
        pass

    #########
    # NODES #
    #########

    G_1.clear()

    assert G_1.graph_name == "KarateGraph"
    assert G_1.graph["foo"] == "bar"
    assert len(G_1.nodes) == len(G_2.nodes)

    person_1 = G_1.nodes["person/1"]
    assert person_1["_key"] == "1"
    assert person_1["_id"] == "person/1"
    assert person_1["club"] == "Mr. Hi"

    assert G_1.nodes["person/2"]["club"]
    assert set(G_1._node.data.keys()) == {"person/1", "person/2"}

    G_1.nodes["person/3"]["club"] = "foo"
    assert db.document("person/3")["club"] == "foo"
    G_1.nodes["person/3"]["club"] = "bar"
    assert db.document("person/3")["club"] == "bar"

    for k in G_1:
        assert G_1.nodes[k] == db.document(k)

    for v in G_1.nodes.values():
        assert v

    G_1.clear()

    assert not G_1._node.data

    for k, v in G_1.nodes.items():
        assert k == v["_id"]

    G_1._node.clear(clear_remote=True)

    assert len(G_1.nodes) == 0

    with pytest.raises(KeyError):
        G_1.nodes["person/1"]

    #########
    # EDGES #
    #########
