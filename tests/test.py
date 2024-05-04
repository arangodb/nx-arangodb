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

def test_bc_no_pull():
    try:
        import phenolrs
    except ModuleNotFoundError:
        pytest.skip("phenolrs not installed")

    G_1 = nxadb.Graph(graph_name="KarateGraph", foo="bar")

    res = nxadb.betweenness_centrality(G_1, pull_graph_on_cpu=False)

    assert len(res) == len(G_1)


def test_pagerank():
    G_1 = nx.karate_club_graph()

    G_2 = nxadb.Graph(incoming_graph_data=G_1)

    r_1 = nx.pagerank(G_1)
    r_2 = nx.pagerank(G_2)
    r_3 = nx.pagerank(G_1, backend="arangodb")
    r_4 = nx.pagerank(G_2, backend="arangodb")

    assert r_1 and r_2 and r_3 and r_4

def test_pagerank_no_pull():
    try:
        import phenolrs
    except ModuleNotFoundError:
        pytest.skip("phenolrs not installed")

    G_1 = nxadb.Graph(graph_name="KarateGraph", foo="bar")

    res = nxadb.pagerank(G_1, pull_graph_on_cpu=False)

    assert len(res) == len(G_1)

def test_louvain():
    G_1 = nx.karate_club_graph()

    G_2 = nxadb.Graph(incoming_graph_data=G_1)

    r_1 = nx.community.louvain_communities(G_1)
    r_2 = nx.community.louvain_communities(G_2)
    r_3 = nx.community.louvain_communities(G_1, backend="arangodb")
    r_4 = nx.community.louvain_communities(G_2, backend="arangodb")

    assert r_1 and r_2 and r_3 and r_4

def test_louvain_no_pull():
    try:
        import phenolrs
    except ModuleNotFoundError:
        pytest.skip("phenolrs not installed")

    G_1 = nxadb.Graph(graph_name="KarateGraph", foo="bar")

    res = nxadb.louvain_communities(G_1, pull_graph_on_cpu=False)

    assert res


def test_crud():
    G_1 = nxadb.Graph(graph_name="KarateGraph", foo="bar")
    G_2 = nx.Graph(nx.karate_club_graph())

    assert G_1.graph_name == "KarateGraph"
    assert G_1.graph["foo"] == "bar"

    #########
    # NODES #
    #########

    assert len(G_1.nodes) == len(G_2.nodes)

    for k, v in G_1.nodes(data=True):
        assert db.document(k) == v

    for k, v in G_1.nodes(data="club"):
        assert db.document(k)["club"] == v

    for k, v in G_1.nodes(data="bad_key", default="boom!"):
        doc = db.document(k)
        assert "bad_key" not in doc
        assert v == "boom!"

    G_1.clear() # clear cache

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

    with pytest.raises(KeyError):
        G_1.nodes["person/unknown"]

    assert G_1.nodes["person/1"]["club"] == "Mr. Hi"
    G_1.add_node("person/1", club="updated value")
    assert G_1.nodes["person/1"]["club"] == "updated value"
    len(G_1.nodes) == len(G_2.nodes)

    G_1.add_node("person/35", foo={'bar': 'baz'})
    len(G_1.nodes) == len(G_2.nodes) + 1
    G_1.clear()
    assert G_1.nodes["person/35"]["foo"] == {'bar': 'baz'}
    # TODO: Support this use case:
    # G_1.nodes["person/35"]["foo"]["bar"] = "baz2"

    G_1.add_nodes_from(['1', '2', '3'], foo='bar')
    G_1.clear()
    assert G_1.nodes["1"]["foo"] == "bar"
    assert G_1.nodes["2"]["foo"] == "bar"
    assert G_1.nodes["3"]["foo"] == "bar"

    assert db.collection(G_1.default_node_type).count() == 3
    assert db.collection(G_1.default_node_type).has("1")
    assert db.collection(G_1.default_node_type).has("2")
    assert db.collection(G_1.default_node_type).has("3")

    #########
    # EDGES #
    #########

    # breakpoint() 

    # assert len(G_1.edges) == len(G_2.edges)
    # assert len(G_1.adj) == len(G_2.adj)


    # breakpoint()