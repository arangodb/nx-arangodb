from typing import Any

import networkx as nx
import pytest

import nx_arangodb as nxadb

from .conftest import db


def test_db(load_graph: Any) -> None:
    assert db.version()


def test_load_graph_from_nxadb():
    graph_name = "KarateGraph"

    db.delete_graph(graph_name, drop_collections=True, ignore_missing=True)

    G_nx = nx.karate_club_graph()

    _ = nxadb.Graph(
        graph_name=graph_name,
        incoming_graph_data=G_nx,
        default_node_type="person",
    )

    assert db.has_graph(graph_name)
    assert db.has_collection("person")
    assert db.has_collection("person_to_person")
    assert db.collection("person").count() == len(G_nx.nodes)
    assert db.collection("person_to_person").count() == len(G_nx.edges)

    db.delete_graph(graph_name, drop_collections=True)


def test_bc(load_graph):
    G_1 = nx.karate_club_graph()
    G_2 = nxadb.Graph(incoming_graph_data=G_1)
    G_3 = nxadb.Graph(graph_name="KarateGraph")

    r_1 = nx.betweenness_centrality(G_1)
    r_2 = nx.betweenness_centrality(G_2)
    r_3 = nx.betweenness_centrality(G_1, backend="arangodb")
    r_4 = nx.betweenness_centrality(G_2, backend="arangodb")
    r_5 = nx.betweenness_centrality.orig_func(G_3)

    assert len(r_1) == len(G_1)
    assert r_1 == r_2
    assert r_2 == r_3
    assert r_3 == r_4
    assert len(r_1) == len(r_5)

    try:
        import phenolrs  # noqa
    except ModuleNotFoundError:
        pytest.skip("phenolrs not installed")

    G_4 = nxadb.Graph(graph_name="KarateGraph")
    r_6 = nx.betweenness_centrality(G_4)

    G_5 = nxadb.Graph(graph_name="KarateGraph")
    r_7 = nxadb.betweenness_centrality(G_5, pull_graph_on_cpu=False)

    G_6 = nxadb.DiGraph(graph_name="KarateGraph")
    r_8 = nx.betweenness_centrality(G_6)

    # assert r_6 == r_7 # this is acting strange. I need to revisit
    assert r_7 == r_8
    assert len(r_6) == len(r_7) == len(r_8) == len(G_4) > 0


def test_pagerank(load_graph: Any) -> None:
    G_1 = nx.karate_club_graph()
    G_2 = nxadb.Graph(incoming_graph_data=G_1)
    G_3 = nxadb.Graph(graph_name="KarateGraph")

    r_1 = nx.pagerank(G_1)
    r_2 = nx.pagerank(G_2)
    r_3 = nx.pagerank(G_1, backend="arangodb")
    r_4 = nx.pagerank(G_2, backend="arangodb")
    r_5 = nx.pagerank.orig_func(G_3)

    assert len(r_1) == len(G_1)
    assert r_1 == r_2
    assert r_2 == r_3
    assert r_3 == r_4
    assert len(r_1) == len(r_5)

    try:
        import phenolrs  # noqa
    except ModuleNotFoundError:
        pytest.skip("phenolrs not installed")

    G_4 = nxadb.Graph(graph_name="KarateGraph")
    r_6 = nx.pagerank(G_4)

    G_5 = nxadb.Graph(graph_name="KarateGraph")
    r_7 = nxadb.pagerank(G_5, pull_graph_on_cpu=False)

    G_6 = nxadb.DiGraph(graph_name="KarateGraph")
    r_8 = nx.pagerank(G_6)

    assert len(r_6) == len(r_7) == len(r_8) == len(G_4) > 0


def test_louvain(load_graph: Any) -> None:
    G_1 = nx.karate_club_graph()
    G_2 = nxadb.Graph(incoming_graph_data=G_1)
    G_3 = nxadb.Graph(graph_name="KarateGraph")

    r_1 = nx.community.louvain_communities(G_1)
    r_2 = nx.community.louvain_communities(G_2)
    r_3 = nx.community.louvain_communities(G_1, backend="arangodb")
    r_4 = nx.community.louvain_communities(G_2, backend="arangodb")
    r_5 = nx.community.louvain_communities.orig_func(G_3)

    assert len(r_1) > 0
    assert len(r_2) > 0
    assert len(r_3) > 0
    assert len(r_4) > 0
    assert len(r_5) > 0

    try:
        import phenolrs  # noqa
    except ModuleNotFoundError:
        pytest.skip("phenolrs not installed")

    G_4 = nxadb.Graph(graph_name="KarateGraph")
    r_6 = nx.community.louvain_communities(G_4)

    G_5 = nxadb.Graph(graph_name="KarateGraph")
    r_7 = nxadb.community.louvain_communities(G_5, pull_graph_on_cpu=False)

    G_6 = nxadb.DiGraph(graph_name="KarateGraph")
    r_8 = nx.community.louvain_communities(G_6)

    assert len(r_5) > 0
    assert len(r_6) > 0
    assert len(r_7) > 0
    assert len(r_8) > 0


def test_shortest_path(load_graph: Any) -> None:
    G_1 = nxadb.Graph(graph_name="KarateGraph")
    G_2 = nxadb.DiGraph(graph_name="KarateGraph")

    r_1 = nx.shortest_path(G_1, source="person/1", target="person/34")
    r_2 = nx.shortest_path(G_1, source="person/1", target="person/34", weight="weight")
    r_3 = nx.shortest_path(G_2, source="person/1", target="person/34")
    r_4 = nx.shortest_path(G_2, source="person/1", target="person/34", weight="weight")

    assert r_1 == r_3
    assert r_2 == r_4
    assert r_1 != r_2
    assert r_3 != r_4


def test_graph_nodes_crud(load_graph: Any) -> None:
    G_1 = nxadb.Graph(graph_name="KarateGraph", foo="bar")
    G_2 = nx.Graph(nx.karate_club_graph())

    assert G_1.graph_name == "KarateGraph"
    assert G_1.graph["foo"] == "bar"

    assert len(G_1.nodes) == len(G_2.nodes)

    for k, v in G_1.nodes(data=True):
        assert db.document(k) == v

    for k, v in G_1.nodes(data="club"):
        assert db.document(k)["club"] == v

    for k, v in G_1.nodes(data="bad_key", default="boom!"):
        doc = db.document(k)
        assert "bad_key" not in doc
        assert v == "boom!"

    G_1.clear()  # clear cache

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

    G_1.add_node("person/35", foo={"bar": "baz"})
    len(G_1.nodes) == len(G_2.nodes) + 1
    G_1.clear()
    assert G_1.nodes["person/35"]["foo"] == {"bar": "baz"}
    # TODO: Support this use case:
    # G_1.nodes["person/35"]["foo"]["bar"] = "baz2"

    G_1.add_nodes_from(["1", "2", "3"], foo="bar")
    G_1.clear()
    assert G_1.nodes["1"]["foo"] == "bar"
    assert G_1.nodes["2"]["foo"] == "bar"
    assert G_1.nodes["3"]["foo"] == "bar"

    assert db.collection(G_1.default_node_type).count() == 3
    assert db.collection(G_1.default_node_type).has("1")
    assert db.collection(G_1.default_node_type).has("2")
    assert db.collection(G_1.default_node_type).has("3")

    G_1.remove_node("1")
    assert not db.collection(G_1.default_node_type).has("1")
    with pytest.raises(KeyError):
        G_1.nodes["1"]

    with pytest.raises(KeyError):
        G_1.adj["1"]

    G_1.remove_nodes_from(["2", "3"])
    assert not db.collection(G_1.default_node_type).has("2")
    assert not db.collection(G_1.default_node_type).has("3")

    with pytest.raises(KeyError):
        G_1.nodes["2"]

    with pytest.raises(KeyError):
        G_1.adj["2"]

    assert len(G_1.adj["person/1"]) > 0
    assert G_1.adj["person/1"]["person/2"]
    edge_id = G_1.adj["person/1"]["person/2"]["_id"]
    G_1.remove_node("person/1")
    assert not db.has_document("person/1")
    assert not db.has_document(edge_id)


def test_graph_edges_crud(load_graph: Any) -> None:
    G_1 = nxadb.Graph(graph_name="KarateGraph")
    G_2 = nx.karate_club_graph()

    assert len(G_1.adj) == len(G_2.adj)
    assert len(G_1.edges) == len(G_2.edges)

    for src, dst, w in G_1.edges.data("weight"):
        assert G_1.adj[src][dst]["weight"] == w

    for src, dst, w in G_1.edges.data("bad_key", default="boom!"):
        assert "bad_key" not in G_1.adj[src][dst]
        assert w == "boom!"

    for k, edge in G_1.adj["person/1"].items():
        assert db.has_document(k)
        assert db.has_document(edge["_id"])

    G_1.add_edge("person/1", "person/1", foo="bar", _edge_type="knows")
    edge_id = G_1.adj["person/1"]["person/1"]["_id"]
    doc = db.document(edge_id)
    assert doc["foo"] == "bar"
    assert G_1.adj["person/1"]["person/1"]["foo"] == "bar"

    del G_1.adj["person/1"]["person/1"]["foo"]
    doc = db.document(edge_id)
    assert "foo" not in doc

    G_1.adj["person/1"]["person/1"].update({"bar": "foo"})
    doc = db.document(edge_id)
    assert doc["bar"] == "foo"

    assert len(G_1.adj["person/1"]["person/1"]) == len(doc)
    adj_count = len(G_1.adj["person/1"])
    G_1.remove_edge("person/1", "person/1")
    assert len(G_1.adj["person/1"]) == adj_count - 1
    assert not db.has_document(edge_id)
    assert "person/1" in G_1

    assert not db.has_document(f"{G_1.default_node_type}/new_node_1")
    col_count = db.collection(G_1.default_edge_type).count()

    G_1.add_edge("new_node_1", "new_node_2", foo="bar")
    G_1.add_edge("new_node_1", "new_node_2", foo="bar", bar="foo")

    bind_vars = {
        "src": f"{G_1.default_node_type}/new_node_1",
        "dst": f"{G_1.default_node_type}/new_node_2",
    }

    result = list(
        db.aql.execute(
            f"FOR e IN {G_1.default_edge_type} FILTER e._from == @src AND e._to == @dst RETURN e",  # noqa
            bind_vars=bind_vars,
        )
    )

    assert len(result) == 1

    result = list(
        db.aql.execute(
            f"FOR e IN {G_1.default_edge_type} FILTER e._from == @dst AND e._to == @src RETURN e",  # noqa
            bind_vars=bind_vars,
        )
    )

    assert len(result) == 0

    assert db.collection(G_1.default_edge_type).count() == col_count + 1
    assert G_1.adj["new_node_1"]["new_node_2"]
    assert G_1.adj["new_node_1"]["new_node_2"]["foo"] == "bar"
    assert G_1.adj["new_node_2"]["new_node_1"]
    assert (
        G_1.adj["new_node_2"]["new_node_1"]["_id"]
        == G_1.adj["new_node_1"]["new_node_2"]["_id"]
    )
    edge_id = G_1.adj["new_node_1"]["new_node_2"]["_id"]
    doc = db.document(edge_id)
    assert db.has_document(doc["_from"])
    assert db.has_document(doc["_to"])
    assert G_1.nodes["new_node_1"]
    assert G_1.nodes["new_node_2"]

    G_1.remove_edge("new_node_1", "new_node_2")
    G_1.clear()
    assert "new_node_1" in G_1
    assert "new_node_2" in G_1
    assert "new_node_2" not in G_1.adj["new_node_1"]

    G_1.add_edges_from(
        [("new_node_1", "new_node_2"), ("new_node_1", "new_node_3")], foo="bar"
    )
    G_1.clear()
    assert "new_node_1" in G_1
    assert "new_node_2" in G_1
    assert "new_node_3" in G_1
    assert G_1.adj["new_node_1"]["new_node_2"]["foo"] == "bar"
    assert G_1.adj["new_node_1"]["new_node_3"]["foo"] == "bar"

    G_1.remove_edges_from([("new_node_1", "new_node_2"), ("new_node_1", "new_node_3")])
    assert "new_node_1" in G_1
    assert "new_node_2" in G_1
    assert "new_node_3" in G_1
    assert "new_node_2" not in G_1.adj["new_node_1"]
    assert "new_node_3" not in G_1.adj["new_node_1"]

    assert G_1["person/1"]["person/2"] == G_1["person/2"]["person/1"]
    new_weight = 1000
    G_1["person/1"]["person/2"]["weight"] = new_weight
    assert G_1["person/1"]["person/2"]["weight"] == new_weight
    assert G_1["person/2"]["person/1"]["weight"] == new_weight
    G_1.clear()
    assert G_1["person/1"]["person/2"]["weight"] == new_weight
    assert G_1["person/2"]["person/1"]["weight"] == new_weight


def test_readme(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph")
    G_nx = nx.karate_club_graph()

    assert len(G.nodes) == len(G_nx.nodes)
    assert len(G.adj) == len(G_nx.adj)
    assert len(G.edges) == len(G_nx.edges)

    G.nodes(data="club", default="unknown")
    G.edges(data="weight", default=1000)

    G.nodes["person/1"]
    G.adj["person/1"]
    G.edges[("person/1", "person/3")]

    G.nodes["person/1"]["name"] = "John Doe"
    G.nodes["person/1"].update({"age": 40})
    del G.nodes["person/1"]["name"]

    G.adj["person/1"]["person/3"]["weight"] = 2
    G.adj["person/1"]["person/3"].update({"weight": 3})
    del G.adj["person/1"]["person/3"]["weight"]

    G.edges[("person/1", "person/3")]["weight"] = 0.5
    assert G.adj["person/1"]["person/3"]["weight"] == 0.5

    G.add_node("person/35", name="Jane Doe")
    G.add_nodes_from(
        [("person/36", {"name": "Jack Doe"}), ("person/37", {"name": "Jill Doe"})]
    )
    G.add_edge("person/1", "person/35", weight=1.5, _edge_type="knows")
    G.add_edges_from(
        [
            ("person/1", "person/36", {"weight": 2}),
            ("person/1", "person/37", {"weight": 3}),
        ],
        _edge_type="knows",
    )

    G.remove_edge("person/1", "person/35")
    G.remove_edges_from([("person/1", "person/36"), ("person/1", "person/37")])
    G.remove_node("person/35")
    G.remove_nodes_from(["person/36", "person/37"])

    G.clear()

    assert len(G.nodes) == len(G_nx.nodes)
    assert len(G.adj) == len(G_nx.adj)
    assert len(G.edges) == len(G_nx.edges)


def test_digraph_nodes_crud() -> None:
    pytest.skip("Not implemented yet")


def test_digraph_edges_crud() -> None:
    pytest.skip("Not implemented yet")
