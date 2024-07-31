from typing import Any, Callable

import networkx as nx
import pytest

import nx_arangodb as nxadb
from nx_arangodb.classes.dict import EdgeAttrDict, NodeAttrDict

from .conftest import db

G_NX = nx.karate_club_graph()


def assert_same_dict_values(
    d1: dict[str | int, float], d2: dict[str | int, float], digit: int
) -> None:
    if type(next(iter(d1.keys()))) == int:
        d1 = {f"person/{k+1}": v for k, v in d1.items()}  # type: ignore

    if type(next(iter(d2.keys()))) == int:
        d2 = {f"person/{k+1}": v for k, v in d2.items()}  # type: ignore

    assert d1.keys() == d2.keys(), "Dictionaries have different keys"
    for key in d1:
        m = f"Values for key '{key}' are not equal up to digit {digit}"
        assert round(d1[key], digit) == round(d2[key], digit), m


def assert_bc(d1: dict[str | int, float], d2: dict[str | int, float]) -> None:
    assert_same_dict_values(d1, d2, 14)


def assert_pagerank(d1: dict[str | int, float], d2: dict[str | int, float]) -> None:
    assert_same_dict_values(d1, d2, 15)


def assert_louvain(l1: list[set[Any]], l2: list[set[Any]]) -> None:
    # TODO: Implement some kind of comparison
    # Reason: Louvain returns different results on different runs
    pass


def assert_k_components(
    d1: dict[int, list[set[Any]]], d2: dict[int, list[set[Any]]]
) -> None:
    assert d1.keys() == d2.keys(), "Dictionaries have different keys"
    assert d1 == d2


def test_db(load_graph: Any) -> None:
    assert db.version()


def test_load_graph_from_nxadb():
    graph_name = "KarateGraph"

    db.delete_graph(graph_name, drop_collections=True, ignore_missing=True)

    _ = nxadb.Graph(
        graph_name=graph_name,
        incoming_graph_data=G_NX,
        default_node_type="person",
    )

    assert db.has_graph(graph_name)
    assert db.has_collection("person")
    assert db.has_collection("person_to_person")
    assert db.collection("person").count() == len(G_NX.nodes)
    assert db.collection("person_to_person").count() == len(G_NX.edges)

    db.delete_graph(graph_name, drop_collections=True)


@pytest.mark.parametrize(
    "algorithm_func, assert_func",
    [
        (nx.betweenness_centrality, assert_bc),
        (nx.pagerank, assert_pagerank),
        (nx.community.louvain_communities, assert_louvain),
    ],
)
def test_algorithm(
    algorithm_func: Callable[..., Any],
    assert_func: Callable[..., Any],
    load_graph: Any,
) -> None:
    G_1 = G_NX
    G_2 = nxadb.Graph(incoming_graph_data=G_1)
    G_3 = nxadb.Graph(graph_name="KarateGraph")
    G_4 = nxadb.DiGraph(graph_name="KarateGraph", symmetrize_edges=True)
    G_5 = nxadb.DiGraph(graph_name="KarateGraph", symmetrize_edges=False)

    r_1 = algorithm_func(G_1)
    r_2 = algorithm_func(G_2)
    r_3 = algorithm_func(G_1, backend="arangodb")
    r_4 = algorithm_func(G_2, backend="arangodb")

    r_5 = algorithm_func.orig_func(G_3)  # type: ignore
    nx.config.backends.arangodb.pull_graph = False
    r_6 = algorithm_func(G_3)
    nx.config.backends.arangodb.pull_graph = True

    assert all([r_1, r_2, r_3, r_4, r_5, r_6])
    assert_func(r_1, r_2)
    assert_func(r_2, r_3)
    assert_func(r_3, r_4)
    assert_func(r_5, r_6)

    try:
        import phenolrs  # noqa
    except ModuleNotFoundError:
        pytest.skip("phenolrs not installed")

    r_7 = algorithm_func(G_3)
    r_8 = algorithm_func(G_4)
    r_9 = algorithm_func(G_5)
    r_10 = algorithm_func(nx.DiGraph(incoming_graph_data=G_NX))

    assert all([r_7, r_8, r_9, r_10])
    assert_func(r_7, r_1)
    assert_func(r_7, r_8)
    assert len(r_8) == len(r_9)
    assert r_8 != r_9
    assert_func(r_8, r_10)


def test_shortest_path_remote_algorithm(load_graph: Any) -> None:
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
    G_2 = nx.Graph(G_NX)

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

    G_1.nodes["person/2"]["object"] = {"foo": "bar", "bar": "foo"}
    assert "_rev" not in G_1.nodes["person/2"]["object"]
    assert isinstance(G_1.nodes["person/2"]["object"], NodeAttrDict)
    assert db.document("person/2")["object"] == {"foo": "bar", "bar": "foo"}

    G_1.nodes["person/2"]["object"]["foo"] = "baz"
    assert db.document("person/2")["object"]["foo"] == "baz"

    del G_1.nodes["person/2"]["object"]["foo"]
    assert "_rev" not in G_1.nodes["person/2"]["object"]
    assert isinstance(G_1.nodes["person/2"]["object"], NodeAttrDict)
    assert "foo" not in db.document("person/2")["object"]

    G_1.nodes["person/2"]["object"].update({"sub_object": {"foo": "bar"}})
    assert "_rev" not in G_1.nodes["person/2"]["object"]["sub_object"]
    assert isinstance(G_1.nodes["person/2"]["object"]["sub_object"], NodeAttrDict)
    assert db.document("person/2")["object"]["sub_object"]["foo"] == "bar"

    G_1.clear()

    assert G_1.nodes["person/2"]["object"]["sub_object"]["foo"] == "bar"
    G_1.nodes["person/2"]["object"]["sub_object"]["foo"] = "baz"
    assert "_rev" not in G_1.nodes["person/2"]["object"]["sub_object"]
    assert db.document("person/2")["object"]["sub_object"]["foo"] == "baz"


def test_graph_edges_crud(load_graph: Any) -> None:
    G_1 = nxadb.Graph(graph_name="KarateGraph")
    G_2 = G_NX

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
    assert db.document(G_1["new_node_1"]["new_node_2"]["_id"])["foo"] == "bar"
    G_1.add_edge("new_node_1", "new_node_2", foo="bar", bar="foo")
    doc = db.document(G_1["new_node_1"]["new_node_2"]["_id"])
    assert doc["foo"] == "bar"
    assert doc["bar"] == "foo"

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

    edge_id = G_1["person/1"]["person/2"]["_id"]
    G_1["person/1"]["person/2"]["object"] = {"foo": "bar", "bar": "foo"}
    assert "_rev" not in G_1["person/1"]["person/2"]["object"]
    assert isinstance(G_1["person/1"]["person/2"]["object"], EdgeAttrDict)
    assert db.document(edge_id)["object"] == {"foo": "bar", "bar": "foo"}

    G_1["person/1"]["person/2"]["object"]["foo"] = "baz"
    assert db.document(edge_id)["object"]["foo"] == "baz"

    del G_1["person/1"]["person/2"]["object"]["foo"]
    assert "_rev" not in G_1["person/1"]["person/2"]["object"]
    assert isinstance(G_1["person/1"]["person/2"]["object"], EdgeAttrDict)
    assert "foo" not in db.document(edge_id)["object"]

    G_1["person/1"]["person/2"]["object"].update({"sub_object": {"foo": "bar"}})
    assert "_rev" not in G_1["person/1"]["person/2"]["object"]["sub_object"]
    assert isinstance(G_1["person/1"]["person/2"]["object"]["sub_object"], EdgeAttrDict)
    assert db.document(edge_id)["object"]["sub_object"]["foo"] == "bar"

    G_1.clear()

    assert G_1["person/1"]["person/2"]["object"]["sub_object"]["foo"] == "bar"
    G_1["person/1"]["person/2"]["object"]["sub_object"]["foo"] = "baz"
    assert "_rev" not in G_1["person/1"]["person/2"]["object"]["sub_object"]
    assert db.document(edge_id)["object"]["sub_object"]["foo"] == "baz"


def test_readme(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")

    assert len(G.nodes) == len(G_NX.nodes)
    assert len(G.adj) == len(G_NX.adj)
    assert len(G.edges) == len(G_NX.edges)

    G.nodes(data="club", default="unknown")
    G.edges(data="weight", default=1000)

    G.nodes["person/1"]
    G.adj["person/1"]
    G.edges[("person/1", "person/3")]

    assert G.nodes["1"] == G.nodes["person/1"] == G.nodes[1]

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

    assert len(G.nodes) == len(G_NX.nodes)
    assert len(G.adj) == len(G_NX.adj)
    assert len(G.edges) == len(G_NX.edges)


@pytest.mark.parametrize(
    "data_type, incoming_graph_data, has_club, has_weight",
    [
        ("dict of dicts", nx.karate_club_graph()._adj, False, True),
        (
            "dict of lists",
            {k: list(v) for k, v in G_NX._adj.items()},
            False,
            False,
        ),
        ("container of edges", list(G_NX.edges), False, False),
        ("iterator of edges", iter(G_NX.edges), False, False),
        ("generator of edges", (e for e in G_NX.edges), False, False),
        ("2D numpy array", nx.to_numpy_array(G_NX), False, True),
        (
            "scipy sparse array",
            nx.to_scipy_sparse_array(G_NX),
            False,
            True,
        ),
        ("Pandas EdgeList", nx.to_pandas_edgelist(G_NX), False, True),
        ("Pandas Adjacency", nx.to_pandas_adjacency(G_NX), False, True),
    ],
)
def test_incoming_graph_data_not_nx_graph(
    data_type: str, incoming_graph_data: Any, has_club: bool, has_weight: bool
) -> None:
    # See nx.convert.to_networkx_graph for the official supported types
    name = "KarateGraph"
    db.delete_graph(name, drop_collections=True, ignore_missing=True)

    G = nxadb.Graph(incoming_graph_data=incoming_graph_data, graph_name=name)

    assert len(G.adj) == len(G_NX.adj) == db.collection(G.default_node_type).count()
    assert (
        len(G.nodes)
        == len(G_NX.nodes)
        == db.collection(G.default_node_type).count()
        == G.number_of_nodes()
    )
    assert (
        len(G.edges)
        == len(G_NX.edges)
        == db.collection(G.default_edge_type).count()
        == G.number_of_edges()
    )
    assert has_club == ("club" in G.nodes["0"])
    assert has_weight == ("weight" in G.adj["0"]["1"])


def test_digraph_nodes_crud() -> None:
    graph_name = "digraph"
    db.delete_graph(graph_name, drop_collections=True, ignore_missing=True)
    G = nxadb.DiGraph(graph_name=graph_name, default_node_type="dinode")

    G.add_node(1, foo="bar")
    G.add_nodes_from([2, 3, 4], bar="foo")
    G.add_edge(1, 2, weight=1)
    G.add_edges_from([(2, 3), (3, 4), (4, 1)], weight=5)

    assert db.collection("dinode").count() == 4
    assert db.collection("dinode_to_dinode").count() == 4

    G.remove_node(1)
    assert db.collection("dinode").count() == 3
    assert db.collection("dinode_to_dinode").count() == 2

    G.remove_edge(2, 3)
    assert db.collection("dinode_to_dinode").count() == 1

    G.remove_edges_from([(3, 4)])
    assert db.collection("dinode_to_dinode").count() == 0
    assert db.collection("dinode").count() == 3

    G.remove_nodes_from([2, 3, 4])
    assert db.collection("dinode").count() == 0


def test_digraph_edges_crud() -> None:
    pytest.skip("Not implemented yet")
