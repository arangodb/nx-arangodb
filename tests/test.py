from typing import Any, Callable

import networkx as nx
import pytest
from arango import DocumentDeleteError

import nx_arangodb as nxadb
from nx_arangodb.classes.dict import EdgeAttrDict, NodeAttrDict

from .conftest import db

G_NX = nx.karate_club_graph()


def create_line_graph(load_attributes: set[str]) -> nxadb.Graph:
    G = nx.Graph()
    G.add_edge(1, 2, my_custom_weight=1)
    G.add_edge(2, 3, my_custom_weight=1)
    G.add_edge(3, 4, my_custom_weight=1000)
    G.add_edge(4, 5, my_custom_weight=1000)

    if load_attributes:
        return nxadb.Graph(
            incoming_graph_data=G,
            graph_name="LineGraph",
            edge_collections_attributes=load_attributes,
        )

    return nxadb.Graph(incoming_graph_data=G, graph_name="LineGraph")


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


def test_load_graph_from_nxadb_w_specific_edge_attribute():
    graph_name = "KarateGraph"

    db.delete_graph(graph_name, drop_collections=True, ignore_missing=True)

    graph = nxadb.Graph(
        graph_name=graph_name,
        incoming_graph_data=G_NX,
        default_node_type="person",
        edge_collections_attributes={"weight"},
    )
    # TODO: re-enable this line as soon as CPU based data caching is implemented
    # graph._adj._fetch_all()

    for _from, adj in graph._adj.items():
        for _to, edge in adj.items():
            assert "weight" in edge
            assert isinstance(edge["weight"], (int, float))

    # call without specifying weight, fallback to weight: 1 for each
    nx.pagerank(graph)

    # call with specifying weight
    nx.pagerank(graph, weight="weight")

    db.delete_graph(graph_name, drop_collections=True)


def test_load_graph_from_nxadb_w_not_available_edge_attribute():
    graph_name = "KarateGraph"

    db.delete_graph(graph_name, drop_collections=True, ignore_missing=True)

    graph = nxadb.Graph(
        graph_name=graph_name,
        incoming_graph_data=G_NX,
        default_node_type="person",
        # This will lead to weight not being loaded into the edge data
        edge_collections_attributes={"_id"},
    )

    # Should just succeed without any errors (fallback to weight: 1 as above)
    nx.pagerank(graph, weight="weight_does_not_exist")

    db.delete_graph(graph_name, drop_collections=True)


def test_load_graph_with_non_default_weight_attribute():
    graph_name = "LineGraph"

    db.delete_graph(graph_name, drop_collections=True, ignore_missing=True)

    graph = create_line_graph(load_attributes={"my_custom_weight"})
    res_custom = nx.pagerank(graph, weight="my_custom_weight")
    res_default = nx.pagerank(graph)

    # to check that the results are different in case of different weights
    # custom specified weights vs. fallback default weight to 1
    assert res_custom != res_default

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

    assert all([r_1, r_2, r_3, r_4])
    assert_func(r_1, r_2)
    assert_func(r_2, r_3)
    assert_func(r_3, r_4)

    try:
        import phenolrs  # noqa
    except ModuleNotFoundError:
        pytest.skip("phenolrs not installed")

    r_7 = algorithm_func(G_3)
    r_7_orig = algorithm_func.orig_func(G_3)  # type: ignore

    r_8 = algorithm_func(G_4)
    r_8_orig = algorithm_func.orig_func(G_4)  # type: ignore

    r_9 = algorithm_func(G_5)
    r_9_orig = algorithm_func.orig_func(G_5)  # type: ignore

    r_10 = algorithm_func(nx.DiGraph(incoming_graph_data=G_NX))

    assert all([r_7, r_7_orig, r_8, r_8_orig, r_9, r_9_orig, r_10])
    assert_func(r_7, r_7_orig)
    assert_func(r_8, r_8_orig)
    assert_func(r_9, r_9_orig)
    assert_func(r_7, r_1)
    assert_func(r_7, r_8)
    assert r_8 != r_9
    assert r_8_orig != r_9_orig
    assert_func(r_8, r_10)
    assert_func(r_8_orig, r_10)


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


@pytest.mark.parametrize(
    "graph_cls",
    [
        (nxadb.Graph),
        (nxadb.DiGraph),
    ],
)
def test_nodes_crud(load_graph: Any, graph_cls: type[nxadb.Graph]) -> None:
    G_1 = graph_cls(graph_name="KarateGraph", foo="bar")
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


def test_digraph_edges_crud(load_graph: Any) -> None:
    G_1 = nxadb.DiGraph(graph_name="KarateGraph")
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
    assert G_1.pred["new_node_2"]["new_node_1"]
    assert "new_node_1" not in G_1.adj["new_node_2"]
    assert (
        G_1.adj["new_node_1"]["new_node_2"]["_id"]
        == G_1.pred["new_node_2"]["new_node_1"]["_id"]
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

    assert "person/1" not in G_1["person/2"]
    assert G_1.succ["person/1"]["person/2"] == G_1.pred["person/2"]["person/1"]
    new_weight = 1000
    G_1["person/1"]["person/2"]["weight"] = new_weight
    assert G_1.succ["person/1"]["person/2"]["weight"] == new_weight
    assert G_1.pred["person/2"]["person/1"]["weight"] == new_weight
    G_1.clear()
    assert G_1.succ["person/1"]["person/2"]["weight"] == new_weight
    assert G_1.pred["person/2"]["person/1"]["weight"] == new_weight

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


def test_graph_dict_init(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    assert db.collection("_graphs").has("KarateGraph")
    graph_document = db.collection("_graphs").get("KarateGraph")
    assert graph_document["_key"] == "KarateGraph"
    assert graph_document["edgeDefinitions"] == [
        {"collection": "knows", "from": ["person"], "to": ["person"]},
        {"collection": "person_to_person", "from": ["person"], "to": ["person"]},
    ]
    assert graph_document["orphanCollections"] == []

    graph_doc_id = G.graph.graph_id
    assert db.has_document(graph_doc_id)


def test_graph_dict_init_extended(load_graph: Any) -> None:
    # Tests that available data (especially dicts) will be properly
    # stored as GraphDicts in the internal cache.
    G = nxadb.Graph(graph_name="KarateGraph", foo="bar", bar={"baz": True})
    G.graph["foo"] = "!!!"
    G.graph["bar"]["baz"] = False
    assert db.document(G.graph.graph_id)["foo"] == "!!!"
    assert db.document(G.graph.graph_id)["bar"]["baz"] is False
    assert "baz" not in db.document(G.graph.graph_id)


def test_graph_dict_clear_will_not_remove_remote_data(load_graph: Any) -> None:
    G_adb = nxadb.Graph(
        graph_name="KarateGraph",
        foo="bar",
        bar={"a": 4},
    )

    G_adb.graph["ant"] = {"b": 5}
    G_adb.graph["ant"]["b"] = 6
    G_adb.clear()
    try:
        G_adb.graph["ant"]
    except KeyError:
        raise AssertionError("Not allowed to fail.")

    assert db.document(G_adb.graph.graph_id)["ant"] == {"b": 6}


def test_graph_dict_set_item(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    try:
        db.collection(G.graph.COLLECTION_NAME).delete(G.name)
    except DocumentDeleteError:
        pass
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

    json_values = [
        "aString",
        1,
        1.0,
        True,
        False,
        {"a": "b"},
        ["a", "b", "c"],
        {"a": "b", "c": ["a", "b", "c"]},
        None,
    ]

    for value in json_values:
        G.graph["json"] = value

        if value is None:
            assert "json" not in db.document(G.graph.graph_id)
        else:
            assert G.graph["json"] == value
            assert db.document(G.graph.graph_id)["json"] == value


def test_graph_dict_update(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    G.clear()

    G.graph["a"] = "b"
    to_update = {"c": "d"}
    G.graph.update(to_update)

    # local
    assert G.graph["a"] == "b"
    assert G.graph["c"] == "d"

    # remote
    adb_doc = db.collection("nxadb_graphs").get(G.graph_name)
    assert adb_doc["a"] == "b"
    assert adb_doc["c"] == "d"


def test_graph_attr_dict_nested_update(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    G.clear()

    G.graph["a"] = {"b": "c"}
    G.graph["a"].update({"d": "e"})
    assert G.graph["a"]["b"] == "c"
    assert G.graph["a"]["d"] == "e"
    assert db.document(G.graph.graph_id)["a"]["b"] == "c"
    assert db.document(G.graph.graph_id)["a"]["d"] == "e"


def test_graph_dict_nested_1(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    G.clear()
    icon = {"football_icon": "MJ7"}

    G.graph["a"] = {"b": icon}
    assert G.graph["a"]["b"] == icon
    assert db.document(G.graph.graph_id)["a"]["b"] == icon


def test_graph_dict_nested_2(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    G.clear()
    icon = {"football_icon": "MJ7"}

    G.graph["x"] = {"y": icon}
    G.graph["x"]["y"]["amount_of_goals"] = 1337

    assert G.graph["x"]["y"]["amount_of_goals"] == 1337
    assert db.document(G.graph.graph_id)["x"]["y"]["amount_of_goals"] == 1337


def test_graph_dict_empty_values(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    G.clear()

    G.graph["empty"] = {}
    assert G.graph["empty"] == {}
    assert db.document(G.graph.graph_id)["empty"] == {}

    G.graph["none"] = None
    assert "none" not in db.document(G.graph.graph_id)
    assert "none" not in G.graph


def test_graph_dict_nested_overwrite(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    G.clear()
    icon1 = {"football_icon": "MJ7"}
    icon2 = {"basketball_icon": "MJ23"}

    G.graph["a"] = {"b": icon1}
    G.graph["a"]["b"]["football_icon"] = "ChangedIcon"
    assert G.graph["a"]["b"]["football_icon"] == "ChangedIcon"
    assert db.document(G.graph.graph_id)["a"]["b"]["football_icon"] == "ChangedIcon"

    # Overwrite entire nested dictionary
    G.graph["a"] = {"b": icon2}
    assert G.graph["a"]["b"]["basketball_icon"] == "MJ23"
    assert db.document(G.graph.graph_id)["a"]["b"]["basketball_icon"] == "MJ23"


def test_graph_dict_complex_nested(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    G.clear()

    complex_structure = {"level1": {"level2": {"level3": {"key": "value"}}}}

    G.graph["complex"] = complex_structure
    assert G.graph["complex"]["level1"]["level2"]["level3"]["key"] == "value"
    assert (
        db.document(G.graph.graph_id)["complex"]["level1"]["level2"]["level3"]["key"]
        == "value"
    )


def test_graph_dict_nested_deletion(load_graph: Any) -> None:
    G = nxadb.Graph(graph_name="KarateGraph", default_node_type="person")
    G.clear()
    icon = {"football_icon": "MJ7", "amount_of_goals": 1337}

    G.graph["x"] = {"y": icon}
    del G.graph["x"]["y"]["amount_of_goals"]
    assert "amount_of_goals" not in G.graph["x"]["y"]
    assert "amount_of_goals" not in db.document(G.graph.graph_id)["x"]["y"]

    # Delete top-level key
    del G.graph["x"]
    assert "x" not in G.graph
    assert "x" not in db.document(G.graph.graph_id)


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
