# type: ignore

import time
from collections import UserDict

import networkx as nx
import pytest
from networkx.utils import edges_equal

import nx_arangodb as nxadb
from nx_arangodb.classes.dict.adj import EdgeAttrDict, EdgeKeyDict

from .conftest import db
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
from .test_graph import get_doc

GRAPH_NAME = "test_graph"


class BaseMultiGraphTester(BaseAttrGraphTester):
    def test_has_edge(self):
        G = self.K3Graph()
        assert G.has_edge(0, 1)
        assert not G.has_edge(0, -1)
        assert G.has_edge(0, 1, 0)
        assert not G.has_edge(0, 1, 1)

    def test_get_edge_data(self):
        G = self.K3Graph()
        edge_id = "test_graph_node_to_test_graph_node/0"
        edge = get_doc(edge_id)
        # edge is not cached, int key not supported
        assert G.get_edge_data(0, 1, 0) is None
        assert G.get_edge_data(0, 1) == {edge_id: edge}
        assert G.get_edge_data(0, 1, edge_id) == edge
        assert G[0][1] == {edge_id: edge}
        assert G[0][1][0] == edge
        assert G.get_edge_data(10, 20) is None
        # edge is cached, int key supported
        assert G.get_edge_data(0, 1, 0) == edge

    def test_adjacency(self):
        G = self.K3Graph()

        edge_0_1_id = "test_graph_node_to_test_graph_node/0"
        edge_0_1 = get_doc(edge_0_1_id)
        edge_0_2_id = "test_graph_node_to_test_graph_node/1"
        edge_0_2 = get_doc(edge_0_2_id)
        edge_1_2_id = "test_graph_node_to_test_graph_node/2"
        edge_1_2 = get_doc(edge_1_2_id)

        assert dict(G.adjacency()) == {
            "test_graph_node/0": {
                "test_graph_node/1": {edge_0_1_id: edge_0_1},
                "test_graph_node/2": {edge_0_2_id: edge_0_2},
            },
            "test_graph_node/1": {
                "test_graph_node/0": {edge_0_1_id: edge_0_1},
                "test_graph_node/2": {edge_1_2_id: edge_1_2},
            },
            "test_graph_node/2": {
                "test_graph_node/0": {edge_0_2_id: edge_0_2},
                "test_graph_node/1": {edge_1_2_id: edge_1_2},
            },
        }

    def deepcopy_edge_attr(self, H, G):
        assert G[1][2][0]["foo"] == H[1][2][0]["foo"]
        G[1][2][0]["foo"].append(1)
        assert G[1][2][0]["foo"] != H[1][2][0]["foo"]

    def shallow_copy_edge_attr(self, H, G):
        assert G[1][2][0]["foo"] == H[1][2][0]["foo"]
        G[1][2][0]["foo"].append(1)
        assert G[1][2][0]["foo"] == H[1][2][0]["foo"]

    def graphs_equal(self, H, G):
        assert G._adj == H._adj
        assert G._node == H._node
        assert G.graph == H.graph
        assert G.name == H.name
        if not G.is_directed() and not H.is_directed():
            assert H._adj[1][2][0] is H._adj[2][1][0]
            assert G._adj[1][2][0] is G._adj[2][1][0]
        else:  # at least one is directed
            if not G.is_directed():
                G._pred = G._adj
                G._succ = G._adj
            if not H.is_directed():
                H._pred = H._adj
                H._succ = H._adj
            assert G._pred == H._pred
            assert G._succ == H._succ
            assert H._succ[1][2][0] is H._pred[2][1][0]
            assert G._succ[1][2][0] is G._pred[2][1][0]

    def same_attrdict(self, H, G):
        # same attrdict in the edgedata
        old_foo = H[1][2][0]["foo"]
        H.adj[1][2][0]["foo"] = "baz"
        assert G._adj == H._adj
        H.adj[1][2][0]["foo"] = old_foo
        assert G._adj == H._adj

        old_foo = H.nodes[0]["foo"]
        H.nodes[0]["foo"] = "baz"
        assert G._node == H._node
        H.nodes[0]["foo"] = old_foo
        assert G._node == H._node

    def different_attrdict(self, H, G):
        # used by graph_equal_but_different
        old_foo = H[1][2][0]["foo"]
        H.adj[1][2][0]["foo"] = "baz"
        assert G._adj != H._adj
        H.adj[1][2][0]["foo"] = old_foo
        assert G._adj == H._adj

        old_foo = H.nodes[0]["foo"]
        H.nodes[0]["foo"] = "baz"
        assert G._node != H._node
        H.nodes[0]["foo"] = old_foo
        assert G._node == H._node

    def test_to_undirected(self):
        pytest.skip("TODO: Revisit graph_equals & copy")

        G = self.K3Graph()
        self.add_attributes(G)
        H = nx.MultiGraph(G)
        self.is_shallow_copy(H, G)
        H = G.to_undirected()
        self.is_deepcopy(H, G)

    def test_to_directed(self):
        pytest.skip("TODO: Revisit graph_equals & copy")

        G = self.K3
        self.add_attributes(G)
        H = nx.MultiDiGraph(G)
        self.is_shallow_copy(H, G)
        H = G.to_directed()
        self.is_deepcopy(H, G)

    def test_number_of_edges_selfloops(self):
        G = self.EmptyGraph()
        G.add_edge(0, 0)
        G.add_edge(0, 0)
        edge_id = G.add_edge(0, 0)

        assert G.number_of_edges() == 3
        assert db.has_document(edge_id)
        G.remove_edge(0, 0, edge_id)
        assert G.number_of_edges() == 2
        assert not db.has_document(edge_id)

        assert G.number_of_edges(0, 0) == 2
        G.remove_edge(0, 0)
        assert G.number_of_edges(0, 0) == 1

    def test_edge_lookup(self):
        G = self.EmptyGraph()
        edge_a_id = G.add_edge(1, 2, foo="bar")
        edge_b_id = G.add_edge(1, 2, "key", foo="biz")
        edge_a_doc = get_doc(edge_a_id)
        edge_b_doc = get_doc(edge_b_id)
        assert edge_a_doc["foo"] == "bar"
        assert edge_b_doc["foo"] == "biz"
        assert edges_equal(G.edges[1, 2, 0], edge_a_doc)
        assert edges_equal(G.edges[1, 2, edge_a_id], edge_a_doc)
        assert edges_equal(G.edges[1, 2, 0], edge_b_doc)
        assert edges_equal(G.edges[1, 2, edge_b_id], edge_b_doc)

    def test_edge_attr(self):
        G = self.EmptyGraph()
        edge_a = G.add_edge(1, 2, key="k1", foo="bar")
        edge_b = G.add_edge(1, 2, key="k2", foo="baz")
        assert get_doc(edge_a)["foo"] == "bar"
        assert get_doc(edge_b)["foo"] == "baz"
        assert isinstance(G.get_edge_data(1, 2), EdgeKeyDict)
        assert all(isinstance(d, EdgeAttrDict) for u, v, d in G.edges(data=True))
        assert edges_equal(
            G.edges(keys=True, data=True),
            [
                ("test_graph_node/1", "test_graph_node/2", edge_a, get_doc(edge_a)),
                ("test_graph_node/1", "test_graph_node/2", edge_b, get_doc(edge_b)),
            ],
        )
        assert edges_equal(
            G.edges(keys=True, data="foo"),
            [
                ("test_graph_node/1", "test_graph_node/2", edge_a, "bar"),
                ("test_graph_node/1", "test_graph_node/2", edge_b, "baz"),
            ],
        )

    def test_edge_attr4(self):
        G = self.EmptyGraph()
        edge_a = G.add_edge(1, 2, key=0, data=7, spam="bar", bar="foo")
        edge_a_doc = get_doc(edge_a)
        assert edge_a_doc["data"] == 7
        assert edge_a_doc["spam"] == "bar"
        assert edge_a_doc["bar"] == "foo"
        assert edges_equal(
            G.edges(data=True),
            [("test_graph_node/1", "test_graph_node/2", get_doc(edge_a))],
        )
        G[1][2][0]["data"] = 10  # OK to set data like this
        assert get_doc(edge_a)["data"] == 10
        assert edges_equal(
            G.edges(data=True),
            [("test_graph_node/1", "test_graph_node/2", get_doc(edge_a))],
        )

        G.adj[1][2][0]["data"] += 20
        assert get_doc(edge_a)["data"] == 30
        assert edges_equal(
            G.edges(data=True),
            [("test_graph_node/1", "test_graph_node/2", get_doc(edge_a))],
        )

        G.edges[1, 2, 0]["data"] = 21  # another spelling, "edge"
        assert get_doc(edge_a)["data"] == 21
        assert edges_equal(
            G.edges(data=True),
            [("test_graph_node/1", "test_graph_node/2", get_doc(edge_a))],
        )
        G.adj[1][2][0]["listdata"] = [20, 200]
        G.adj[1][2][0]["weight"] = 20
        edge_a_doc = get_doc(edge_a)
        assert edge_a_doc["listdata"] == [20, 200]
        assert edge_a_doc["weight"] == 20
        assert edge_a_doc["data"] == 21
        assert edge_a_doc["spam"] == "bar"
        assert edge_a_doc["bar"] == "foo"
        assert edges_equal(
            G.edges(data=True),
            [
                (
                    "test_graph_node/1",
                    "test_graph_node/2",
                    edge_a_doc,
                )
            ],
        )


class TestMultiGraph(BaseMultiGraphTester, _TestGraph):
    def setup_method(self):
        self.Graph = nx.MultiGraph
        # build K3
        ed1, ed2, ed3 = ({0: {}}, {0: {}}, {0: {}})
        self.k3adj = {0: {1: ed1, 2: ed2}, 1: {0: ed1, 2: ed3}, 2: {0: ed2, 1: ed3}}
        self.k3nodes = ["test_graph_node/0", "test_graph_node/1", "test_graph_node/2"]
        self.K3 = self.Graph()
        self.K3._adj = self.k3adj
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}

        def nxadb_graph_constructor(*args, **kwargs) -> nxadb.MultiGraph:
            db.delete_graph(GRAPH_NAME, drop_collections=True, ignore_missing=True)
            G = nxadb.MultiGraph(*args, **kwargs, name=GRAPH_NAME)
            # Experimenting with a delay to see if it helps with CircleCI...
            time.sleep(0.10)
            return G

        self.K3Graph = lambda *args, **kwargs: nxadb_graph_constructor(
            *args, **kwargs, incoming_graph_data=self.K3
        )
        self.Graph = self.K3Graph
        self.EmptyGraph = lambda *args, **kwargs: nxadb_graph_constructor(
            *args, **kwargs
        )

    def test_data_input(self):
        G = self.EmptyGraph({1: [2], 2: [1]})
        assert G.number_of_edges() == 1
        assert G.number_of_nodes() == 2
        assert sorted(G.adj.items()) == [
            ("test_graph_node/1", G.adj[1]),
            ("test_graph_node/2", G.adj[2]),
        ]

    def test_data_multigraph_input(self):
        # standard case with edge keys and edge data
        edata0 = {"w": 200, "s": "foo"}
        edata1 = {"w": 201, "s": "bar"}
        keydict = {0: edata0, 1: edata1}
        dododod = {"a": {"b": keydict}}

        for multigraph_input in [True, None]:
            G = self.EmptyGraph(dododod, multigraph_input=multigraph_input)
            assert G.number_of_edges() == 2
            edge_a_b_0 = G.adj["a"]["b"][0]["_id"]
            edge_a_b_1 = G.adj["a"]["b"][1]["_id"]
            assert G["a"]["b"][0]["w"] == edata0["w"]
            assert G["a"]["b"][0]["s"] == edata0["s"]
            assert G["a"]["b"][1]["w"] == edata1["w"]
            assert G["a"]["b"][1]["s"] == edata1["s"]

            # TODO: Figure out why it's either (a, b) or (b, a)...
            multiple_edge_a_b = [
                (
                    "test_graph_node/a",
                    "test_graph_node/b",
                    edge_a_b_0,
                    get_doc(edge_a_b_0),
                ),
                (
                    "test_graph_node/a",
                    "test_graph_node/b",
                    edge_a_b_1,
                    get_doc(edge_a_b_1),
                ),
            ]

            multiple_edge_b_a = [
                (
                    "test_graph_node/b",
                    "test_graph_node/a",
                    edge_a_b_0,
                    get_doc(edge_a_b_0),
                ),
                (
                    "test_graph_node/b",
                    "test_graph_node/a",
                    edge_a_b_1,
                    get_doc(edge_a_b_1),
                ),
            ]

            edges = list(G.edges(keys=True, data=True))
            for edge in edges:
                # TODO: Need to revisit. I don't like this...
                assert edge in multiple_edge_a_b or edge in multiple_edge_b_a

        G = self.EmptyGraph(dododod, multigraph_input=False)
        assert G.number_of_edges() == 1
        edge_a_b_0 = G.adj["a"]["b"][0]["_id"]
        single_edge_a_b = (
            "test_graph_node/a",
            "test_graph_node/b",
            edge_a_b_0,
            get_doc(edge_a_b_0),
        )
        single_edge_b_a = (
            "test_graph_node/b",
            "test_graph_node/a",
            edge_a_b_0,
            get_doc(edge_a_b_0),
        )
        edges = list(G.edges(keys=True, data=True))
        assert len(edges) == 1
        # TODO: Need to revisit. I don't like this...
        assert edges[0] == single_edge_a_b or edges[0] == single_edge_b_a

        # test round-trip to_dict_of_dict and MultiGraph constructor
        G = self.EmptyGraph(dododod, multigraph_input=True)
        dod = nx.to_dict_of_dicts(G)  # NOTE: This is currently failing...
        H = self.EmptyGraph(dod)
        assert nx.is_isomorphic(G, H) is True  # test that default is True
        for mgi in [True, False]:
            H = self.EmptyGraph(nx.to_dict_of_dicts(G), multigraph_input=mgi)
            assert nx.is_isomorphic(G, H) == mgi

    # Set up cases for when incoming_graph_data is not multigraph_input
    etraits = {"w": 200, "s": "foo"}
    egraphics = {"color": "blue", "shape": "box"}
    edata = {"traits": etraits, "graphics": egraphics}
    dodod1 = {"a": {"b": edata}}
    dodod2 = {"a": {"b": etraits}}
    dodod3 = {"a": {"b": {"traits": etraits, "s": "foo"}}}
    dol = {"a": ["b"]}

    multiple_edge = [("a", "b", "traits", etraits), ("a", "b", "graphics", egraphics)]
    single_edge = [("a", "b", 0, {})]  # type: ignore[var-annotated]
    single_edge1 = [("a", "b", 0, edata)]
    single_edge2 = [("a", "b", 0, etraits)]
    single_edge3 = [("a", "b", 0, {"traits": etraits, "s": "foo"})]

    cases = [  # (dod, mgi, edges)
        (dodod1, True, multiple_edge),
        (dodod1, False, single_edge1),
        (dodod2, False, single_edge2),
        (dodod3, False, single_edge3),
        (dol, False, single_edge),
    ]

    # def test_non_multigraph_input(self, dod, mgi, edges):
    # pass
    # TODO: Implement

    def test_non_multigraph_input_mgi_none(self):
        etraits = {"w": 200, "s": "foo"}
        egraphics = {"color": "blue", "shape": "box"}
        edata = {"traits": etraits, "graphics": egraphics}
        dodod1 = {"a": {"b": edata}}
        dodod2 = {"a": {"b": etraits}}
        dodod3 = {"a": {"b": {"traits": etraits, "s": "foo"}}}

        # test constructor without to_networkx_graph for mgi=None
        G = self.EmptyGraph(dodod1, multigraph_input=None)
        assert G.number_of_edges() == 2
        assert G["a"]["b"][0]["w"] == etraits["w"]
        assert G["a"]["b"][0]["s"] == etraits["s"]
        assert G["a"]["b"][1]["color"] == egraphics["color"]
        assert G["a"]["b"][1]["shape"] == egraphics["shape"]

        G = self.EmptyGraph(dodod2, multigraph_input=None)
        assert G.number_of_edges() == 1
        assert G["a"]["b"][0]["w"] == etraits["w"]
        assert G["a"]["b"][0]["s"] == etraits["s"]

        G = self.EmptyGraph(dodod3, multigraph_input=None)
        assert G.number_of_edges() == 1
        assert G["a"]["b"][0]["traits"] == etraits
        assert G["a"]["b"][0]["s"] == dodod3["a"]["b"]["s"]

    def test_getitem(self):
        G = self.K3Graph()
        assert G[0] == {"test_graph_node/1": G[0][1], "test_graph_node/2": G[0][2]}
        with pytest.raises(KeyError):
            G.__getitem__("j")
        with pytest.raises(TypeError):
            G.__getitem__(["A"])

    def test_remove_node(self):
        G = self.K3Graph()
        assert 0 in G.nodes
        assert len(G[0]) == 2
        assert G.number_of_nodes() == 3
        G.remove_node(0)
        assert 0 not in G.nodes
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert len(G[1][2]) == 1
        edge_1_2 = get_doc(list(G[1][2])[0])
        assert G.adj == {
            "test_graph_node/1": {"test_graph_node/2": {edge_1_2["_id"]: edge_1_2}},
            "test_graph_node/2": {"test_graph_node/1": {edge_1_2["_id"]: edge_1_2}},
        }
        with pytest.raises(nx.NetworkXError):
            G.remove_node(-1)

    def test_add_edge(self):
        G = self.EmptyGraph()
        edge_id = G.add_edge(0, 1)
        assert G.number_of_edges() == 1
        assert len(G[0][1]) == 1
        assert G[0][1][edge_id] == get_doc(edge_id)

        G = self.EmptyGraph()
        edge_id = G.add_edge(*(0, 1))
        assert G.number_of_edges() == 1
        assert len(G[0][1]) == 1
        assert G[0][1][edge_id] == get_doc(edge_id)

        with pytest.raises(ValueError):
            G.add_edge(None, "anything")

    def test_add_edge_conflicting_key(self):
        G = self.EmptyGraph()
        G.add_edge(0, 1, key=1)
        G.add_edge(0, 1)
        assert G.number_of_edges() == 2
        G = self.EmptyGraph()
        G.add_edges_from([(0, 1, 1, {})])
        G.add_edges_from([(0, 1)])
        assert G.number_of_edges() == 2

    def test_add_edges_from(self):
        G = self.EmptyGraph()
        G.add_edges_from([(0, 1), (0, 1, {"weight": 3})])
        assert len(G[0][1]) == 2
        assert "weight" not in G[0][1][0]
        assert G[0][1][1]["weight"] == 3
        edge_0_1_0 = get_doc(G[0][1][0]["_id"])
        edge_0_1_1 = get_doc(G[0][1][1]["_id"])
        assert G.adj == {
            "test_graph_node/0": {
                "test_graph_node/1": {
                    edge_0_1_0["_id"]: edge_0_1_0,
                    edge_0_1_1["_id"]: edge_0_1_1,
                }
            },
            "test_graph_node/1": {
                "test_graph_node/0": {
                    edge_0_1_0["_id"]: edge_0_1_0,
                    edge_0_1_1["_id"]: edge_0_1_1,
                }
            },
        }
        G.add_edges_from([(0, 1), (0, 1, {"weight": 3})], weight=2)
        assert len(G[0][1]) == 4
        assert G[0][1][2]["weight"] == 2
        assert G[0][1][3]["weight"] == 3
        edge_0_1_2 = get_doc(G[0][1][2]["_id"])
        edge_0_1_3 = get_doc(G[0][1][3]["_id"])
        assert G.adj == {
            "test_graph_node/0": {
                "test_graph_node/1": {
                    edge_0_1_0["_id"]: edge_0_1_0,
                    edge_0_1_1["_id"]: edge_0_1_1,
                    edge_0_1_2["_id"]: edge_0_1_2,
                    edge_0_1_3["_id"]: edge_0_1_3,
                }
            },
            "test_graph_node/1": {
                "test_graph_node/0": {
                    edge_0_1_0["_id"]: edge_0_1_0,
                    edge_0_1_1["_id"]: edge_0_1_1,
                    edge_0_1_2["_id"]: edge_0_1_2,
                    edge_0_1_3["_id"]: edge_0_1_3,
                }
            },
        }

        G = self.EmptyGraph()
        edges = [
            (0, 1, {"weight": 3}),
            (0, 1, (("weight", 2),)),
            (0, 1, 5),
            (0, 1, "s"),
        ]
        G.add_edges_from(edges)
        assert G.number_of_edges() == 4
        assert G[0][1][0]["weight"] == 3
        assert G[0][1][1]["weight"] == 2
        assert G[0][1][2]["_id"] != 5  # custom key not supported
        assert G[0][1][3]["_id"] != "s"  # custom key not supported

        # too few in tuple
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0,)])
        # too many in tuple
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0, 1, 2, 3, 4)])
        # not a tuple
        with pytest.raises(TypeError):
            G.add_edges_from([0])

    def test_multigraph_add_edges_from_four_tuple_misordered(self):
        """add_edges_from expects 4-tuples of the format (u, v, key, data_dict).

        Ensure 4-tuples of form (u, v, data_dict, key) raise exception.
        """
        G = self.EmptyGraph()
        with pytest.raises(TypeError):
            # key/data values flipped in 4-tuple
            G.add_edges_from([(0, 1, {"color": "red"}, 0)])

    def test_remove_edge(self):
        G = self.K3Graph()
        edge_id = list(G[0][1])[0]
        assert db.has_document(edge_id)
        G.remove_edge(0, 1)
        assert not db.has_document(edge_id)
        with pytest.raises(KeyError):
            G[0][1]
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(-1, 0)
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(0, 2, key=1)

    def test_remove_edges_from(self):
        G = self.K3Graph()
        edges_0_1 = list(G[0][1])
        assert len(edges_0_1) == 1
        G.remove_edges_from([(0, 1)])
        assert not db.has_document(edges_0_1[0])
        with pytest.raises(KeyError):
            G[0][1]

        G.remove_edges_from([(0, 0)])  # silent fail
        G.add_edge(0, 1)
        G.remove_edges_from(list(G.edges(data=True, keys=True)))
        assert G.adj == {
            "test_graph_node/0": {},
            "test_graph_node/1": {},
            "test_graph_node/2": {},
        }

        G = self.K3Graph()
        G.remove_edges_from(list(G.edges(data=False, keys=True)))
        assert G.adj == {
            "test_graph_node/0": {},
            "test_graph_node/1": {},
            "test_graph_node/2": {},
        }

        G = self.K3Graph()
        G.remove_edges_from(list(G.edges(data=False, keys=False)))
        assert G.adj == {
            "test_graph_node/0": {},
            "test_graph_node/1": {},
            "test_graph_node/2": {},
        }

        G = self.K3Graph()
        assert len(G[0][1]) == 1
        G.add_edge(0, 1)
        assert len(G[0][1]) == 2
        edge_0_1_0 = list(G[0][1])[0]
        edge_0_2_0 = list(G[0][2])[0]
        assert db.has_document(edge_0_1_0)
        assert len(G[0][2]) == 1
        assert len(G[1][2]) == 1
        G.remove_edges_from([(0, 1, edge_0_1_0), (0, 2, edge_0_2_0, {}), (1, 2)])
        assert not db.has_document(edge_0_1_0)
        assert not db.has_document(edge_0_2_0)
        assert edge_0_1_0 not in G[0][1]
        assert len(G[0][1]) == 1
        with pytest.raises(KeyError):
            G[0][2]
        with pytest.raises(KeyError):
            G[1][2]

    def test_remove_multiedge(self):
        G = self.K3Graph()
        edge_id = G.add_edge(0, 1)
        assert db.has_document(edge_id)
        G.remove_edge(0, 1, key=edge_id)
        assert not db.has_document(edge_id)
        last_edge = list(G[0][1])[-1]
        assert db.has_document(last_edge)
        G.remove_edge(0, 1)
        assert not db.has_document(last_edge)
        with pytest.raises(KeyError):
            G[0][1]
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(0, 1)
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(-1, 0)


# TODO: Revisit
# Subgraphing not implemented yet
# class TestEdgeSubgraph:
#     """Unit tests for the :meth:`MultiGraph.edge_subgraph` method."""

#     def setup_method(self):
#         # Create a doubly-linked path graph on five nodes.
#         G = nx.MultiGraph()
#         nx.add_path(G, range(5))
#         nx.add_path(G, range(5))
#         # Add some node, edge, and graph attributes.
#         for i in range(5):
#             G.nodes[i]["name"] = f"node{i}"
#         G.adj[0][1][0]["name"] = "edge010"
#         G.adj[0][1][1]["name"] = "edge011"
#         G.adj[3][4][0]["name"] = "edge340"
#         G.adj[3][4][1]["name"] = "edge341"
#         G.graph["name"] = "graph"
#         # Get the subgraph induced by one of the first edges and one of
#         # the last edges.
#         self.G = G
#         self.H = G.edge_subgraph([(0, 1, 0), (3, 4, 1)])

#     def test_correct_nodes(self):
#         """Tests that the subgraph has the correct nodes."""
#         assert [0, 1, 3, 4] == sorted(self.H.nodes())

#     def test_correct_edges(self):
#         """Tests that the subgraph has the correct edges."""
#         assert [(0, 1, 0, "edge010"), (3, 4, 1, "edge341")] == sorted(
#             self.H.edges(keys=True, data="name")
#         )

#     def test_add_node(self):
#         """Tests that adding a node to the original graph does not
#         affect the nodes of the subgraph.

#         """
#         self.G.add_node(5)
#         assert [0, 1, 3, 4] == sorted(self.H.nodes())

#     def test_remove_node(self):
#         """Tests that removing a node in the original graph does
#         affect the nodes of the subgraph.

#         """
#         self.G.remove_node(0)
#         assert [1, 3, 4] == sorted(self.H.nodes())

#     def test_node_attr_dict(self):
#         """Tests that the node attribute dictionary of the two graphs is
#         the same object.

#         """
#         for v in self.H:
#             assert self.G.nodes[v] == self.H.nodes[v]
#         # Making a change to G should make a change in H and vice versa.
#         self.G.nodes[0]["name"] = "foo"
#         assert self.G.nodes[0] == self.H.nodes[0]
#         self.H.nodes[1]["name"] = "bar"
#         assert self.G.nodes[1] == self.H.nodes[1]

#     def test_edge_attr_dict(self):
#         """Tests that the edge attribute dictionary of the two graphs is
#         the same object.

#         """
#         for u, v, k in self.H.edges(keys=True):
#             assert self.G._adj[u][v][k] == self.H._adj[u][v][k]
#         # Making a change to G should make a change in H and vice versa.
#         self.G._adj[0][1][0]["name"] = "foo"
#         assert self.G._adj[0][1][0]["name"] == self.H._adj[0][1][0]["name"]
#         self.H._adj[3][4][1]["name"] = "bar"
#         assert self.G._adj[3][4][1]["name"] == self.H._adj[3][4][1]["name"]

#     def test_graph_attr_dict(self):
#         """Tests that the graph attribute dictionary of the two graphs
#         is the same object.

#         """
#         assert self.G.graph is self.H.graph


# class CustomDictClass(UserDict):
#     pass


# class MultiGraphSubClass(nx.MultiGraph):
#     node_dict_factory = CustomDictClass  # type: ignore[assignment]
#     node_attr_dict_factory = CustomDictClass  # type: ignore[assignment]
#     adjlist_outer_dict_factory = CustomDictClass  # type: ignore[assignment]
#     adjlist_inner_dict_factory = CustomDictClass  # type: ignore[assignment]
#     edge_key_dict_factory = CustomDictClass  # type: ignore[assignment]
#     edge_attr_dict_factory = CustomDictClass  # type: ignore[assignment]
#     graph_attr_dict_factory = CustomDictClass  # type: ignore[assignment]


# TODO: Figure out where this is used
# class TestMultiGraphSubclass(TestMultiGraph):
#     def setup_method(self):
#         self.Graph = MultiGraphSubClass
#         # build K3
#         self.k3edges = [(0, 1), (0, 2), (1, 2)]
#         self.k3nodes = [0, 1, 2]
#         self.K3 = self.Graph()
#         self.K3._adj = self.K3.adjlist_outer_dict_factory(
#             {
#                 0: self.K3.adjlist_inner_dict_factory(),
#                 1: self.K3.adjlist_inner_dict_factory(),
#                 2: self.K3.adjlist_inner_dict_factory(),
#             }
#         )
#         self.K3._pred = {0: {}, 1: {}, 2: {}}
#         for u in self.k3nodes:
#             for v in self.k3nodes:
#                 if u != v:
#                     d = {0: {}}
#                     self.K3._adj[u][v] = d
#                     self.K3._adj[v][u] = d
#         self.K3._node = self.K3.node_dict_factory()
#         self.K3._node[0] = self.K3.node_attr_dict_factory()
#         self.K3._node[1] = self.K3.node_attr_dict_factory()
#         self.K3._node[2] = self.K3.node_attr_dict_factory()
