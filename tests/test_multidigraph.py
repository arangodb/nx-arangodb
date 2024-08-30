# type: ignore

import time
from collections import UserDict

import networkx as nx
import pytest
from networkx.utils import edges_equal

import nx_arangodb as nxadb

from .conftest import db
from .test_graph import GRAPH_NAME, get_doc

# from .test_multigraph import TestEdgeSubgraph as _TestMultiGraphEdgeSubgraph
from .test_multigraph import BaseMultiGraphTester
from .test_multigraph import TestMultiGraph as _TestMultiGraph


class BaseMultiDiGraphTester(BaseMultiGraphTester):
    def test_adjacency(self):
        G = self.K3Graph()

        edge_0_1_id = list(G[0][1])[0]
        edge_0_1 = get_doc(edge_0_1_id)
        edge_0_2_id = list(G[0][2])[0]
        edge_0_2 = get_doc(edge_0_2_id)
        edge_1_0_id = list(G[1][0])[0]
        edge_1_0 = get_doc(edge_1_0_id)
        edge_2_0_id = list(G[2][0])[0]
        edge_2_0 = get_doc(edge_2_0_id)
        edge_1_2_id = list(G[1][2])[0]
        edge_1_2 = get_doc(edge_1_2_id)
        edge_2_1_id = list(G[2][1])[0]
        edge_2_1 = get_doc(edge_2_1_id)

        assert dict(G.adjacency()) == {
            "test_graph_node/0": {
                "test_graph_node/1": {edge_0_1_id: edge_0_1},
                "test_graph_node/2": {edge_0_2_id: edge_0_2},
            },
            "test_graph_node/1": {
                "test_graph_node/0": {edge_1_0_id: edge_1_0},
                "test_graph_node/2": {edge_1_2_id: edge_1_2},
            },
            "test_graph_node/2": {
                "test_graph_node/0": {edge_2_0_id: edge_2_0},
                "test_graph_node/1": {edge_2_1_id: edge_2_1},
            },
        }

    def get_edges_data(self, G):
        edges_data = []
        for src, dst, _ in G.edges:
            edge_id = list(G[src][dst])[0]
            edges_data.append((src, dst, get_doc(edge_id)))

        return sorted(edges_data)

    def test_edges(self):
        G = self.K3Graph()
        assert edges_equal(G.edges(), self.edges_all)
        assert edges_equal(G.edges(0), self.edges_0)
        assert edges_equal(G.edges([0, 1]), self.edges_0_1)
        pytest.raises((KeyError, nx.NetworkXError), G.edges, -1)

    def test_edges_data(self):
        G = self.K3Graph()
        edges_data = self.get_edges_data(G)
        edges_data_0 = edges_data[0:2]
        assert sorted(G.edges(data=True)) == edges_data
        assert sorted(G.edges(0, data=True)) == edges_data_0
        pytest.raises((KeyError, nx.NetworkXError), G.neighbors, -1)

    def test_edges_multi(self):
        G = self.K3Graph()
        assert sorted(G.edges()) == sorted(self.edges_all)
        assert sorted(G.edges(0)) == sorted(self.edges_0)
        assert G.number_of_edges() == 6
        edge_id = G.add_edge(0, 1)
        assert db.has_document(edge_id)
        assert G.number_of_edges() == 7
        assert sorted(G.edges()) == sorted(
            self.edges_all + [("test_graph_node/0", "test_graph_node/1")]
        )

    def test_out_edges(self):
        G = self.K3Graph()
        assert sorted(G.out_edges()) == sorted(self.edges_all)
        assert sorted(G.out_edges(0)) == sorted(self.edges_0)
        pytest.raises((KeyError, nx.NetworkXError), G.out_edges, -1)
        edges_0_with_keys = [
            (src, dst, G[src][dst][0]["_id"]) for src, dst in self.edges_0
        ]
        assert sorted(G.out_edges(0, keys=True)) == edges_0_with_keys

    def test_out_edges_multi(self):
        G = self.K3Graph()
        assert sorted(G.out_edges()) == sorted(self.edges_all)
        assert sorted(G.out_edges(0)) == sorted(self.edges_0)
        edge_id = G.add_edge(0, 1, 2)
        assert edge_id != 2
        assert db.has_document(edge_id)
        assert sorted(G.edges()) == sorted(
            self.edges_all + [("test_graph_node/0", "test_graph_node/1")]
        )

    def test_out_edges_data(self):
        G = self.K3Graph()

        edges_data = self.get_edges_data(G)
        edges_data_0 = edges_data[0:2]

        assert sorted(G.edges(0, data=True)) == edges_data_0
        G.remove_edge(0, 1)
        edge_0_1_new_id = G.add_edge(0, 1, data=1)
        edge_0_1_new = get_doc(edge_0_1_new_id)
        edge_0_data_new = [
            ("test_graph_node/0", "test_graph_node/1", edge_0_1_new),
            edges_data_0[1],
        ]
        assert sorted(G.edges(0, data=True)) == edge_0_data_new
        assert sorted(G.edges(0, data="data")) == [
            ("test_graph_node/0", "test_graph_node/1", 1),
            ("test_graph_node/0", "test_graph_node/2", None),
        ]
        assert sorted(G.edges(0, data="data", default=-1)) == [
            ("test_graph_node/0", "test_graph_node/1", 1),
            ("test_graph_node/0", "test_graph_node/2", -1),
        ]

    def test_in_edges(self):
        G = self.K3Graph()

        edges_0_in = [
            ("test_graph_node/1", "test_graph_node/0"),
            ("test_graph_node/2", "test_graph_node/0"),
        ]

        assert sorted(G.in_edges()) == sorted(self.edges_all)
        assert sorted(G.in_edges(0)) == edges_0_in
        pytest.raises((KeyError, nx.NetworkXError), G.in_edges, -1)
        G.add_edge(0, 1, 2)
        assert sorted(G.in_edges()) == sorted(
            self.edges_all + [("test_graph_node/0", "test_graph_node/1")]
        )
        assert sorted(G.in_edges(0, keys=True)) == [
            (
                "test_graph_node/1",
                "test_graph_node/0",
                "test_graph_node_to_test_graph_node/0",
            ),
            (
                "test_graph_node/2",
                "test_graph_node/0",
                "test_graph_node_to_test_graph_node/1",
            ),
        ]

    def test_in_edges_no_keys(self):
        G = self.K3Graph()
        edges_0_in = [
            ("test_graph_node/1", "test_graph_node/0"),
            ("test_graph_node/2", "test_graph_node/0"),
        ]

        assert sorted(G.in_edges()) == sorted(self.edges_all)
        assert sorted(G.in_edges(0)) == edges_0_in
        G.add_edge(0, 1, 2)
        assert sorted(G.in_edges()) == sorted(
            self.edges_all + [("test_graph_node/0", "test_graph_node/1")]
        )

        edges_data = self.get_edges_data(G)
        in_edges_data = G.in_edges(data=True, keys=False)
        assert len(in_edges_data) == len(edges_data)
        assert len(list(in_edges_data)[0]) == 3

    def test_in_edges_data(self):
        G = self.K3Graph()
        list(G[2][0])
        list(G[1][0])
        edge_2_0 = get_doc(G[2][0][0]["_id"])
        edge_1_0 = get_doc(G[1][0][0]["_id"])
        assert sorted(G.in_edges(0, data=True)) == [
            ("test_graph_node/1", "test_graph_node/0", edge_1_0),
            ("test_graph_node/2", "test_graph_node/0", edge_2_0),
        ]
        G.remove_edge(1, 0)
        G.add_edge(1, 0, data=1)
        edge_1_0 = get_doc(G[1][0][0]["_id"])
        assert sorted(G.in_edges(0, data=True)) == [
            ("test_graph_node/1", "test_graph_node/0", edge_1_0),
            ("test_graph_node/2", "test_graph_node/0", edge_2_0),
        ]
        assert sorted(G.in_edges(0, data="data")) == [
            ("test_graph_node/1", "test_graph_node/0", 1),
            ("test_graph_node/2", "test_graph_node/0", None),
        ]
        assert sorted(G.in_edges(0, data="data", default=-1)) == [
            ("test_graph_node/1", "test_graph_node/0", 1),
            ("test_graph_node/2", "test_graph_node/0", -1),
        ]

    def is_shallow(self, H, G):
        # graph
        assert G.graph["foo"] == H.graph["foo"]
        G.graph["foo"].append(1)
        assert G.graph["foo"] == H.graph["foo"]
        # node
        assert G.nodes[0]["foo"] == H.nodes[0]["foo"]
        G.nodes[0]["foo"].append(1)
        assert G.nodes[0]["foo"] == H.nodes[0]["foo"]
        # edge
        assert G[1][2][0]["foo"] == H[1][2][0]["foo"]
        G[1][2][0]["foo"].append(1)
        assert G[1][2][0]["foo"] == H[1][2][0]["foo"]

    def is_deep(self, H, G):
        # graph
        assert G.graph["foo"] == H.graph["foo"]
        G.graph["foo"].append(1)
        assert G.graph["foo"] != H.graph["foo"]
        # node
        assert (
            G.nodes["test_graph_node/0"]["foo"] == H.nodes["test_graph_node/0"]["foo"]
        )
        G.nodes["test_graph_node/0"]["foo"].append(1)
        assert (
            G.nodes["test_graph_node/0"]["foo"] != H.nodes["test_graph_node/0"]["foo"]
        )
        # edge
        edge_id = G[1][2][0]["_id"]
        assert (
            G[1][2][0]["foo"]
            == H["test_graph_node/1"]["test_graph_node/2"][edge_id]["foo"]
        )
        G[1][2][edge_id]["foo"].append(1)
        assert (
            G[1][2][0]["foo"]
            != H["test_graph_node/1"]["test_graph_node/2"][edge_id]["foo"]
        )

    def test_to_undirected(self):
        # MultiDiGraph -> MultiGraph changes number of edges so it is
        # not a copy operation... use is_shallow, not is_shallow_copy
        G = self.K3Graph()
        self.add_attributes(G)
        H = nxadb.MultiGraph(G)
        # self.is_shallow(H,G)
        # the result is traversal order dependent so we
        # can't use the is_shallow() test here.
        try:
            assert edges_equal(
                H.edges(),
                [
                    ("test_graph_node/0", "test_graph_node/2"),
                    ("test_graph_node/0", "test_graph_node/1"),
                    ("test_graph_node/1", "test_graph_node/2"),
                ],
            )
        except AssertionError:
            assert edges_equal(
                H.edges(),
                [
                    ("test_graph_node/0", "test_graph_node/1"),
                    ("test_graph_node/1", "test_graph_node/2"),
                    ("test_graph_node/1", "test_graph_node/2"),
                    ("test_graph_node/2", "test_graph_node/0"),
                ],
            )
        H = G.to_undirected()
        self.is_deep(H, G)

    def test_has_successor(self):
        G = self.K3Graph()
        assert G.has_successor(0, 1)
        assert not G.has_successor(0, -1)

    def test_successors(self):
        G = self.K3Graph()
        assert sorted(G.successors(0)) == ["test_graph_node/1", "test_graph_node/2"]
        pytest.raises((KeyError, nx.NetworkXError), G.successors, -1)

    def test_has_predecessor(self):
        G = self.K3Graph()
        assert G.has_predecessor(0, 1)
        assert not G.has_predecessor(0, -1)

    def test_predecessors(self):
        G = self.K3Graph()
        assert sorted(G.predecessors(0)) == ["test_graph_node/1", "test_graph_node/2"]
        pytest.raises((KeyError, nx.NetworkXError), G.predecessors, -1)

    def test_degree(self):
        G = self.K3Graph()
        assert sorted(G.degree()) == [
            ("test_graph_node/0", 4),
            ("test_graph_node/1", 4),
            ("test_graph_node/2", 4),
        ]
        assert dict(G.degree()) == {
            "test_graph_node/0": 4,
            "test_graph_node/1": 4,
            "test_graph_node/2": 4,
        }
        assert G.degree(0) == 4
        assert list(G.degree(iter([0]))) == [("test_graph_node/0", 4)]
        edge_id = G.add_edge(0, 1, weight=0.3, other=1.2)
        doc = db.document(edge_id)
        assert doc["weight"] == 0.3
        assert doc["other"] == 1.2
        assert sorted(G.degree(weight="weight")) == [
            ("test_graph_node/0", 4.3),
            ("test_graph_node/1", 4.3),
            ("test_graph_node/2", 4),
        ]
        assert sorted(G.degree(weight="other")) == [
            ("test_graph_node/0", 5.2),
            ("test_graph_node/1", 5.2),
            ("test_graph_node/2", 4),
        ]

    def test_in_degree(self):
        G = self.K3Graph()
        assert sorted(G.in_degree()) == [
            ("test_graph_node/0", 2),
            ("test_graph_node/1", 2),
            ("test_graph_node/2", 2),
        ]
        assert dict(G.in_degree()) == {
            "test_graph_node/0": 2,
            "test_graph_node/1": 2,
            "test_graph_node/2": 2,
        }
        assert G.in_degree(0) == 2
        assert list(G.in_degree(iter([0]))) == [("test_graph_node/0", 2)]
        assert G.in_degree(0, weight="weight") == 2

    def test_out_degree(self):
        G = self.K3Graph()
        assert sorted(G.out_degree()) == [
            ("test_graph_node/0", 2),
            ("test_graph_node/1", 2),
            ("test_graph_node/2", 2),
        ]
        assert dict(G.out_degree()) == {
            "test_graph_node/0": 2,
            "test_graph_node/1": 2,
            "test_graph_node/2": 2,
        }
        assert G.out_degree(0) == 2
        assert list(G.out_degree(iter([0]))) == [("test_graph_node/0", 2)]
        assert G.out_degree(0, weight="weight") == 2

    def test_size(self):
        G = self.K3Graph()
        assert G.size() == 6
        assert G.number_of_edges() == 6
        G.add_edge(0, 1, weight=0.3, other=1.2)
        assert G.number_of_edges() == 7
        assert round(G.size(weight="weight"), 2) == 6.3
        assert round(G.size(weight="other"), 2) == 7.2

    def test_to_undirected_reciprocal(self):
        G = self.EmptyGraph()
        assert G.number_of_edges() == 0
        G.add_edge(1, 2)
        assert G.number_of_edges() == 1

        G_undirected = G.to_undirected()
        assert G_undirected.number_of_edges() == 1
        assert G_undirected.has_edge("test_graph_node/1", "test_graph_node/2")
        assert G_undirected.has_edge("test_graph_node/2", "test_graph_node/1")

        G_undirected_reciprocal = G.to_undirected(reciprocal=True)
        assert G_undirected_reciprocal.number_of_edges() == 0
        assert not G_undirected_reciprocal.has_edge(
            "test_graph_node/1", "test_graph_node/2"
        )

        edge_2_1_id = G.add_edge("test_graph_node/2", "test_graph_node/1", foo="bar")
        assert G.number_of_edges() == 2
        G_undirected_reciprocal = G.to_undirected(reciprocal=True)
        assert G_undirected_reciprocal.number_of_edges() == 2
        assert G_undirected_reciprocal.has_edge(
            "test_graph_node/1", "test_graph_node/2"
        )
        assert G_undirected_reciprocal.has_edge(
            "test_graph_node/2", "test_graph_node/1"
        )
        # notice how edge_1_2 now has the same data as edge_2_1 (+ the same _id)
        edge_1_2 = G_undirected_reciprocal["test_graph_node/1"]["test_graph_node/2"][
            edge_2_1_id
        ]
        edge_2_1 = G_undirected_reciprocal["test_graph_node/2"]["test_graph_node/1"][
            edge_2_1_id
        ]
        assert edge_1_2 == edge_2_1
        assert edge_1_2["foo"] == "bar"

    def test_reverse_copy(self):
        G = self.EmptyGraph([(0, 1), (0, 1)])
        R = G.reverse()
        assert sorted(R.edges()) == [
            ("test_graph_node/1", "test_graph_node/0"),
            ("test_graph_node/1", "test_graph_node/0"),
        ]
        R.remove_edge("test_graph_node/1", "test_graph_node/0")
        assert sorted(R.edges()) == [("test_graph_node/1", "test_graph_node/0")]
        assert sorted(G.edges()) == [
            ("test_graph_node/0", "test_graph_node/1"),
            ("test_graph_node/0", "test_graph_node/1"),
        ]

    def test_reverse_nocopy(self):
        G = self.EmptyGraph([(0, 1), (0, 1)])
        with pytest.raises(NotImplementedError):
            G.reverse(copy=False)  # nocopy not supported yet
        pytest.skip("NotImplementedError: In-place reverse is not supported yet.")
        # assert sorted(R.edges()) == [
        #     ("test_graph_node/1", "test_graph_node/0"),
        #     ("test_graph_node/1", "test_graph_node/0"),
        # ]
        # pytest.raises(nx.NetworkXError, R.remove_edge, 1, 0)

    def test_di_attributes_cached(self):
        G = self.K3Graph().copy()
        assert id(G.in_edges) == id(G.in_edges)
        assert id(G.out_edges) == id(G.out_edges)
        assert id(G.in_degree) == id(G.in_degree)
        assert id(G.out_degree) == id(G.out_degree)
        assert id(G.succ) == id(G.succ)
        assert id(G.pred) == id(G.pred)


class TestMultiDiGraph(BaseMultiDiGraphTester, _TestMultiGraph):
    def setup_method(self):
        self.Graph = nx.MultiDiGraph
        # build K3
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._succ = {0: {}, 1: {}, 2: {}}
        # K3._adj is synced with K3._succ
        self.K3._pred = {0: {}, 1: {}, 2: {}}
        for u in self.k3nodes:
            for v in self.k3nodes:
                if u == v:
                    continue
                d = {0: {}}
                self.K3._succ[u][v] = d
                self.K3._pred[v][u] = d
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}

        def nxadb_graph_constructor(*args, **kwargs) -> nxadb.MultiDiGraph:
            db.delete_graph(GRAPH_NAME, drop_collections=True, ignore_missing=True)
            G = nxadb.MultiDiGraph(*args, **kwargs, name=GRAPH_NAME, write_async=False)
            return G

        self.K3Graph = lambda *args, **kwargs: nxadb_graph_constructor(
            *args, **kwargs, incoming_graph_data=self.K3
        )
        self.EmptyGraph = lambda *args, **kwargs: nxadb_graph_constructor(
            *args, **kwargs
        )
        self.k3nodes = ["test_graph_node/0", "test_graph_node/1", "test_graph_node/2"]

        self.edges_all = [
            ("test_graph_node/0", "test_graph_node/1"),
            ("test_graph_node/1", "test_graph_node/0"),
            ("test_graph_node/0", "test_graph_node/2"),
            ("test_graph_node/2", "test_graph_node/0"),
            ("test_graph_node/1", "test_graph_node/2"),
            ("test_graph_node/2", "test_graph_node/1"),
        ]
        self.edges_0 = [
            ("test_graph_node/0", "test_graph_node/1"),
            ("test_graph_node/0", "test_graph_node/2"),
        ]
        self.edges_0_1 = [
            ("test_graph_node/0", "test_graph_node/2"),
            ("test_graph_node/0", "test_graph_node/1"),
            ("test_graph_node/1", "test_graph_node/2"),
            ("test_graph_node/1", "test_graph_node/0"),
        ]

    def test_add_edge(self):
        G = self.EmptyGraph()
        edge_id = G.add_edge(0, 1)
        edge_doc = get_doc(edge_id)
        assert G._adj == {
            "test_graph_node/0": {"test_graph_node/1": {edge_id: edge_doc}},
            "test_graph_node/1": {},
        }
        assert G._succ == {
            "test_graph_node/0": {"test_graph_node/1": {edge_id: edge_doc}},
            "test_graph_node/1": {},
        }
        assert G._pred == {
            "test_graph_node/0": {},
            "test_graph_node/1": {"test_graph_node/0": {edge_id: edge_doc}},
        }
        G = self.EmptyGraph()
        edge_id = G.add_edge(*(0, 1))
        edge_doc = get_doc(edge_id)
        assert G._adj == {
            "test_graph_node/0": {"test_graph_node/1": {edge_id: edge_doc}},
            "test_graph_node/1": {},
        }
        assert G._succ == {
            "test_graph_node/0": {"test_graph_node/1": {edge_id: edge_doc}},
            "test_graph_node/1": {},
        }
        assert G._pred == {
            "test_graph_node/0": {},
            "test_graph_node/1": {"test_graph_node/0": {edge_id: edge_doc}},
        }
        with pytest.raises(ValueError, match="Key cannot be None"):
            G.add_edge(None, 3)

    def test_add_edges_from(self):
        G = self.EmptyGraph()
        G.add_edges_from([(0, 1), (0, 1, {"weight": 3})])
        edge_0_1_0_id = G[0][1][0]["_id"]
        edge_0_1_0 = get_doc(edge_0_1_0_id)
        edge_0_1_1_id = G[0][1][1]["_id"]
        edge_0_1_1 = get_doc(edge_0_1_1_id)
        assert edge_0_1_1["weight"] == 3
        assert G._adj == {
            "test_graph_node/0": {
                "test_graph_node/1": {
                    edge_0_1_0_id: edge_0_1_0,
                    edge_0_1_1_id: edge_0_1_1,
                }
            },
            "test_graph_node/1": {},
        }

        assert G._succ == {
            "test_graph_node/0": {
                "test_graph_node/1": {
                    edge_0_1_0_id: edge_0_1_0,
                    edge_0_1_1_id: edge_0_1_1,
                }
            },
            "test_graph_node/1": {},
        }

        assert G._pred == {
            "test_graph_node/1": {
                "test_graph_node/0": {
                    edge_0_1_0_id: edge_0_1_0,
                    edge_0_1_1_id: edge_0_1_1,
                }
            },
            "test_graph_node/0": {},
        }

        G.add_edges_from([(0, 1), (0, 1, {"weight": 3})], weight=2)
        edge_0_1_2_id = G[0][1][2]["_id"]
        edge_0_1_2 = get_doc(edge_0_1_2_id)
        assert edge_0_1_2["weight"] == 2

        edge_0_1_3_id = G[0][1][3]["_id"]
        edge_0_1_3 = get_doc(edge_0_1_3_id)
        assert edge_0_1_3["weight"] == 3

        assert G._adj == {
            "test_graph_node/0": {
                "test_graph_node/1": {
                    edge_0_1_0_id: edge_0_1_0,
                    edge_0_1_1_id: edge_0_1_1,
                    edge_0_1_2_id: edge_0_1_2,
                    edge_0_1_3_id: edge_0_1_3,
                }
            },
            "test_graph_node/1": {},
        }

        assert G._succ == {
            "test_graph_node/0": {
                "test_graph_node/1": {
                    edge_0_1_0_id: edge_0_1_0,
                    edge_0_1_1_id: edge_0_1_1,
                    edge_0_1_2_id: edge_0_1_2,
                    edge_0_1_3_id: edge_0_1_3,
                }
            },
            "test_graph_node/1": {},
        }

        assert G._pred == {
            "test_graph_node/1": {
                "test_graph_node/0": {
                    edge_0_1_0_id: edge_0_1_0,
                    edge_0_1_1_id: edge_0_1_1,
                    edge_0_1_2_id: edge_0_1_2,
                    edge_0_1_3_id: edge_0_1_3,
                }
            },
            "test_graph_node/0": {},
        }

        G = self.EmptyGraph()
        edges = [
            (0, 1, {"weight": 3}),
            (0, 1, (("weight", 2),)),
            (0, 1, 5),
            (0, 1, "s"),
        ]
        G.add_edges_from(edges)

        edge_0_1_0_id = G[0][1][0]["_id"]
        edge_0_1_0 = get_doc(edge_0_1_0_id)
        assert edge_0_1_0["weight"] == 3

        edge_0_1_1_id = G[0][1][1]["_id"]
        edge_0_1_1 = get_doc(edge_0_1_1_id)
        assert edge_0_1_1["weight"] == 2

        edge_0_1_2_id = G[0][1][2]["_id"]
        edge_0_1_2 = get_doc(edge_0_1_2_id)
        assert edge_0_1_2_id != 5
        assert "weight" not in edge_0_1_2

        edge_0_1_3_id = G[0][1][3]["_id"]
        edge_0_1_3 = get_doc(edge_0_1_3_id)
        assert edge_0_1_3_id != "s"
        assert "weight" not in edge_0_1_3

        assert G._succ == {
            "test_graph_node/0": {
                "test_graph_node/1": {
                    edge_0_1_0_id: edge_0_1_0,
                    edge_0_1_1_id: edge_0_1_1,
                    edge_0_1_2_id: edge_0_1_2,
                    edge_0_1_3_id: edge_0_1_3,
                }
            },
            "test_graph_node/1": {},
        }

        assert G._pred == {
            "test_graph_node/1": {
                "test_graph_node/0": {
                    edge_0_1_0_id: edge_0_1_0,
                    edge_0_1_1_id: edge_0_1_1,
                    edge_0_1_2_id: edge_0_1_2,
                    edge_0_1_3_id: edge_0_1_3,
                }
            },
            "test_graph_node/0": {},
        }

        # too few in tuple
        pytest.raises(nx.NetworkXError, G.add_edges_from, [(0,)])
        # too many in tuple
        pytest.raises(nx.NetworkXError, G.add_edges_from, [(0, 1, 2, 3, 4)])
        # not a tuple
        pytest.raises(TypeError, G.add_edges_from, [0])
        with pytest.raises(ValueError, match="Key cannot be None"):
            G.add_edges_from([(None, 3), (3, 2)])

    def test_remove_edge(self):
        G = self.K3Graph()
        assert db.has_document(list(G[0][1])[0])
        G.remove_edge(0, 1)
        with pytest.raises(KeyError):
            G[0][1]

        edge_1_0_id = list(G[1][0])[0]
        edge_1_0 = get_doc(edge_1_0_id)

        edge_0_2_id = list(G[0][2])[0]
        edge_0_2 = get_doc(edge_0_2_id)

        edge_2_0_id = list(G[2][0])[0]
        edge_2_0 = get_doc(edge_2_0_id)

        edge_1_2_id = list(G[1][2])[0]
        edge_1_2 = get_doc(edge_1_2_id)

        edge_2_1_id = list(G[2][1])[0]
        edge_2_1 = get_doc(edge_2_1_id)

        assert G._succ == {
            "test_graph_node/0": {"test_graph_node/2": {edge_0_2_id: edge_0_2}},
            "test_graph_node/1": {
                "test_graph_node/0": {edge_1_0_id: edge_1_0},
                "test_graph_node/2": {edge_1_2_id: edge_1_2},
            },
            "test_graph_node/2": {
                "test_graph_node/0": {edge_2_0_id: edge_2_0},
                "test_graph_node/1": {edge_2_1_id: edge_2_1},
            },
        }

        assert G._pred == {
            "test_graph_node/0": {
                "test_graph_node/1": {edge_1_0_id: edge_1_0},
                "test_graph_node/2": {edge_2_0_id: edge_2_0},
            },
            "test_graph_node/1": {
                "test_graph_node/2": {edge_2_1_id: edge_2_1},
            },
            "test_graph_node/2": {
                "test_graph_node/0": {edge_0_2_id: edge_0_2},
                "test_graph_node/1": {edge_1_2_id: edge_1_2},
            },
        }

        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, -1, 0)
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, 0, 2, key=1)


# TODO: Revisit
# Subgraphing not implemented yet
# class TestEdgeSubgraph(_TestMultiGraphEdgeSubgraph):
#     """Unit tests for the :meth:`MultiDiGraph.edge_subgraph` method."""

#     def setup_method(self):
#         # Create a quadruply-linked path graph on five nodes.
#         G = nx.MultiDiGraph()
#         nx.add_path(G, range(5))
#         nx.add_path(G, range(5))
#         nx.add_path(G, reversed(range(5)))
#         nx.add_path(G, reversed(range(5)))
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


# class CustomDictClass(UserDict):
#     pass


# class MultiDiGraphSubClass(nx.MultiDiGraph):
#     node_dict_factory = CustomDictClass  # type: ignore[assignment]
#     node_attr_dict_factory = CustomDictClass  # type: ignore[assignment]
#     adjlist_outer_dict_factory = CustomDictClass  # type: ignore[assignment]
#     adjlist_inner_dict_factory = CustomDictClass  # type: ignore[assignment]
#     edge_key_dict_factory = CustomDictClass  # type: ignore[assignment]
#     edge_attr_dict_factory = CustomDictClass  # type: ignore[assignment]
#     graph_attr_dict_factory = CustomDictClass  # type: ignore[assignment]


# class TestMultiDiGraphSubclass(TestMultiDiGraph):
#     def setup_method(self):
#         self.Graph = MultiDiGraphSubClass
#         # build K3
#         self.k3edges = [(0, 1), (0, 2), (1, 2)]
#         self.k3nodes = [0, 1, 2]
#         self.K3 = self.Graph()
#         self.K3._succ = self.K3.adjlist_outer_dict_factory(
#             {
#                 0: self.K3.adjlist_inner_dict_factory(),
#                 1: self.K3.adjlist_inner_dict_factory(),
#                 2: self.K3.adjlist_inner_dict_factory(),
#             }
#         )
#         # K3._adj is synced with K3._succ
#         self.K3._pred = {0: {}, 1: {}, 2: {}}
#         for u in self.k3nodes:
#             for v in self.k3nodes:
#                 if u == v:
#                     continue
#                 d = {0: {}}
#                 self.K3._succ[u][v] = d
#                 self.K3._pred[v][u] = d
#         self.K3._node = self.K3.node_dict_factory()
#         self.K3._node[0] = self.K3.node_attr_dict_factory()
#         self.K3._node[1] = self.K3.node_attr_dict_factory()
#         self.K3._node[2] = self.K3.node_attr_dict_factory()
