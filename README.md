Development Sandbox:

<a href="https://colab.research.google.com/drive/1gIfJDEumN6UdZou_VlSbG874xGkHwtU2?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

What's currently possible:
- ArangoDB CRUD Interface for `nx.Graph`
- Algorithm dispatching to `nx` & `nxcg` (`betweenness_centrality`, `pagerank`, `louvain_communities`)
- Algorithm dispatching to ArangoDB (`shortest_path`)
- Data Load from ArangoDB to `nx` object
- Data Load from ArangoDB to `nxcg` object
- Data Load from ArangoDB via dictionary-based remote connection

Next steps:
- Generalize `nxadb`'s support for `nx` & `nxcg` algorithms
- Improve support for `nxadb.DiGraph`
- CRUD Interface Improvements

Planned:
- Support for `nxadb.MultiGraph` & `nxadb.MultiDiGraph`
- Data Load from `nx` to ArangoDB
- Data Load from `nxcg` to ArangoDB

```py
import os
import networkx as nx
import nx_arangodb as nxadb

os.environ["DATABASE_HOST"] = "http://localhost:8529"
os.environ["DATABASE_USERNAME"] = "root"
os.environ["DATABASE_PASSWORD"] = "password"
os.environ["DATABASE_NAME"] = "_system"

G = nxadb.Graph(graph_name="KarateGraph")

G_nx = nx.karate_club_graph()
assert len(G.nodes) == len(G_nx.nodes)
assert len(G.adj) == len(G_nx.adj)
assert len(G.edges) == len(G_nx.edges)

nx.betweenness_centrality(G)
nx.pagerank(G)
nx.community.louvain_communities(G)
nx.shortest_path(G, "person/1", "person/34")
nx.all_neighbors(G, "person/1")

G.nodes(data='club', default='unknown')
G.edges(data='weight', default=1000)

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
```