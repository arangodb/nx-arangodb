Development Sandbox:

<a href="https://colab.research.google.com/drive/1gIfJDEumN6UdZou_VlSbG874xGkHwtU2?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

What's currently possible:
- Algorithm dispatching for GPU & CPU (`betweenness_centrality`, `pagerank`, `louvain_communities`)
- Data Load from ArangoDB to `nx`
- Data Load from ArangoDB to `nxcg`

Next Milestone:
- NetworkX CRUD Interface for ArangoDB

Planned, but not yet scopped:
- NetworkX Graph Query Method
- Data Write to ArangoDB from `nx`
- Data Write to ArangoDB from `nxcg`

```py

import networkx as nx
import nx_arangodb as nxadb

G_1 = nx.karate_club_graph()

G_2 = nxadb.Graph(G_1)

bc_1 = nx.betweenness_centrality(G_1)
bc_2 = nx.betweenness_centrality(G_2)
bc_3 = nx.betweenness_centrality(G_1, backend="arangodb")
bc_4 = nx.betweenness_centrality(G_2, backend="arangodb")
```
