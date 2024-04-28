My GPU Dev Environment:

<a href="https://colab.research.google.com/drive/1gIfJDEumN6UdZou_VlSbG874xGkHwtU2?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Doesn't do much for now 

```py

import networkx as nx
import nx_arangodb as nxadb

G_1 = nx.karate_club_graph()

G_2 = nxadb.Graph(G_1)

bc_1 = nx.betweenness_centrality(G_1)
bc_2 = nx.betweenness_centrality(G_2)
bc_3 = nx.betweenness_centrality(G_1, backend="arangodb")
bc_4 = nx.betweenness_centrality(G_2, backend="arangodb")

assert bc_1 == bc_2 == bc_3 == bc_4

```
