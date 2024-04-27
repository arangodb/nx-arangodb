Doesn't do much for now 

```py

import networkx as nx
import nx_arangodb as nxadb

G_1 = nx.karate_club_graph()

G_2 = nxadb.Graph(G_1)

bc_1 = nx.betweenness_centrality(G_1)
bc_2 = nx.betweenness_centrality(G_2) # Goes through dispatching

assert bc_1 == bc_2

```
