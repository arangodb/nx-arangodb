# algorithms

This is an experimental module seeking to provide server-side algorithms for `nx-arangodb` Graphs. The goal is to provide a set of algorithms that can be delegated to the server for processing, rather than having to pull all the data to the client and process it there.

Currently, the module is in a very early stage and only provides a single algorithm: `shortestPath`. This is simply to demonstrate the potential of the module and to provide a starting point for further development.

```python
import os
import networkx as nx
from nx_arangodb as nxadb

# os.environ ...

G = nxadb.Graph(name="MyGraph")

nx.pagerank(G) # Runs on the client
nx.shortest_path(G, source="A", target="B") # Runs on the DB server
nx.shortest_path.orig_func(G, source="A", target="B") # Runs on the client
```

As ArangoDB continues to grow its Graph Analytics capabilities, this module will be updated to take advantage of those features. Stay tuned!