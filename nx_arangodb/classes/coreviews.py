"""Experimental overrides of the NetworkX Views that represent the
core data structures such as nested Mappings (e.g. dict-of-dicts).

Overriding these classes allows us to implement custom logic for
data filtering and updating in the database, instead of in Python.

These classes are a work-in-progress. The main goal is to try
to delegate data processing to ArangoDB, whenever possible.

To use these experimental views, you must set **use_arango_views=True**
when creating a new graph object:
>>> G = nxadb.Graph(name="MyGraph", use_arango_views=True)
"""

import networkx as nx


class ArangoAdjacencyView(nx.classes.coreviews.AdjacencyView):
    """The ArangoAdjacencyView class is an experimental subclass of
    the AdjacencyView class.

    Contrary to the original AdjacencyView class, the ArangoAdjacencyView
    is writable to allow for bulk updates to the graph in the DB.
    """

    def update(self, data):
        """Update a set of edges within the graph.

        The benefit of this method is that it allows for bulk API updates,
        as opposed to `G.add_edges_from`, which currently makes
        one API request per edge.

        Example
        -------
        >>> G = nxadb.Graph(name="MyGraph")
        >>> G.adj.update(
            {
                'node/1': {
                    'node/2': {"node_to_node/1", "foo": "bar"},
                    'node/3': {"node_to_node/2", "foo": "baz"},
                    ...
                },
                ...
            })
        """
        return self._atlas.update(data)

    def __getitem__(self, name):
        return ArangoAtlasView(self._atlas[name])


class ArangoAtlasView(nx.classes.coreviews.AtlasView):
    """The ArangoAtlasView class is an experimental subclass of the
    AtlasView class.

    Contrary to the original AtlasView class, the ArangoAtlasView is
    writable to allow for bulk updates to the graph in the DB.
    """

    def update(self, data):
        """Update a set of edges within the graph for a specific node.

        Example
        -------
        >>> G = nxadb.Graph(name="MyGraph")
        >>> G.adj['node/1'].update(
            {
                'node/2': {"node_to_node/1", "foo": "bar"},
                'node/3': {"node_to_node/2", "foo": "baz"},
                ...
            })
        """
        return self._atlas.update(data)
