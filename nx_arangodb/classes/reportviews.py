"""Experimental overrides of the NetworkX Views that represent the
nodes and edges of the graph.

Overriding these classes allows us to implement custom logic for
data filtering and updating in the database, instead of in Python.

These classes are a work-in-progress. The main goal is to try
to delegate data processing to ArangoDB, whenever possible.

To use these experimental views, you must set **use_arango_views=True**
when creating a new graph object:
>>> G = nxadb.Graph(name="MyGraph", use_arango_views=True)
"""

from __future__ import annotations

import networkx as nx

import nx_arangodb as nxadb


class ArangoNodeView(nx.classes.reportviews.NodeView):
    """The ArangoNodeView class is an experimental subclass of the
    NodeView class.

    Contrary to the original NodeView class, the ArangoNodeView is
    writable to allow for bulk updates to the graph in the DB.
    """

    # DataView method
    def __call__(self, data=False, default=None):
        if data is False:
            return self
        return ArangoNodeDataView(self._nodes, data, default)

    def data(self, data=True, default=None):
        """Return a read-only view of node data.

        Parameters
        ----------
        data : bool or node data key, default=True
            If ``data=True`` (the default), return a `NodeDataView` object that
            maps each node to *all* of its attributes. `data` may also be an
            arbitrary key, in which case the `NodeDataView` maps each node to
            the value for the keyed attribute. In this case, if a node does
            not have the `data` attribute, the `default` value is used.
        default : object, default=None
            The value used when a node does not have a specific attribute.

        Returns
        -------
        NodeDataView
            The layout of the returned NodeDataView depends on the value of the
            `data` parameter.

        Notes
        -----
        If ``data=False``, returns a `NodeView` object without data.

        See Also
        --------
        NodeDataView
        """
        if data is False:
            return self
        return ArangoNodeDataView(self._nodes, data, default)

    def update(self, data):
        """Update a set of nodes within the graph.

        The benefit of this method is that it allows for bulk API updates,
        as opposed to `G.add_nodes_from`, which currently makes
        one API request per node.

        Example
        -------
        >>> G = nxadb.Graph(name="MyGraph")
        >>> G.nodes.update(
            {
                'node/1': {"node/1", "foo": "bar"},
                'node/2': {"node/2", "foo": "baz"},
                ...
            })
        """
        return self._nodes.update(data)


class ArangoNodeDataView(nx.classes.reportviews.NodeDataView):
    """The ArangoNodeDataView class is an experimental subclass of the
    NodeDataView class.

    The main use for this class is to iterate through node-data pairs.
    The data can be the entire data-dictionary for each node, or it
    can be a specific attribute (with default) for each node.

    In the event that the data is a specific attribute, the data is
    filtered server-side, instead of in Python. This is done by using
    the ArangoDB Query Language (AQL) to filter the data.
    """

    def __iter__(self):
        data = self._data
        if data is False:
            return iter(self._nodes)
        if data is True:
            return iter(self._nodes.items())

        ######################
        # NOTE: Monkey Patch #
        ######################

        # Old:
        # return (
        #     (n, dd[data] if data in dd else self._default)
        #     for n, dd in self._nodes.items()
        # )

        # New:
        return iter(self._nodes.items(data=data, default=self._default))

        # Reason: We can utilize AQL to filter the data we
        # want to return, instead of filtering it in Python

        ###########################


class ArangoEdgeDataView(nx.classes.reportviews.EdgeDataView):
    """The ArangoEdgeDataView class is an experimental subclass of the
    EdgeDataView class.

    This view is primarily used to iterate over the edges reporting
    edges as node-tuples with edge data optionally reported.

    In the event that the data is a specific attribute, the data is
    filtered server-side, instead of in Python. This is done by using
    the ArangoDB Query Language (AQL) to filter the data.
    """

    def __iter__(self):
        ######################
        # NOTE: Monkey Patch #
        ######################

        if self._nbunch is None and self._data not in [None, True, False]:
            # Reason: We can utilize AQL to filter the data we
            # want to return, instead of filtering it in Python
            # This is hacky for now, but it's meant to show that
            # the data can be filtered server-side.
            # We solve this by relying on self._adjdict, which
            # is the AdjListOuterDict object that has a custom
            # items() method that can filter data with AQL.

            yield from self._adjdict.items(data=self._data, default=self._default)
        else:
            yield from super().__iter__()


class ArangoEdgeView(nx.classes.reportviews.EdgeView):
    """The ArangoEdgeView class is an experimental subclass of the
    EdgeView class.

    The __len__ method is overridden to count the number of edges
    in the graph by querying the database, instead of iterating
    through the edges in Python.
    """

    dataview = ArangoEdgeDataView

    def __len__(self):

        ######################
        # NOTE: Monkey Patch #
        ######################

        # Old:
        # num_nbrs = (len(nbrs) + (n in nbrs) for n, nbrs in self._nodes_nbrs())
        # return sum(num_nbrs) // 2

        # New:
        G: nxadb.Graph = self._graph
        return sum(
            [
                G.db.collection(ed["edge_collection"]).count()
                for ed in G.adb_graph.edge_definitions()
            ]
        )

        # Reason: We can utilize AQL to count the number of edges
        # instead of making individual requests to the database
        # i.e avoid having to do `n in nbrs` for each node

        ######################
