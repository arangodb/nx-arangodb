"""
An override of the NodeView, NodeDataView, EdgeView, and EdgeDataView classes
to allow for custom data filtering in the database instead of in Python.
"""

from __future__ import annotations

import networkx as nx

import nx_arangodb as nxadb

from .function import get_node_id


class CustomNodeView(nx.classes.reportviews.NodeView):
    def __call__(self, data=False, default=None):
        if data is False:
            return self
        return CustomNodeDataView(self._nodes, data, default)

    def data(self, data=True, default=None):
        if data is False:
            return self
        return CustomNodeDataView(self._nodes, data, default)

    def update(self, data):
        return self._nodes.update(data)


class CustomNodeDataView(nx.classes.reportviews.NodeDataView):
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


class CustomEdgeDataView(nx.classes.reportviews.EdgeDataView):

    ######################
    # NOTE: Monkey Patch #
    ######################

    def __iter__(self):
        if self._nbunch is None and self._data not in [None, True, False]:
            # Reason: We can utilize AQL to filter the data we
            # want to return, instead of filtering it in Python
            # This is hacky for now, but it's meant to show that
            # the data can be filtered server-side.
            # We solve this by relying on self._adjdict, which
            # is the AdjListOuterDict object that has a custom
            # items() method that can filter data with AQL.

            # Filter for self._data  server-side
            yield from self._adjdict.items(data=self._data, default=self._default)
        else:
            yield from super().__iter__()


class CustomEdgeView(nx.classes.reportviews.EdgeView):
    dataview = CustomEdgeDataView

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
