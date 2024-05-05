import networkx as nx


class CustomNodeView(nx.classes.reportviews.NodeView):
    def __call__(self, data=False, default=None):
        if data is False:
            return self
        return CustomNodeDataView(self._nodes, data, default)

    def data(self, data=True, default=None):
        if data is False:
            return self
        return CustomNodeDataView(self._nodes, data, default)


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
