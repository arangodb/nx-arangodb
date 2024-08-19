import networkx as nx


class CustomAdjacencyView(nx.classes.coreviews.AdjacencyView):

    def update(self, data):
        return self._atlas.update(data)
