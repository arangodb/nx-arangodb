import networkx as nx


class CustomAdjacencyView(nx.classes.coreviews.AdjacencyView):

    def update(self, data):
        return self._atlas.update(data)

    def __getitem__(self, name):
        return CustomAtlasView(self._atlas[name])


class CustomAtlasView(nx.classes.coreviews.AtlasView):

    def update(self, data):
        return self._atlas.update(data)
