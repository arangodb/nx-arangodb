import networkx as nx


# TODO: WIP - This is a placeholder for now
class CustomAdjacencyView(nx.classes.coreviews.AdjacencyView):

    def __getitem__(self, name):
        return super().__getitem__(name)

    def copy(self):
        return super().copy()
