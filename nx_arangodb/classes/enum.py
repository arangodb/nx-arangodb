from enum import Enum, auto


class TraversalDirection(Enum):
    OUTBOUND = auto()
    INBOUND = auto()
    ANY = auto()


class GraphType(Enum):
    Graph = auto()
    DiGraph = auto()
    MultiGraph = auto()
    MultiDiGraph = auto()


DIRECTED_GRAPH_TYPES = {GraphType.DiGraph.name, GraphType.MultiDiGraph.name}
MULTIGRAPH_TYPES = {GraphType.MultiGraph.name, GraphType.MultiDiGraph.name}
