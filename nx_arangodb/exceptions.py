class NetworkXArangoDBException(Exception):
    pass


class GraphDoesNotExist(NetworkXArangoDBException):
    pass


class DatabaseNotSet(NetworkXArangoDBException):
    pass


class GraphNameNotSet(NetworkXArangoDBException):
    pass


class GraphNotEmpty(NetworkXArangoDBException):
    pass


class InvalidTraversalDirection(NetworkXArangoDBException):
    pass


class EdgeAlreadyExists(NetworkXArangoDBException):
    pass


class AQLMultipleResultsFound(NetworkXArangoDBException):
    pass


class ArangoDBAlgorithmError(NetworkXArangoDBException):
    pass


class MultipleEdgesFound(NetworkXArangoDBException):
    pass


class EdgeTypeAmbiguity(NetworkXArangoDBException):
    pass


class InvalidDefaultNodeType(NetworkXArangoDBException):
    pass
