class NetworkXArangoDBException(Exception):
    pass


class GraphDoesNotExist(NetworkXArangoDBException):
    pass


class DatabaseNotSet(NetworkXArangoDBException):
    pass


class GraphNameNotSet(NetworkXArangoDBException):
    pass


class InvalidTraversalDirection(NetworkXArangoDBException):
    pass


EDGE_ALREADY_EXISTS_ERROR_CODE = 1210


class EdgeAlreadyExists(NetworkXArangoDBException):
    pass


class AQLMultipleResultsFound(NetworkXArangoDBException):
    pass


class ArangoDBAlgorithmError(NetworkXArangoDBException):
    pass


class ShortestPathError(ArangoDBAlgorithmError):
    pass
