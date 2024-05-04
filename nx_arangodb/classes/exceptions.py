class NetworkXArangoDBException(Exception):
    pass


EDGE_ALREADY_EXISTS_ERROR_CODE = 1210


class EdgeAlreadyExists(NetworkXArangoDBException):
    """Raised when trying to add an edge that already exists in the graph."""

    pass


class AQLMultipleResultsFound(NetworkXArangoDBException):
    """Raised when multiple results are returned from a query that was expected to return a single result."""

    pass
