from __future__ import annotations

import os
import sys
from functools import partial
from typing import Any, Callable, Protocol, Set

import networkx as nx
from networkx.utils.backends import _load_backend, _registered_algorithms

import nx_arangodb as nxadb
from nx_arangodb.logger import logger

# Avoid infinite recursion when testing
_IS_TESTING = os.environ.get("NETWORKX_TEST_BACKEND") in {"arangodb"}


class NetworkXFunction(Protocol):
    graphs: dict[str, Any]
    name: str
    list_graphs: Set[str]
    orig_func: Callable[..., Any]
    _returns_graph: bool


class BackendInterface:
    @staticmethod
    def convert_from_nx(graph: nx.Graph, *args: Any, **kwargs: Any) -> nxadb.Graph:
        return nxadb._to_nxadb_graph(graph, *args, **kwargs)

    @staticmethod
    def convert_to_nx(obj: Any, *args: Any, **kwargs: Any) -> nx.Graph:
        if not isinstance(obj, nxadb.Graph):
            return obj

        return nxadb._to_nx_graph(obj, *args, **kwargs)

    def __getattr__(self, attr: str, *, from_backend_name: str = "arangodb") -> Any:
        """
        Dispatching mechanism for all networkx algorithms. This avoids having to
        write a separate function for each algorithm.
        """
        if (
            attr not in _registered_algorithms
            or _IS_TESTING
            and attr in {"empty_graph"}
        ):
            raise AttributeError(attr)

        if from_backend_name != "arangodb":
            raise ValueError(f"Unsupported source backend: '{from_backend_name}'")

        return partial(_auto_func, attr)


def _auto_func(func_name: str, /, *args: Any, **kwargs: Any) -> Any:
    """
    Function to automatically dispatch to the correct backend for a given algorithm.

    :param func_name: The name of the algorithm to run.
    :type func_name: str
    """
    dfunc = _registered_algorithms[func_name]

    backend_priority: list[str] = []
    if nxadb.convert.GPU_AVAILABLE and nx.config.backends.arangodb.use_gpu:
        backend_priority.append("cugraph")

    for backend in backend_priority:
        if not _should_backend_run(backend, func_name, *args, **kwargs):
            continue

        try:
            return _run_with_backend(
                backend,
                dfunc,
                args,
                kwargs,
            )

        except NotImplementedError:
            logger.debug(f"'{func_name}' not implemented for backend '{backend}'")
            pass

    default_backend = "networkx"
    logger.debug(f"'{func_name}' running on default backend '{default_backend}'")
    return _run_with_backend(default_backend, dfunc, args, kwargs)


def _should_backend_run(
    backend_name: str, func_name: str, *args: Any, **kwargs: Any
) -> bool:
    """
    Determine if a specific backend should be used for a given algorithm.

    Copied from networkx-3.4.1/networkx/utils/backends.py#L1514-L1535
    to patch the implementation for backwards compatibility, as the signature of
    this function in NetworkX 3.3 is different from the one in NetworkX 3.4.

    :param backend_name: The name of the backend to check.
    :type backend_name: str
    :param func_name: The name of the algorithm/function to be run.
    :type func_name: str
    :param args: Variable length argument list for the function.
    :type args: Any
    :param kwargs: Arbitrary keyword arguments for the function.
    :type kwargs: Any
    :returns: Whether the backend should be used.
    :rtype: bool
    """
    if backend_name == "networkx":
        return True

    backend = _load_backend(backend_name)
    should_run = backend.should_run(func_name, args, kwargs)
    if isinstance(should_run, str) or not should_run:
        reason = f", because: {should_run}" if isinstance(should_run, str) else ""
        logger.debug(f"Backend '{backend_name}' not used for '{func_name}' ({reason})")
        return False

    return True


def _run_with_backend(
    backend_name: str,
    dfunc: NetworkXFunction,
    args: Any,
    kwargs: Any,
) -> Any:
    """
    :param backend: The name of the backend to run the algorithm on.
    :type backend: str
    :param dfunc: The function to run.
    :type dfunc: NetworkXFunction
    """
    func_name = dfunc.name
    backend_func = (
        dfunc.orig_func
        if backend_name == "networkx"
        else getattr(_load_backend(backend_name), func_name)
    )

    graphs_resolved = {
        gname: val
        for gname, pos in dfunc.graphs.items()
        if (val := args[pos] if pos < len(args) else kwargs.get(gname)) is not None
    }

    if dfunc.list_graphs:
        graphs_converted = {
            gname: (
                [_convert_to_backend(g, backend_name) for g in val]
                if gname in dfunc.list_graphs
                else _convert_to_backend(val, backend_name)
            )
            for gname, val in graphs_resolved.items()
        }
    else:
        graphs_converted = {
            gname: _convert_to_backend(graph, backend_name)
            for gname, graph in graphs_resolved.items()
        }

    converted_args = list(args)
    converted_kwargs = dict(kwargs)

    for gname, val in graphs_converted.items():
        if gname in kwargs:
            converted_kwargs[gname] = val
        else:
            converted_args[dfunc.graphs[gname]] = val

    result = backend_func(*converted_args, **converted_kwargs)

    # TODO: Convert to nxadb.Graph?
    # What would this look like? Create a new graph in ArangoDB?
    # Or just establish a remote connection?
    # For now, if dfunc._returns_graph is True, it will return a
    # regular nx.Graph object.
    # if dfunc._returns_graph:
    #     raise NotImplementedError("Returning Graphs not implemented yet")

    return result


def _convert_to_backend(G_from: Any, backend_name: str) -> Any:
    if backend_name == "networkx":
        return nxadb._to_nx_graph(G_from)

    if backend_name == "cugraph":
        return nxadb._to_nxcg_graph(G_from)

    raise ValueError(f"Unsupported backend: '{backend_name}'")


backend_interface = BackendInterface()
