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

    # TODO: Use `nx.config.backends.arangodb.backend_priority` instead
    backend_priority = []
    if nxadb.convert.GPU_ENABLED:
        backend_priority.append("cugraph")

    for backend in backend_priority:
        if not dfunc.__wrapped__._should_backend_run(backend, *args, **kwargs):
            logger.warning(f"'{func_name}' cannot be run on backend '{backend}'")
            continue

        try:
            return _run_with_backend(
                backend,
                dfunc,
                args,
                kwargs,
            )

        except NotImplementedError:
            logger.warning(f"'{func_name}' not implemented for backend '{backend}'")
            pass

    default_backend = "networkx"
    logger.debug(f"'{func_name}' running on default backend '{default_backend}'")
    return _run_with_backend(default_backend, dfunc, args, kwargs)


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

    if dfunc._returns_graph:
        raise NotImplementedError("Not implemented yet")
        # if to_backend is not None:
        #     result = to_backend.convert_to_nx(result)

        # result = from_backend.convert_from_nx(
        #     result,
        #     preserve_edge_attrs=True,
        #     preserve_node_attrs=True,
        #     preserve_graph_attrs=True,
        #     name=func_name,
        # )

    return result


def _convert_to_backend(G_from: Any, backend_name: str) -> Any:
    if backend_name == "networkx":
        pull_graph = nx.config.backends.arangodb.pull_graph
        return nxadb._to_nx_graph(G_from, pull_graph=pull_graph)

    if backend_name == "cugraph":
        return nxadb._to_nxcg_graph(G_from)

    raise ValueError(f"Unsupported backend: '{backend_name}'")


backend_interface = BackendInterface()
