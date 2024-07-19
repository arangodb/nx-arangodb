from __future__ import annotations

import os
import sys
from functools import partial
from typing import Any, Callable, Protocol, Set

import networkx as nx
from networkx.utils.backends import _load_backend, _registered_algorithms

import nx_arangodb as nxadb

# Avoid infinite recursion when testing
_IS_TESTING = os.environ.get("NETWORKX_TEST_BACKEND") in {"arangodb"}


class NetworkXFunction(Protocol):
    graphs: dict[str, Any]
    name: str
    list_graphs: Set[str]
    orig_func: Callable[..., Any]
    _returns_graph: bool


class AbstractBackendInterface(Protocol):
    @staticmethod
    def convert_from_nx(
        graph: Any, *args: Any, **kwargs: Any
    ) -> nxadb.Graph | nxadb.DiGraph: ...

    @staticmethod
    def convert_to_nx(
        obj: nx.Graph | nx.DiGraph | nxadb.Graph | nxadb.DiGraph,
        *,
        name: str | None = None,
    ) -> nx.Graph | nx.DiGraph: ...


class BackendInterface:
    # Required conversions
    @staticmethod
    def convert_from_nx(
        graph: Any, *args: Any, **kwargs: Any
    ) -> nxadb.Graph | nxadb.DiGraph:
        return nxadb.from_networkx(graph, *args, **kwargs)

    @staticmethod
    def convert_to_nx(
        obj: nx.Graph | nx.DiGraph | nxadb.Graph | nxadb.DiGraph,
        *,
        name: str | None = None,
    ) -> nx.Graph | nx.DiGraph:
        if isinstance(obj, nxadb.Graph):
            return nxadb.to_networkx(obj)
        return obj

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
        return partial(_auto_func, from_backend_name, attr)


def _auto_func(
    from_backend_name: str, func_name: str, /, *args: Any, **kwargs: Any
) -> Any:
    """
    Function to automatically dispatch to the correct backend for a given algorithm.

    :param from_backend_name: The source backend.
    :type from_backend_name: str
    :param func_name: The name of the algorithm to run.
    :type func_name: str
    """
    dfunc = _registered_algorithms[func_name]

    # TODO: Use `nx.config.backends.arangodb.backend_priority` instead
    backend_priority = []
    if nxadb.convert.GPU_ENABLED:
        backend_priority.append("cugraph")

    for to_backend_name in backend_priority:
        if not dfunc.__wrapped__._should_backend_run(to_backend_name, *args, **kwargs):
            continue

        try:
            return _run_with_backend(
                from_backend_name,
                to_backend_name,
                dfunc,
                args,
                kwargs,
            )

        except NotImplementedError:
            pass

    return _run_with_backend(from_backend_name, "networkx", dfunc, args, kwargs)


def _run_with_backend(
    from_backend_name: str,
    to_backend_name: str,
    dfunc: NetworkXFunction,
    args: Any,
    kwargs: Any,
) -> Any:
    """
    :param from_backend_name: The source backend.
    :type from_backend_name: str
    :param to_backend_name: The name of the backend to run the algorithm on.
    :type to_backend_name: str
    :param dfunc: The function to run.
    :type dfunc: Callable
    """

    from_backend = _load_backend(from_backend_name)
    to_backend = (
        _load_backend(to_backend_name) if to_backend_name != "networkx" else None
    )

    graphs_resolved = {
        gname: val
        for gname, pos in dfunc.graphs.items()
        if (val := args[pos] if pos < len(args) else kwargs.get(gname)) is not None
    }

    func_name = dfunc.name
    if dfunc.list_graphs:
        graphs_converted = {
            gname: (
                [
                    _convert_to_backend(g, from_backend, to_backend, func_name)
                    for g in val
                ]
                if gname in dfunc.list_graphs
                else _convert_to_backend(val, from_backend, to_backend, func_name)
            )
            for gname, val in graphs_resolved.items()
        }
    else:
        graphs_converted = {
            gname: _convert_to_backend(graph, from_backend, to_backend, func_name)
            for gname, graph in graphs_resolved.items()
        }

    converted_args = list(args)
    converted_kwargs = dict(kwargs)

    for gname, val in graphs_converted.items():
        if gname in kwargs:
            converted_kwargs[gname] = val
        else:
            converted_args[dfunc.graphs[gname]] = val

    backend_func = (
        dfunc.orig_func if to_backend is None else getattr(to_backend, func_name)
    )

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


def _convert_to_backend(G_from, from_backend, to_backend, func_name):
    if to_backend is None:  # NetworkX
        pull_graph = nx.config.backends.arangodb.pull_graph
        return nxadb.convert._to_nx_graph(G_from, pull_graph=pull_graph)

    return nxadb.convert._to_nxcg_graph(G_from)
