import sys
import time
from io import StringIO

import cupy
import networkx as nx
import pytest

import nx_arangodb as nxadb

from .conftest import create_grid_graph


# Taken from:
# https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class Capturing(list[str]):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def assert_same_dict_values(
    d1: dict[str | int, float], d2: dict[str | int, float], digit: int
) -> None:
    if type(next(iter(d1.keys()))) == int:
        d1 = {f"person/{k+1}": v for k, v in d1.items()}  # type: ignore

    if type(next(iter(d2.keys()))) == int:
        d2 = {f"person/{k+1}": v for k, v in d2.items()}  # type: ignore

    assert d1.keys() == d2.keys(), "Dictionaries have different keys"
    for key in d1:
        m = f"Values for key '{key}' are not equal up to digit {digit}"
        assert round(d1[key], digit) == round(d2[key], digit), m


def assert_pagerank(d1: dict[str | int, float], d2: dict[str | int, float]) -> None:
    assert d1
    assert d2
    assert_same_dict_values(d1, d2, 5)


@pytest.mark.parametrize(
    "graph_cls",
    [
        (nxadb.Graph),
        (nxadb.DiGraph),
        (nxadb.MultiGraph),
        (nxadb.MultiDiGraph),
    ],
)
def test_adb_graph_gpu_pagerank(graph_cls: type[nxadb.Graph]) -> None:
    graph = create_grid_graph(graph_cls)
    # sleep 1s to make sure potential async operations are finished (writes to adb)
    # first iteration needs to write the graph to adb, whereas the other iterations
    # can directly read the graph from adb
    time.sleep(1)

    res_gpu = None
    res_cpu = None

    # Measure GPU execution time
    nxadb.convert.GPU_ENABLED = True
    start_gpu = time.time()
    with Capturing() as output_gpu:
        res_gpu = nx.pagerank(graph)

    assert any(
        "nx_cugraph.Graph" in line for line in output_gpu
    ), "Expected output not found in GPU execution"
    assert any(
        "NXCG Graph construction took" in line for line in output_gpu
    ), "Expected output not found in GPU execution"

    gpu_time = time.time() - start_gpu

    # Disable GPU and measure CPU execution time
    nxadb.convert.GPU_ENABLED = False
    start_cpu = time.time()
    with Capturing() as output_cpu:
        res_cpu = nx.pagerank(graph)

    assert all(
        "nx_cugraph.Graph" not in line for line in output_cpu
    ), "Unexpected GPU-related output found in CPU execution"
    assert all(
        "NXCG Graph construction took" not in line for line in output_cpu
    ), "Unexpected GPU-related output found in CPU execution"

    cpu_time = time.time() - start_cpu

    assert gpu_time < cpu_time, "GPU execution should be faster than CPU execution"
    assert_pagerank(res_gpu, res_cpu)
