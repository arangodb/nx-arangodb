import sys
import time
from io import StringIO

import cupy
import networkx as nx
import pytest

import nx_arangodb as nxadb
from tests.conftest import assert_pagerank, create_grid_graph


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

    # Note: While this works, we should use the logger or some alternative
    # approach testing this. Via stdout is not the best way to test this.
    with Capturing() as output_gpu:
        res_gpu = nx.pagerank(graph)

    assert any(
        "NXCG Graph construction took" in line for line in output_gpu
    ), "Expected output not found in GPU execution"

    gpu_time = time.time() - start_gpu

    # Disable GPU and measure CPU execution time
    nxadb.convert.GPU_ENABLED = False
    start_cpu = time.time()
    with Capturing() as output_cpu:
        res_cpu = nx.pagerank(graph)

    output_cpu_list = list(output_cpu)
    assert len(output_cpu_list) == 1
    assert "Graph 'GridGraph' load took" in output_cpu_list[0]

    cpu_time = time.time() - start_cpu

    assert gpu_time < cpu_time, "GPU execution should be faster than CPU execution"
    assert_pagerank(res_gpu, res_cpu)
