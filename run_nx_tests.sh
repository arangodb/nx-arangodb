# Copied from nx-cugraph
set -e

# TODO: address the following tests
# --pyargs networkx.algorithms.community.louvain \

DATABASE_HOST=http://localhost:8529
DATABASE_USERNAME=root
DATABASE_PASSWOR=test
DATABASE_NAME=_system

NETWORKX_GRAPH_CONVERT=arangodb \
NETWORKX_TEST_BACKEND=arangodb \
NETWORKX_FALLBACK_TO_NX=True \
    pytest \
    --pyargs networkx.classes \
    --pyargs networkx.algorithms.centrality \
    --pyargs networkx.algorithms.link_analysis \
    --pyargs networkx.algorithms.shortest_paths \
    --cov-config=$(dirname $0)/pyproject.toml \
    --cov=nx_arangodb \
    --cov-report= \
    "$@"
coverage report \
    --include="*/nx_arangodb/algorithms/*" \
    --omit=__init__.py \
    --show-missing \
    --rcfile=$(dirname $0)/pyproject.toml
