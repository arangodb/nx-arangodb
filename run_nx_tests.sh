# Copied from nx-cugraph

NETWORKX_GRAPH_CONVERT=arangodb \
NETWORKX_TEST_BACKEND=arangodb \
NETWORKX_FALLBACK_TO_NX=True \
    pytest \
    --pyargs networkx.classes networkx.algorithms.centrality \
    --cov-config=$(dirname $0)/pyproject.toml \
    --cov=nx_arangodb \
    --cov-report= \
    "$@"
coverage report \
    --include="*/nx_arangodb/algorithms/*" \
    --omit=__init__.py \
    --show-missing \
    --rcfile=$(dirname $0)/pyproject.toml
