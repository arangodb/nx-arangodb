set -e

NETWORKX_GRAPH_CONVERT=arangodb \
NETWORKX_TEST_BACKEND=arangodb \
NETWORKX_FALLBACK_TO_NX=True \
    pytest \
    --pyargs networkx.classes \
    --cov-config=$(dirname $0)/pyproject.toml \
    --cov=nx_arangodb \
    --cov-report= \
    "$@"
coverage report \
    --include="*/nx_arangodb/classes/*" \
    --omit=__init__.py \
    --show-missing \
    --rcfile=$(dirname $0)/pyproject.toml
