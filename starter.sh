#!/bin/bash

# Starts a local ArangoDB server or cluster (community or enterprise).
# Useful for testing the python-arango driver against a local ArangoDB setup.

# Usage:
#   ./starter.sh [single|cluster] [community|enterprise]
# Example:
#   ./starter.sh cluster enterprise

extra_ports="-p 8539:8539 -p 8549:8549"
image_name="enterprise"
conf_file="cluster.conf"

docker run -d \
  --name arango \
  -p 8528:8528 \
  -p 8529:8529 \
  $extra_ports \
  -v "$(pwd)/tests/static/":/tests/static \
  -v /tmp:/tmp \
  "arangodb/$image_name:latest" \
  /bin/sh -c "arangodb --configuration=/tests/static/$conf_file"
