Quickstart
==========

1. Set up ArangoDB
2. Set environment variables
3. Instantiate a NetworkX-ArangoDB Graph

1. Set up ArangoDB
------------------

**Option A: Local Instance via Docker**

Appears on ``localhost:8529`` with the user ``root`` & password ``openSesame``.

More info: `arangodb.com/download-major <https://arangodb.com/download-major/>`_.

.. code-block:: bash

    docker run -e ARANGO_ROOT_PASSWORD=openSesame -p 8529:8529 arangodb/arangodb

**Option B: ArangoDB Cloud Trial**

`ArangoGraph <https://dashboard.arangodb.cloud/home>`_ is ArangoDB's Cloud offering to use ArangoDB as a managed service.

A 14-day trial is available upon sign up.

**Option C: Temporary Cloud Instance via Python**

A temporary cloud database can be provisioned using the `adb-cloud-connector <https://github.com/arangodb/adb-cloud-connector?tab=readme-ov-file#arangodb-cloud-connector>`_ Python package.

.. code-block:: bash

    pip install adb-cloud-connector

.. code-block:: python

    from adb_cloud_connector import get_temp_credentials

    credentials = get_temp_credentials()

    print(credentials)

2. Set environment variables
----------------------------

Connecting to ArangoDB requires the following environment variables:

1. ``DATABASE_HOST``: The host URL of the ArangoDB instance.
2. ``DATABASE_USERNAME``: The username to connect to the ArangoDB instance.
3. ``DATABASE_PASSWORD``: The password to connect to the ArangoDB instance.
4. ``DATABASE_NAME``: The name of the database to connect to.

For example, using Option 1 from above:

.. code-block:: bash

    export DATABASE_HOST=http://localhost:8529
    export DATABASE_USERNAME=root
    export DATABASE_PASSWORD=openSesame
    export DATABASE_NAME=_system

Or using Option 3 from above:

.. code-block:: python

    import os
    from adb_cloud_connector import get_temp_credentials

    credentials = get_temp_credentials()

    os.environ["DATABASE_HOST"] = credentials["url"]
    os.environ["DATABASE_USERNAME"] = credentials["username"]
    os.environ["DATABASE_PASSWORD"] = credentials["password"]
    os.environ["DATABASE_NAME"] = credentials["dbName"]

3. Instantiate a NetworkX-ArangoDB Graph
----------------------------------------

Instantiating a NetworkX-ArangoDB Graph is similar to instantiating a NetworkX Graph.

Providing the ``name`` parameter will create a new graph in ArangoDB if it does not already exist.

Providing the ``incoming_graph_data`` in combination with the ``name`` parameter will create a new graph in ArangoDB
with the provided data. If the graph already exists, an error will be raised.

.. code-block:: python

    import networkx as nx
    import nx_arangodb as nxadb

    G = nxadb.Graph(name="MyGraph") # New ArangoDB Graph
    G2 = nxadb.Graph(incoming_graph_data=nx.karate_club_graph()) # Regular NetworkX Graph
    G3 = nxadb.Graph(incoming_graph_data=nx.karate_club_graph(), name="KarateGraph") # New ArangoDB Graph

From here, you can use the conventional NetworkX API to interact with the graph.

Assuming you already have a graph in ArangoDB named `MyGraph`, you can reload it as follows:

.. code-block:: python

    import nx_arangodb as nxadb

    G = nxadb.Graph(name="MyGraph")

    print(G.number_of_nodes(), G.number_of_edges())
