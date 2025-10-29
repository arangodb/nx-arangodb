Conversions between NetworkX, nx-arangodb, and nx-cugraph
=========================================================

This page describes how to convert graphs across ecosystems using
``nx_arangodb.convert`` for interoperability and performance.

Overview
--------

- ``_to_nx_graph``: ``nxadb.Graph`` | ``nx.Graph`` -> ``nx.Graph``
- ``_to_nxadb_graph``: ``nx.Graph`` | ``nxadb.Graph`` -> ``nxadb.Graph``
- ``_to_nxcg_graph``: ``nxadb.Graph`` | ``nxcg.Graph`` -> ``nxcg.Graph``
  (requires ``nx-cugraph`` and CuPy)

Behavior notes
--------------

- Database-backed ``nxadb`` graphs: converting to NetworkX pulls node and
  adjacency dictionaries from ArangoDB and constructs a plain NetworkX graph.
- Local ``nxadb`` graphs (``graph_exists_in_db`` is ``False``) are already
  NetworkX-compatible and may be returned as-is.
- Directed/multi settings are preserved; ``as_directed`` forces directed
  variants without changing multiplicity.
- If ``nx-cugraph`` or CuPy is unavailable, GPU conversions are disabled and
  ``_to_nxcg_graph`` raises ``NotImplementedError``.

Performance and caching
-----------------------

- Database pulls are timed and logged.
- ``nxadb_to_nx`` returns a standard NetworkX graph and does not re-wrap the
  custom dict implementations in ``nx_arangodb.classes.dict``.
- ``nxadb_to_nxcg`` can cache the constructed GPU graph when
  ``G.use_nxcg_cache`` is ``True`` on the ``nxadb.Graph`` instance.

Examples
--------

.. code-block:: python

   import networkx as nx
   import nx_arangodb as nxadb

   # Start with a NetworkX graph
   G_nx = nx.karate_club_graph()

   # Convert to nx-arangodb (optionally force directed)
   G_adb = nxadb.convert._to_nxadb_graph(G_nx, as_directed=False)

   # Convert back to NetworkX
   G_nx2 = nxadb.convert._to_nx_graph(G_adb)

   # Convert to nx-cugraph if available
   try:
       G_cg = nxadb.convert._to_nxcg_graph(G_adb)
   except NotImplementedError:
       # nx-cugraph/CuPy not available
       pass

API Reference
-------------

.. autofunction:: nx_arangodb.convert._to_nx_graph

.. autofunction:: nx_arangodb.convert._to_nxadb_graph

.. autofunction:: nx_arangodb.convert._to_nxcg_graph

.. autofunction:: nx_arangodb.convert.nx_to_nxadb

.. autofunction:: nx_arangodb.convert.nxadb_to_nx

.. autofunction:: nx_arangodb.convert.nxadb_to_nxcg


