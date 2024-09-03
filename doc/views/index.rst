.. _views:

**************
ArangoDB Views
**************

Having a database as a backend to NetworkX allows us to delegate
certain operations to the database.

This can be applied to the concept of NetworkX Views.

Below are a set of experimental overrides of the NetworkX Views that represent the
nodes and edges of the graph. Overriding these classes allows us to
implement custom logic for data filtering and updating in the database.

These classes are a work-in-progress. The main goal is to try
to delegate data processing to ArangoDB, whenever possible.

To use these experimental views, you must set **use_arango_views=True**
when creating a new graph object:

.. code-block:: python

   import nx_arangodb as nxadb

   G = nxadb.Graph(name="MyGraph", use_arango_views=True)


.. toctree::
   :maxdepth: 1

   coreviews
   reportviews