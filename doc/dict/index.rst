.. _dict:

************
Dictionaries
************

The ``dict`` module provides a set of ``UserDict``-based classes that extend the traditional dictionary functionality to maintain a remote connection to an ArangoDB Database.

NetworkX Graphs rely on dictionary-based structures to store their data, which are defined by their factory functions:

1. ``node_dict_factory``
2. ``node_attr_dict_factory``
3. ``adjlist_outer_dict_factory``
4. ``adjlist_inner_dict_factory``
5. ``edge_key_dict_factory`` (Only for MultiGraphs)
6. ``edge_attr_dict_factory``
7. ``graph_attr_dict_factory``

These factories are used to create the dictionaries that store the data of the nodes, edges, and the graph itself.

This module contains the following classes:

1. ``NodeDict``
2. ``NodeAttrDict``
3. ``AdjListOuterDict``
4. ``AdjListInnerDict``
5. ``EdgeKeyDict``
6. ``EdgeAttrDict``
7. ``GraphDict``
8. ``GraphAttrDict``

Each class extends the functionality of the corresponding dictionary factory by adding methods to interact with the data in ArangoDB. Think of it as a CRUD interface for ArangoDB. This is done by overriding the primary dunder methods of the ``UserDict`` class.

By using this strategy in addition to subclassing the ``nx.Graph`` class, we're able to preserve the original functionality of the NetworkX Graphs while adding ArangoDB support.

.. toctree::
   :maxdepth: 1

   adj
   node
   graph
