.. _api/dtypes:

==========
Data Types
==========

.. currentmodule:: alphagradient.dtypes

The ``alphagradient.dtypes`` module is a collection of intermediate data types
used by the top-level alphagradient module. Most notably, this module contains
classes that are returned as objects from calls to top-level classes, such
as ``alphagradient.dtypes.Backtest``, ``alphagradient.dtypes.Performance``,
``alphagradient.dtypes.Position``, ``alphagradient.dtypes.UniverseView``, and more.

.. warning::

    ``alphagradient.dtypes`` exposes otherwise private alphagradient classes,
    and is primarily intended for documentation and type checking. Instantiating
    ``alphagradient.dtypes`` members directly is not supported; instability
    should be expected.

Errors and Exceptions
---------------------
.. toctree::
    :maxdepth: 2

    Exceptions

Type Annotations
----------------
.. toctree::
    :maxdepth: 2

    Annotations


Data
----
.. toctree::
    :maxdepth: 2

    data/index


Financial Positions
-------------------
.. toctree::
    :maxdepth: 2

    positions/index


Views
-----
.. toctree::
    :maxdepth: 2

    views/index


Enumerations
------------
.. toctree::
    :maxdepth: 2
    
    enums/index


Abstract Asset Types
--------------------
.. toctree::
    :maxdepth: 2

    abstract/index


Stats and Metrics
-----------------
.. toctree::
    :maxdepth: 2

    metrics/index


Universe Operations
-------------------
.. toctree::
    :maxdepth: 2

    universe/index
