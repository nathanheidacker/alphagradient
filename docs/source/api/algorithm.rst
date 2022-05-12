.. _api.algorithm:

==========
Algorithms
==========

.. currentmodule:: alphagradient

Constructor
-----------
.. autosummary::
    :toctree: ../generated
    :template: class.rst

    Algorithm


Backtesting
-----------
.. autosummary::
    :toctree: ../generated

    Algorithm.__call__


Attributes
----------
.. autosummary::
    :toctree: ../generated

    Algorithm.active
    Algorithm.date
    Algorithm.end
    Algorithm.env
    Algorithm.start
    Algorithm.type
    

Abstract Methods
----------------
.. autosummary::
    :toctree: ../generated

    Algorithm.setup
    Algorithm.cycle


Time Control
------------
.. autosummary::
    :toctree: ../generated

    Algorithm.default_run


Algorithm Development Helpers
-----------------------------
.. autosummary::
    :toctree: ../generated
    
    Algorithm.initialize_inputs


Internal Functionality
----------------------
.. deprecated:: 0.0.2

    These functions will be made private in the upcoming update

.. autosummary::
    :toctree: ../generated

    Algorithm.validate_end
    Algorithm.validate_resolution