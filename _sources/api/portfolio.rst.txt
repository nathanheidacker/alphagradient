.. _api.portfolio:

==========
Portfolios
==========

.. currentmodule:: alphagradient

Constructor
-----------
.. autosummary::
    :toctree: ../generated/
    :template: class.rst

    Portfolio

Attributes
----------
.. autosummary::
    :toctree: ../generated/

    Portfolio.base
    Portfolio.base_symbol
    Portfolio.cash
    Portfolio.date
    Portfolio.liquid
    Portfolio.longs
    Portfolio.positions
    Portfolio.shorts
    Portfolio.value
    Portfolio.type


Internal Methods
----------------
.. deprecated:: 0.0.2

    These methods will be private in the next version

.. autosummary::
    :toctree: ../generated/

    Portfolio.reset
    Portfolio.validate_transaction
    Portfolio.update_history
    Portfolio.update_positions


Financial Transactions
----------------------
.. autosummary::
    :toctree: ../generated/

    Portfolio.buy
    Portfolio.cover
    Portfolio.covered_call
    Portfolio.invest
    Portfolio.liquidate
    Portfolio.sell
    Portfolio.short


Position Access
---------------
.. autosummary::
    :toctree: ../generated/

    Portfolio.get_position
    Portfolio.get_related_positions