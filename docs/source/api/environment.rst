.. _api.environment:

============
Environments
============

.. currentmodule:: alphagradient

Constructor
-----------
.. autosummary::
    :toctree: ../generated/
    :template: class.rst

    Environment

Attributes
----------
.. autosummary::
    :toctree: ../generated/

    Environment.assets
    Environment.base
    Environment.date
    Environment.end
    Environment.open
    Environment.portfolios
    Environment.resolution
    Environment.start
    Environment.status
    Environment.times
    Environment.type

Internal Methods
----------------
.. autosummary::
    :toctree: ../generated/

    Environment.finalize


Time Control
------------
.. autosummary::
    :toctree: ../generated/

    Environment.autosync
    Environment.next
    Environment.optimal_start
    Environment.step
    Environment.sync


Data Access
-----------
.. autosummary::
    :toctree: ../generated/

    Environment.data


Tracking from Outside
---------------------
.. deprecated:: 0.0.2

    In the next update, environments will no longer provide functionality for
    tracking assets that were instantiated outside of their domain.

.. autosummary::
    :toctree: ../generated/

    Environment.track
    Environment.portfolio


Redirected Transactions
-----------------------
.. deprecated:: 0.0.2

    These functions are already adequately handled by internal methods. Their
    explicit definitions in the class will be removed in the next update
    (functionality will remain identical)

.. autosummary::
    :toctree: ../generated/

    Environment.buy
    Environment.cover
    Environment.sell
    Environment.short

.. autosummary::
    :toctree: ../generated/
    :template:

    Environment.AssetDict