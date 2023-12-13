.. _api.utils:

=================
Utility Functions
=================

.. currentmodule:: alphagradient

Batching
--------

.. note::

    AlphaGradient auto-batching functions are built to be effective at splitting
    large iterables into batches to be used in **multiprocessing**. If you do
    not require batches for this purpose, it is unlikely that the autobatching
    functions will provide what you need. 

.. autosummary::
    :toctree: ../generated/

    utils.auto_batch
    utils.auto_batch_size
    utils.get_batches


Conversions
-----------
.. autosummary::
    :toctree: ../generated/

    utils.get_time
    utils.read_timestring
    utils.read_twelve_hour_timestring
    utils.timestring
    utils.to_datetime
    utils.to_time


Time and Datetime Manipulation
------------------------------
.. autosummary::
    :toctree: ../generated/

    utils.get_weekday
    utils.nearest_expiry
    utils.optimal_start
    utils.set_time
    utils.to_step


TypeGuards
----------
.. autosummary::
    :toctree: ../generated/

    utils.is_func


Miscellaneous
-------------
.. autosummary::
    :toctree: ../generated/

    utils.bounded
    utils.progress_print

.. autosummary::
    :toctree: ../generated/
    :template: class.rst
    
    utils.NullClass


Internal Use
------------
.. autosummary::
    :toctree: ../generated/

    utils.deconstruct_dt

