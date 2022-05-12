.. _algolib:

=================
Algorithm Library
=================

.. currentmodule:: alphagradient

AlphaGradient's algorithm library is a collection of financial algorithms built using AlphaGradient. These algorithms are public domain and free to use. We highly encourage the open-sourcing of algorithms, as it contributes to a vibrant and lively community and helps others learn. If you've developed an algorithm using AlphaGradient that you'd like to make publicly available to all AlphaGradient users, see our page on :ref:`publishing an algorithm <algolib.publishing>`.

All of the algorithms contained within the algorithm library are licensed under the Apache 2.0 license. See `alphagradient.license` for more information.


Standard Algorithms
-------------------
This section contains algorithms that are 'standard', in that they operate within standard financial exchanges and use almost exclusively asset classes which are either contained in the standard asset library or derivative of those assets with minimal changes. Most of the algorithms in this section of the library operate on United States financial markets.

.. toctree::
    :maxdepth: 2

    standard/index


Non-Standard Algorithms
-----------------------
These are algorithms which do not fit the criteria of the category above in some fashion. They either operate out of non-standard financial markets (eg. virtual exchanges, video game economies) or deal primarily with asset classes that are **not** familiar to the majority of traders (eg. highly custom assets). Algorithms in this section could really pertain to anything--be sure to read documentation thoroughly.

.. toctree::
    :maxdepth: 2

    nonstandard/index


Tutorial Algorithms
-------------------
This section contains algorithms that are present in AlphaGradient documentation, used to demonstrate fundamental algorithm design principles or how the library functions as a whole.

.. toctree::
    :maxdepth: 2

    tutorials/index


Publishing an Algorithm
-----------------------
If you've developed something really cool using AlphaGradient, please share it! Visit our section on publishing an algorithm to get started.

.. toctree::
    :maxdepth: 2

    publishing
    TemplateAlgo