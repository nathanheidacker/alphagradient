.. _about:

=====
About
=====

.. currentmodule:: alphagradient

Design and Intended Audience
----------------------------
AlphaGradient was built with an intention to open the world of algorithmic trading to anyone with an interest in finance. From the beginning, AlphaGradient's API was designed with user experience at the forefront, such that users of **any** programming skill level could take up the library and start building real financial algorithms within a single session.

In it's current state, the library is far from offering all of the functionality that it ultimately intends--but the backbone is there. Unlike other engines and trading platforms which are specialized to accomodate real-world economies, AlphaGradient is built on a set of more robust abstract principles; it can model essentially any system of economic transactions (see the User Guide for examples).

Ultimately, AlphaGradient is built for anyone that wants to use it. To successfully implement algorithms, some elementary knowledge of both finance and python are required, but nothing to the extent that anyone would be ruled out. If you have an understanding of how securities are traded or have ever used a brokerage account, you can use AlphaGradient.

The Road Ahead
--------------
Of course, this is still just an extremely early look at what AlphaGradient eventually hopes to become. Current features and functionality for most of the primitive financial objects implemented by AlphaGradient are sparse at best, so one of the most immediate goals is to flesh those tools out as the library matures into a more respectable, robust finance toolkit. Namely, we're in the process of developing far more advanced analytical tools for these primitive objects. 

Performance Analytics
+++++++++++++++++++++
Most importantly, :class:`Backtest` objects will provide a framework for extremely detailed and customizable performance analytics reporting, as well as tools for natively outputting these reports to human readable (and hopefully aesthetically pleasing!) output formats, like interactive HTML. 

We're also considering developing complete third party tools for creating viewing, and interacting with algorithm backtest reports. We find that the use of a modern, minimalist GUI is dramatically more attractive for the average potential user--especially non-programmers.

In the future, these more concrete applications can even be used to implement a visual scripting interface for algorithm development, not disimilar to Unreal Engine's 'blueprint' system. While we don't want to walk away from the accessibility and open-source nature of the python identity, ultimately the goal is to bring effortless and easy-to-understand algorithmic trading to the largest audience possible.


Data Sourcing
+++++++++++++
As of right now, AlphaGradient makes no guarantee to provide data for the Assets that it instantiates--the responsbility to provide data useful to the outcome of algorithm backtests lies largely with the user. However, for certain asset classes, AlphaGradient does make an effort to provide data automatically where convenient, such as it currently does with stocks, currencies, and options. 

However, this data is of limited quality, owing to the fact that it uses public APIs that offer data of limited resolution and limited history. Data sourcing for assets classes whose nature permits automatic sourcing could definitely be improved--not just by the implementation of new features and functionality for customizing the type of data with which those assets are automatically instantiated, but by the implementation of new data sources themselves. There are a host of other APIs that offer similar kinds of data that may be more powerful used collectively than independently. 

Data sources for other kinds of assets which do not currently support automatic data at time of instantiation may also exist, which, if implemented, may allow AlphaGradient to eventually handle everything behind the scenes automatically, dramatically reducing the burden of the user to supply anything on their own.

If the community grows large enough, AlphaGradient may eventually choose to support a paid tier of data sourcing by partnering with various data sources, the use of which would automatically be integrated with AlphaGradient via the use of a private API key.

Expanding the Standard Asset Library
++++++++++++++++++++++++++++++++++++
The current standard library is very limited in scope. While the AlphaGradient asset base class is designed to be very easily extendable, it would likely be to the benefit of users to offer a much larger variety of basic asset classes. The first official release of AlphaGradient is guaranteed to include at least the following.

New Concrete Asset Types:
    * Bonds
    * Commodities
    * Cryptocurrencies
    * Futures
    * Swaps
    * Loans

New Abstract Asset Types:
    * Virtual Assets
    * Unique Assets (one of a kind, or extremely limited quantity)
    * Stochastics

Stochastic assets are must larger implementation because they will allow for the creation of entire synthetic datasets whos statistical characteristics model those of real economies when taken as a whole. This will allow algorithms to test the robustness of their theory; does it only work in historical data, or does it work any (resonably conceived) data? Initially, AlphaGradient will ship with only a single Stochastic asset type: ``BrownianStock``. These assets will operate in exactly the same way as normal :class:`Stock` objects already do, but their data will be generated by underlying processes of geometric brownian motion. 

Algorithm Improvements
++++++++++++++++++++++
Being the star of the show, its important that :class:`Algorithms <Algorithm>` are feature-complete. As of right now, there are a few limitations of their use that prevents them from being nearly as powerful as they could be.

Simplifying Assumptions
***********************
In their current form, algorithms make some pretty large assumptions in order to create a simplified 'perfect world'. While this world of perfect assumptions makes the process of creating algorithms and understanding their outputs far easier, it also makes their outputs significantly less realistic and sometimes even impossible. Below are just a few of the simplifying assumptions made by algorithms that do not accurately reflect real world trading circumstances.

Simplifying Assumptions:
    * All assets have infinite volume and infinite liquidity
    * | Consequently, no transaction has the capacity to affect any change in
      | price of the asset involved in the transaction.
    * Trades occur instantly 
    * Assets have absolute prices, rather than bid ask spreads

While these assumptions make things extremely simple for beginners, power users may find themselves disappointed by the lack of realism when dealing with algorithms that operate with extremely high frequencies, transaction quantities, or both. The first official release of AlphaGradient will include a suite of toggle-able options that allow users to change what assumptions are made at algorithm runtime.

Asynchronous Algorithms
***********************
The current standard algorithm implementation runs synchronously; it evaluates everything at every time step before moving to the next. While this format of operation is certainly the most intuitive, it is extremely suboptimal in regard to computational efficiency, especially for use in a machine learning context where thousands of backtest results need to be generated to build effective datasets.

Next on the to-do list for algorithms is the implementation of the ``AsyncAlgorithm``, which operates with similar, but slightly modified functionality to allow it to perform valuations at different points in time in a vastly more optimized asynchronous fashion. Async algos will require the user to define a function that, when given a far more limited set of data, outputs a set of actions to be performed rather than modifying an Environment object in place. These action sets can then be compiled and valuated in parllalel, allowing for algorithms with an order of magnitude (or more) improvement in backtest speed.

Machine Learning
++++++++++++++++
The first official v1.0.0 release of AlphaGradient will be marked by the completion of its machine learning module, ``alphagradient.ml``. This is the **Gradient** in Alpha\ **Gradient**. ``ml`` will implement a host of features for intuitively using AlphaGradient primitives with the most popular ml libraries, such as PyTorch, Tensorflow, and Keras.

Planned Features:
    * Asset metadata profiles for use in training
    * | Asset Dataloaders which automatically generate properly formatted
      | historical asset data for use in training.
    * | Pipelines for combining loaders with models and automatically
      | interpretting and formatting ouputs
    * Pre-tuned models which can be selectively trained on any asset
    * Pre-trained models for asset price prediction


About Me
--------
Hi, I'm Nathan, and this library is something of a passion project of mine. I'm a recent graduate of Northwestern University (Go Cats!) with a bachelors in cognitive science and artificial intelligence. If any of my work interests you feel free to reach out at `this email <nathanheidacker2022@u.northwestern.edu>`_.