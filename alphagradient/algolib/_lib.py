# -*- coding: utf-8 -*-
"""
AlphaGradient's algorithm library is a collection of financial algorithms built 
using AlphaGradient. These algorithms are public domain and free to use. We 
highly encourage the open-sourcing of algorithms, as it contributes to a 
vibrant and lively community and helps others learn. If you've developed an 
algorithm using AlphaGradient that you'd like to make publicly available to 
all AlphaGradient users, see our page on :ref:`publishing an algorithm <algolib.publish>`.

All of the algorithms contained within the algorithm library are licensed under 
the Apache 2.0 license. See `alphagradient.license` for more information.
"""

# Standard Imports
import os
import math
from datetime import datetime, timedelta

# Third Party imports
import numpy as np

# Local Imports
from .. import _proxy as ag

# Typing
from typing import (
    Any,
)


class IndexHold(ag.Algorithm):
    """A tutorial algorithm! Buy and Hold!"""

    def setup(self, start: datetime, **kwargs: Any) -> ag.Environment:
        """
        This algorithm only requires a single index fund to invest in, so it
        uses the default AG benchmark SPY. No other assets are instantiated.
        Only a single portfolio ("MAIN") is used.

        Parameters:
            start:
                The starting datetime of the backtest is required to properly
                setup the environment
        """
        # Our initial balance
        initial = 1_000_000
        spy = ag.Stock("SPY")

        # Creating an environment object
        env = ag.Environment(assets=[spy])

        # identical to env.main.invest(initial)
        env.invest(initial)

        # Calling this drastically improves runtime performance
        env.finalize()

        return env

    def cycle(self, start: datetime, end: datetime, **kwargs: Any) -> None:
        """
        The goals of this algorithm are very simple,
        """
        # Buying at the start...
        if self.date <= datetime.fromisoformat("2010-01-04 16:00:00"):

            # Determining how much we can afford
            to_buy = math.floor(self.env.liquid / self.env.stock.spy.value)

            # This buys the asset on the main portfolio
            self.env.buy(self.env.stock.spy, to_buy)

        # And holding till the end!
        elif self.date >= (end - timedelta(days=1)):

            # Selling everything
            self.env.liquidate()


class ThetaGang(ag.Algorithm):
    """
    An example algorithm in the algorithm library used to demonstrate some of
    AlphaGradient's standard features and best practices

    This is a tutorial algorithm that seeks to demonstrate some of AlphaGradient's
    features and standard design practices. This algorithm sells the maximum number
    of covered calls on SPY that it can, with a bounded strike price to prevent from
    selling calls that could lose money when assigned

    Here's a breakdown:

        #. | At the beginning of the day, buy as many shares of SPY as we can to the
           | nearest multiple of 100

        #. | Using SPY shares as collateral, sells 1 DTE covered calls on SPY where
           | the strike is determined by SPY's current value. The algorithm will never
           | sell a call with a strike below it's average cost for the shares it owns.
           | This prevents it from losing money in the case of call assignment.

        #. | The strike bounding component of 2) is toggle-able by instantiating with
           | bounded=False
    """

    def __init__(self, *args: Any, bounded: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Determines whether or not a lower bound should be placed on the strike
        self.bounded = bounded

    def setup(self, start: datetime, **kwargs: Any) -> ag.Environment:
        """This is the environement setup that is performed before each backtest. Must return an environment object"""

        # Creating a basket with the given start parameter
        env = ag.Environment()

        # Creating SPY stock, attaching it to self (will be referenced frequently)
        # This call to the stock() method both instantiates the stock within the environment, AND returns it
        self.spy = env.stock("SPY")

        # Creating the stock normally like so:
        # self.spy = ag.finance.Stock("SPY")
        # will NOT allow the environment to track it or update its value data as time progresses

        # Initial investment into the primary portfolio
        env.invest(self.spy.value * 150)

        # We only want the algorithm to evaluate at market open and close of each day
        # Finalizing will dramatically increase execution time, but is not necessary
        env.finalize(manual=["9:30 AM", "4:00 PM"])

        return env

    def cycle(self, **kwargs: Any) -> None:
        """The actions to perform at every valuation point"""

        # Selling as many covered calls on SPY as we can
        self.env.covered_call(self.generate_call())

        # The above line of code is a shortcut for:
        # self.env.main.covered_call(self.generate_call())

        # Showing the changes at every time step
        self.print(self.stats.change_report())

    def generate_call(self, delta: float = 1) -> ag.Call:
        """Generates the ideal SPY call to be sold based on current circumstances"""

        # Getting our current position in the Asset <STOCK SPY>
        spy_position = self.env.get_position(self.spy)

        # Determining our optimal strike price
        optimal = math.floor(self.spy.value) + delta

        # Determining a lower bound for our strike price (the ceiling of our basis)
        lower_bound = optimal
        if spy_position and self.bounded:
            lower_bound = math.ceil(spy_position.average_cost)

        # Determining our strike price
        strike = max(optimal, lower_bound)

        # Determining the call expiry date (1 DTE)
        expiry = self.env.date + timedelta(days=1)

        # We can't sell calls with expiries on weekends or outside of market hours
        expiry = ag.utils.nearest_expiry(expiry)

        # Creating the call using the environment so that it doesnt have to be added retroactively
        return self.env.call(self.spy, strike, expiry)


class TemplateAlgo(ag.Algorithm):
    """
    Shows how to create and document an AlphaGradient algorithm

    This section of the algorithm should contain documentation about what the
    algorithm aims to accomplish in general, the theory that it operates on in
    order to accomplish those aims, as well as important implementation details
    on how specifically this algorithm models that theory.

    Docstrings for classes, functions, properties, and attributes alike should
    follow standard rst, google, or numpy docstring formats such that they are
    compatible with sphinx autodocumentation

    The primary docstring for the algorithm should also document its __init__
    function if it has been overridden. Include a parameters section inside
    the class docstring for parameters taken in for __init__.

    All Algorithms seeking publication should be properly type annotated.

    Parameters:
        template_requires:
            An example of a parameter that an algorithm may require during
            instantiation to initialize some class setting or control behavior.
            This example sets a private attribute that underlies the function
            of the algorithm's 'template_property' property. This is a fairly
            standard use case.
    """

    def __init__(self, *args: Any, template_requires: Any, **kwargs: Any) -> None:
        """
        Shows how to create and document an __init__ function for an algorithm

        Documentation for init functions in algorithms should be included in the
        algorithm's main docstring below the class definition. Be sure to
        document all new parameters thoroughly, as well as any extra
        initialization behavior. Docstrings for __init__, as well as other
        dunder methods, are not required unless they perform unexpected
        behavior.
        """
        self._prop = template_requires
        super().__init__(*args, **kwargs)

    def setup(self, **kwargs: Any) -> ag.Environment:
        """
        Shows how to create and document a setup function

        The setup function should be capable of accepting 'start' and 'end'
        keyword arguments, so if you don't accept those directly, you must
        accept **kwargs. They will additionally be passed any arguments that
        are passed to the algorithm's __call__ method.

        Setup functions should always return an environment object which will be
        bound to the Algorithm's 'env' attribute. This occurs when the
        algorithm is instantiated, as well as before every backtest.

        Documentation for setup functions should primarily concern what kind
        of environment is being set up, and why the instantiated assets are
        required. Why are certain assets or asset classes being used? How
        does the algorithm take advantage of them? If there are multiple
        portfolios in the environment, why, and what is the purpose of each one?
        Questions like this should be answerable from a reading the setup
        docstring.

        Parameters:
            example:
                Include any additional parameters required by the setup function
                in their own parameters section, using this format.
        """
        return ag.Environment()

    def cycle(self, **kwargs: Any) -> None:
        """
        Shows how to create and document a cycle function

        The cycle function should be capable of accepting 'start' and 'end'
        keyword arguments, so if you don't accept those directly, you must
        accept **kwargs. They will additionally be passed any arguments that
        are passed to the algorithm's __call__ method.

        Cycle functions should have no return value; their goal is to modify the
        Algorithm's bound environment in-place (accessible at algo.env). The
        bound environment will perform the actions contained in cycle at every
        time step, and the results will be recorded by the algorithm.

        Documentation for cycle functions should primarily concern specific
        implementation details of the theory outlined in the class' primary
        docstring. What specific decisions were made to implement the theory
        specified in the main docstring? Are there any nuances or shortcomings
        of this approach, or limitations to its accuracy? Questions like this
        should be answerable from a reading of the cycle docsting.

        Parameters:
            example:
                Include any additional parameters required by the cycle function
                in their own parameters section, using this format.
        """
        return None

    def run(self, **kwargs: Any) -> None:
        """
        Shows how to create and document a custom run function

        Algorithms do not require custom run functions, but will use them if
        provided. By default, algorithms provide a default_run version of run
        that cycles through time steps by determining the next best time to
        valuate at, and then calling cycle at each new step in time. Run
        functions are vastly more error prone to implement than their
        setup and cycle counterparts, so users should only look to override
        the standard behavior when absolutely necessary.

        Run functions should also be capable of accepting start and end
        keyword arguments. They are also passed any arguments given during
        the algorithm's __call__ method.

        Documentation for run functions should explain why new time control
        is necessarily or optimal for the function of this specific
        algorithm, as well as how it has been implemented.

        Parameters:
            example:
                Include any additional parameters required by the run function
                in their own parameters section, using this format.
        """
        return None

    @property
    def template_property(self) -> Any:
        """
        Shows how to create and document an algorithm property

        Algorithm properties should contain their own docstrings, rather than
        being documented in an 'Attributes' section in the class docstring.

        Documentation should describe what the property is, as well as its
        purpose/function in the algorithm. Properties require no parameters or
        return sections.
        """
        return self._prop

    def template_method(self, *some_args: Any, **some_kwargs: Any) -> Any:
        """
        Shows how to create and document an algorithm method

        Algorithms can and should be treated like any other class; rather than
        writing all of the algorithm code in the cycle function, it more often
        makes sense to create some class methods to compartmentalize and reuse
        behavior.

        Documentation for algorithm methods should describe their function
        within the algorithm, as well as thoroughly explain their inputs and
        outputs. If any parameters exist, they should be documented in
        a Parameters section. Unless the method's computation is trivial to
        the extent that the docstring begins with "Returns", a Returns section
        should be included to describe the output

        Parameters:
            some_args:
                All parameters should be thoroughly explained invidually
                in the parameters section

        Returns:
            Unless the docstring begins with "Returns", a Returns section such
            as this one should be included to elaborate on the return value.
        """
        return None
