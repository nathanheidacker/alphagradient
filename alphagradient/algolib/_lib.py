# -*- coding: utf-8 -*-
"""
AlphaGradient's algorithm library, containing vetted and high quality financial 
algorithms to be used freely.

To publish an algorithm to the AlgoLib, please contact 
nathanheidacker2022@u.northwestern.edu
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
    """An example algorithm in the algorithm library used to demonstrate some
    of AlphaGradient's standard features and best practices

    This is a tutorial algorithm that seeks to demonstrate some of
    AlphaGradient's features and standard design practices. This algorithm
    sells the maximum numnber of covered calls on SPY that it can, with a
    bounded strike price to prevent from selling calls that could lose money
    when assigned

    Heres a breakdown:
        #. | At the beginning of the day, buy as many shares of SPY as we can to
           | the nearest multiple of 100

        #. | Using SPY shares as collateral, sells 1 DTE covered calls on SPY
           | where the strike is determined by SPY's current value. The
           | algorithm will never sell a call with a strike below it's average
           | cost for the shares it owns. This prevents it from losing money in
           | the case of call assignment.
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
