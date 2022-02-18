# -*- coding: utf-8 -*-
"""AlphaGradient's algorithm library, containing vetted and high 
quality financial algorithms to be used freely.

Todo:
    * NA
"""

# Standard Imports
import os
import math
from datetime import timedelta

# Third Party imports

# Local Imports
os.chdir("..")
import alphagradient as ag

class SpyCoveredCalls(ag.Algorithm):

    def setup(self, start, end):
        env = ag.finance.Basket(start=start)
        self.spy = env.stock("SPY")
        self.p = env.portfolio(self.spy.value * 101)

        return env

    def run(self, start, end):
        day = 1
        while self.date < end:

            # Buying spy if we can buy spy
            to_buy = math.floor((self.p.liquid / self.spy.value) / 100)
            to_buy *= 100
            if to_buy > 0:
                self.env.buy(self.spy, to_buy)

            # Selling calls if we can sell calls
            position = self.p.longs.get("STOCK_SPY", False)
            if position and position.quantity >= 100:
                to_sell = math.floor(position.quantity / 100)
                self.env.short(self.spycall(), to_sell)

            # Go to the next trading day
            self.verbose(f"DAY {day}")
            self.verbose(self.p.positions, "\n")
            self.env.step()
            day += 1

    def spycall(self, offset=1, delta=1):
        strike = self.spy.value + offset
        call = self.env.call(self.spy, strike, delta)
        return call






