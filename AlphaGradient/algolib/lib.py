# -*- coding: utf-8 -*-
"""AlphaGradient's algorithm library, containing vetted and high 
quality financial algorithms to be used freely.

Todo:
    * NA
"""

# Standard Imports
import os
import math
from datetime import datetime, date, timedelta

# Third Party imports
import numpy as np

# Local Imports
os.chdir("..")
import alphagradient as ag

class SpyCoveredCalls(ag.Algorithm):

    def setup(self, start, end):
        env = ag.finance.Basket(start=start, force=True)
        self.spy = env.stock("SPY")
        self.initial = self.spy.value * 101
        self.p = env.portfolio(self.initial)

        return env

    def run(self, start, end):
        day = 1
        while self.date < end:

            # Buying spy if we can buy spy
            to_buy = math.floor((self.p.liquid / self.spy.value) / 100) * 100
            print(f"CASH {self.p.liquid} | SPY {self.spy.value} | PURCHASE {to_buy}")
            print([pos.expired for pos in self.p.positions.values()])
            print(self.p._positions["CASH"], self.p.positions["CASH"])
            if to_buy > 0:
                self.env.buy(self.spy, to_buy)

            # Selling calls if we can sell calls
            position = self.p.longs.get("STOCK_SPY", False)
            available = 0
            if position:
                available = position.quantity - (sum([pos.quantity for pos in list(self.p.call.values())]) * 100)
            to_sell = 0
            if available >= 100:
                to_sell = math.floor(available / 100)
                to_sell = min(to_sell, 100)
                self.env.short(self.spycall(), to_sell)

            # Go to the next trading day
            self.verbose(f"DAY {day}: {self.date.date(), ag.utils.timestring(self.date.time())}, {ag.utils.get_weekday(self.date)}")
            self.verbose(f"BOUGHT: {to_buy} | SOLD: {to_sell}")
            self.verbose(self.p.positions, "\n")
            self.env.next()
            """
            if self.date.hour == 9:
                self.env.step(timedelta(seconds=60 * 60 * 7))
            else:
                self.env.step(timedelta(seconds=60 * 60 * 17))
            """
            day += 1

        self.p.liquidate()
        print(ag.finance.Cash(self.initial), self.p.cash, self.p.cash - ag.finance.Cash(self.initial))

    def spycall(self, offset=1, delta=1):
        strike = self.spy.value + offset
        delta = self.spy.date + timedelta(days=delta)
        if delta.weekday() >= 4 and delta.weekday() <= 6:
            delta += timedelta(days=(7 - delta.weekday()))
        call = self.env.call(self.spy, int(strike), delta)
        return call






