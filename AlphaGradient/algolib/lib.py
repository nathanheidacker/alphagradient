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
        self.initial = self.spy.value * 150
        self.p = env.portfolio(self.initial)

        return env

    def run(self, start, end):
        step = 1
        while self.date < end:

            if self.env.open:

                # Buying spy if we can buy spy
                to_buy = math.floor((self.p.liquid / self.spy.value) / 100) * 100
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
                    to_sell = min(to_sell, 10)
                    self.env.short(self.spycall(), to_sell)

            # Go to the next trading day
            self.verbose(f"STEP {step}: ")
            self.p.print_changes()
            self.verbose(self.p.positions)
            self.verbose(self.env.assets, "\n\n")
            self.env.next()

            step += 1

        self.p.liquidate(force=True)
        print(f"Initial: {ag.finance.Cash(self.initial)} | Current: {self.p.cash} | Profit: {self.p.cash - ag.finance.Cash(self.initial)} | Return: {round(((self.p.cash.quantity - self.initial) / self.initial)* 100, 2)}")

    def spycall(self, offset=1, delta=1):
        strike = self.spy.value + offset
        delta = self.spy.date + timedelta(days=delta)
        if delta.weekday() >= 4 and delta.weekday() <= 6:
            delta += timedelta(days=(7 - delta.weekday()))
        call = self.env.call(self.spy, int(strike), delta)
        return call






