# -*- coding: utf-8 -*-
"""AG module containing all standard asset types

This module contains ag.Asset implementations of all of the most standard classes of assets typically used in financial algorithms.

Todo:
    * Type Hints
    * Implement other standard types at the bottom
    * Figure out how currency/base currency works
"""

# Standard imports
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from numbers import Number
from random import random
import math
import json

# Third party imports
import yfinance as yf
import pandas as pd
import numpy as np

# Local imports
from .asset import Asset, types
from ..data import datatools

with open("AlphaGradient/finance/standard_asset_settings.json") as f:
    settings = json.load(f)
"""settings (dict): dictionary of settings for initializing standard asset subclasses"""


def _getinfo(base="USD"):

    def mapper(code):
        try:
            return yf.Ticker(f"{code}{base}=X").history(period="1d", interval="1d")["Close"].item()
        except:
            return -1

    info = pd.read_pickle("AlphaGradient/finance/currency_info_old.p")
    info["VALUE"] = info["CODE"].map(mapper)
    info = info[info["VALUE"] > 0]

    with open("AlphaGradient/finance/currency_info2.p", "wb") as p:
        info.to_pickle(p)

    return info


class Currency(Asset, settings=settings["CURRENCY"]):

    base = "USD"
    info = pd.read_pickle("AlphaGradient/finance/currency_info.p")

    @classmethod
    def update_info(cls):
        pass

    def __new__(cls, *args, **kwargs):
        if not args:
            args = (None,)
        if args[0] is None:
            args = list(args)
            args[0] = cls.base
            args = tuple(args)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, code):
        self.validate_code(code, error=True)
        self.code = code
        self.symbol = self.info[self.info["CODE"] == self.code]["SYMBOL"].item()
        super().__init__(code)

    def __str__(self):
        return f"<{self.type} {self.symbol}{self.code}: {self.price}>"

    def valuate(self, *args, **kwargs):
        return self.convert(self.code, self.base)


    def online_data(self, *args, **kwargs):
        if self.is_base:
            return None
        else:
            data = yf.Ticker(f"{self.code}{self.base}=X").history(period="max")
            return datatools.AssetData(Currency, data)

    @classmethod
    def get_symbol(cls, base=None):
        base = cls.base if base is None else base
        return cls.info[cls.info["CODE"] == base]["SYMBOL"].item()

    @classmethod
    def validate_code(cls, code, error=False):
        if code in list(cls.info["CODE"]):
            return True
        if error:
            raise ValueError(f"{code} is not a recognized as a valid currency code")
        return False

    @classmethod
    def code_value(cls, code):
        return cls.info[cls.info["CODE"] == code]["VALUE"].item()

    def convert(self, target, base=None):
        """Converts target currency to base currency

        Returns the value of the target currency in units of the base currrency

        Returns:
            value (Number): target currency value in base currency"""
        base = self.code if base is None else base
        return self.code_value(target) / self.code_value(base)

    def rate(self, target, base=None):
        """Converts this currency to the target

        Returns the value of the base currency in units of the target currency

        Returns:
            value (Number): base currency value in target currency"""
        return 1 / self.convert(target, base)

    @property
    def is_base(self):
        return self.code == self.base


class Stock(Asset, settings=settings["STOCK"]):
    """A financial asset representing stock in a publicly traded company

    An asset representing a single share of a publicly traded company. Supports online data from yahoo finance using yfinance.

    Attributes:
        ticker (str): This stock's ticker
    """
    def valuate(self):
        return self.value

    def online_data(self):
        data = yf.Ticker(self.name).history(period="max")
        return datatools.AssetData(Stock, data)

    @property
    def ticker(self):
        """Alias for name, relevant to stocks"""
        return self.name.upper()

    def call(self, strike, expiry):
        return Call(self, strike, expiry)

    def put(self, strike, expiry):
        return Put(self, strike, expiry)


class Option(Asset, ABC, settings=settings["OPTION"]):

    def __new__(cls, *args, **kwargs):
        underlying = args[0] if args else kwargs["underlying"]
        strike = args[1] if args and len(args) > 1 else kwargs["strike"]
        expiry = args[2] if args and len(args) > 2 else kwargs["expiry"]

        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry)
        elif isinstance(expiry, int):
            expiry = underlying.date + timedelta(days=expiry)
        elif isinstance(expiry, timedelta):
            expiry = underlying.date + expiry

        if not isinstance(expiry, datetime):
            raise TypeError(
                f'''Invalid input {expiry=} 
                for initialization of {underlying.name} {cls.__name__}''')

        key = f"{underlying.name}{strike}{cls.__name__[0]}{expiry.date().__str__()}"

        return super().__new__(cls, key)


    def __init__(self, underlying, strike, expiry):

        if isinstance(strike, Number):
            self.strike = round(strike, 1)
        else:
            raise TypeError(f"Inavlid non-numerical input for {strike}")

        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry)
        elif isinstance(expiry, int):
            expiry = underlying.date + timedelta(days=expiry)
        elif isinstance(expiry, timedelta):
            expiry = underlying.date + expiry

        if not isinstance(expiry, datetime):
            raise TypeError(
                f'''Invalid input {expiry=} 
                for initialization of {underlying.name} {self.__class__.__name__}''')

        self.expiry = expiry
        self.underlying = underlying
        self._mature = False

        super().__init__(self.key, date=underlying.date, base=underlying.base)

    @property
    def key(self):
        return f"{self.underlying.name}{self.strike}{self.__class__.__name__[0]}{self.expiry.date().__str__()}"

    @staticmethod
    def cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def _bsbase(spot, strike, rfr, dy, ttm, vol):
        '''initialization of black scholes d1 and d2 for option valuation'''

        # Calculation of d1, d2
        ttm = ttm if ttm > 0 else 1
        d1 = (math.log(spot / strike) + ((rfr - dy + ((vol * vol) / 2)) * ttm)) / (vol * math.sqrt(ttm))
        d2 = d1 - (vol * math.sqrt(ttm))

        return d1, d2

    @property
    def expired(self):
        return self.date >= self.expiry

    @staticmethod
    def exact_days(delta):
        spd = 86400
        if isinstance(delta, timedelta):
            return delta.total_seconds() / spd
        raise TypeError(f"{self.__class__.__name__}.exact_days method only accepts timdelta objects. Received {delta.__class__.__name__} {delta=}")

    @property
    def ttm(self):
        return max(self.expiry - self.date, timedelta())

    @property
    def spot(self):
        return self.underlying.value

    def reset(self):
        self._mature = False


class Call(Option, settings=settings["CALL"]):

    def valuate(self, date):
        if self._mature:
            return self.value

        elif self.expired:
            self._mature = True
            if self.strike < self.underlying.value:
                return (self.underlying.value - self.strike) * 100
            else:
                return 0
        else:
            return self.black_scholes(date)

    def black_scholes(self, date):
        ttm = self.exact_days(self.ttm) / 365
        div_yield = 0.01

        d1, d2 = self._bsbase(
                              spot = self.spot, 
                              strike = self.strike, 
                              rfr = self.rfr, 
                              dy = div_yield, 
                              ttm = ttm, 
                              vol = 0.3
                              )

        call_premium = (self.cdf(d1) * self.spot * math.pow(math.e, -1 * (div_yield * ttm))) - (self.cdf(d2) * self.strike * math.pow(math.e, -1 * (self.rfr * ttm)))

        # Each contract is for 100 shares
        return call_premium * 100


    def expire(self, portfolio, position):

        # Expired call positions are only assigned if they are ITM
        if not position.asset.itm:
            return None

        underlying = position.asset.underlying
        key = f"{underlying.key}_LONG"
        quantity = position.quantity * 100
        cost = quantity * position.asset.strike
        underlying_pos = portfolio._positions.get(key, None)
        call_pos = position.__class__(underlying, quantity)
        call_pos.cost = cost

        if position.short:
            if underlying_pos and underlying_pos.quantity >= quantity:
                portfolio._positions[key] -= call_pos
                portfolio.cash += cost
            else:
                portfolio.cash += position.value
        else:
            if portfolio.cash > call_pos.value:
                if underlying_pos:
                    portfolio._positions[key] += call_pos
                else:
                    portfolio._positions[key] = call_pos
                portfolio.cash -= cost
            else:
                portfolio.cash += position.value

    @property
    def itm(self):
        return self.strike < self.underlying.value




class Put(Option, settings=settings["OPTION"]):

    def valuate(self, date):
        if self._mature:
            return self.value

        elif self.expired:
            self._mature = True
            if self.strike > self.underlying.value:
                return (self.strike - self.underlying.value) * 100
            else:
                return 0
        else:
            return self.black_scholes(date)

    def black_scholes(self, date):
        ttm = self.exact_days(self.ttm) / 365
        div_yield = 0.01

        d1, d2 = self._bsbase(
                              spot = self.spot, 
                              strike = self.strike, 
                              rfr = self.rfr, 
                              dy = div_yield, 
                              ttm = ttm, 
                              vol = 0.3
                              )

        put_premium = (self.cdf(-1 * d2) * self.strike * math.pow(math.e, -1 * (self.rfr * ttm))) - (self.cdf(-1 * d1) * self.spot * math.pow(math.e, -1 * (div_yield * ttm)))

        # Each contract is for 100 shares
        return put_premium * 100


    def expire(self, portfolio, position):

        # Expired put positions are only assigned if they are ITM
        if not position.asset.itm:
            return None

        underlying = position.asset.underlying
        key = f"{underlying.key}_LONG"
        quantity = position.quantity * 100
        cost = quantity * position.asset.strike
        underlying_pos = portfolio._positions.get(key, None)
        put_pos = position.__class__(underlying, quantity)
        put_pos.cost = cost

        if position.short:
            if underlying_pos:
                portfolio._positions[key] += put_pos
                portfolio.cash -= cost
            else:
                portfolio.cash += position.value
        else:
            if underlying_pos and underlying_pos.quantity >= quantity:
                portfolio._positions[key] -= put_pos
                portfolio.cash += cost
            else:
                portfolio.cash += position.value


    @property
    def itm(self):
        return self.strike > self.underlying.value

# BROWNIAN STOCK IMPLEMENTATION BELOW

"""
class BrownianStock(Asset, settings=settings["BROWNIANSTOCK"]):
    def __init__(self, ticker=None, date=None, resolution=timedelta(days=1), base=None):
        self.resolution = resolution
        if ticker is None:
            ticker = self._generate_ticker()
            while ticker in TYPES.BROWNIANSTOCK.instances:
                ticker = self._generate_ticker()

        super().__init__(ticker, date=date, base=base)

        self.rng = np.random.default_rng()
        self.price = self.rng.gamma(1.5, 100)


    def _generate_ticker(self, random=False, last=[None]):
        if random:
            return ''.join([chr(np.random.randint(65, 91))
                                 for _ in range(4)])
        else:
            name = None

            if last[0] is None:
                name = [65, 65, 65]

            else:
                name = [ord(char) for char in last[0]]
                for i, char in enumerate(name):
                    if char < 91:
                        name[i] += 1
                        break
                    else:
                        name[i] = 65
                else:
                    name =  [65 for _ in range((len(name) + 1))]
                    
            name = ''.join([chr(char) for char in name])
            last[0] = name
            return name
        

    def valuate(self, date=None):
        date = self.normalize_date(date)
        if self.date + self.resolution > date:
            #do something
            return self.price

        return self.price
"""


"""
TO BE IMPLEMENTED IN THE FUTURE
-- TODO --

class Commodity(Asset):
    def __init__(self, item):
        super().__init__(item)

    def _valuate(self):
        return NotImplemented

class Future(Asset):
    def __init__(self):
        raise NotImplementedError

class RealEstate(Asset):
    def __init__(self):
        raise NotImplementedError

class Crypto(Asset):
    def __init__(self):
        raise NotImplementedError

class Virtual(Asset):
    def __init__(self):
        raise NotImplementedError

class Unique(Asset):
    def __init__(self):
        raise NotImplementedError

"""
