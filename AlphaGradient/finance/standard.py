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
        return f"<{self.type} {self.symbol}{self.code}: {self.get_symbol()}{self.roundprice}>"

    def valuate(self, *args, **kwargs):
        return self.convert(self.code, self.base)

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


class Currency2:

    _instance = []
    base = "USD"
    info = pd.read_pickle("AlphaGradient/finance/currency_info.p")

    def __init__(self):

        Currency2._instance.append(self)





class Stock(Asset, settings=settings["STOCK"]):
    """A financial asset representing stock in a publicly traded company

    An asset representing a single share of a publicly traded company. Supports online data from yahoo finance using yfinance.

    Attributes:
        ticker (str): This stock's ticker
    """
    def valuate(self):
        return self.price

    def online_data(self):
        data = yf.Ticker(self.name).history(period="max")
        return datatools.AssetData(Stock, data)

    @property
    def ticker(self):
        """Alias for name, relevant to stocks"""
        return self.name.upper()


class BrownianStock(Asset, settings=settings["BROWNIANSTOCK"]):
    def __init__(self, ticker=None, date=None, resolution=timedelta(days=1)):
        self.resolution = resolution
        if ticker is None:
            ticker = self._generate_ticker()
            while ticker in TYPES.BROWNIANSTOCK.instances:
                ticker = self._generate_ticker()

        super().__init__(ticker, date=date)

        self.rng = np.random.default_rng()
        self.price = self.rng.gamma(1.5, 100)


    def _generate_ticker(self):
        return ''.join([chr(np.random.randint(65, 91))
                                 for _ in range(4)])
        

    def valuate(self, date=None):
        date = self.normalize_date(date)
        if self.date + self.resolution > date:
            #do something
            return self.price

        return self.price


class Option(Asset, settings=settings["OPTION"]):

    def __init__(self, underlying, strike, expiry):
        super().__init__(underlying.name)

        if not isinstance(strike, (float, int)):
            try:
                strike = float(strike)
            except TypeError:
                raise f'''Invalid input type {strike=} 
                for initialization of {underlying.name} {self.__class__.__name__}'''
            except ValueError:
                raise f'''Unsuccessful conversion of {strike=} 
                to numeric type during initialization of {underlying.name} {self.type}'''
        self.strike = strike

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

    def _black_scholes(self, spot, strike, rfr, dy, ttm, vol):
        '''initialization of black scholes d1 and d2 for option valuation'''

        # Standard cumulative distribution function
        def cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        # Calculation of d1, d2
        d1 = (math.log(spot / strike) +
              ((rfr - dy + ((vol * vol) / 2)) * ttm)) / (vol * math.sqrt(ttm))
        d2 = d1 - (vol * math.sqrt(ttm))

        return d1, d2


class Call(Option, settings=settings["CALL"]):
    def __init__(self, underlying, strike, expiry):
        super().__init__(underlying, strike, expiry)

    def valuate(self):
        return NotImplemented


class Put(Option, settings=settings["OPTION"]):
    def __init__(self):
        raise NotImplementedError

    def valuate(self):
        return NotImplemented


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
