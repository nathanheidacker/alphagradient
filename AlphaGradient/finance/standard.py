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
import numpy as np

# Local imports
from .asset import Asset, types
from ..data import datatools

with open("AlphaGradient/finance/standard_asset_settings.json") as f:
    settings = json.load(f)
"""settings (dict): dictionary of settings for initializing standard asset subclasses"""

class Currency(Asset, settings=settings["CURRENCY"]):

    to_code = {
        '$': 'USD',
        'GBP': 'GBP',
        'YEN': 'YEN',
    }

    to_symbol = {v:k for k, v in to_code.items()}

    def __init__(self, identifier='USD'):

        if not isinstance(identifier, str):
            raise TypeError(
                f'''{identifier.__class__.__name__} {identifier} is not a valid currency identifier. 
                Please use a currency code or symbol as a string''')

        code = None
        symbol = None

        if len(identifier) == 1:
            code = self.to_code[identifier]
            symbol = identifier

        elif len(identifier) == 3:
            code = identifier
            symbol = self.to_symbol[identifier]

        else:
            raise ValueError(
                f'''{identifier} is not a valid currency identifier. 
                Please use a currency symbol or three digit currency code''')

        super().__init__(code, date=datetime.today(), require_data=False)

    def valuate(self):
        return NotImplemented


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
