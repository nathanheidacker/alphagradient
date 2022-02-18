# -*- coding: utf-8 -*-
"""AG module containing classes for holding and organizing assets

Todo:
    * na
"""
# Standard imports
from numbers import Number
from datetime import datetime, timedelta
from weakref import WeakValueDictionary as WeakDict
from enum import Enum, auto

# Third Party imports
import pandas as pd
import numpy as np

# Local Imports
from .._globals import __globals as glbs
from .asset import Asset, types
from .portfolio import Portfolio, Cash
from .standard import Currency, Stock, Call, Put

class Basket:

    class Status(Enum):
        NONE = auto()
        SINGLE = auto()
        MULTIPLE = auto()

        @classmethod
        def get(cls, n):
            if n == 0:
                return cls.NONE
            elif n == 1:
                return cls.SINGLE
            else:
                return cls.MULTIPLE

    class AssetDict(WeakDict):

        def __init__(self, cls, basket):
            self._name = cls.__name__
            self.c = cls
            self._basket = basket
            super().__init__()

        def __call__(self, *args, **kwargs):
            new = self.c(*args, **kwargs)
            if not getattr(new, "underlying", False):
                new._valuate(self._basket.start)
            new.base = self._basket.base.code
            self._basket.assets.append(new)
            self[new.name] = new
            return new

        def __str__(self):
            return str(dict(self))

        def __repr__(self):
            return self.__str__()

        def __getattr__(self, attr):
            try:
                return self[attr]
            except KeyError:
                raise AttributeError(f"AlphaGradient Basket has no {attr.capitalize()} instance {attr}")
    
    def __init__(self, start=None, end=None, resolution=None, base=None, assets=None, portfolios=None):
        self._start = glbs.start if start is None else self.validate_date(start)
        self._date = self.start
        self._end = glbs.end if end is None else self.validate_date(end)
        self._resolution = glbs.resolution if resolution is None else self.validate_resolution(resolution)
        self._assets = [] if not isinstance(assets, list) else assets
        self._portfolios = [] if not isinstance(assets, list) else portfolios
        self._base = glbs.base if not isinstance(base, Currency) else base

    def __getattr__(self, attr):
        instantiable = {c.__name__.lower(): c for c in types.instantiable()}
        if attr in instantiable:
            assetdict = self.AssetDict(instantiable[attr], self)
            setattr(self, attr, assetdict)
            return assetdict
        elif attr in self.Status._member_names_:
            return self.status is self.Status[attr]
        elif getattr(Portfolio, attr, False):
            return self._redirect(attr)
        else:
            raise AttributeError(f"AlphaGradient Basket object has no attribute {attr}")

    @property
    def start(self):
        return self._start

    @property
    def date(self):
        return self._date

    @property
    def end(self):
        return self._end

    @property
    def resolution(self):
        return self._resolution

    @property
    def base(self):
        return self._base

    @property
    def assets(self):
        return self._assets

    @property
    def portfolios(self):
        return {p.name : p for p in self._portfolios}

    @property
    def status(self):
        return self.Status.get(len(self.portfolios))

    @start.setter
    def start(self, date):
        self._start = self.validate_date(date)

    @date.setter
    def date(self, date):
        date = self.validate_date(date)
        if not isinstance(date, datetime):
            raise TypeError(f"date must be a datetime, received {date=}")
        self._date = date

    @end.setter
    def end(self, date):
        self._end = self.validate_date(date)

    @resolution.setter
    def resolution(self, delta):
        self._resolution = self.validate_resolution(delta)

    @base.setter
    def base(self, code):
        self._base = Currency.validate_code(code, error=True)

    def portfolio(self, initial, name=None, date=None, base=None):
        if isinstance(initial, Portfolio):
            self._portfolios.append(initial)
            return initial
        else:
            base = self.base.code if base is None else Currency.validate_code(base)
            date = self.start if date is None else self.validate_date(date)
            new = Portfolio(initial, name, date, base)
            self._portfolios.append(new)
            return new

    def data(self, dtype="dict"):
        dtype = dtype.lower() if isinstance(dtype, str) else dtype
        if dtype in [dict, "dict"]:
            return {asset.key:asset.data for asset in self.assets if asset.data}
        elif dtype in [list, "list"]:
            return [asset.data for asset in self.assets if asset.data]
        else:
            raise ValueError(f"Unsupported type {dtype=}. Basket data can only be returned as a list or a dictionary")

    @staticmethod
    def validate_date(date):
        if isinstance(date, str):
            return datetime.fromisoformat(date)

        elif isinstance(date, datetime):
            return date

        else:
            raise TypeError(f"Could not normalize {date.__class__.__name__} {date} to datetime. Dates should be datetimes or isoformat date-strings")

    @staticmethod
    def validate_resolution(resolution):
        if isinstance(resolution, timedelta):
            return resolution
        else:
            raise TypeError(f"Resolution must be a datetime.timedelta object. Received {resolution.__class__.__name__} {resolution}")

    def default_start(self):
        data = [dataset.index[0] for dataset in self.data(dtype=list)]
        return max(data).to_pydatetime() if data else glbs.start

    def default_end(self):
        data = [dataset.index[-1] for dataset in self.data(dtype=list)]
        return min(data).to_pydatetime() if data else glbs.start

    def default_resolution(self):
        return timedelta(days=1)

    def auto(self):
        self.start = self.default_start()
        self.end = self.default_end()
        self.resolution = self.default_resolution()

    def sync(self, date=None):

        date = self.start if date is None else self.validate_date(date)

        for portfolio in self._portfolios:
            portfolio.date = date
            portfolio.reset()

        for asset in self.assets:
            if isinstance(asset, (Call, Put)):
                asset.reset()
            asset._valuate(date)

    def autosync(self):
        self.auto()
        self.sync()

    def step(self, delta=None):

        delta = self.resolution if delta is None else self.validate_resolution(delta)

        for portfolio in self._portfolios:
            portfolio.date = portfolio.date + delta
            portfolio.update_history()

        for asset in self._assets:
            asset._valuate(asset.date + delta)
            asset._step(asset.date + delta)

        self.date = self.date + delta

    def buy(self, asset, quantity, name=None):
        if self.NONE:
            raise ValueError(f"This basket has no active portfolios. Please instantiate one in order to make transactions")
        elif self.SINGLE:
            portfolio = self._portfolios[0]
            portfolio.buy(asset, quantity)
        elif self.MULTIPLE:
            if name is None:
                try:
                    self.portfolios["MAIN"].buy(asset, quantity)
                except KeyError:
                    raise ValueError(f"This basket has multiple portfolios. The portfolio name for this transaction must be specified.")
            else:
                try:
                    portfolio = self.portfolios[name]
                    portfolio.buy(asset, quantity)
                except KeyError:
                    raise ValueError(f"Basket has no portfolio instance named {name.__repr__()}")

    def sell(self, asset, quantity, name=None):
        if self.NONE:
            raise ValueError(f"This basket has no active portfolios. Please instantiate one in order to make transactions")
        elif self.SINGLE:
            portfolio = self._portfolios[0]
            portfolio.sell(asset, quantity)
        elif self.MULTIPLE:
            if name is None:
                try:
                    self.portfolios["MAIN"].sell(asset, quantity)
                except KeyError:
                    raise ValueError(f"This basket has multiple portfolios. The portfolio name for this transaction must be specified.")
            else:
                try:
                    portfolio = self.portfolios[name]
                    portfolio.sell(asset, quantity)
                except KeyError:
                    raise ValueError(f"Basket has no portfolio instance named {name.__repr__()}")

    def short(self, asset, quantity, name=None):
        if self.NONE:
            raise ValueError(f"This basket has no active portfolios. Please instantiate one in order to make transactions")
        elif self.SINGLE:
            portfolio = self._portfolios[0]
            portfolio.short(asset, quantity)
        elif self.MULTIPLE:
            if name is None:
                try:
                    self.portfolios["MAIN"].short(asset, quantity)
                except KeyError:
                    raise ValueError(f"This basket has multiple portfolios. The portfolio name for this transaction must be specified.")
            else:
                try:
                    portfolio = self.portfolios[name]
                    portfolio.short(asset, quantity)
                except KeyError:
                    raise ValueError(f"Basket has no portfolio instance named {name.__repr__()}")

    def cover(self, asset, quantity, name=None):
        if self.NONE:
            raise ValueError(f"This basket has no active portfolios. Please instantiate one in order to make transactions")
        elif self.SINGLE:
            portfolio = self._portfolios[0]
            portfolio.cover(asset, quantity)
        elif self.MULTIPLE:
            if name is None:
                try:
                    self.portfolios["MAIN"].cover(asset, quantity)
                except KeyError:
                    raise ValueError(f"This basket has multiple portfolios. The portfolio name for this transaction must be specified.")
            else:
                try:
                    portfolio = self.portfolios[name]
                    portfolio.cover(asset, quantity)
                except KeyError:
                    raise ValueError(f"Basket has no portfolio instance named {name.__repr__()}")

    def _redirect(self, attr):
        if self.NONE:
            raise AttributeError(f"Basket has no active portfolios. Must instantiate at least one portfolio to access portfolio attribute {attr}")
        elif self.SINGLE:
            return getattr(self._portfolios[0], attr)
        else:
            return {name: getattr(p, attr) for name, p in self.portfolios.items()}

    def reset(self):
        self.date = self.start
        self.sync()


# Using a protected keyword, attr must be set outside of the class
setattr(Basket, "type", types.basket)
setattr(types.basket, "c", Basket)



class Universe:
    pass

# Using a protected keyword, attr must be set outside of the class
setattr(Universe, "type", types.universe)
setattr(types.universe, "c", Universe)
