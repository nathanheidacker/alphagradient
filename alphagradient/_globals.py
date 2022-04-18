# -*- coding: utf-8 -*-
"""AG module containing globally useful functions and constants used
throughout the package

Todo:
    * Implement updating/refreshing asset data for locally saved assets
        - One at a time and for entire exchanges
"""
# Standard imports
from datetime import datetime, timedelta, time
from collections import deque
from pathlib import Path
import requests
import os

# Third party imports
from p_tqdm import p_map
from bs4 import BeautifulSoup as bs
import pandas as pd

# Local imports
from .finance import (
                      Portfolio,
                      Basket,
                      Universe,
                      types,
                      Asset,
                      Currency,
                      Call,
                      Put,
                      Stock
                      )
from .finance.standard import Option
from .data.datatools import AssetData
from .algorithm import Algorithm, Run, Performance
from . import utils

GLOBAL_DEFAULT_START = datetime.fromisoformat("2000-01-03")
GLOBAL_DEFAULT_END = utils.set_time(datetime.today(), "0:0:0")
GLOBAL_DEFAULT_RESOLUTION = timedelta(days=1)
GLOBAL_DEFAULT_BASE = "USD"

class Globals:
    """A set of global conditions and functions that act on all
    alphagradient objects

    Attributes:
        start (datetime): the default beginning for asset initialization
        end (datetime): the default end for algorithm backtesting
        resolution (timedelta): the default time difference between each
            valuation step when using globals.step()
        base (Currency): The currency object that acts as the base
            which all asset values are converted to
        rfr (Number): The current risk free rate
    """
    def _mkprop(self, attr):
        return property(lambda this: getattr(self, attr))

    def _shareprop(self, attr, obj, name=None):
        setattr(obj, (name or attr), self._mkprop(attr))

    def __init__(self):

        # Initializing global attrs
        self._start = GLOBAL_DEFAULT_START
        self._end = GLOBAL_DEFAULT_END
        self._resolution = GLOBAL_DEFAULT_RESOLUTION
        self._date = GLOBAL_DEFAULT_START
        self._persistent = self._find_persistent()

        for cls in [Asset, Basket, Algorithm]:
            self._shareprop("start", cls, name="_global_start")

        for cls in [Basket, Algorithm]:
            self._shareprop("end", cls, name="_global_end")

        for cls in [Asset, AssetData, Basket, Algorithm]:
            self._shareprop("resolution", cls, name="_global_res")

        for cls in [Asset, Portfolio, Run, Performance]:
            self._shareprop("date", cls)

        for cls in [Asset, Universe, utils]:
            self._shareprop("persistent", cls, name="_global_persistent_path")

        # Baskets must also have the ability to set the global date
        date_getter = lambda this: getattr(self, "date")
        date_setter = lambda this, dt: setattr(self, "date", dt)
        setattr(Basket, "date", property(date_getter, date_setter))

        self._base_code = GLOBAL_DEFAULT_BASE
        #self._shareprop("_base_code", Currency, name="base")
        self._shareprop("_base_code", Asset, name="_global_base")

        self._path = Path(__file__).parent
        self._shareprop("path", Asset, name="_basepath")
        self._shareprop("path", Universe, name="_basepath")

        self._base = Currency(Currency.base)
        self._shareprop("base", Basket, name="_global_base")

        self._rfr = self._get_rfr()
        self._shareprop("rfr", Asset)

        self._benchmark = Stock("SPY")
        self._shareprop("benchmark", Asset)
        self._shareprop("benchmark", Universe)

    def __str__(self):
        return str({k[1:]:v for k, v in self.__dict__.items()})

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _get_rfr():
        try:
            search = "https://www.google.com/search?q=3+month+t+bill+rate"
            data = requests.get(search)
            data = bs(data.text, "lxml")
            data = data.find_all("td", class_="IxZjcf sjsZvd s5aIid OE1use")
            return round(float(data[0].text[:-1]) / 100, 5)
        except Exception as e:
            return 0.0

    @staticmethod
    def all_assets():
        """Returns all assets currently in memory"""
        return (asset for t in types if t.c in types.instantiable() for asset in t.instances.values())

    @staticmethod
    def all_data():
        """Returns all asset datasets"""
        return (asset.data for t in types if t.c in types.instantiable() for asset in t.instances.values() if asset.data)

    @staticmethod
    def normalize_date(date):
        """coerces reasonable inputs into datetimes

        Args:
            date (str | datetime): the date to coerce

        Returns:
            date (datetime): The coerced date

        Raises:
            TypeError: when the date can not be coerced into datetime
        """
        if isinstance(date, str):
            return datetime.fromisoformat(date)

        elif isinstance(date, datetime):
            return date

        raise TypeError(f"Date input of type {type(date)} could not be"
                        "normalized")

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, dt):
        if isinstance(dt, datetime):
            self._date = dt

        elif isinstance(dt, pd.Timestamp):
            self._date = dt.to_pydatetime()

        elif isinstance(dt, str):
            self._date = datetime.fromisoformat(dt)

        else:
            raise TypeError(f"Unable to set global date to {dt}, invalid type {type(dt).__name__}")

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, date=None):
        date = self.default_start() if not isinstance(date, datetime) else date
        self._start = date
        setattr(Asset, "_global_start", self.start)

        for algo in types.algorithm.instances.values():
            algo.start = max(algo.start, self._start)

    def default_start(self):
        """The default start if nothing else is provided

        If any assets are instantiated, returns the earliest available
        date for which data is available in any asset dataset. If no
        assets are instantiated, returns todays date less 10 years

        Returns:
            date (datetime): the default start date to be used
        """
        data = [data.first for data in self.all_data()]

        if data:
            start = max(data).to_pydatetime()
            return utils.set_time(start, "9:30 AM")

        return GLOBAL_DEFAULT_START

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, date=None):
        date = self.default_end() if not isinstance(date, datetime) else date
        self._end = date

        for algo in types.algorithm.instances.values():
            algo.end = min(algo.end, self._end)

    def default_end(self):
        """The default end if nothing else is provided

        If any assets are instantiated, returns the last available
        date for which data is available in any asset dataset. If no
        assets are instantiated, returns todays date

        Returns:
            date (datetime): the default end date to be used
        """
        data = [data.last for data in self.all_data()]

        if data:
            end = min(data).to_pydatetime()
            return utils.set_time(end, "4:30 PM")

        return GLOBAL_DEFAULT_END

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, delta=None):
        delta = self.default_resolution() if not isinstance(delta, timedelta) else timedelta
        self._resolution = delta

    def default_resolution(self):
        """The default universal time resolution to be used

        If any assets are instantiated, finds the lowest time
        resolution (timedelta between datetime indexed data) present
        in any asset dataset. Otherwise, defaults to 1 day

        Returns:
            delta (timedelta): The default time resolution
        """
        data = [data.resolution for data in self.all_data()]

        if data:
            return min(data)

        return GLOBAL_DEFAULT_RESOLUTION

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, code):
        if Currency.validate_code(code, error=True):
            Currency.base = code
            self._base = Currency(code)
            for currency in types.currency.instances.values():
                currency.base = code
                currency._valuate()

    @property
    def rfr(self):
        """An alias for the global risk free rate"""
        return self._rfr

    @rfr.setter
    def rfr(self, rate):
        setattr(Option, "rfr", rate)
        self._rfr = rate

    @property
    def benchmark(self):
        return self._benchmark

    @benchmark.setter
    def benchmark(self, benchmark):
        if isinstance(benchmark, Stock):
            self._benchmark = benchmark
        elif isinstance(benchmark, str):
            try:
                self._benchmark = Stock(benchmark)
            except Exception as e:
                raise RuntimeError(f"Unable to use {benchmark} as a benchmark because the following error occurred during initialization: {e}") from e
        else:
            raise TypeError(f"Benchmark must be a Stock. Received {benchmark.__class__.__name__}")

    @property
    def path(self):
        return self._path

    def auto(self):
        """Automatically sets global start, end, and resolution to
        their defaults based on what assets are instantiated

        Returns:
            Modifies global variables in place, returns nothing
        """
        self._start = self.default_start()
        self._end = self.default_end()
        self._resolution = self.default_resolution()

    def sync(self, date):
        """Synchronizes all alphagradient objects globally to the
        given datetime

        PLEASE NOTE THAT SYNCING TO THE START MAY CAUSE ERRORS WHEN RUNNING ALGORITHMS, AS NO DATA
        IS AVAILABLE FOR CALCULATIONS THAT REQUIRE CHANGE ACROSS SOME TIME INTERVAL. FOR EXAMPLE.
        CALCULATIONS OF AN ASSET'S VOLATILITY BECOME IMPOSSIBLE BECAUSE THEY CAN ONLY ACCESS A
        SINGLE DATA POINT. THIS IS EXPECTED BEHAVIOR

        Args:
            date (datetime): The date to synchronize to

        Returns:
            modifies all currently instantiated assets and portfolios
            in-place
        """
        self.date = self.normalize_date(date)

        def sync_asset(asset):
            if getattr(asset, "reset", False):
                asset.reset()
            asset._valuate()

        def sync_portfolio(portfolio):
            portfolio.reset()

        assets = list(self.all_assets())
        portfolios = list(types.portfolio.instances.values())

        if assets:
            deque(map(sync_asset, assets), maxlen=0)

        if portfolios:
            deque(map(sync_portfolio, portfolios), maxlen=0)

    def autosync(self, t=None):
        """automatically determines best global variables for start, end, and resolution based on
        currently instantiated assets, and syncs them all to an optimal starting period"""

        # Determining absolute start and end of available data
        self.auto()

        # Determining the optimal start period. To avoid errors, we will not sync to the beginning
        optimal_difference = (self.end - self.start) / 2
        optimal_date = self.start + optimal_difference

        # Setting the optimal date's time to midnight unless specified otherwise
        t = "00:00:00" if t is None else t
        utils.set_time(optimal_date, t)

        # Bounding the date to acceptable minimums and maximums
        lower_bound = utils.set_time(self.start + timedelta(days=1), t)
        upper_bound = utils.set_time(self.start + timedelta(days=365), t)
        sync_date = utils.bounded(optimal_date, lower=lower_bound, upper=upper_bound)

        self.sync(sync_date)

    def step(self, delta=None):
        """Takes a single time step for all assets

        Valuates all currently instantiated assets one time step
        (of magnitude delta) ahead of their previous valuation. When
        not provided with a delta explicitly, uses the global
        resolution to determine time step magnitude.

        Args:
            delta (timedelta): The magnitude of the time step taken

        Returns:
            Modifies assets in place, returns nothing
        """
        if isinstance(delta, str):
            try:
                delta = datetime.fromisoformat(delta)
            except ValueError:
                delta = time.fromisoformat(delta)

        if isinstance(delta, (datetime, pd.Timestamp)):
            delta = delta - self.date

        elif isinstance(delta, time):
            if delta >= self.date.time():
                delta = utils.set_time(self.date, delta)
            else:
                delta = utils.set_time(self.date - timedelta(days=1), delta)

        delta = self.resolution if delta is None else self.validate_resolution(delta)

        self.date = self.date + delta

        for asset in self.all_assets():
            asset._valuate()
            asset._step()

        for portfolio in types.portfolio.instances.values():
            portfolio.update_positions()
            portfolio.update_history()

        for algo in types.algorithm.instances.values():
            algo.stats.update()

    def refresh(self):
        pass

    def refresh_local(self):
        pass

    def refresh_all(self):
        pass

    def scan(self):
        pass

    @staticmethod
    def validate_resolution(resolution):
        """Determines whether or not the input resolution is valid.
        If is is, returns it as a native python timedelta object

        Args:
            resolution (Number | str | timedelta): The resolution to be
                validated and converted

        Returns:
            resolution (timedelta): The convered, valid resolution

        Raises:
            TypeError: When the resolution is inconvertible
        """
        if isinstance(resolution, timedelta):
            return resolution
        elif isinstance(resolution, int):
            return timedelta(days=resolution)
        else:
            raise TypeError(f"Resolution must be a datetime.timedelta "
                            "object. Received "
                            f"{resolution.__class__.__name__} "
                            f"{resolution}")

    def persist(self, path=None):
        """Allows persistent storage of ag assets

        Directs alphagradient to persist ag asset datasets in memory at the given path, which
        defaults to the current working directory when none is specified. AlphaGradient will create
        a directory in the path called "alphagradient.persistent" in which it will store pickles to easily reinstantiate assets

        Parameters:
            path (path-like): Where to create/use the pickle storage directory.

        Returns:
            modifies the global instance in place by saving the storage path
        """
        if path is None:
            path = self._default_persistent()
        else:
            path = path.joinpath("alphagradient.persistent/")
        if not os.path.isdir(path):
            path.mkdir()
        self._persistent = path

    @property
    def persistent(self):
        return self._persistent

    def _find_persistent(self):
        path = self._default_persistent()
        if os.path.isdir(path):
            return path
        return None

    def _default_persistent(self):
        return Path(os.getcwd()).joinpath("alphagradient.persistent/")



__globals = Globals()
"""The global variables class instance"""