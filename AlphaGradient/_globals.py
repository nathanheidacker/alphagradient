# -*- coding: utf-8 -*-
"""AG module containing globally useful functions and constants used
throughout the package

Todo:
    * Implement updating/refreshing asset data for locally saved assets
        - One at a time and for entire exchanges
"""
# Standard imports
from datetime import datetime, timedelta

# Third party imports

# Local imports
from .finance import types, Asset, Currency, Call, Put, Stock
from .finance.standard import Option
from .data.datatools import AssetData
from . import utils

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

    def __init__(self):
        self._start = self.default_start()
        setattr(Asset, "_global_start", self.start)
        self._end = self.default_end()
        self._resolution = self.default_resolution()
        setattr(AssetData, "_global_res", self.resolution)
        setattr(Asset, "_global_res", self.resolution)
        self._base = Currency(Currency.base)
        self.RISK_FREE_RATE = 0.0
        setattr(Asset, "rfr", self.rfr)
        self._benchmark = Stock("SPY")
        setattr(Asset, "benchmark", self._benchmark)

    def __str__(self):
        return str({k[1:]:v for k, v in self.__dict__.items()})

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def all_assets():
        """Returns all assets currently in memory"""
        return [asset for t in types if t.c in types.instantiable() for asset in t.instances.values()]

    @staticmethod
    def all_data():
        """Returns all asset datasets"""
        return [asset.data for t in types if t.c in types.instantiable() for asset in t.instances.values() if asset.data]

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
    def start(self):
        return self._start

    @start.setter
    def start(self, date=None):
        date = self.default_start() if not isinstance(date, datetime) else date
        self._start = date
        setattr(Asset, "_global_start", self.start)

    def default_start(self):
        """The default start if nothing else is provided

        If any assets are instantiated, returns the earliest available
        date for which data is available in any asset dataset. If no
        assets are instantiated, returns todays date less 10 years

        Returns:
            date (datetime): the default start date to be used
        """
        data = [data.index[0] for data in self.all_data()]
        default = datetime.today() - timedelta(days=3650)
        start = max(data).to_pydatetime() if data else default
        start = utils.set_time(start, "9:30 AM")
        return start

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, date=None):
        date = self.default_end() if not isinstance(date, datetime) else date
        self._end = date

    def default_end(self):
        """The default end if nothing else is provided

        If any assets are instantiated, returns the last available
        date for which data is available in any asset dataset. If no
        assets are instantiated, returns todays date

        Returns:
            date (datetime): the default end date to be used
        """
        data = [data.index[-1] for data in self.all_data()]
        default = datetime.today()
        end = min(data).to_pydatetime() if data else default
        end = utils.set_time(end, "4:30 PM")
        return end

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
        data = None
        default = timedelta(days=1)
        return min(data) if data else default

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
        return self.RISK_FREE_RATE

    @rfr.setter
    def rfr(self, rate):
        setattr(Option, "rfr", rate)
        self.RISK_FREE_RATE = rate

    @property
    def benchmark(self):
        return self._benchmark

    @benchmark.setter
    def benchmark(self, benchmark):
        if isinstance(benchmark, Stock):
            self._benchmark = benchmark
            setattr(Asset, "benchmark", self._benchmark)
        elif isinstance(benchmark, str):
            try:
                self._benchmark = Stock(benchmark)
                setattr(Asset, "benchmark", self._benchmark)
            except Exception as e:
                raise RuntimeError(f"Unable to use {benchmark} as a benchmark because the following error occurred during initialization: {e}") from e
        else:
            raise TypeError(f"Benchmark must be a Stock. Received {benchmark.__class__.__name__}")

    def auto(self):
        """Automatically sets global start, end, and resolution to
        their defaults based on what assets are instantiated

        Returns:
            Modifies global variables in place, returns nothing
        """
        self._start = self.default_start()
        self._end = self.default_end()
        self._resolution = self.default_resolution()

    def sync(self, date=None):
        """Synchronizes all alphagradient objects globally to the
        given datetime

        Args:
            date (datetime): The date to synchronize to

        Returns:
            modifies all currently instantiated assets and portfolios
            in-place
        """
        if date is not None:
            date = self.normalize_date(date)
        date = date if isinstance(date, datetime) else self.start

        for asset in self.all_assets():
            if isinstance(asset, (Call, Put)):
                asset.reset()
            asset._valuate(date)

        for portfolio in list(types.portfolio.instances.values()):
            portfolio.date = date
            portfolio.reset()

    def autosync(self):
        """automatically determines best global variables for
        instantiated assets, and syncs them all to that date"""
        self.auto()
        self.sync()

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
        delta = delta if isinstance(delta, timedelta) else self.resolution

        for asset in self.all_assets():
            asset._valuate(asset.date + delta)
            asset._step(asset.date + delta)

        for portfolio in types.portfolio.instances.values():
            portfolio.date = portfolio.date + delta
            portfolio.update_history()


__globals = Globals()
"""The global variables class instance"""