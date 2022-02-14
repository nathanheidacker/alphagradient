# -*- coding: utf-8 -*-
"""AG module containing globally useful functions and constants used throughout the package

Todo:
    * na
"""
# Standard imports
from datetime import datetime, timedelta

# Third party imports

# Local imports
from .finance import types, Asset, Currency

class Globals:
    """A set of global conditions and functions that act on all alphagradient objects

    Attributes:
        start (datetime): the default beginning for asset initialization
        end (datetime): the default end for algorithm backtesting
        resolution (timedelta): the default time difference between each
            valuation step when using globals.step()
    """

    def __init__(self):
        self._start = self.default_start()
        setattr(Asset, "_global_start", self.start)
        self._end = self.default_end()
        self._resolution = self.default_resolution()
        self._base = Currency(Currency.base)

    def __str__(self):
        return str({k[1:]:v for k, v in self.__dict__.items()})

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def all_assets():
        """Returns all assets currently in memory"""
        return [asset for t in types if Asset in t.c.__bases__ for asset in t.instances.values()]

    @staticmethod
    def all_data():
        """Returns all asset datasets"""
        return [asset.data for t in types if Asset in t.c.__bases__ for asset in t.instances.values() if asset.data]

    @staticmethod
    def normalize_date(date):
        """coerces reasonable inputs into datetimes"""
        if isinstance(date, str):
            return datetime.fromisoformat(date)

        elif isinstance(date, datetime):
            return date

        else:
            raise TypeError(
                f"Date input of type {type(date)} could not be normalized")

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, date=None):
        date = default_start() if not isinstance(date, datetime) else date
        self._start = date
        setattr(Asset, "_global_start", self.start)

    def default_start(self):
        """The default start if nothing else is provided

        If any assets are instantiated, returns the earliest available date for which data is available in any asset dataset. If no assets are instantiated, returns todays date less 10 years

        Returns:
            date (datetime): the default start date to be used
        """
        data = [data.index[0] for data in self.all_data()]
        default = datetime.today() - timedelta(days=3650)
        return min(data) if data else default

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, date=None):
        date = self.default_end() if not isinstance(date, datetime) else date
        self._end = date

    def default_end(self):
        """The default end if nothing else is provided

        If any assets are instantiated, returns the last available date for which data is available in any asset dataset. If no assets are instantiated, returns todays date

        Returns:
            date (datetime): the default end date to be used
        """
        data = [data.index[-1] for data in self.all_data()]
        default = datetime.today()
        return min(data) if data else default

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, delta=None):
        delta = self.default_resolution() if not isinstance(delta, timedelta) else timedelta
        self._resolution = delta

    def default_resolution(self):
        """The default universal time resolution to be used

        If any assets are instantiated, finds the lowest time resolution (timedelta between datetime indexed data) present in any asset dataset. Otherwise, defaults to 1 day

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

    def auto(self):
        """Automatically sets global start, end, and resolution to their defaults based on what assets are instantiated

        Returns:
            Modifies global variables in place, returns nothing
        """
        self._start = self.default_start()
        self._end = self.default_end()
        self._resolution = self.default_resolution()

    def sync(self, date=None):
        """Synchronizes all alphagradient objects globally to the given datetime

        Args:
            date (datetime): The date to synchronize to

        Returns:
            modifies all currently instantiated assets and portfolios in-place
        """
        date = date if isinstance(date, datetime) else self.start

        for asset in self.all_assets():
            asset._valuate(date)

    def autosync(self):
        """automatically determines best global variables for instantiated assets, and syncs them all to that date"""
        self.auto()
        self.sync()

    def step(self, delta=None):
        """Takes a single time step for all assets

        Valuates all currently instantiated assets one time step (of magnitude delta) ahead of their previous valuation. When not provided with a delta explicitly, uses the global resolution to determine time step magnitude.

        Args:
            delta (timedelta): The magnitude of the time step taken

        Returns:
            Modifies assets in place, returns nothing
        """
        delta = delta if isinstance(delta, timedelta) else self.resolution

        for asset in self.all_assets():
            asset._valuate(asset.date + delta)


__globals = Globals()
"""The global variables class instance"""

