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
from ._finance import (
    Portfolio,
    Environment,
    Universe,
    types,
    Asset,
    Currency,
    Stock,
)
from ._finance._standard import Option
from ._data._datatools import AssetData
from ._algorithm import Algorithm, Backtest, Performance
from . import utils

# Typing
from typing import Union, Any, Optional, Generator

from .utils import DatetimeLike, TimeLike, DateOrTime, PyNumber, Number

GLOBAL_DEFAULT_START = datetime.today().replace(minute=0, second=0, microsecond=0)
"""The global default start date"""

GLOBAL_DEFAULT_END = datetime.today()
"""The global default end date"""

GLOBAL_DEFAULT_RESOLUTION = timedelta(days=1)
"""The global default time resolution"""

GLOBAL_DEFAULT_BASE = "USD"
"""The global default currency base (as a currency code)"""


class Globals:
    """A set of global conditions and functions that act on all
    alphagradient objects

    This object is required for initializing alphagradient assets,
    portfolios, universes, and environments properly, but also contains
    attributes with instantiated versions of these classes. For this reason,
    alphagradient dynamically shares attributes of this object with other
    classes to avoid circular import logic. Other classes may have attributes
    shared from this object, such as self.date

    This object is not to be instantiated by end users, and only has
    internal functionality

    Attributes:
        start (datetime):
            The default beginning for asset initialization

        end (datetime):
            The default end for algorithm backtesting

        resolution (timedelta):
            The default time difference between each valuation step when
            using globals.step()

        base (Currency):
            The currency object that acts as the base which all asset values
            are converted to

        rfr (Number):
            The current risk free rate
    """

    # Not sure how to properly type hint a property return value...
    # TODO: | THIS IS FUNCTIONALITY THAT SHOULD BE ABSTRACTED TO GLOBALS VIA A
    #      | 'BIND' COMMAND OR SOMETHING SIMILAR
    def _mkprop(self, attr: str) -> Any:
        return property(lambda this: getattr(self, attr))

    def _shareprop(self, attr: str, obj: Any, name: str = None) -> None:
        setattr(obj, (name or attr), self._mkprop(attr))

    def __init__(self) -> None:
        # Initializing global attrs
        self._date = GLOBAL_DEFAULT_START
        self._persistent = self._find_persistent()

        cls: Any  # This is supposed to represent an AG class
        for cls in [Asset, Environment, Algorithm]:
            self._shareprop("start", cls, name="_global_start")

        for cls in [Environment, Algorithm]:
            self._shareprop("end", cls, name="_global_end")

        for cls in [Asset, AssetData, Environment, Algorithm]:
            self._shareprop("resolution", cls, name="_global_res")

        for cls in [Asset, Portfolio, Backtest, Performance]:
            self._shareprop("date", cls, "_date")

        for cls in [Asset, Universe, utils]:
            self._shareprop("persistent", cls, name="_global_persistent_path")

        # Environments must also have the ability to set the global date
        date_getter = lambda this: getattr(self, "date")
        date_setter = lambda this, dt: setattr(self, "date", dt)
        setattr(Environment, "_date", property(date_getter, date_setter))

        self._base_code = GLOBAL_DEFAULT_BASE
        # self._shareprop("_base_code", Currency, name="base")
        self._shareprop("_base_code", Asset, name="_global_base")

        self._path: Path = Path(__file__).parent
        self._shareprop("path", Asset, name="_basepath")
        self._shareprop("path", Universe, name="_basepath")

        self._base: Currency = Currency(Currency.base)
        self._shareprop("base", Environment, name="_global_base")

        self._rfr: float = self._get_rfr()
        self._shareprop("rfr", Asset, name="_rfr")

        self._benchmark: Asset = Stock("SPY")
        self._shareprop("benchmark", Asset, name="_benchmark")
        self._shareprop("benchmark", Universe)

    def __str__(self) -> str:
        return str({k[1:]: v for k, v in self.__dict__.items()})

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _get_rfr() -> float:
        try:
            search = "https://www.google.com/search?q=3+month+t+bill+rate"
            data: Any = requests.get(search)
            data = bs(data.text, "lxml")
            data = data.find_all("td", class_="IxZjcf sjsZvd s5aIid OE1use")
            return round(float(data[0].text[:-1]) / 100, 5)
        except Exception as e:
            return 0.0

    @staticmethod
    def all_assets() -> Generator[Asset, None, None]:
        """Returns all assets currently in memory"""
        return (
            asset
            for t in types  # type: ignore[attr-defined]
            if t.c in types.instantiable()
            for asset in t.instances.values()
        )

    @staticmethod
    def all_data() -> Generator[AssetData, None, None]:
        """Returns all asset datasets"""
        return (
            asset.data
            for t in types  # type: ignore[attr-defined]
            if t.c in types.instantiable()
            for asset in t.instances.values()
            if asset.data
        )

    @property
    def date(self) -> datetime:
        "The date for the global environment"
        return self._date

    @date.setter
    def date(self, dt: DatetimeLike) -> None:
        self._date = utils.to_datetime(dt)

    @property
    def start(self) -> datetime:
        """The default start for the global environment"""
        data = [dataset.first for dataset in self.all_data()]
        return min(data) if data else GLOBAL_DEFAULT_START

    @property
    def end(self) -> datetime:
        """The default end for the global environment"""
        data = [dataset.last for dataset in self.all_data()]
        return max(data) if data else GLOBAL_DEFAULT_END

    @property
    def resolution(self) -> timedelta:
        """The default resolution for the global environment"""
        data = [dataset.resolution for dataset in self.all_data()]
        return min(data) if data else GLOBAL_DEFAULT_RESOLUTION

    @property
    def base(self) -> Currency:
        """The base currency for the global environment"""
        return self._base

    @base.setter
    def base(self, code: str) -> None:
        if Currency.validate_code(code, error=True):
            Currency.base = code
            self._base = Currency(code)
            for currency in types.currency.instances.values():
                currency.base = code
                currency._valuate()

    @property
    def rfr(self) -> float:
        """An alias for the global risk free rate"""
        return self._rfr

    @rfr.setter
    def rfr(self, rate: float) -> None:
        """TODO: RFR SHOULD BE SHARED DURING INIT, NO NEED TO UPDATE"""
        setattr(Option, "rfr", rate)
        self._rfr = rate

    @property
    def benchmark(self) -> Asset:
        """The benchmark asset for the global environment, used in calculations
        of alpha, beta, etc."""
        return self._benchmark

    @benchmark.setter
    def benchmark(self, benchmark: Union[Asset, str]) -> None:
        if isinstance(benchmark, Asset):
            assert benchmark.data is not None
            self._benchmark = benchmark
        elif isinstance(benchmark, str):
            try:
                new_benchmark = Stock(benchmark)
                assert new_benchmark.data is not None
                self._benchmark = new_benchmark
            except Exception as e:
                raise RuntimeError(
                    f"Unable to use {benchmark} as a benchmark because the "
                    f"following error occurred during initialization: {e}"
                ) from e
        else:
            raise TypeError(
                "Benchmark must be a Stock. Received " f"{benchmark.__class__.__name__}"
            )

    @property
    def path(self) -> Path:
        """The path to the root directory of this installation of AlphaGradient"""
        return self._path

    def sync(self, date: DatetimeLike) -> None:
        """
        Synchronizes all alphagradient objects globally to the given datetime

        .. warning::
            PLEASE NOTE THAT SYNCING TO THE START MAY CAUSE ERRORS WHEN RUNNING
            ALGORITHMS, AS NO DATA IS AVAILABLE FOR CALCULATIONS THAT REQUIRE
            CHANGE ACROSS SOME TIME INTERVAL. FOR EXAMPLE. CALCULATIONS OF AN
            ASSET'S VOLATILITY BECOME IMPOSSIBLE BECAUSE THEY CAN ONLY ACCESS
            A SINGLE DATA POINT. THIS IS **EXPECTED** BEHAVIOR

        Optimally, use autosync() to avoid the issues above. At minimum, sync
        to a date that is one day ahead of the global start date
        (start = ag.globals.start + timedelta(days=1))

        Args:
            date (datetime): The date to synchronize to

        Returns:
            modifies all currently instantiated assets and portfolios
            in-place
        """
        self._date = utils.to_datetime(date)

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

    def autosync(
        self, end: Optional[DatetimeLike] = None, t: Optional[TimeLike] = None
    ) -> None:
        """
        automatically determines best global variables for start, end, and
        resolution based on currently instantiated assets, and syncs them all
        to an optimal starting period
        """
        sync_date = self.start
        data = list(self.all_data())
        if data:
            max_start = max([dataset.first for dataset in data])
            min_end = min([dataset.last for dataset in data])
            sync_date = utils.optimal_start(
                start=self.start, max_start=max_start, min_end=min_end, end=end, t=t
            )
        self.sync(sync_date)

    def step(self, delta: Union[DateOrTime, timedelta, PyNumber] = None) -> None:
        """
        Takes a single time step for all assets

        Valuates all currently instantiated assets one time step
        (of magnitude delta) ahead of their previous valuation. When
        not provided with a delta explicitly, uses the global
        resolution to determine time step magnitude.

        Parameters:
            delta (Union[DateOrTime, timedelta, float]):
                The magnitude of the time step taken
        """
        self._date += (
            self.resolution if delta is None else utils.to_step(self.date, delta)
        )

        for asset in self.all_assets():
            asset._valuate()
            asset._step(self.date)

        for portfolio in types.portfolio.instances.values():
            portfolio.update_positions()
            portfolio.update_history()

        for algo in types.algorithm.instances.values():
            algo.stats._update()

    def refresh(self) -> None:
        """TODO: Should reinstantiate all persistent ag objects"""
        return None

    def scan(self) -> None:
        """TODO: Should scan a path for ag files and add them to persistent"""
        return None

    def persist(self, path: Optional[Union[Path, str]] = None) -> None:
        """
        Allows persistent storage of ag assets

        Directs alphagradient to persist ag asset datasets in memory at the
        given path, which defaults to the current working directory when none
        is specified. AlphaGradient will create a directory in the path called
        "alphagradient.persistent" in which it will store pickles to easily
        reinstantiate assets

        Parameters:
            path: Where to create/use the pickle storage directory.

        Returns:
            modifies the global instance in place by saving the storage path
        """
        # Converting to valid path object
        def to_path(path: Union[Path, str]) -> Path:
            if isinstance(path, str):
                path = Path(path)
            return path.joinpath("alphagradient.persistent/")

        # Getting the default if none is passed
        path = self._default_persistent() if path is None else to_path(path)

        # Checking that the dir exists, making it if not
        if not os.path.isdir(path):
            path.mkdir()

        # Setting the global peristent path
        self._persistent = path

    @property
    def persistent(self) -> Union[Path, None]:
        """The global environment's persistent path"""
        return self._persistent

    def _find_persistent(self) -> Union[Path, None]:
        """Looks for (and sets, if found) a persistent directory"""
        path = self._default_persistent()
        if os.path.isdir(path):
            return path
        return None

    def _default_persistent(self) -> Path:
        """Returns the default persistent path (where AG expects to find it)"""
        return Path(os.getcwd()).joinpath("alphagradient.persistent/")


__globals = Globals()
"""The global variables class instance"""
