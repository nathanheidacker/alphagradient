# -*- coding: utf-8 -*-
"""AG module containing classes for holding and organizing assets

Todo:
    * Fix basket.validate_resolution to actually validate resolutions
        that are not timedelta objects
"""
# Standard imports
from numbers import Number
from datetime import datetime, time, timedelta
from weakref import WeakValueDictionary as WeakDict
from enum import Enum, auto
from urllib import request
from pathos.multiprocessing import ProcessingPool as Pool
from p_tqdm import p_map
from collections import OrderedDict
from copy import copy
import io
import os
import math

# Third Party imports
import yfinance as yf
import yfinance.shared as shared
import pandas as pd
import numpy as np

# Local Imports
from .._globals import __globals as glbs
from .. import utils
from .asset import Asset, types
from .portfolio import Portfolio, Cash
from .standard import Currency, Stock, Call, Put
from ..data import datatools

class Basket:
    """A 'basket' of assets and other AlphaGradient objects that all
    share the same date and time resolution

    This object essentially represents an isolated financial
    environment that ensures all instantiated assets/objects are
    compatible with eachother and are valuated at the same moment in
    time.

    Baskets should be used in algorithms for creating isolated
    financial environments that keep AG objects conveniently
    accessible in one place.

    Other attributes not included in the list below include all
    instantiable asset types (eg. basket.stock, basket.call, etc.),
    which will return a dictionary of assets of that type which exist
    in this basket, where keys are the asset.key. These attributes
    double as functions for instantiation of that asset type within
    the basket. Instantiating an asset by calling these attributes
    will take/accept the same arguments for instantiation as when
    instantiated normally through alphagradient.finance.

    Similarly to how the the global types enum works, baskets also
    permit attribute access of specific assets when accessed as the
    attribute of a type (eg. basket.stock.SPY will access SPY, if this
    basket is tracking it).

    Baskets can also directly access the attributes of their portfolios
    with attribute access. When a basket keeps track of multiple
    portfolios, accessing the portfolio attribute will instead return
    a dictionary corresponding to that attribute, where keys are the
    portfolio names and values are the respective attribute.

    Attributes:
        date (datetime): This basket's current date (the last
            valuation date of all currently tracked AlphaGradient
            objects). Any assets that are newly instantiated inside of
            this basket will automatically be valuated on this date
        start (datetime): This basket's start date, which all tracked
            assets will valuate to if no steps have been taken.
            basket.date returns to this date when reset is called.
        end (datetime): This basket's end date, which algorithms will
            use as a default to determine when backtesting should end
            if none is provided.
        resolution (timedelta): The default period of time inbetween
            timesteps.
        base (str): The base currency used by this basket represented
            as a currency code. All newly tracked/insantiated assets
            and objects will use this currency as a default base if
            none are provided during intantiation
        assets (list(Asset)): A list of assets currently within or
            tracked by this basket.
        portfolios (dict(Portfolio)): A dictionary of portfolios in
            which keys are the portfolio names, which are provided
            during instantiation. If no names are provided,
            defaults to "MAIN", then "P0", "P1", "P2", ..., "PN"
        status (Status): A member of the Status enum which corresponds
            to how many portfolios this basket is currently tracking.
            Controls behavior of basket bound methods that facilitate
            portfolio transactions such as buy, sell, short, and cover.
    """

    class Status(Enum):
        """Denotes a basket's portfolio status, controlling basket
        bound methods for portfolio transactions"""
        NONE = auto()
        SINGLE = auto()
        MULTIPLE = auto()

        @classmethod
        def get(cls, n):
            """Returns the appropriate status member based on the
            quantity of portfolios passsed"""
            if n == 0:
                return cls.NONE
            elif n == 1:
                return cls.SINGLE
            else:
                return cls.MULTIPLE

    class AssetDict(WeakDict):
        """A weakref dictionary of assets belonging to one asset
        subclass. Allows baskets attribute access to specific asset
        classes, as well as asset instantiation"""
        def __init__(self, cls, basket):
            self._name = cls.__name__
            self.c = cls
            self._basket = basket
            super().__init__()

        def __call__(self, *args, **kwargs):
            force = self._basket.force
            if kwargs.get("force"):
                force = kwargs["force"]
                kwargs.pop("force")
            new = self.c(*args, force=force, **kwargs)
            new._valuate(self._basket.date)
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
                raise AttributeError(f"AlphaGradient Basket has no \
                                     {attr.capitalize()} instance {attr}")

    def __init__(self,
                 start=None,
                 end=None,
                 resolution=None,
                 base=None,
                 assets=None,
                 portfolios=None,
                 force=False):
        self._start = glbs.start if start is None else self.validate_date(start)
        self._date = self.start
        self._end = glbs.end if end is None else self.validate_date(end)
        self._resolution = glbs.resolution if resolution is None else self.validate_resolution(resolution)
        self._assets = [] if not isinstance(assets, list) else assets
        self._portfolios = [] if not isinstance(assets, list) else portfolios
        self._base = glbs.base if not isinstance(base, Currency) else base
        self.force = force

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
        last = max(asset.data.last for asset in self.assets)
        return self._end if self._end < last else last

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

    @property
    def open(self):
        return all(asset.open for asset in self.assets) and self.date.weekday() < 5

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
        """Instantiates a portfolio within this basket

        Args:
            initial (Number): The initial quantity of the base currency
            name (str): This portfolio's name, used for indexing
            date (datetime): This portfolio's starting valuation date
            base (str): A currency code representing this portfolio's
                base currency

        Returns:
            new (Portfolio): returns a new portfolio based on the
                inputs, and adds it to this basket's tracked portfolios
        """
        if isinstance(initial, Portfolio):
            self._portfolios.append(initial)
            return initial
        else:
            base = self.base.code if base is None else Currency.validate_code(base)
            date = self.start if date is None else self.validate_date(date)
            name = self if name is None else name
            new = Portfolio(initial, name, date, base)
            self._portfolios.append(new)
            return new

    def data(self, dtype="dict"):
        """Returns all of the datasets in the assets tracked by this basket

        Args:
            dtype (str | type): the dtype of the returned data object

        Returns:
            data (dict | list): Returns a list or dict of asset
                datasets, depending on the dtype input

        Raises:
            ValueError: When the dtype input is unrecognized/invalid
        """
        dtype = dtype.lower() if isinstance(dtype, str) else dtype
        if dtype in [dict, "dict"]:
            return {asset.key:asset.data for asset in self.assets if asset.data}
        elif dtype in [list, "list"]:
            return [asset.data for asset in self.assets if asset.data]
        else:
            raise ValueError(f"Unsupported type {dtype=}. Basket data "
                             "can only be returned as a list or a "
                             "dictionary")

    @staticmethod
    def validate_date(date):
        """Determines whether or not the input date is valid. If it is,
        returns it as a native python datetime object

        Args:
            date (datetime | str): The date to be validated

        Returns:
            date (datetime): the valid datetime

        Raises:
            TypeError: When the date input is invalid
        """
        if isinstance(date, str):
            return datetime.fromisoformat(date)

        elif isinstance(date, datetime):
            return date

        else:
            raise TypeError(f"Could not normalize "
                            f"{date.__class__.__name__} {date} to "
                            "datetime. Dates should be datetimes or "
                            "isoformat date-strings")

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
        else:
            raise TypeError(f"Resolution must be a datetime.timedelta "
                            "object. Received "
                            f"{resolution.__class__.__name__} "
                            f"{resolution}")

    def default_start(self):
        """Determines the default start datetime based on the tracked assets

        Returns the maximum start date of all asset datasets currently
        inside of this basket. If there are none, returns the global start

        Returns:
            start (datetime): the default start for this basket
        """
        data = [dataset.index[0] for dataset in self.data(dtype=list)]
        return max(data).to_pydatetime() if data else glbs.start

    def default_end(self):
        """Determines the default end datetime based on the tracked assets

        Returns the minimum end date of all asset datasets currently
        inside of this basket. If there are none, returns the global end

        Returns:
            end (datetime): the default end for this basket
        """
        data = [dataset.index[-1] for dataset in self.data(dtype=list)]
        return min(data).to_pydatetime() if data else glbs.start

    def default_resolution(self):
        """Determines the default resolution based on the tracked assets

        Returns the minimum time resolution of all asset datasets
        currently inside of this basket. If there are none, returns
        the global resolution

        Returns:
            resolution (timedelta): the default resolution for this
                basket
        """
        return timedelta(days=1)

    def auto(self):
        """Automatically sets the start, end, and resolution of this
        basket to their defaults based on currently tracked assets"""
        self.start = self.default_start()
        self.end = self.default_end()
        self.resolution = self.default_resolution()

    def sync(self, date=None):
        """Syncs all alphagradient objects in this basket to the given datetime

        Valuates all assets to the given date, and sets the date of all
        portfolios to the given date. This only occurs for objects
        within this basket, rather than globally. Date defaults to
        baskets current date if none is provided

        Args:
            date (datetime): The date to sync to

        Returns:
            None (NoneType): Modifies this basket in place
        """
        date = self.date if date is None else self.validate_date(date)

        for portfolio in self._portfolios:
            portfolio.date = date
            portfolio.reset()

        for asset in self.assets:
            if isinstance(asset, (Call, Put)):
                asset.reset()
            asset._valuate(date)

        self.date = date

    def autosync(self):
        """Combines auto and sync

        Automatically determines appropriate start, end, and resolution,
        and automatically sync all objects to the newly determined
        start date

        Returns:
            None (NoneType): Modifies this basket in place
        """
        self.auto()
        self.sync(self.start)

    def step(self, delta=None):
        """Takes a single time step in this basket, moving all
        alphagradient objects forward by the given delta

        The function that should be called in algorithms to iterate
        forward in time after everything has been accomplished and
        evaluated in the current period. Automatically moves all ag
        objects in this basket forward in time by the given delta,
        which defaults to the basket.resolution if none is provided.

        Args:
            delta (timedelta): The step size

        Returns:
            None (NoneType): Modifies this basket in place
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

        for asset in self._assets:
            asset._valuate(asset.date + delta)
            asset._step(asset.date + delta)

        for portfolio in self._portfolios:
            portfolio.date = portfolio.date + delta
            portfolio.update_history()

        self.date = self.date + delta

        # Cleaning out expired assets
        self._assets = [asset for asset in self.assets if not asset.expired]

    def next(self):
        nextt = min([asset.next for asset in self.assets])
        self.step(nextt)

    def buy(self, asset, quantity, name=None):
        """Buys an asset using this basket's main portfolio, unless
        specified otherwise.

        Creates a long position in the given asset with a purchase
        volume given by 'quantity' within the respective portfolio

        Args:
            asset (Asset): The asset in which to create a long position
            quantity (Number): The purchase quantity
            name (str | None): The name of the Portfolio where the
                transaction will take place

        Returns:
            None (NoneType): Modifies this basket in place

        Raises:
            ValueError: If basket has no active portfolios, or if name
            is not specified when there are multiple portfolios none
            of which are named "MAIN"
        """
        # Transactions require a portfolio
        if self.NONE:
            raise ValueError("This basket has no active portfolios. "
                             "Please instantiate one in order to make "
                             "transactions")

        # Transaction can only refer to one portfolio
        elif self.SINGLE:
            portfolio = self._portfolios[0]
            portfolio.buy(asset, quantity)

        # Portfolio of transaction must be specified
        elif self.MULTIPLE:

            # Default to main if none is provided
            if name is None:
                try:
                    self.portfolios["MAIN"].buy(asset, quantity)
                except KeyError:
                    raise ValueError("This basket has multiple "
                                     "portfolios. The portfolio name "
                                     "for this transaction must be "
                                     "specified")

            # Try to access portfolio by name
            else:
                try:
                    portfolio = self.portfolios[name]
                    portfolio.buy(asset, quantity)
                except KeyError:
                    raise ValueError(f"Basket has no portfolio "
                                     "instance named "
                                     f"{name.__repr__()}")

    def sell(self, asset, quantity, name=None):
        """Sells an asset using this basket's main portfolio, unless
        specified otherwise.

        Decrements a long position in the given asset by 'quantity'.
        Maximum sale quantity is the amount owned by the portfolio.

        Args:
            asset (Asset): The asset of the corresponding decremented
                position
            quantity (Number): The sale quantity
            name (str | None): The name of the Portfolio where the
                transaction will take place

        Returns:
            None (NoneType): Modifies this basket in place

        Raises:
            ValueError: If basket has no active portfolios, or if name
                is not specified when there are multiple portfolios none
                of which are named "MAIN"
        """
        # Transactions require a portfolio
        if self.NONE:
            raise ValueError("This basket has no active portfolios. "
                             "Please instantiate one in order to make "
                             "transactions")

        # Transaction can only refer to one portfolio
        elif self.SINGLE:
            portfolio = self._portfolios[0]
            portfolio.sell(asset, quantity)

        # Portfolio of transaction must be specified
        elif self.MULTIPLE:

            # Default to main if none is provided
            if name is None:
                try:
                    self.portfolios["MAIN"].sell(asset, quantity)
                except KeyError:
                    raise ValueError("This basket has multiple "
                                     "portfolios. The portfolio name "
                                     "for this transaction must be "
                                     "specified")

            # Try to access portfolio by name
            else:
                try:
                    portfolio = self.portfolios[name]
                    portfolio.sell(asset, quantity)
                except KeyError:
                    raise ValueError(f"Basket has no portfolio "
                                     "instance named "
                                     f"{name.__repr__()}")

    def short(self, asset, quantity, name=None):
        """Shorts an asset using this basket's main portfolio, unless
        specified otherwise.

        Creates a short position in the given asset with a short sale
        volume given by 'quantity' within the respective portfolio

        Args:
            asset (Asset): The asset in which to create a short position
            quantity (Number): The short sale quantity
            name (str | None): The name of the Portfolio where the
                transaction will take place

        Returns:
            None (NoneType): Modifies this basket in place

        Raises:
            ValueError: If basket has no active portfolios, or if name
                is not specified when there are multiple portfolios
                none of which are named "MAIN"
        """
        # Transactions require a portfolio
        if self.NONE:
            raise ValueError("This basket has no active portfolios. "
                             "Please instantiate one in order to make "
                             "transactions")

        # Transaction can only refer to one portfolio
        elif self.SINGLE:
            portfolio = self._portfolios[0]
            portfolio.short(asset, quantity)

        # Portfolio of transaction must be specified
        elif self.MULTIPLE:

            # Default to main if none is provided
            if name is None:
                try:
                    self.portfolios["MAIN"].short(asset, quantity)
                except KeyError:
                    raise ValueError("This basket has multiple "
                                     "portfolios. The portfolio name "
                                     "for this transaction must be "
                                     "specified")

            # Try to access portfolio by name
            else:
                try:
                    portfolio = self.portfolios[name]
                    portfolio.short(asset, quantity)
                except KeyError:
                    raise ValueError(f"Basket has no portfolio "
                                     "instance named "
                                     f"{name.__repr__()}")

    def cover(self, asset, quantity, name=None):
        """Covers the short sale of an asset using this basket's main
        portfolio, unless specified otherwise.

        Decrements a long position in the given asset by 'quantity'.
        Maximum sale quantity is the amount owned by the portfolio.

        Args:
            asset (Asset): The asset of the corresponding decremented
                position
            quantity (Number): The sale quantity
            name (str | None): The name of the Portfolio where the
                transaction will take place

        Returns:
            None (NoneType): Modifies this basket in place

        Raises:
            ValueError: If basket has no active portfolios, or if name
                is not specified when there are multiple portfolios
                none of which are named "MAIN"
        """
        # Transactions require a portfolio
        if self.NONE:
            raise ValueError("This basket has no active portfolios. "
                             "Please instantiate one in order to make "
                             "transactions")

        # Transaction can only refer to one portfolio
        elif self.SINGLE:
            portfolio = self._portfolios[0]
            portfolio.cover(asset, quantity)

        # Portfolio of transaction must be specified
        elif self.MULTIPLE:

            # Default to main if none is provided
            if name is None:
                try:
                    self.portfolios["MAIN"].cover(asset, quantity)
                except KeyError:
                    raise ValueError("This basket has multiple "
                                     "portfolios. The portfolio name "
                                     "for this transaction must be "
                                     "specified")

            # Try to access portfolio by name
            else:
                try:
                    portfolio = self.portfolios[name]
                    portfolio.cover(asset, quantity)
                except KeyError:
                    raise ValueError(f"Basket has no portfolio "
                                     "instance named "
                                     f"{name.__repr__()}")

    def _redirect(self, attr):
        """Redirects attribute access to a the proper portfolio when
        user attempts to access portfolio attributes through the
        basket

        Args:
            attr (str): The attribute being accessed

        Returns:
            attr: The attribute on the respective portfolio

        Raises:
            AttributeError: When the attribute does not exist
        """
        if self.NONE:
            raise AttributeError(f"Basket has no active portfolios. "
                                 "Must instantiate at least one "
                                 "portfolio to access portfolio "
                                 f"attribute {attr}")
        elif self.SINGLE:
            return getattr(self._portfolios[0], attr)
        else:
            return {name: getattr(p, attr) for name, p in self.portfolios.items()}

    def reset(self):
        """Resets this basket to its own start date, and syncs all
        assets back to that start date"""
        self.date = self.start
        self.sync()


# Using a protected keyword, attr must be set outside of the class
setattr(Basket, "type", types.basket)
setattr(types.basket, "c", Basket)

def _get_exchange_info():
    base = "ftp://ftp.nasdaqtrader.com/symboldirectory/"
    dfs = []
    for file in ["nasdaqlisted.txt", "otherlisted.txt"]:
        data = io.StringIO(request.urlopen(f"{base}{file}").read().decode())
        data = pd.read_csv(data, sep="|")
        dfs.append(data[:-1])

    dfs = [df[df["Test Issue"] == "N"].drop("Test Issue", axis=1) for df in dfs]

    nasdaq, other = dfs

    nasdaq["Exchange"] = "NASDAQ"
    nasdaq.drop(["Market Category", "Financial Status", "NextShares"], axis=1, inplace=True)

    converter = {"A": "NYSEMKT",
                 "N": "NYSE",
                 "P": "NYSEARCA",
                 "Z": "BATS",
                 "V": "IEXG"}
    other["Exchange"] = other["Exchange"].map(lambda s: converter[s])
    other.drop(["CQS Symbol", "NASDAQ Symbol"], axis=1, inplace=True)
    other.rename({"ACT Symbol": "Symbol"}, axis=1, inplace=True)

    data = pd.concat(dfs).sort_values("Symbol")

    return data.reset_index(drop=True)


class Universe(dict):
    """A collection of stocks that can be efficiently filtered"""
    exchange_info = None
    try:
        raise Exception
        exchange_info = _get_exchange_info()
        with open("alphagradient/finance/exchange_info.p", "wb") as f:
            exchange_info.to_pickle(f)
    except Exception as e:
        exchange_info = pd.read_pickle("alphagradient/finance/exchange_info.p")

    def __init__(self, tickers="local", refresh=False, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.refresh = refresh
        self._tickers = []
        self._errors = []
        self.update()

        tickers, stock_input = self._ticker_input(tickers)
        self.print(f"Initializing Universe: {len(tickers)} Stocks")
        if not stock_input:
            self.add(tickers)

        self.filter = _filter(self)

    def __getattr__(self, attr):
        try:
            glbs.benchmark.__getattribute__(attr)
            return _filterexp(attr)
        except AttributeError:
            raise AttributeError(f"Universe object has no attribute {attr}")

    def __copy__(self):
        return Universe(list(self.values()), refresh=self.refresh, verbose=False)

    def __deepcopy__(self):
        return self.__copy__()

    @property
    def tickers(self):
        return self._tickers

    @property
    def errors(self):
        return self._errors

    @property
    def coverage(self):
        return 1

    @property
    def print(self):
        return print if self.verbose else lambda *args, **kwargs: None

    @property
    def supported(self):
        return self.exchange_info["Exchange"].unique.to_list()

    def add(self, tickers, refresh=None):
        refresh = refresh or self.refresh
        tickers, stock_input = self._ticker_input(tickers)
        tickers = [ticker.upper() for ticker in tickers if ticker not in self._tickers]
        self._tickers = sorted(self._tickers + tickers)
        if not stock_input:
            errors = self._get_stocks(tickers, refresh=refresh)
            for error in errors:
                self._tickers.remove(error)
                self._errors.append(error)
        self._remove_errors()


    def update(self):
        if self.refresh:
            self.local_p = []
            self.local_csv = []
            self.local = []
        else:
            self.local_p = sorted([f[6:-2] for f in os.listdir("alphagradient/data/pickles/") if f.startswith("STOCK_") and f.endswith(".p")])
            self.local_csv = sorted([f[6:-4] for f in os.listdir("alphagradient/data/raw/") if f.startswith("STOCK_") and f.endswith(".csv")])
            self.local = sorted(list(set(self.local_p + self.local_csv)))


    def _ticker_input(self, tickers):
        stock_input = False
        if isinstance(tickers, str):
            if tickers.lower() == "all":
                tickers = self.exchange_info["Symbol"].to_list()
            elif tickers.lower() == "local":
                tickers = self.local
            else:
                tickers = self._get_listings(tickers)

        elif isinstance(tickers, list):
            if all(isinstance(ticker, Stock) for ticker in tickers):
                stocks = tickers
                tickers = [stock.name for stock in stocks]
                for stock, ticker in zip(stocks, tickers):
                    self[ticker] = stock
                stock_input = True

            elif not all(isinstance(ticker, str) for ticker in tickers):
                invalid = str([type(ticker) for ticker in tickers if type(ticker) not in [Stock, str]])[1:-1]
                raise TypeError(f"Ticker list inputs must contain only strings or only Stocks. List contained: {invalid}")

        elif isinstance(tickers, Stock):
            self[tickers.name] = tickers
            tickers = [tickers.name]
            stock_input = True

        elif isinstance(tickers, Number):
            n = int(tickers)
            tickers = self.local[:(n if n < len(self.local) else len(self.local))]
            if len(tickers) <= n:
                n = n if n < len(self.exchange_info) else len(self.exchange_info)
                info = self.exchange_info["Symbol"].to_list()[:n]
                i = 0
                while len(tickers) < n:
                    if info[i] not in tickers:
                        tickers.append(info[i])
                    i += 1

            tickers.sort()

        else:
            raise TypeError(f"Invalid input type {type(tickers).__name__} for tickers. Tickers must be a list of strings, a list of AlphaGradient Stocks, or the name of an exchange (currently supports NYSE and NASDAQ)")

        return tickers, stock_input


    def _get_stocks(self, tickers, refresh=False):
        all_tickers = tickers
        ncols = 100
        self.print(f"Adding {len(all_tickers)} Stocks to Universe")
        local = [ticker for ticker in all_tickers if ticker in self.local]
        online = [ticker for ticker in all_tickers if ticker not in local]

        def get_local(tickers):
            self.print(f"Initializing {len(tickers)} Stocks from local data")
            stocks = p_map(lambda ticker: Stock(ticker), tickers, ncols=ncols, disable=(not self.verbose))
            for stock in stocks:
                self[stock.name] = stock
            self.print()

        def get_online(tickers, timeout=5):
            self.print(f"Initializing {len(tickers)} Stocks from online data")
            for i, batch in enumerate(utils.auto_batch(tickers)):
                self.print(f"Batch {i + 1}: downloading {len(batch)} stocks")
                data = yf.download(
                                   " ".join(batch),
                                   group_by="Ticker",
                                   auto_adjust=False,
                                   progress=self.verbose,
                                   show_errors=False,
                                   timeout=timeout
                                   )

                to_remove = list(shared._ERRORS.keys())
                for ticker in to_remove:
                    batch.remove(ticker)

                self.print(f"Initializing {len(batch)} Stocks from downloaded batch ({len(to_remove)} failures)")

                if batch:
                    stocks = p_map(lambda ticker: Stock(ticker, data=data[ticker].dropna(how="all")),
                                   batch, ncols=ncols,
                                   disable=(not self.verbose))
                else:
                    stocks = []

                for stock in stocks:
                    self[stock.name] = stock
                self.print()

        def get_online_mp(tickers, timeout=5):
            size = utils.auto_batch_size(tickers)
            batches = math.ceil(len(tickers) / size)
            self.print(f"Initializing {len(tickers)} Stocks from online data ({batches} batches)")

            def get_batch(batch):
                data = yf.download(
                                   " ".join(batch),
                                   group_by="Ticker",
                                   auto_adjust=False,
                                   progress=False,
                                   show_errors=False,
                                   timeout=timeout
                                   )
                to_remove = list(shared._ERRORS.keys())
                errors = {
                      "timeout": [ticker for ticker, error in shared._ERRORS.items() if error == "No data found for this date range, symbol may be delisted"],
                      "delisted": [ticker for ticker, error in shared._ERRORS.items() if error == "No data found, symbol may be delisted"]
                }
                for ticker in to_remove:
                    batch.remove(ticker)

                return ([Stock(ticker, data=data[ticker].dropna(how="all")) for ticker in batch], errors)

            batches = list(auto_batch(tickers))
            stocks = p_map(get_batch, batches, ncols=ncols, disable=(not self.verbose))
            errors = {
                      "delisted": [error for value in stocks for error in value[1]["delisted"]],
                      "timeout": [error for value in stocks for error in value[1]["timeout"]]
            }
            stocks = [stock for value in stocks for stock in value[0]]
            for stock in stocks:
                self[stock.ticker] = stock

            return errors

        if refresh:
            errors = get_online_mp(all_tickers)

        else:
            if local:
                get_local(local)
            if online:
                errors = get_online_mp(online)
            else:
                errors = {"delisted": [], "timeout": []}

        num_errors = len(errors["delisted"]) + len(errors["timeout"])
        delisted = errors["delisted"][:]
        timeouts = errors["timeout"][:]


        self.print(f"Successfully added {(len(all_tickers) - num_errors)} of {len(all_tickers)} stocks ({len(delisted)} failures, {len(timeouts)} timeouts, {num_errors} total errors))")

        previous = []
        while timeouts:
            self.print(f"\nRetrying {len(timeouts)} timeouts")
            errors = get_online_mp(timeouts, timeout=10)

            n_timeout = len(errors["timeout"])
            n_delisted = len(errors["delisted"])
            n_attempts = len(timeouts)
            n_errors = n_timeout + n_delisted
            n_success = n_attempts - n_errors
            self.print(f"{n_success} / {n_attempts} successful timeout reattempts ({n_delisted} failures, {n_timeout} timeouts, {n_errors} total errors)")

            delisted += errors["delisted"]
            previous = timeouts[:]
            timeouts = []
            for ticker in errors["timeout"]:
                if ticker in previous:
                    delisted.append(ticker)
                else:
                    timeouts.append(ticker)


        return delisted


    def _get_listings(self, exchange):
        info = self.exchange_info
        return info[info["Exchange"] == exchange]["Symbol"].to_list()

    def _remove_errors(self):
        for symbol in self.errors:
            ei = self.exchange_info[self.exchange_info["Symbol"] == symbol].index
            self.exchange_info.drop(ei, inplace=True)
        with open("alphagradient/finance/exchange_info.p", "wb") as f:
            self.exchange_info.to_pickle(f)


class _filter:
    def __init__(self, universe):
        self.universe = universe

    def __getitem__(self, item):
        if utils.isiter(item):
            item = list(item)
        else:
            item = [item]

        if self._validate_filters(item):
            universe = copy(self.universe)
            for filterr in item:
                print(filterr)
                if filterr.called:
                    universe = self.filter_mp(universe, filterr)
                else:
                    universe = self.filter(universe, filterr)


        return universe

    @staticmethod
    def filter_mp(universe, filterr):

        def dictable(stock):
            return (stock.name, filterr.exec(stock))

        filtered = []

        with Pool() as pool:
            filtered = dict(pool.map(dictable, universe.values()))
        filtered = [v for k, v in universe.items() if filtered[k]]
        return UniverseView(universe, filtered, filterr)

    @staticmethod
    def filter_mp_v2(universe, filterr):

        def process_batch(batch):
            return [(stock, filterr.exec(stock)) for stock in batch]

        batches = list(utils.auto_batch(list(universe.values())))
        filtered = []
        with Pool() as pool:
            filtered = pool.map(process_batch, batches)

        filtered = [stock for batch in filtered for stock, success in batch if success]

        return UniverseView(universe, filtered, filterr)


    @staticmethod
    def filter(universe, filterr):
        filtered = [v for v in universe.values() if filterr.exec(v)]
        return UniverseView(universe, filtered, filterr)


    @staticmethod
    def _validate_filters(filters):

        def validate(filterr):
            if isinstance(filterr, list):
                if all(isinstance(obj, str) for obj in filterr):
                    filterr = _filterexp(filterr, special="strlist")

                elif all(isinstance(obj, Stock) for obj in filterr):
                    filterr = _filterexp(filterr, special="stocklist")

            elif isinstance(filterr, dict):
                if all(((isinstance(k, str), isinstance(v, Stock)) == (True, True)) for k, v in filterr.items()):
                    filterr = _filterexp(filerr, special="dict")

            if not isinstance(filterr, _filterexp):
                raise TypeError(f"Invalid filter {filterr}")

            return filterr

        return [validate(filterr) for filterr in filters]


class _filterexp:

    def __init__(self, attr, special=None):
        self.attr = attr
        self.operation = None
        self.condition = None
        self.exp = None
        self.args = None
        self.kwargs = None
        self.called = False
        self.is_other = False

        if special is not None:
            self.attr = "direct"
            self.operation = "filter"
            if special == "strlist":
                self.condition = attr
            elif special == "stocklist":
                self.condition = [stock.name for stock in attr]
            elif special == "dict":
                self.condition = list(attr.keys())

    def __str__(self):
        convert = {"__lt__": "<", "__le__": "<=", "__eq__": "==", "__ne__": "!=", "__gt__": ">", "__ge__": ">=", "filter": "filter"}
        attr = self.attr_string()

        if self.is_other:
            return attr

        elif not (self.operation and self.condition):
            return f"if {attr}"

        return f"{attr} {convert[self.operation]} {self.condition}"

    def __hash__(self):
        kwargs = None
        if self.kwargs:
            kwargs = ((k, v) for k, v in self.kwargs)
        return (self.attr, self.args, kwargs, self.operation, self.condition).__hash__()

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.called = True
        return self

    def exec(self, *args, **kwargs):
        if self.exp is None:
            self.exp = self.build_exp()
        result = self.exp(*args, **kwargs)
        try:
            return bool(result)
        except TypeError as e:
            raise TypeError(f"Filter expression '{self}' does not return a boolean or boolean-convertible value") from e

    def __bool__(self):
        if self.is_other:
            return True
        return NotImplemented

    def __lt__(self, other):
        if self.is_other:
            return NotImplemented
        if isinstance(other, _filterexp):
            other.is_other = True
        if not self.operation: self.operation = "__lt__"
        if not self.condition: self.condition = other
        return self

    def __le__(self, other):
        if self.is_other:
            return NotImplemented
        if isinstance(other, _filterexp):
            other.is_other = True
        if not self.operation: self.operation = "__le__"
        if not self.condition: self.condition = other
        return self

    def __eq__(self, other):
        if self.is_other:
            return NotImplemented
        if isinstance(other, _filterexp):
            other.is_other = True
        if not self.operation: self.operation = "__eq__"
        if not self.condition: self.condition = other
        return self

    def __ne__(self, other):
        if self.is_other:
            return NotImplemented
        if isinstance(other, _filterexp):
            other.is_other = True
        if not self.operation: self.operation = "__ne__"
        if not self.condition: self.condition = other
        return self

    def __gt__(self, other):
        if self.is_other:
            return NotImplemented
        if isinstance(other, _filterexp):
            other.is_other = True
        if not self.operation: self.operation = "__gt__"
        if not self.condition: self.condition = other
        return self

    def __ge__(self, other):
        if self.is_other:
            return NotImplemented
        if isinstance(other, _filterexp):
            other.is_other = True
        if not self.operation: self.operation = "__ge__"
        if not self.condition: self.condition = other
        return self

    def build_exp(self):
        func = lambda stock: True
        args = tuple()
        other_args = tuple()
        kwargs = {}
        other_kwargs = {}
        if self.called:
            args = self.args or args
            kwargs = self.kwargs or kwargs

        if isinstance(self.condition, _filterexp) and self.condition.called:
            other_args = self.condition.args or other_args
            other_kwargs = self.condition.kwargs or other_kwargs

        if not (self.condition and self.operation):
            if self.called:
                func = lambda stock: getattr(stock, self.attr)(*args, **kwargs)
            else:
                return lambda stock: stock.__getattribute__(self.attr)

        elif self.operation == "filter":
            func = lambda stock: stock.name in self.condition

        elif isinstance(self.condition, _filterexp):
            if self.called and self.condition.called:
                func = lambda stock: getattr(getattr(stock, self.attr)(*args, **kwargs), self.operation)(getattr(stock, self.condition.attr)(*other_args, **other_kwargs))
            elif self.called:
                func = lambda stock: getattr(getattr(stock, self.attr)(*args, **kwargs), self.operation)(getattr(stock, self.condition.attr))
            elif self.condition.called:
                func = lambda stock: getattr(getattr(stock, self.attr), self.operation)(getattr(stock, self.condition.attr)(*other_args, **other_kwargs))
            else:
                func = lambda stock: getattr(getattr(stock, self.attr), self.operation)(getattr(stock, self.condition.attr))

        else:
            if self.called:
                func = lambda stock: getattr(getattr(stock, self.attr)(*args, **kwargs), self.operation)(self.condition)

            else:
                func = lambda stock: getattr(getattr(stock, self.attr), self.operation)(self.condition)

        return func

    @staticmethod
    def from_string(string):

        def is_numeric_char(char):
            return char.isnumeric or char in [".", "-"]

        def is_numeric(string):
            return all(is_numeric_char(char) for char in string)

        items = string.split(" ")
        attr = items[0]
        operation = items[1]
        condition = " ".join(items[2:])
        if condition.startswith("["):
            condition = condition[1:-1].split(" ")
            return _filterexp(condition, special="strlist")
        elif is_numeric(condition):
            try:
                condition = float(condition)
                exp = _filterexp(attr)
                exp.operation = operation
                exp.condition = condition
                return exp
            except ValueError:
                pass

        exp = _filterexp(attr)
        exp.operation = operation
        exp.condition = condition

        return exp

    def attr_string(self):
        result = f"stock.{self.attr}"
        args = ""
        kwargs = ""
        if self.args is not None:
            end = -1
            if len(self.args) < 2: end -= 1
            args = str(self.args)[1:end]
        if self.kwargs is not None:
            kwargs = str([f"{k}={v}" for k,v in self.kwargs.items()])[1:-1]
            called = True
        if args or kwargs:
            if args:
                result = f"{result}({args}"
                if kwargs:
                    result = f"{result}, {kwargs}"
                result += ")"
            else:
                result = f"{result}({kwargs})"
        elif self.called:
            result = f"{result}()"

        return result



class UniverseView(Universe):

    def __init__(self, base, stocklist, filterr):
        super().__init__(stocklist, refresh=False, verbose=False)
        if type(base) is Universe:
            self.universe = base
            self.filters = OrderedDict({filterr: len(self)})
        elif type(base) is UniverseView:
            self.universe = base.universe
            self.filters = base.filters
            self.filters[filterr] = len(self)
        else:
            raise TypeError(f"Unacceptable base for a universe view {type(base)=}")


    @property
    def coverage(self):
        return len(self) / len(self.universe)

    @property
    def history(self):
        initial = len(self.universe)
        result = f"Universe: {initial} stocks"
        result += "\n" + ("-" * len(result))
        prev = initial
        for filterr, curr in self.filters.items():
            diff = prev - curr
            step_percentage = round((diff / prev) * 100, 3) if prev > 0 else 0
            total_percentage = round((1 - (curr / initial)) * 100, 3) if initial > 0 else 0
            result += f"\n{filterr.__str__()}: {curr} results"
            result += f" | {step_percentage}% ({diff}) removed"
            result += f" | {total_percentage}% ({initial - curr}) total removed"
            prev = curr

        return result





# Using a protected keyword, attr must be set outside of the class
setattr(Universe, "type", types.universe)
setattr(types.universe, "c", Universe)
