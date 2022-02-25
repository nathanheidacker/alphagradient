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

# Third Party imports
import pandas as pd
import numpy as np

# Local Imports
from .._globals import __globals as glbs
from .. import utils
from .asset import Asset, types
from .portfolio import Portfolio, Cash
from .standard import Currency, Stock, Call, Put

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

        for portfolio in self._portfolios:
            portfolio.date = portfolio.date + delta
            portfolio.update_history()

        for asset in self._assets:
            asset._valuate(asset.date + delta)
            asset._step(asset.date + delta)

        self.date = self.date + delta

    def next(self):
        nextt = min([asset.next for asset in self.assets])
        print(nextt)
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

class Universe:
    """A collection of stocks that can be efficiently filtered"""
    pass

# Using a protected keyword, attr must be set outside of the class
setattr(Universe, "type", types.universe)
setattr(types.universe, "c", Universe)
