# -*- coding: utf-8 -*-
"""AG module containing base Asset class and associated enums

This module contains all of the necessary components for the proper
function of standard asset classes within AlphaGradient, as well as
the API.

Todo:
    * Complete google style docstrings for all module components
    * Complete function/class header typing
    * Replace is_numeric with Number instance checks
    * Interpret more than just isoformat for normalizing datestrings
    * IMPLEMENT SAVING ASSETS LOCALLY TO BE REUSED, REFRESH/ONLINE INPUT
"""

# Standard imports
from abc import ABC, abstractmethod
from datetime import datetime, time, timedelta
from numbers import Number
from weakref import WeakValueDictionary as WeakDict
import math
import pickle
import os
from pathlib import Path
from functools import lru_cache
#import typing

# Third Party imports
from aenum import Enum, unique, auto, extend_enum
import pandas as pd
import numpy as np

# Local imports
from ..data import datatools
from .. import utils

_currency_info_path = Path(__file__).parent.joinpath("currency_info.p")
_currency_info = pd.read_pickle(_currency_info_path)

@unique
class types(Enum):
    """A enumeration with members for all asset subclasses

    The types enum is a special enumeration which dynamically creates
    new members for all subclasses of ag.Asset. Enum members store a
    weakly referential dictionary to all instances of that asset
    subclass, which can be accessed as attributes or dynamically
    through indexing.

    Examples:
        * TYPES.STOCK.instances returns all instances of Stock
        * TYPES.STOCK.SPY returns Spy Stock (if instantiated)
        * TYPES.STOCK["SPY"] does the same

    """

    def _generate_next_value_(name, *args):
        """Determines how new enum members are generated when new asset
        subclasses are created"""

        class Instances(WeakDict):
            """A weakly referential dictionary of all instances of the
            subclass to which the enum member corresponds"""
            def __getattr__(self, item):
                try:
                    return self[item.upper()]
                except KeyError:
                    raise AttributeError(
                        f"Asset type \'{name}\' has no instance \'{item}\'")

            def __str__(self):
                return str(dict(self))

            @property
            def base(self):
                """The monetary basis of this asset subclass,
                represented by a currency code. Only used by Currency
                subclass"""
                if self and getattr(list(self.values())[0], "base", False):
                    return [c for c in self.values() if c.is_base][0]
                else:
                    raise AttributeError(f"Asset type \'{self}\' has no instance \'base\'")

        return (name, Instances())

    # Non-asset types that need to be instantiated manually
    undefined = auto() # Used when the subclass is hidden
    portfolio = auto()

    # Reconsider if these are necessary
    algorithm = auto()
    basket = auto()
    universe = auto()

    def __init__(self, *args, **kwargs):
        self.c = object
        super().__init__(*args, **kwargs)

    def __str__(self):
        return self.name.upper()

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        return self.instances[item]

    @property
    def instances(self):
        """A list of all instances of a certain asset type"""
        return self.value[1]

    @classmethod
    def list(cls):
        """Returns the types enum as a list of enum members with attribute access

        Args:
            types_enum (Enum): The types enum

        Returns:
            types_list (TypeList): The list created from types
        """
        class TypeList(list):
            """A list of types enum members with attribute access for enum members"""
            def __init__(self, types_enum):
                self += [t for t in types_enum][1:]
                self.enum = types_enum

            def __getitem__(self, item):
                if item.__class__ is str:
                    try:
                        return self.enum[item]
                    except KeyError:
                        raise KeyError(f'Asset type \'{item}\' does not exist')
                return super().__getitem__(item)

            def __getattr__(self, item):
                try:
                    return self.enum[item]
                except KeyError:
                    raise AttributeError(f'Asset type \'{item}\' does not exist')

        return TypeList(cls)

    @classmethod
    def instantiable(cls):
        """A list of all instantiable asset subclasses

        Returns:
            instantiable (list): All asset subclasses which contain no
                abstract methods.
        """
        check = Asset.__subclasses__()
        instantiable = []
        for sub in check:
            check += sub.__subclasses__()
            if ABC not in sub.__bases__:
                instantiable.append(sub)

        return instantiable

class AssetDuplicationError(Exception):
    """Raised when asset duplication is forced via copy methods"""
    def __init__(self, asset):
        message = (f"Attempted duplication of {asset.name} "
        f"{asset.type}. Multiple instances of this asset are not "
        "permitted")
        super().__init__(message)

class DataProtocol(Enum):
    """Enumeration of different data requirement protocols for assets

    This enumeration and its members control data requirements for the
    instantiation of different asset subclasses.
        * REQUIRE: Data MUST be supplied in some form, or the asset will
            fail to instantiate. For assets that require that the
            asset.data attribute is a pandas dataframe object
        * FLEXIBLE: Asset will be instantiated whether or not data is
            supplied, asset.data attribute can be None or pd.DataFrame
        * PROHIBITED: Asset will fail to instantiate if any data is
            supplied. For assets that require that the asset.data
            attribute is None (eg. simulated assets)
    """
    REQUIRED = auto()
    FLEXIBLE = auto()
    PROHIBITED = auto()

    @classmethod
    def _get(cls, require_data, prohibit_data):
        """Returns the appropriate data protocol based on asset
        subclass initialization settings

        Args:
            require_data (bool): Whether or not the asset requires data
            prohibit_data (bool): Whether or not the asset prohibits
                data

        Returns:
            member (DataProtocol): The member protocol corresponding to
                the settings

        Raises:
            TypeError: raised when inputs are not booleans
        """
        if require_data is prohibit_data:
            return cls.FLEXIBLE
        elif require_data and not prohibit_data:
            return cls.REQUIRED
        elif not require_data and prohibit_data:
            return cls.PROHIBITED

        raise TypeError(f"inputs must be booleans")

    @classmethod
    def _decompose(cls, member):
        """Returns the respective subclass initialization settings that
        correspond to each protocol member

        Args:
            member (DataProtocol): the member protocol to be decomposed

        Returns:
            require, prohibit (Tuple(bool)): The asset subclass
                initialization settings corresponding to the input member

        Raises:
            KeyError: When member is not a valid member of DataProtocol
        """
        if member is cls.FLEXIBLE:
            return False, False
        elif member is cls.REQUIRED:
            return True, False
        elif member is cls.PROHIBIT:
            return False, True

        raise KeyError(f"{member} is not a valid member of the DataProtocol enumeration")


class Asset(ABC):
    """Abstract base class representing a financial asset

    The ABC underlying all standard and custom asset classes within
    AlphaGradient, designed to be used in conjunction with other
    standard ag objects such as portfolios and algorithms. All Assets
    are singletons.

    When defining a new Asset subclass, the valuate method

    Attributes:
        _args (dict): Arguments used in most recent initialization.
            Referenced to determine whether arguments have changed materially since previous
            instantiation
        name (str): Name of the asset
        value (Number): The value of this asset as a number,
            representing a quantity of the global base currency.
        price (str): A print friendly version of the asset value with
            the asset's base currency symbol attached
        close (bool): Whether most recent valuation represents the
            close (end) of an interval.
        base (str): The base currency of the asset, represented as a
            currency code.
        data (pd.DataFrame | NoneType): The historical data for this
            asset, which will determine its value over time when valuated.
        date (datetime): The date of this asset's most recent
            valuation, which to which its current value corresponds
        key (str): A string for use in dictionaries which index this
            asset
        expired (bool): Whether or not this asset is currently expired.
    """

    def __init_subclass__(
                          cls,
                          *args,
                          hidden=False,
                          require_data=None,
                          prohibit_data=None,
                          required=None,
                          optional=None,
                          open_value=None,
                          close_value=None,
                          market_open=None,
                          market_close=None,
                          units=None,
                          settings=None,
                          **kwargs):
        """Controls behavior for instantiation of Asset subclasses.

        Creates new enumerations within the TYPES enum for newly
        created subclassses of Asset. Also sets some class level
        attributes that control behavior during instantiation.

        The following are all **class** level attributes for Asset and
        all Asset subclasses.

        Attributes:
            type (types): The types enum member corresponding to this
                asset subclass, with instances attached.
            data_protocol (DataProtocol): DataProtocol enum member
                indicating what kind of data inputs are acceptable
            required (list of str): the columns that MUST be present
                in the data input for asset initialization to occur.
            optional (list of str): the columns that will also be
                accepted/kept in the input, but are not required.
            open_value (str): the string corresponding to the name of
                the column that holds price data for the beginning of
                a time interval
            close_value (str): the string corresponding to the name of
                the column that holds price data for the end of a time
                interval
            settings (dict): a dictionary of all of the class
                attributes above
            **kwargs: list of other arbitrary args in class header
        """

        # Asset subclass (class-level) attribute initialization
        cls.required = required
        cls.optional = optional
        cls.open_value = open_value
        cls.close_value = close_value
        cls.market_open = market_open
        cls.market_close = market_close
        cls.unit, cls.units = units if units is not None else (None, None)

        # Ensuring safe access if nothing is passed
        settings = {} if settings is None else settings

        # Hiding the asset from the types enumeration
        hidden = hidden or settings.get("hidden", False)

        # Using the passed in settings object to set class attributes
        # in the case they are not provided by kwargs
        if settings:
            for attr in ["required", "optional", "open_value", "close_value", "market_open", "market_close"]:
                if not getattr(cls, attr, False):
                    try:
                        setattr(cls, attr, settings[attr])
                    except KeyError as e:
                        pass

            if require_data is None:
                require_data = settings.get("require_data", False)

            if prohibit_data is None:
                prohibit_data = settings.get("prohibit_data", False)

            if settings.get("units", False):
                cls.unit, cls.units = settings["units"]

        # Setting the data protocol
        cls.data_protocol = DataProtocol._get(require_data, prohibit_data)

        # Setting asset-level market close and open
        for attr_name in ("market_open", "market_close"):
            attr = getattr(cls, attr_name, None)
            if attr is not None:
                try:
                    setattr(cls, attr_name, utils.get_time(attr))
                except ValueError:
                    raise ValueError(f"Invalid input for {attr_name} during initialization of {cls.__name__} asset subclass. Unable to convert {attr} to a time object.")
            else:
                setattr(cls, attr_name, time(minute=0, second=0, microsecond=0))

        # What will become the name of this subclass' type
        TYPE = cls.__name__.lower()

        # Extending the enum to accomodate new type
        if ABC not in cls.__bases__ and not hidden:
            if TYPE not in [t.name for t in types]:
                extend_enum(types, TYPE)
            cls.type = types[TYPE]
            cls.type.c = cls

        # Used when a new asset subclass is hidden from the AG api
        if not getattr(cls, 'type', None):
            cls.type = types.undefined

        if cls.unit is None or cls.units is None:
            cls.unit = "unit"
            cls.units = "units"

    def __new__(cls, *args, **kwargs):
        # Seeing if this asset is already instantiated
        if args or kwargs.get("name", False):
            name = kwargs["name"] if kwargs.get("name") else args[0]

            # Returning the asset if exists
            if name in cls.type.instances:
                return cls.type.instances[name]

        # Returning a new asset
        return super().__new__(cls)

    def __init__(
            self,
            name,
            date=None,
            data=None,
            columns=None,
            force=False,
            base=None):

        # Standard style guideline for all asset names. Ensures that
        # they are accessible via attribute access.
        self.initialized = False
        name = str(name).upper().replace(' ', '_')

        # Checks if arguments have changed materially from previous
        # initialization, and skip initialization if they haven't.
        if self.type.instances.get(name) is self:

            skip = True
            if data is not None:

                if isinstance(data, pd.DataFrame) and isinstance(self._args["data"], pd.DataFrame):
                    if not data.equals(self._args["data"]):
                        skip = False

                elif type(data) is not type(self._args["data"]):
                    skip = False

                elif data != self._args["data"]:
                    skip = False

            if skip:
                return

        # Saving new args, storing new instance
        self._args = locals()
        del self._args["self"]
        self.type.instances[name] = self

        # Attribute Initialization
        self.name = name
        self.base = base if base in list(_currency_info["CODE"]) else self._global_base
        self._value = data if isinstance(data, Number) else 0
        self.close = True
        self.data = None

        # Data entry is prohibited, data must always be None
        if self.data_protocol is DataProtocol.PROHIBITED:
            self.data = None

        # Data entry is not prohibited, initialize dataset
        else:
            if data is None:

                # First attempt to get data from saved files
                data = datatools.get_data(self)

                # Second attempt to get data from online data
                if data is None and getattr(self, "online_data", False):
                    try:
                        data = self.online_data()
                    except ValueError:
                        pass

                # Third attempt to get data from nothing...?
                if data is None and self.data_protocol is DataProtocol.REQUIRED:

                    # When we need to force the instantiation of an
                    # asset that requires data, but no data is available
                    if force:
                        data = datatools.AssetData(self.__class__, 1)
                    else:
                        raise ValueError(f"{self.name} {self.type} "
                                         "could not be initialized "
                                         "without data. If this is "
                                         "the first time this asset is "
                                         "being instantiated, please "
                                         "provide a valid dataset or "
                                         "instantiate with force=True.")

                self.data = data

            # Explicit reinstancing from online data
            elif isinstance(data, str) and data.lower() == "online":
                try:
                    data = self.online_data()
                except AttributeError as e:
                    raise ValueError(f"{self.name} unable to retrieve online data, {self.type} has not implemented an online_data method") from e
                if data is None:
                    raise ValueError(f"Unable to retreive online data for {self.type} \"{self.name}\", initialization failed")

                self.data = data


            # Data input received, make a new asset dataset
            else:
                self.data = datatools.AssetData(self.__class__, data, columns)

            # Data verification when required
            if self.data_protocol is DataProtocol.REQUIRED and not self.data:
                raise ValueError(f"{self.name} {self.type} could not "
                                 "be initialized without data. If this "
                                 "is the first time this asset is "
                                 "being instantiated, please provide "
                                 "a valid dataset or instantiate with "
                                 "force=True.")

        # Ensures that the initial price is set properly
        self._valuate()
        self.initialized = True
        self._save()

    def __str__(self):
        return f'<{self.type} {self.name}: {self.price} /{self.unit}>'

    def __hash__(self):
        return self.key.__hash__()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self is other

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __copy__(self):
        raise AssetDuplicationError(self)

    def __deepcopy__(self):
        raise AssetDuplicationError(self)

    @property
    def value(self):
        return self._value

    @property
    def next(self):
        if self.data:
            return self.data.next(self.date)
        else:
            return self.date + self._global_res

    @value.setter
    def value(self, value):
        if isinstance(value, Number):
            if value != float("nan"):
                self._value = value
        else:
            raise TypeError(f"Can not update value of {self.name} "
                            f"{self.type} to "
                            f"{value.__class__.__name__} "
                            f"{value}. Value must be a numnber")

    @property
    def price(self):
        symbol = types.currency.instances[self.base].symbol
        price = abs(self._value)

        if price != 0:
            r = 2
            while price < 1:
                r += 1
                price *= 10

            price = round(self._value, r)

        return f"{symbol}{price}"

    @property
    def key(self):
        """Returns a key used for accessing stored files relevant to this asset

        Creates a key from information unique to this asset. These
        keys are used to access locally stored data relevant to this
        asset

        Returns:
            key (str): a key unique to this asset
        """

        return f"{self.type}_{self.name}"

    @property
    def expired(self):
        """Whether or not this asset is expired

        Most assets will never expire, so the default behavior is to
        always return false. Some assets like options can expire,
        however. Overwrite this property to determine when an asset
        expires

        Returns:
            expired (bool): A boolean representing the expiration status
                of this asset
        """
        return False

    @property
    def open(self):
        return self.date.time() >= self.market_open and self.date.time() <= self.market_close if self.market_close != self.market_open else True

    def _valuate(self):
        """Updates asset prices when time steps take place

        This is the method that is actually called under the hood when
        time steps take place, which properly directs valuation
        behavior to either valuate or _data_valuate depending on the
        asset type.

        Args:
            date: date-like object that will become a datetime.
                Determines at what point in time the valuation will be

        Returns:
            price (float): Updates the instance's price inplace, and
                also returns it
        """
        self.value = self.quote(self.date)

    def _data_valuate(self, date=None):
        """Determines how asset prices update when using data

        Determines how assets are valuated when data is available. Keep
        track of/updates when the asset is at the beginning or the end
        of a time interval, and valuates accordingly.

        Args:
            date: date-like object that will become a datetime.
                Determines at what point in time the valuation will be

        Returns:
            price (float): Updates the instance's price inplace, and
                also returns it
        """
        date = self.normalize_date(date)
        value = float(self.data.valuate(date, self))
        if not math.isnan(value):
            return value
        else:
            if not self.initialized:
                data = self.data.asof(self.data.index[0])
                return data[self.close_value]
            else:
                return self.value

    @abstractmethod
    def valuate(self, *args, **kwargs):
        """Determines how asset prices update when not using data

        This is the method that defines non-data-based valuation
        behavior for an asset subclass. The standard implementation
        essentially does nothing--prices stay constant. New asset
        subclasses are required to replace this method in order to be
        instantiable.

        Args:
            *args: see below
            **kwargs: Valuate methods for different assets will likely
                require different arguments to be passed. Accepting
                all of them allows
        """
        return self.value

    def quote(self, date):
        date = self.normalize_date(date)
        if self.data:
            return self.data.valuate(date, self)
        else:
            return self.valuate(date)

    def normalize_date(self, date=None):
        """Standardizes different date input dtypes

        Args:
            date: The date to be transformed

        Returns:
            date: a datetime.datetime object equivalent to the input

        Raises:
            TypeError: When the type can not be coerced to datetime
        """
        if date is None:
            return self.date

        # TODO: Interpret datestring format, accept more templates
        elif isinstance(date, str):
            return datetime.fromisoformat(date)

        elif isinstance(date, datetime):
            return date

        elif isinstance(date, pd.Timestamp):
            return date.to_pydatetime()

        elif isinstance(date, np.datetime64):
            return pd.Timestamp(date).to_pydatetime()

        else:
            raise TypeError(
                f"Date input of type {type(date)} could not be normalized")

    @classmethod
    def get_settings(cls, unpack=False):
        """Returns a dictionary of class attributes

        This settings object is the same one used in the class header for defining class attributes.

        Args:
            unpack (bool): When true, provides the values unpacked into
                a list, for easy unpacking

        Returns:
            settings (dict): class-level settings for this asset type
        """
        require, prohibit = DataProtocol._decompose(cls.data_protocol)

        settings = {
            "hidden": cls.type is types.undefined,
            "require_data": require,
            "prohibit_data": prohibit,
            "required": cls.required,
            "optional": cls.optional,
            "close_value": cls.close_value,
            "open_value": cls.open_value
        }

        return settings.values() if unpack else settings

    def _save(self):
        """Saves this asset's data locally"""
        if self.data and self._global_persistent_path is not None:
            path = self._global_persistent_path.joinpath(f"{self.key}.p")
            with open(path, "wb") as p:
                self.data._data.to_pickle(p)

    def _step(self, *args, **kwargs):
        """Automatically called before this asset is valuated during time steps

        This function is a hook to perform some behavior on the asset
        prior to its valuation at each time step.

        Args:
            date (datetime): The date of the valuation that will occur
                after this function is executed

        Returns:
            None (NoneType): Modifies the asset in place.
        """
        return None


    def expire(self, portfolio, position):
        """Controls the behavior of this asset and positions in this
        asset when it expires inside of a portfolio

        Positions in portfolios are automatically removed from that
        portfolio when they expire. Their expiration is determined
        based on two conditions: Whether or not the position.quantity > 0,
        and whether or not the position's underlying asset is expired.

        Changing this method in an asset subclass will change its
        behavior as it expires. The conditions for expiring can also
        be changed by overloading the asset's 'expired' property
        (must be a property)

        Args:
            portfolio (Portfolio): The portfolio holding a position in
                this assets
            position (Position): The above portfolio's current position
                in this asset that is becoming expired after this call.

        Returns:
            None (NoneType): Modifies the asset/portfolio/position in
                place
        """
        return None

    def range(self, start, end):
        """Returns a datetime-indexed dataframe of asset values from start to end"""

        if self.data:
            return self.data.range(start, end)

        @np.vectorize
        @lru_cache
        def vquote(date):
            return self.valuate(date)

        dates = pd.date_range(start, end)
        prices = vquote(dates)
        data = pd.DataFrame(data=prices,
                            index=dates,
                            columns=[self.close_value])

        if len(data) > 1:
            shifted = data[self.close_value].shift(1)
            shifted[0] = shifted[1]
            data["CHANGE"] = (data[self.close_value] / shifted) - 1

        elif not data.empty:
            data["CHANGE"] = [0]

        else:
            data["CHANGE"] = []

        return data

    def ma(self, days=365):
        """Returns the moving average of this asset's value over the period given by days

        Args:
            days (Number): A number indicating the number of days for
                which the moving average should be calculated

        Returns:
            ma (float): A floating point value representing the average
                price over the period given by days.
        """
        start = self.date - timedelta(days=days)
        data = self.range(start, self.date)
        return data[self.close_value].mean()

    def vol(self, days=365):
        """Returns the historical volatility of this asset's value over the period given by days, as a percentage of the current valuation

        Args:
            days (Number): A number indicating the number of days for
                which the historical volatility should be calculated

        Returns:
            hv (float): A floating point number representing the average
                deviation of this asset's price from its moving average over the same period, expressed as a percentage of the current value.
        """

        ### TODO ### THIS MUST CONSIDER THE INTERVALS OF TIME BETWEEN EACH INDEX, RIGHT NOW ASSUMES THAT ALL VALUES ARE EQUIDISTANT / DAILY. TIME RESOLUTION OF THE DATA WILL AFFECT THE STD

        # 252 Trading days per year
        multiplier = ((days / (252 / 365)) ** 0.5)
        start = self.date - timedelta(days=days)
        data = self.range(start, self.date)
        return data["CHANGE"].std() * multiplier

    def beta(self, days=365, benchmark=None):
        """Returns the beta for the period

        Returns the beta for the period. Calculated by weighting the covariance of this asset and the benchmark asset by the ratio of their volatilities.

        Args:
            days (Number): The period across which to calculate beta
            benchmark (Asset): The benchmark asset to compare to

        Returns:
            beta (Number): This asset's beta for the given period
        """
        benchmark = benchmark or self.benchmark
        start = self.date - timedelta(days=days)
        self_vol = self.vol(days)
        bench_vol = benchmark.vol(days)
        self_data = self.range(start, self.date)["CHANGE"]
        bench_data = benchmark.range(start, self.date)["CHANGE"].asof(self_data.index)
        r = self_data.corr(bench_data)
        if bench_vol == 0:
            return 0
        return r * (self_vol / bench_vol)

    def roi(self, days=365):
        """This assets return on investment for the input period

        Returns the difference between the starting value and the current value as a percentage of the starting value

        Args:
            days (Number): the period across which to calculate

        Returns:
            roi (Number): The percentage difference from start to end
        """
        start = self.date - timedelta(days=days)
        initial = self.quote(start)
        if initial == 0:
            return 0
        return (self.value / initial) - 1

    def cagr(self, days=365):
        """Returns this asset's compounding annualized growth rate for the given period

        Args:
            days (Number): period across which to calculate cagr

        Returns:
            cagr (Number): compounding annualized growth rate for this asset
        """
        return (self.roi(days) + 1) ** (365 / days) - 1

    def alpha(self, days=365, benchmark=None):
        """Returns this asset's alpha for the given period

        Calculates the return of this asset relative to its risk adjusted expected return, using some other asset a basis.

        Args:
            days (Number): The period across which to calculate alpha
            benchmark (Asset): The benchmark to act as a basis fot the
                calculation of risk adjusted expected return

        Returns:
            alpha (Number): This asset's alpha for the period
        """
        benchmark = benchmark or self.benchmark
        asset_roi = self.roi(days)
        bench_roi = self.benchmark.roi(days)
        expected_roi = self.beta(days, benchmark) * (bench_roi - self.rfr)
        return asset_roi - expected_roi
