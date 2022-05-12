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
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, time, timedelta
from functools import lru_cache
import math
from numbers import Number
from pathlib import Path
from weakref import WeakValueDictionary as WeakDict

# Third Party imports
from aenum import Enum, unique, auto, extend_enum
import pandas as pd
import numpy as np

# Local imports
from .._data import get_data, AssetData, ValidData
from .. import utils

# Typing
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    no_type_check,
    NoReturn,
    Optional,
    Type,
    Union,
)

from ..utils import DateOrTime, DatetimeLike

if TYPE_CHECKING:
    from ._standard import Currency
    from ._portfolio import Portfolio, Position


_currency_info_path: Path = Path(__file__).parent.joinpath("currency_info.p")
"""The path to local currency info"""

_currency_info: pd.DataFrame = pd.read_pickle(_currency_info_path)
"""Currency information stored locally"""


class Instances(WeakDict):
    """A weakly referential dictionary of all instances of the
    subclass to which the enum member corresponds"""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__()

    def __getattr__(self, attr: str) -> Any:
        try:
            return self[attr.upper()]
        except KeyError:
            raise AttributeError(f"Asset type '{self.name}' has no instance '{attr}'")

    def __str__(self) -> str:
        return str(dict(self))

    @property
    def base(self) -> Currency:
        """The monetary basis of this asset subclass,
        represented by a currency code. Only used by Currency
        subclass"""
        if self and getattr(list(self.values())[0], "base", False):
            return [c for c in self.values() if c.is_base][0]
        else:
            raise AttributeError(f"Asset type '{self.name}' has no instance 'base'")


@unique
class types(Enum):
    """
    A enumeration with members for all asset subclasses

    The types enum is a special enumeration which dynamically creates
    new members for all subclasses of ag.Asset. Enum members store a
    weakly referential dictionary to all instances of that asset
    subclass, which can be accessed as attributes or dynamically
    through indexing.

    Examples:
        * TYPES.STOCK.instances returns all instances of Stock
        * TYPES.STOCK.SPY returns Spy Stock (if instantiated)
        * TYPES.STOCK["SPY"] does the same

        .. code:: python

            spy = ag.Stock("SPY")
            dia = ag.Stock("DIA")
            qqq = ag.Stock("QQQ")


        .. code:: pycon

            >>> ag.stock
            {'SPY': <STOCK SPY: $143.53 /share>, 'DIA': <STOCK DIA: $112.28 /share>, 'QQQ': <STOCK QQQ: $92.0 /share>}

            >>> ag.stock.spy
            <STOCK SPY: $143.53 /share>

            >>> spy is ag.stock.spy
            True

            >>> ag.types.stock
            STOCK

            >>> ag.types.stock.spy is ag.stock.spy
            True
    """

    def _generate_next_value_(name: str, *args: Any) -> tuple[str, WeakDict]:
        """Determines how new enum members are generated when new asset
        subclasses are created"""

        class _Instances(WeakDict):
            """A weakly referential dictionary of all instances of the
            subclass to which the enum member corresponds"""

            def __getattr__(self, attr: str) -> Any:
                try:
                    return self[attr.upper()]
                except KeyError:
                    raise AttributeError(
                        f"Asset type '{name}' has no instance '{attr}'"
                    )

            def __str__(self) -> str:
                return str(dict(self))

            @property
            def base(self) -> Currency:
                """The monetary basis of this asset subclass,
                represented by a currency code. Only used by Currency
                subclass"""
                if self and getattr(list(self.values())[0], "base", False):
                    return [c for c in self.values() if c.is_base][0]
                else:
                    raise AttributeError(f"Asset type '{self}' has no instance 'base'")

        return (name, Instances(name))

    # Non-asset types that need to be instantiated manually
    undefined = auto()  # Used when the subclass is hidden
    portfolio = auto()

    # Reconsider if these are necessary
    algorithm = auto()
    environment = auto()
    universe = auto()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.c = object
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return self.name.upper()

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, item: str) -> Any:
        return self.instances[item]

    @property
    def instances(self) -> WeakDict[str, Any]:
        """A list of all instances of a certain asset type"""
        return self.value[1]

    @classmethod
    def to_list(cls) -> list[types]:
        """
        Returns the types enum as a list of enum members with attribute access

        Returns:
            types_list (TypeList):
                The list created from types with special attribute access methods
        """

        class TypeList(list[types]):
            """A list of types enum members with attribute access for enum members"""

            def __init__(self, types_enum: Type[types]) -> None:
                self += [t for t in types_enum][1:]  # type: ignore[attr-defined]
                self.enum = types_enum

            def __getitem__(self, item: Any) -> Any:
                if item.__class__ is str:
                    try:
                        return self.enum[item]  # type: ignore[index]
                    except KeyError:
                        raise KeyError(f"Asset type '{item}' does not exist")
                return super().__getitem__(item)

            def __getattr__(self, item: str) -> Any:
                try:
                    return self.enum[item]  # type: ignore[index]
                except KeyError:
                    raise AttributeError(f"Asset type '{item}' does not exist")

        return TypeList(cls)

    @classmethod
    def instantiable(cls) -> list[Type]:
        """
        A list of all instantiable asset subclasses

        Returns all subclasses of ag.Asset that are currently instantiable,
        meaning they have successfully defined all abstract methods.

        Returns:
            All asset subclasses which contain no abstract methods.
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

    def __init__(self, asset: Asset) -> None:
        message = (
            f"Attempted duplication of {asset.name} "  # type: ignore[attr-defined]
            f"{asset.type}. Multiple instances of this asset are not "
            "permitted"
        )
        super().__init__(message)


class DataProtocol(Enum):
    """
    Enumeration of different data requirement protocols for assets

    This enumeration and its members control data requirements for the
    instantiation of different asset subclasses.

    Attributes:
        REQUIRE (DataProtocol):
            Data MUST be supplied in some form, or the asset will fail to
            instantiate. For assets that require that the asset.data attribute
            is a pandas dataframe object

        FLEXIBLE (DataProtocol):
            Asset will be instantiated whether or not data is supplied,
            asset.data attribute can be None or pd.DataFrame

        PROHIBITED (DataProtocol):
            Asset will fail to instantiate if any data is supplied. For assets
            that require that the asset.data attribute is None (eg. simulated
            assets)
    """

    REQUIRED = auto()
    FLEXIBLE = auto()
    PROHIBITED = auto()

    @classmethod
    def _get(cls, require_data: bool, prohibit_data: bool) -> DataProtocol:
        """
        Returns the appropriate data protocol based on asset
        subclass initialization settings

        Parameters:
            require_data:
                Whether or not the asset requires data

            prohibit_data:
                Whether or not the asset prohibits data

        Returns:
            The member protocol corresponding to the settings

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
    def _decompose(cls, member: DataProtocol) -> tuple[bool, bool]:
        """
        Returns the respective subclass initialization settings that
        correspond to each protocol member

        Parameters:
            member:
                The member protocol to be decomposed

        Returns:
            The DataProtocol member decomoposes in to a tuple (structured as
            tuple(require_data: bool, prohibit_data: bool))

        Raises:
            KeyError: When member is not a valid member of DataProtocol
        """
        if member is cls.FLEXIBLE:
            return False, False
        elif member is cls.REQUIRED:
            return True, False
        elif member is cls.PROHIBIT:
            return False, True

        raise KeyError(
            f"{member} is not a valid member of the DataProtocol enumeration"
        )


class Asset(ABC):
    """
    Abstract base class representing a financial asset

    The abstract base class underlying all standard and custom asset classes
    within AlphaGradient, designed to be used in conjunction with other standard
    ag objects such as portfolios and algorithms. All Assets are singletons.

    When defining a new Asset subclass, the user must define a valuate method,
    which takes in a datetime input and returns a float representing the price
    of the asset at that datetime. For Assets that intend to require data for
    valuation, this method can simply return the current value (self.value)

    Parameters:
        name:
            The name of the asset. This asset will be accessible through the
            global api under this name. For example, instantiating MyAsset("test")
            will allow the user to reference that object at ag.myasset.test,
            so long as it has not been garbage collected.

        data (Optional[ValidData]):
            The data input for this asset. The input will be coerced into an
            AssetData object. When passed as "online", the asset intialization
            will force the asset to gather data through its online_data()
            method, if one has been defined.

        columns:
            When passing in an array-like input for data, use this parameter
            to specify what elements of the array refer to what columns.
            Columns are interpreted/assigned in the same order that they come
            in via data.

        force:
            Force an Asset to be instantiated when it would otherwise fail.
            When data initialization fails, forces the asset to instantiate
            by creating a fake AssetData input with only a single data point.

        base:
            The base currency this object's value and price history are
            represented in. If passing in data that is not represented in the
            global default currency (defaults to USD), use this parameter to
            specify what currency the data is represented in.

    Examples:

        .. code:: python

            import alphagradient as ag

            # Creating a class which will require a data input
            class MyAsset(ag.Asset, require_data=True):

                def valuate(self, date):
                    return self.value
    """

    _args: dict[str, Any]
    _base: str
    _benchmark: Asset
    _close_value: str = "CLOSE"
    _data: Optional[AssetData]
    _data_protocol: DataProtocol = DataProtocol.FLEXIBLE
    _date: datetime
    _global_base: str
    _global_res: timedelta
    _global_persistent_path: Path
    _market_close: time = time.fromisoformat("00:00:00")
    _market_open: time = time.fromisoformat("00:00:00")
    _name: str
    _open_value: str = "OPEN"
    _optional: list[str] = [_open_value]
    _required: list[str] = [_close_value]
    _rfr: float
    _value: float
    _unit: str = "unit"
    _units: str = "units"

    def __init_subclass__(
        cls,
        *args: Any,
        hidden: bool = False,
        require_data: Optional[bool] = None,
        prohibit_data: Optional[bool] = None,
        required: Optional[list[str]] = None,
        optional: Optional[list[str]] = None,
        open_value: Optional[str] = None,
        close_value: Optional[str] = None,
        market_open: Optional[DateOrTime] = None,
        market_close: Optional[DateOrTime] = None,
        units: tuple[str, str] = None,
        settings: dict[str, Any] = None,
        **kwargs: Any,
    ):
        """
        Controls behavior for instantiation of Asset subclasses.

        Creates new enumerations within the TYPES enum for newly
        created subclassses of Asset. Also sets some class level
        attributes that control behavior during instantiation of that type.
        """
        # Asset subclass (class-level) attribute initialization
        # These will all be coerced to appropriate types later in this function,
        # so this initial assignment is for convenience. MyPy doesnt like it though.
        cls._required = required  # type: ignore[assignment]
        cls._optional = optional  # type: ignore[assignment]
        cls._open_value = open_value  # type: ignore[assignment]
        cls._close_value = close_value  # type: ignore[assignment]
        cls._market_open = market_open  # type: ignore[assignment]
        cls._market_close = market_close  # type: ignore[assignment]
        cls._unit, cls._units = units if units is not None else (None, None)  # type: ignore[assignment]

        # Ensuring safe access if nothing is passed
        settings = {} if settings is None else settings

        # Hiding the asset from the types enumeration
        hidden = hidden or settings.get("hidden", False)

        # Using the passed in settings object to set class attributes
        # in the case they are not provided by kwargs
        if settings:
            for attr_name in [
                "required",
                "optional",
                "open_value",
                "close_value",
                "market_open",
                "market_close",
            ]:
                if not getattr(cls, "_" + attr_name, False):
                    try:
                        setattr(cls, "_" + attr_name, settings[attr_name])
                    except KeyError as e:
                        pass

            if require_data is None:
                require_data = settings.get("require_data", False)

            if prohibit_data is None:
                prohibit_data = settings.get("prohibit_data", False)

            if settings.get("units", False):
                cls._unit, cls._units = settings["units"]

        # Setting the data protocol
        # MyPy
        assert isinstance(require_data, bool) and isinstance(prohibit_data, bool)
        cls._data_protocol = DataProtocol._get(require_data, prohibit_data)

        # Setting asset-level market close and open
        for attr_name in ("market_open", "market_close"):
            attr = getattr(cls, "_" + attr_name, None)
            if attr is not None:
                try:
                    setattr(cls, "_" + attr_name, utils.get_time(attr))
                except ValueError:
                    raise ValueError(
                        f"Invalid input for {attr_name} during initialization of {cls.__name__} asset subclass. Unable to convert {attr} to a time object."
                    )
            else:
                setattr(cls, attr_name, time(minute=0, second=0, microsecond=0))

        # What will become the name of this subclass' type
        TYPE = cls.__name__.lower()

        # Extending the enum to accomodate new type
        if ABC not in cls.__bases__ and not hidden:
            if TYPE not in [t.name for t in types]:  # type: ignore[attr-defined]
                extend_enum(types, TYPE)
            cls.type = types[TYPE]  # type: ignore
            cls.type.c = cls  # type: ignore[attr-defined]

        # Used when a new asset subclass is hidden from the AG api
        if not getattr(cls, "type", None):
            cls.type = types.undefined  # type: ignore[attr-defined]

        if cls.unit is None or cls.units is None:
            cls.unit = "unit"
            cls.units = "units"

    def __new__(cls, *args: Any, **kwargs: Any) -> Asset:
        # Seeing if this asset is already instantiated
        if args or kwargs.get("name", False):
            name = kwargs["name"] if kwargs.get("name") else args[0]

            # Returning the asset if exists
            if name in cls.type.instances:  # type: ignore[attr-defined]
                return cls.type.instances[name]  # type: ignore[attr-defined]

        # Returning a new asset
        return cast(Asset, super().__new__(cls))

    def __init__(
        self,
        name: str,
        data: Optional[ValidData] = None,
        columns: Optional[list[str]] = None,
        force: bool = False,
        base: Optional[str] = None,
    ) -> None:

        # Standard style guideline for all asset names. Ensures that
        # they are accessible via attribute access through the global env
        # eg. ag.stock.stocknamehere
        self._initialized = False
        name = str(name).upper().replace(" ", "_")

        # Checks if arguments have changed materially from previous
        # initialization, and skip initialization if they haven't.
        if self.type.instances.get(name) is self:  # type: ignore[attr-defined]

            skip = True
            if data is not None:

                if isinstance(data, pd.DataFrame) and isinstance(
                    self._args["data"], pd.DataFrame
                ):
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
        self.type.instances[name] = self  # type: ignore[attr-defined]

        # Attribute Initialization
        self._name = name
        valid_bases: list[str] = list(_currency_info["CODE"])
        self._base: str = base if base in valid_bases else self._global_base
        self._value: float = data if isinstance(data, (int, float)) else 0
        self.close: bool = True
        self._data = None

        # Data entry is prohibited, data must always be None
        if self.protocol is DataProtocol.PROHIBITED:
            self._data = None

        # Data entry is not prohibited, initialize dataset
        else:
            if data is None:

                # First attempt to get data from saved files
                data = get_data(self)

                # Second attempt to get data from online data
                if data is None and getattr(self, "online_data", False):
                    try:
                        data = self.online_data()  # type: ignore[attr-defined]
                    except ValueError:
                        pass

                # Third attempt to get data from nothing...?
                if data is None and self.protocol is DataProtocol.REQUIRED:

                    # When we need to force the instantiation of an
                    # asset that requires data, but no data is available
                    if force:
                        data = AssetData(self.__class__, 1)
                    else:
                        raise ValueError(
                            f"{self.name} {self.type} "  # type: ignore[attr-defined]
                            "could not be initialized "
                            "without data. If this is "
                            "the first time this asset is "
                            "being instantiated, please "
                            "provide a valid dataset or "
                            "instantiate with force=True."
                        )

                assert isinstance(data, (AssetData, type(None)))
                self._data = data

            # Explicit reinstancing from online data
            elif isinstance(data, str) and data.lower() == "online":
                try:
                    data = self.online_data()  # type: ignore[attr-defined]
                except AttributeError as e:
                    raise ValueError(
                        f"{self.name} unable to retrieve online data, "  # type: ignore[attr-defined]
                        f"{self.type} has not implemented an online_data method"
                    ) from e
                if data is None:
                    raise ValueError(
                        f"Unable to retreive online data for {self.type} "  # type: ignore[attr-defined]
                        f'"{self.name}", initialization failed'
                    )

                self._data = data

            # Data input received, make a new asset dataset
            else:
                self._data = AssetData(self.__class__, data, columns)

            # Data verification when required
            if self.protocol is DataProtocol.REQUIRED and not self.data:
                raise ValueError(
                    f"{self.name} {self.type} could not "  # type: ignore[attr-defined]
                    "be initialized without data. If this "
                    "is the first time this asset is "
                    "being instantiated, please provide "
                    "a valid dataset or instantiate with "
                    "force=True."
                )

        # Ensures that the initial price is set properly
        self._valuate()
        self._initialized = True
        self._save()

    def __str__(self) -> str:
        return f"<{self.type} {self.name}: {self.price} /{self.unit}>"  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return self.key.__hash__()

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        return self is other

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state

    def __copy__(self) -> NoReturn:
        raise AssetDuplicationError(self)

    def __deepcopy__(self) -> NoReturn:
        raise AssetDuplicationError(self)

    @property
    def base(self) -> str:
        """
        The asset's base currency

        Represented as an international currency code; a string of uppercase
        alphabetic characters of length 2-5.
        """
        return self._base

    @property
    def benchmark(self) -> Asset:
        """
        The benchmark asset to use in calculations of alpha, beta, etc
        """
        return self._benchmark

    @property
    def close_value(self) -> str:
        """The name of the column to associate with market close prices for this asset type"""
        return self._close_value

    @property
    def data(self) -> Optional[AssetData]:
        """
        The historical price data for the this asset

        Either None or an instance of AssetData. Depending on their ``protocol``,
        Assets may either require the presence of historical data (eg. Stocks),
        forbid it (eg. BrownianStocks), or operate regardless of circumstance. AssetData
        objects always evaluate as ``True``, so it is safe to evaluate ``self.data``
        as a boolean, unlike a pandas DataFrame.
        """
        return self._data

    @property
    def date(self) -> datetime:
        """The date of this asset's most recent valuation, which to which its
        current value corresponds"""
        return self._date

    @property
    def expired(self) -> bool:
        """
        Whether or not this asset is expired

        Most assets will never expire, so the default behavior is to
        always return false. Some assets like options can expire,
        however. Overwrite this property to determine when an asset
        expires
        """
        return False

    @property
    def key(self) -> str:
        """
        Returns a key used for accessing stored files relevant to this asset

        Creates a key from information unique to this asset. These
        keys are used to access locally stored data relevant to this
        asset
        """
        return f"{self.type}_{self.name}"  # type: ignore[attr-defined]

    @property
    def market_close(self) -> time:
        """The market closing time for this asset type"""
        return self._market_close

    @property
    def market_open(self) -> time:
        """The market opening time for this asset type"""
        return self._market_open

    @property
    def name(self) -> str:
        """
        The name of this asset

        A string of uppercase alphanumeric characters that denotes the name of this
        asset. Names are unique to assets by type; no two assets of the same type
        may share the same name (but assets of different types may be named
        identically). Used to access this asset in the global environment, as well
        as encode its key for storage in positions.

        Examples:

            .. code:: pycon

                >>> spy = ag.Stock(name="SPY")
                >>> spy.name
                SPY

                >>> spy
                <STOCK SPY: $429.06 /share>

                >>> ag.stock.spy
                <STOCK SPY: $429.06 /share>

                >>> spy is ag.stock.spy
                True

                >>> spy.key
                STOCK_SPY
        """
        return self._name

    @property
    def next(self) -> datetime:
        """The next available datetime"""
        if self.data:
            return self.data.next(self.date)
        else:
            return self.date + self._global_res

    @property
    def open(self) -> bool:
        """Whether or not this asset is tradeable based on the current date"""
        return (
            self.date.time() >= self.market_open
            and self.date.time() <= self.market_close
            if self.market_close != self.market_open
            else True
        )

    @property
    def open_value(self) -> str:
        """The name of the column to associate with market open prices for this asset type"""
        return self._open_value

    @property
    def optional(self) -> list[str]:
        """A list of optional columns for any data input to this asset type"""
        return self._optional

    @property
    def price(self) -> str:
        """
        A print friendly version of the asset value with the asset's base
        currency symbol attached
        """
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
    def protocol(self) -> DataProtocol:
        """The data requirement protocol that this asset type operates under"""
        return self._data_protocol

    @property
    def required(self) -> list[str]:
        """A list of required columns for any data input to this asset type"""
        return self._required

    @property
    def rfr(self) -> float:
        """The risk free rate used for this asset type"""
        return self._rfr

    @property
    def value(self) -> float:
        """This asset's current value"""
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        if isinstance(value, Number):
            if value != float("nan"):
                self._value = value
        else:
            raise TypeError(
                f"Can not update value of {self.name} "  # type: ignore[attr-defined]
                f"{self.type} to "
                f"{value.__class__.__name__} "
                f"{value}. Value must be a numnber"
            )

    @property
    def unit(self) -> str:
        """How to refer to a single unit of this asset type"""
        return self._unit

    @property
    def units(self) -> str:
        """How to refer to multiple units of this asset type"""
        return self._units

    def _data_valuate(self, date: Optional[DatetimeLike] = None) -> float:
        """
        Determines how asset prices update when using data

        Determines how assets are valuated when data is available. Keep
        track of/updates when the asset is at the beginning or the end
        of a time interval, and valuates accordingly.

        Parameters:
            date (DatetimeLike):
                DatetimeLike object that determines at what point in time the
                valuation will be

        Returns:
            The price as of the given datetime
        """
        assert self.data is not None
        date = self.date if date is None else utils.to_datetime(date)
        value = float(self.data.valuate(date, self))  # TODO Should this use quote??
        if not math.isnan(value):
            return value
        else:
            if not self._initialized:
                data = self.data.asof(self.data.index[0])
                return data[self.close_value]
            else:
                return self.value

    def _save(self) -> None:
        """Saves this asset's data locally"""
        if self.data and self._global_persistent_path is not None:
            path = self._global_persistent_path.joinpath(f"{self.key}.p")
            with open(path, "wb") as p:
                self.data._data.to_pickle(p)

    def _step(self, date: DatetimeLike) -> None:
        """
        Automatically called before this asset is valuated during time steps

        This function is a hook to perform some behavior on the asset
        prior to its valuation at each time step. It should return None;
        it modifies this asset in place.

        Parameters:
            date (DatetimeLike):
                The date of the valuation that will occur after this function
                is executed
        """
        return None

    def _valuate(self) -> None:
        """
        Updates asset prices when time steps take place

        This is the method that is actually called under the hood when
        time steps take place, which properly directs valuation
        behavior to either valuate or _data_valuate depending on the
        asset type.
        """
        self.value = self.quote(self.date)

    def alpha(self, days: float = 365, benchmark: Asset = None) -> float:
        """
        Returns this asset's alpha for the given period

        Calculates the return of this asset relative to its risk adjusted
        expected return, using some other asset a basis.

        Parameters:
            days:
                The period across which to calculate alpha

            benchmark:
                The benchmark to act as a basis fot the calculation of risk
                adjusted expected return

        Returns:
            This asset's alpha for the period
        """
        benchmark = benchmark or self.benchmark
        assert benchmark.data is not None
        asset_roi = self.roi(days)
        bench_roi = self.benchmark.roi(days)
        expected_roi = self.beta(days, benchmark) * (bench_roi - self.rfr)
        return asset_roi - expected_roi

    def beta(self, days: float = 365, benchmark: Asset = None) -> float:
        """
        Returns the beta for the period

        Returns the beta for the period. Calculated by weighting the covariance
        of this asset and the benchmark asset by the ratio of their
        volatilities.

        Parameters:
            days:
                The period across which to calculate beta

            benchmark:
                The benchmark asset to compare to

        Returns:
            This asset's beta for the given period
        """
        benchmark = benchmark or self.benchmark
        assert benchmark.data is not None
        start = self.date - timedelta(days=days)
        self_vol = self.vol(days)
        bench_vol = benchmark.vol(days)
        self_data = self.range(start, self.date)["CHANGE"]
        bench_data = benchmark.range(start, self.date)["CHANGE"].asof(self_data.index)
        r = self_data.corr(bench_data)
        if bench_vol == 0:
            return 0
        return r * (self_vol / bench_vol)

    def cagr(self, days: float = 365) -> float:
        """
        Returns this asset's compounding annualized growth rate for the given period

        Parameters:
            days:
                period across which to calculate cagr

        Returns:
            Compounding annualized growth rate for this asset
        """
        return (self.roi(days) + 1) ** (365 / days) - 1

    def expire(self, portfolio: Portfolio, position: Position) -> None:
        """
        Controls the behavior of this asset and positions in this asset when
        it expires inside of a portfolio

        Positions in portfolios are automatically removed from that
        portfolio when they expire. Their expiration is determined
        based on two conditions: Whether or not the position.quantity > 0,
        and whether or not the position's underlying asset is expired.

        Changing this method in an asset subclass will change its
        behavior as it expires. The conditions for expiring can also
        be changed by overloading the asset's 'expired' property
        (must be a property)

        This function should return None; it modifies this asset, as well as the
        given Portfolio and Position in-place

        Parameters:
            portfolio:
                The portfolio holding a position in this assets

            position:
                The above portfolio's current position in this asset that is
                becoming expired after this call.
        """
        return None

    @classmethod
    def get_settings(cls, unpack: bool = False) -> Union[list, dict[str, Any]]:
        """
        Returns a dictionary of class attributes

        This settings object is the same one used in the class header for
        defining class attributes.

        Parameters:
            unpack:
                When true, provides the values unpacked into a list, for easy
                unpacking

        Returns:
            Class-level settings for this asset type
        """
        require, prohibit = DataProtocol._decompose(cls._data_protocol)

        settings = {
            "hidden": cls.type is types.undefined,  # type: ignore[attr-defined]
            "require_data": require,
            "prohibit_data": prohibit,
            "required": cls._required,
            "optional": cls._optional,
            "close_value": cls._close_value,
            "open_value": cls._open_value,
        }

        return list(settings.values()) if unpack else settings

    def ma(self, days: float = 365) -> float:
        """
        Returns the moving average of this asset's value over the period given by days

        Parameters:
            days:
                A number indicating the number of days for which the moving
                average should be calculated

        Returns:
            A floating point value representing the average price over the
            period given by days.
        """
        start = self.date - timedelta(days=days)
        data = self.range(start, self.date)
        return data[self.close_value].mean()

    def quote(self, date: DatetimeLike) -> float:
        """
        Returns the value of this asset on the given date

        Parameters:
            date (DatetimeLike):
                The date on which to return the value of this asset

        Returns:
            The asset's value on the given date
        """
        date = utils.to_datetime(date)
        if self.data:
            return self.data.valuate(date, self)
        else:
            return self.valuate(date)

    def range(self, start: DatetimeLike, end: DatetimeLike) -> pd.DataFrame:
        """
        Returns a datetime-indexed dataframe of asset values from start to end

        TODO: Inefficient implementation, not necessary.

        Parameters:
            start (DatetimeLike):
                The date corresponding to the beginning of the period

            end:
                The date corresponding to the end of the period

        Returns (DatetimeLike):
            The range of date corresponding to the period defined by start and
            end
        """

        if self.data:
            return self.data.range(start, end)

        @np.vectorize
        @lru_cache
        def vquote(date):
            return self.valuate(date)

        dates = pd.date_range(start, end)
        prices = vquote(dates)
        data = pd.DataFrame(data=prices, index=dates, columns=[self.close_value])

        if len(data) > 1:
            shifted = data[self.close_value].shift(1)
            shifted[0] = shifted[1]
            data["CHANGE"] = (data[self.close_value] / shifted) - 1

        elif not data.empty:
            data["CHANGE"] = [0]

        else:
            data["CHANGE"] = []

        return data

    def roi(self, days: float = 365) -> float:
        """
        This assets return on investment for the input period

        Returns the difference between the starting value and the current value
        as a percentage of the starting value

        Parameters:
            days:
                The period across which to calculate

        Returns:
            The percentage difference from start to end
        """
        start = self.date - timedelta(days=days)
        initial = self.quote(start)
        if initial == 0:
            return 0
        return (self.value / initial) - 1

    @no_type_check
    @abstractmethod
    def valuate(self, date: DatetimeLike) -> float:
        """
        Determines how asset prices update when not using data

        This is the method that defines non-data-based valuation
        behavior for an asset subclass. The standard implementation
        essentially does nothing--prices stay constant. New asset
        subclasses are required to replace this method in order to be
        instantiable.

        Parameters:
            date (DatetimeLike):
                The date on which to valuate this asset

        Returns:
            Value information for this asset at the given datetime
        """
        return self.value

    def vol(self, days: float = 365) -> float:
        """Returns the historical volatility of this asset's value over the
        period given by days, as a percentage of the current valuation

        Parameters:
            days:
                A number indicating the number of days for which the historical
                volatility should be calculated

        Returns:
            A floating point number representing the average deviation of this
            asset's price from its moving average over the same period,
            expressed as a percentage of the current value.
        """

        ### TODO ### THIS MUST CONSIDER THE INTERVALS OF TIME BETWEEN EACH
        # INDEX, RIGHT NOW ASSUMES THAT ALL VALUES ARE EQUIDISTANT / DAILY.
        # TIME RESOLUTION OF THE DATA WILL AFFECT THE STD

        # 252 Trading days per year
        multiplier = (days / (252 / 365)) ** 0.5
        start = self.date - timedelta(days=days)
        data = self.range(start, self.date)
        return data["CHANGE"].std() * multiplier
