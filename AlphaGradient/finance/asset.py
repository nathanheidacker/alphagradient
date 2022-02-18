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
from datetime import datetime
from numbers import Number
from weakref import WeakValueDictionary as WeakDict
#import typing

# Third Party imports
from aenum import Enum, unique, auto, extend_enum
import pandas as pd

# Local imports
from ..data import datatools

_currency_info = pd.read_pickle("AlphaGradient/finance/currency_info.p")

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
                    return self[item]
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
            Referenced to determine whether
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

        # Ensuring safe access if nothing is passed
        settings = {} if settings is None else settings

        # Hiding the asset from the types enumeration
        hidden = hidden or settings.get("hidden", False)

        # Using the passed in settings object to set class attributes
        # in the case they are not provided by kwargs
        if settings:
            for attr in ["required", "optional", "open_value", "close_value"]:
                if not getattr(cls, attr, False):
                    try:
                        setattr(cls, attr, settings[attr])
                    except KeyError as e:
                        pass

            if require_data is None:
                require_data = settings.get("require_data", False)

            if prohibit_data is None:
                prohibit_data = settings.get("prohibit_data", False)


        # Setting the data protocol
        cls.data_protocol = DataProtocol._get(require_data, prohibit_data)

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
        name = str(name).upper().replace(' ', '_')

        # Checks if arguments have changed materially from previous
        # initialization, and skip initialization if they haven't.
        if self.type.instances.get(name) is self:

            skip = True

            if data != self._args["data"] and data is not None:
                skip = False

            if date != self._args["date"] and date is not None:
                self._valuate(date)

            if skip:
                return

        # Saving new args, storing new instance
        self._args = locals()
        self.type.instances[name] = self

        # Attribute Initialization
        self.name = name
        self.base = base if base in list(_currency_info["CODE"]) else types.currency.instances.base.base
        self._value = data if isinstance(data, Number) else 0
        self.close = True

        # Accept isoformat datestrings as well as datetimes
        date = date if date else self._global_start
        self.date = date if isinstance(
            date, datetime) else datetime.fromisoformat(date)

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
                    data = self.online_data()

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
        self._valuate(date)

    def __str__(self):
        return f'<{self.type} {self.name}: {self.price}>'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self is other

    def __copy__(self):
        raise AssetDuplicationError(self)

    def __deepcopy__(self):
        raise AssetDuplicationError(self)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if isinstance(value, Number):
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

    def _valuate(self, date=None):
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
        self.date = self.normalize_date(date)

        if self.data:
            self.value = self._data_valuate(self.date)
        else:
            self.value = self.valuate(self.date)

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
        data = self.data.asof(date)

        # Switching between beginning and the end of a period
        """TODO: Im actually not sure this makes a lot of sense. We
        would need to valuate twice for the same period or something
        in order for this to work properly. There are a few better
        solutions to consider:
            * If we can assume time intervals are consistent, we can
                separate the open and close of a period by exactly
                that interval
        """
        if self.close:
            self.value = data[self.data.open_value]
        else:
            self.value = data[self.data.close_value]

        self.close = not self.close

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


    def _step(self, date):
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

