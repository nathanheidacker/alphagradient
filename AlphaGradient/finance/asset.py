from ..constants import VERBOSE, CONSTANTS, is_numeric
from ..data import datatools
from aenum import Enum, unique, auto, extend_enum
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from datetime import datetime, timedelta
import dataclasses
import weakref
import pandas as pd


@unique
class TYPES(Enum):
    '''Enum of Asset types that have been declared, 
    containing references to all instances of assets belonging to that type'''

    def _generate_next_value_(name, *args):
        '''Used to determine how enum values are 
        automatically created when new enum members are added'''

        class Instances(weakref.WeakValueDictionary):
            '''A weakly referential dictionary that keeps track of 
            all in-memory instances of a given asset type'''

            # Treat attribute access like item access, also adding a more
            # descriptive error message
            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError:
                    raise AttributeError(
                        f'Asset type \'{name}\' has no instance \'{item}\'')

            def __str__(self):
                return str([x for x in self.items()])[1:-1]

        return (name, Instances())

    # Used when the asset type is not specified
    UNDEFINED = auto()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    # Item access is directed to the enum's associated weakly referential
    # dictionary
    def __getitem__(self, item):
        return self.value[1][item]

    # User access to the weakly referential dictionary
    @property
    def instances(self):
        return self.value[1]


class Hidden:
    '''
    Dummy class for hiding children of asset class from the AlphaGradient TYPES access.

    To use, add this class as a parent in the class declaration:

    class MyAsset(AlphaGradient.Asset, AlphaGradient.Hidden):
            pass
    '''
    pass


class AssetDuplicationError(Exception):
    '''
    An error that occurs when more than a single instance of an asset is being instantiated. 
    All assets are singletons.
    '''

    def __init__(self, asset):
        message = f'''Attempted duplication of {asset.name} {asset.type}. Multiple instances 
        of this asset are not permitted'''
        super().__init__(message)


class Asset(ABC):
    '''Base class representing a financial asset. 
    Used as the basis for all assets within AlphaGradient.'''

    def __init_subclass__(cls, **kwargs):
        '''Creates new enumerations within the TYPES enum for newly created subclasses of Asset'''

        # All TYPES should be upper, style guideline
        TYPE = cls.__name__.upper()

        # Extending the enum to accomodate new type, assigning it to the class
        if all(base not in cls.__bases__ for base in [ABC, Hidden]):
            if TYPE not in [t.name for t in TYPES]:
                extend_enum(TYPES, TYPE)
            cls.type = TYPES[TYPE]

        # Used when a new asset subclass is hidden from the AG api using the
        # 'Hidden' class
        if not getattr(cls, 'type', None):
            cls.type = TYPES.UNDEFINED

    # All assets must have their initialization behavior defined. All
    # subclasses should call super().__init__() in their init
    @abstractmethod
    def __init__(
            self,
            name,
            date=None,
            data=None,
            require_data=False,
            required=None,
            optional=None,
            allow_duplicates=False):

        # Attribute Initialization
        self.name = str(name)
        self.price = data if is_numeric(data) else 0
        self.valuate_on = datatools.AssetData.COLUMNS.CLOSE

        # Check if an instance of this asset already exists
        if name not in self.type.instances:
            if allow_duplicates:
                self.type.instances[name] = [self]
            else:
                self.type.instances[name] = self
        elif allow_duplicates:
            self.type.instances[name] += self
        else:
            raise AssetDuplicationError(self)

        # Accept isoformat datestrings as well as datetimes
        self.date = date if isinstance(
            date, datetime) else datetime.fromisoformat(date)

        # Initialize a dataset based on the input, then search for
        if data is None:
            data = datatools.get_data(self.type, self.name)
            if getattr(self, '_online_data', None):
                data = self._online_data()
            self.data = data if data else datatools.AssetData(
                None, required, optional)
        else:
            self.data = datatools.AssetData(data, required, optional)
            # Pickling / updating the ledger should probably be done outside of
            # the initialization of the class instance
            if data:
                ledger_id = datatools.Ledger.id(self.type, self.name)
                pd.to_pickle(
                    self.data,
                    f'Alphagradient/data/pickles/{ledger_id}')

        # Data verification when required
        if not self.data and require_data:
            raise ValueError(
                f'''{self.type} {self.name} requires data for initialization, 
                but was not provided with a viable dataset''')

        self.valuate()

    def __str__(self):
        return f'({self.name} {self.type}: ${self.price})'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self is other
        else:
            return NotImplemented

    def valuate(self, date=None):
        '''Updates asset prices when time steps take place'''
        date = self._normalize_date_input(date)

        if self.data:
            return self._data_valuate(date)
        else:
            return self._valuate()

    def _data_valuate(self, date=None):
        '''Determines how asset prices update when using data'''
        date = self._normalize_date_input(date)
        data = self.data.asof(date)
        self.price = data[self.valuate_on]
        return self.price

    @abstractmethod
    def _valuate(self):
        '''Determines how asset prices update when not using data'''
        return self.price

    def _normalize_date_input(self, date=None):
        '''Standardizes different modalities of date input'''

        if date is None:
            return self.date

        # Only accepts isoformatted datestrings for the time being, but may be
        # updated to accept other types of datestring inputs
        elif isinstance(date, str):
            return datetime.fromisoformat(date)

        elif isinstance(date, datetime):
            return date

        else:
            raise TypeError(
                f'date input of type {type(date)} could not be normalized')


class AssetData:
    '''Strictly formatted datasets for alphagradient assets

    Given a table, should return a properly formatted (time-indexed) 
    dataset for use in ag assets if possible.'''

    def __init__(self, data):
        self.data = data
        # Read table, read pickle, read array-like
