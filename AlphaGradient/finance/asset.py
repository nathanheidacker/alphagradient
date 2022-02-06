# -*- coding: utf-8 -*-
"""AG module containing base Asset class and associated enums

This module contains all of the necessary components for the proper function of standard asset classes within AlphaGradient, as well as the API.

Todo:
	* Complete google style docstrings for all module components
	* Complete function/class header typing
	* Replace is_numeric with Number instance checks
	* Interpret more than just isoformat for normalizing datestrings
"""

# Standard imports
from abc import ABC, abstractmethod
from datetime import datetime
#from numbers import Number
import weakref
#import typing

# Third Party imports
from aenum import Enum, unique, auto, extend_enum
import pandas as pd

# Local imports
from ..constants import is_numeric # will be replaced by Number
from ..data import datatools


@unique
class TYPES(Enum):
	"""A enumeration with members for all asset subclasses

	The types enum is a special enumeration which dynamically creates new members for all subclasses of ag.Asset. Enum members store a weakly referential dictionary to all instances of that asset subclass, which can be accessed as attributes or dynamically through indexing.

	Examples:
		* TYPES.STOCK.instances returns all instances of Stock
		* TYPES.STOCK.SPY returns Spy Stock (if instantiated)
		* TYPES.STOCK["SPY"] does the same

	"""

	def _generate_next_value_(name, *args):
		"""Determines how new enum members are generated when new asset subclasses are created"""

		class Instances(weakref.WeakValueDictionary):
			"""A weakly referential dictionary of all instances of the subclass to which the enum member corresponds"""
			def __getattr__(self, item):
				try:
					return self[item]
				except KeyError:
					raise AttributeError(
						f"Asset type \'{name}\' has no instance \'{item}\'")

			def __str__(self):
				return str(list(self.items()))[1:-1]

		return (name, Instances())

	# Used when the asset subclass is hidden from the types enum
	UNDEFINED = auto()

	def __str__(self):
		return self.name

	def __repr__(self):
		return self.__str__()

	def __getitem__(self, item):
		return self.value[1][item]

	@property
	def instances(self):
		"""A list of all instances of a certain asset type"""
		return self.value[1]


class AssetDuplicationError(Exception):
	"""Raised when asset duplication is forced via copy methods"""
	def __init__(self, asset):
		message = f"""Attempted duplication of {asset.name} {asset.type}. Multiple instances of this asset are not permitted"""
		super().__init__(message)


class DataProtocol(Enum):
	"""Enumeration of different data requirement protocols for assets"""
	REQUIRED = auto()
	FLEXIBLE = auto()
	PROHIBITED = auto()

	@classmethod
	def _get(cls, require_data, prohibit_data):
		if require_data is prohibit_data:
			return cls.FLEXIBLE
		elif require_data and not prohibit_data:
			return cls.REQUIRED
		elif not require_data and prohibit_data:
			return cls.PROHIBITED

		raise TypeError(f"inputs must be booleans")

	@classmethod
	def _decompose(cls, member):
		if member is cls.FLEXIBLE:
			return False, False
		elif member is cls.REQUIRED:
			return True, False
		elif member is cls.PROHIBIT:
			return False, True

		raise KeyError(f"{member} is not a valid member of the DataProtocol enumeration")


class Asset(ABC):
	"""Abstract base class representing a financial asset

	The ABC underlying all standard and custom asset classes within AlphaGradient, designed to be used in conjunction with other standard ag objects such as portfolios and algorithms. All Assets are singletons.

	When defining a new Asset subclass, the valuate method

	Attributes:
		_args (dict): Arguments used in most recent initialization.
			Referenced to determine whether 
		name (str): Name of the asset
		price (Number): Price of the asset in USD
		close (bool): Whether most recent valuation represents the 
			close (end) of an interval.
	"""

	def __init_subclass__(
						  cls, 
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

		Creates new enumerations within the TYPES enum for newly created subclassses of Asset. Also sets some class level attributes that control behavior during instantiation. The following are all CLASS-level attributes for Asset and all Asset subclasses.

		Attributes:
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

		settings = {} if settings is None else settings
		hidden = hidden or settings.get("hidden", False)

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


		cls.data_protocol = DataProtocol._get(require_data, prohibit_data)

		# All TYPES should be upper, style guideline
		TYPE = cls.__name__.upper()

		# Extending the enum to accomodate new type
		if ABC not in cls.__bases__ and not hidden:
			if TYPE not in [t.name for t in TYPES]:
				extend_enum(TYPES, TYPE)
			cls.type = TYPES[TYPE]

		# Used when a new asset subclass is hidden from the AG api
		if not getattr(cls, 'type', None):
			cls.type = TYPES.UNDEFINED

	def __new__(cls, *args, **kwargs):
		if args or kwargs.get("name", False):
			name = kwargs["name"] if kwargs.get("name") else args[0]
			if name in cls.type.instances:
				return cls.type.instances[name]
		return super().__new__(cls)

	def __init__(
			self,
			name,
			date=None,
			data=None,
			columns=None,
			force=False):

		# Checks if arguments have changed materiall from previous initialization
		if name in self.type.instances:

			skip = True

			if data != self._args["data"] and data is not None:
				skip = False

			if date != self._args["date"] and date is not None:
				self.valuate(date)

			if skip:
				return
		
		# Saving new args, storing new instance
		self._args = locals()
		self.type.instances[name] = self

		# Attribute Initialization
		self.name = str(name)
		self.price = data if is_numeric(data) else 0
		self.close = True

		# Accept isoformat datestrings as well as datetimes
		date = date if date else datetime.today()
		self.date = date if isinstance(
			date, datetime) else datetime.fromisoformat(date)

		# Data entry is prohibited
		if self.data_protocol is DataProtocol.PROHIBITED:
			self.data = None

		# Data entry is not prohibited
		else:
			if data is None:

				# First attempt to get data from saved files
				data = datatools.get_data(self)

				# Second attempt to get data from online data
				if data is None and getattr(self, "online_data", False):
					data = self.online_data()

				# Third attempt to get data from nothing...?
				if data is None and self.data_protocol is DataProtocol.REQUIRED:

					if force:
						data = datatools.AssetData(self.__class__, 1, columns)
					else:
						raise ValueError(f"{self.name} {self.type} could not be initialized without data. If this is the first time this asset is being instantiated, please provide a valid dataset or instantiate with force=True.")

				self.data = data

			# Data input received, make a new asset dataset
			else:
				self.data = datatools.AssetData(self.__class__, data, columns)

			# Data verification when required
			if self.data_protocol is DataProtocol.REQUIRED and not self.data:
				raise ValueError(f"{self.name} {self.type} could not be initialized without data. If this is the first time this asset is being instantiated, please provide a valid dataset or instantiate with force=True.")

		self._valuate()

	def __str__(self):
		return f'<{self.type} {self.name}: ${self.price}>'

	def __repr__(self):
		return self.__str__()

	def __eq__(self, other):
		if self.__class__ is other.__class__:
			return self is other
		else:
			return NotImplemented

	def __copy__(self):
		raise AssetDuplicationError(self)

	def __deepcopy__(self):
		raise AssetDuplicationError(self)

	def _key(self):
		"""Returns a key used for accessing stored files relevant to this asset

		Creates a key from information unique to this asset. These keys are used to access locally stored data relevant to this asset

		Returns:
			key (str): a key unique to this asset
		"""

		return f"{self.type}_{self.name}"

	def _valuate(self, date=None):
		"""Updates asset prices when time steps take place

		This is the method that is actually called under the hood when time steps take place, which properly directs valuation behavior to either valuate or _data_valuate depending on the asset type.

		Args:
			date: date-like object that will become a datetime.
				Determines at what point in time the valuation will be

		Returns:
			price (float): Updates the instance's price inplace, and
				also returns it
		"""
		date = self.normalize_date(date)

		if self.data:
			return self._data_valuate(date)
		else:
			return self.valuate()

	def _data_valuate(self, date=None):
		"""Determines how asset prices update when using data

		Determines how assets are valuated when data is available. Keep track of/updates when the asset is at the beginning or the end of a time interval, and valuates accordingly.

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
		"""TODO: Im actually not sure this makes a lot of sense. We would need to valuate twice for the same period or something in order for this to work properly. There are a few better solutions to consider:
			* If we can assume time intervals are consistent, we can
				separate the open and close of a period by exactly that interval
		"""
		if self.close:
			self.price = data[self.data.open_value]
		else:
			self.price = data[self.data.close_value]

		self.close = not self.close

		return self.price

	@abstractmethod
	def valuate(self, *args, **kwargs):
		"""Determines how asset prices update when not using data

		This is the method that defines non-data-based valuation behavior for an asset subclass. The standard implementation essentially does nothing--prices stay constant. New asset subclasses are required to replace this method in order to be instantiable.

		Args:
			*args: see below
			**kwargs: Valuate methods for different assets will likely
				require different arguments to be passed. Accepting all of them allows 
		"""
		return self.price

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
			"hidden": cls.type is TYPES.UNDEFINED,
			"require_data": require,
			"prohibit_data": prohibit,
			"required": cls.required,
			"optional": cls.optional,
			"close_value": cls.close_value,
			"open_value": cls.open_value
		}

		return settings.values() if unpack else settings

	