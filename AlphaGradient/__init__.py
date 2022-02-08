from .algorithm import Algorithm
from . import finance
from .finance.asset import TYPES as TYPES_ENUM
from .constants import VERBOSE, CONSTANTS
from . import algolib
from .data import datatools
#from typing import Any, Union, List


def __getattr__(name):

	# When accesing AlphaGradient's defined types, we want to return a list of
	# types rather than an enum object, but one that has all the same access
	# abilities as the enum.
	class TypeList(list):
		def __init__(self, TYPES_ENUM):
			self += [TYPE for TYPE in TYPES_ENUM][1:]
			self.ENUM = TYPES_ENUM

		def __getitem__(self, item):
			if item.__class__ is str:
				try:
					return self.ENUM[item]
				except KeyError:
					raise KeyError(f'Asset type \'{item}\' does not exist')
			return super().__getitem__(item)

		def __getattr__(self, item):
			try:
				return self.ENUM[item]
			except KeyError:
				raise AttributeError(f'Asset type \'{item}\' does not exist')

	if name == 'TYPES':
		return TypeList(TYPES_ENUM)

	# Returning a list containing ALL assets currently in memory
	if name == 'ASSETS':
		return [v for TYPE in TYPES_ENUM for _, v in TYPE.instances.items()]

	if name == "COLUMNS":
		return datatools.AssetData.COLUMNS

	# For accessing the instances of a type, if the type exists
	try:
		return TYPES_ENUM[name].instances
	except KeyError:
		raise AttributeError(
			f'AttributeError: module \'{__name__}\' has no attribute \'{name}\'')
