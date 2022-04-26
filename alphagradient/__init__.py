# -*- coding: utf-8 -*-
"""Base initializer for the whole AlphaGradient package"""

# Local imports
from ._finance._asset import types as _types
from ._algorithm import Algorithm
from ._globals import __globals
from ._data import _datatools
from ._finance import (
    Asset,
    Portfolio,
    Cash,
    Currency,
    Stock,
    Call,
    Put,
    Environment,
    Universe
)
from . import utils
from . import algolib
from . import dtypes

def __getattr__(name):
    if name == "types":
        return _types.list()

    # Returning a list containing ALL assets currently in memory
    if name == "assets":
        return list(__globals.all_assets())

    if name == "globals":
        return __globals

    # For accessing the instances of a type, if the type exists
    try:
        return _types[name].instances
    except KeyError as ke:
        raise AttributeError(f"AttributeError: module \'{__name__}\' "
                             f"has no attribute \'{name}\''") from ke


__all__ = [
    # FINANCE IMPORTS
    'Asset',
    'Portfolio', 
    'Cash', 
    'Currency', 
    'Stock', 
    'Call',
    'Put',
    'Environment',
    'Universe',

    # OTHER TOP LEVEL CLASSES
    'Algorithm',
    ''
]