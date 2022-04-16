# -*- coding: utf-8 -*-
"""A proxy for the ag package for use in the algolibrary

Todo:
    * organize imports
"""

# Local imports
from .finance.asset import types as _types
from .algorithm import Algorithm
from ._globals import __globals
from .data import datatools
from . import finance
from . import utils

def __getattr__(name):
    if name == "types":
        return _types.list()

    # Returning a list containing ALL assets currently in memory
    if name == "assets":
        return __globals.all_assets()

    if name == "globals":
        return __globals

    # For accessing the instances of a type, if the type exists
    try:
        return _types[name].instances
    except KeyError as ke:
        raise AttributeError(f"AttributeError: module \'{__name__}\' "
                             f"has no attribute \'{name}\''") from ke
