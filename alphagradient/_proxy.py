# -*- coding: utf-8 -*-
"""A proxy for the ag package for use in the algolibrary and typechecking purposes."""

# Standard imports
from __future__ import annotations

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
    Universe,
)
from . import utils
from . import algolib
from . import dtypes as _dtypes_module

# Typing
from typing import (
    Any,
)

# Controls module level access to AG types, types enum
def __getattr__(name: str) -> Any:
    if name == "types":
        return _types.list()

    # Returning a list containing ALL assets currently in memory
    if name == "assets":
        return list(__globals.all_assets())

    if name == "globals":
        return __globals

    # For accessing the instances of a type, if the type exists
    try:
        return _types[name].instances  # type: ignore[misc]
    except KeyError as ke:
        raise AttributeError(
            f"AttributeError: module '{__name__}' has no attribute '{name}'"
        ) from ke
