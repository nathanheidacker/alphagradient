# -*- coding: utf-8 -*-
"""A proxy for the ag package for use in the algolibrary and typechecking purposes."""
# ==================================================================================
# TYPING SETUP
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from ._algorithm import Performance, Backtest, Stats, Algorithm
    from ._data._datatools import AssetData
    from ._finance._asset import AssetDuplicationError, DataProtocol, Asset
    from ._finance._collections import Environment, Universe
    from ._finance._portfolio import View, Position, Cash, Portfolio
    from ._finance._standard import Currency, Stock, Option, Call, Put
# ==================================================================================

# Local imports
from ._finance._asset import types as _types
from ._algorithm import Algorithm
from ._globals import __globals
from ._data import _datatools
from . import _finance
from . import utils


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
        return _types[name].instances
    except KeyError as ke:
        raise AttributeError(
            f"AttributeError: module '{__name__}' " f"has no attribute '{name}''"
        ) from ke
