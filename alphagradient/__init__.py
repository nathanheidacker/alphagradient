# -*- coding: utf-8 -*-
"""
The top-level AlphaGradient module for creating standard AlphaGradient 
objects, as well as interacting with the AlphaGradient global Environment 
and global API

Examples:
    .. highlight:: python
    .. code:: python
         
         import alphagradient as ag
         from datetime import datetime
         
         # Creating some AG objects
         spy = ag.Stock("SPY")
         portfolio = ag.Portfolio(10000)
         myEnv = ag.Environment(assets=[spy], portfolios=[portfolio])
         uni = ag.Universe()

         # Creating a new algorithm
         class MyAlgorithm(ag.Algorithm):
             def setup(**kwargs):
                 ...

             def cycle(**kwargs):
                 ... 

         algo = MyAlgorithm()
         backtest = algo(end="2022-01-03 16:00:00")

         # Global API interaction
         ag.globals.sync(datetime.today())

         # Going forward by one week
         ag.globals.step(7)
"""

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
    Type,
)


class _dtypes(tuple):
    """
    Controls access to the dtypes module by rerouting attribute access.
    """

    def __new__(cls, types_list: list[type], organized: dict[str, list]) -> _dtypes:
        return super().__new__(cls, tuple(types_list))  # type: ignore[arg-type]

    def __init__(self, to_tuple: list[type], organized: dict[str, list]) -> None:
        self.organized = organized

    def __getattr__(self, attr: str) -> Any:
        try:
            return getattr(_dtypes_module, attr)
        except AttributeError as e:
            try:
                return self.organized[attr]
            except KeyError:
                raise e


dtypes = _dtypes(_dtypes_module._alltypes, _dtypes_module._alltypes_organized)


def __getattr__(name: str) -> Any:
    if name == "types":
        return _types.to_list()

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
            f"AttributeError: module '{__name__}' " f"has no attribute '{name}'"
        ) from ke


# Organization of objects in All determines order in the documentation.
# Please make sure they are ordered correctly when adding new items
__all__ = [
    "Algorithm",
    "Asset",
    "Currency",
    "Stock",
    "Call",
    "Put",
    "Portfolio",
    "Cash",
    "Environment",
    "Universe",
]
