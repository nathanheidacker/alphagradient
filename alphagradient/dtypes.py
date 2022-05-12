# -*- coding: utf-8 -*-
"""Miscellaneous AlphaGradient types not otherwise accessible"""

# Standard Imports
from dataclasses import dataclass
from datetime import date, time, datetime
from pathlib import Path
import sys

# Third Party Imports
import pandas as pd
import numpy as np

# Local Imports
from ._algorithm import (
    Performance,
    Backtest,
    Stats,
)
from ._data._datatools import AssetData
from ._finance._asset import (
    AssetDuplicationError as AssetDuplicationError,
    DataProtocol,
    Asset,
)
from ._finance._portfolio import (
    PositionView,
    Position,
    Portfolio,
)
from ._finance._standard import (
    Option,
    Stock,
)

from ._finance._collections import (
    UniverseView,
    Filter,
    FilterExpression,
)

# Typing
from typing import (
    TypeVar,
    Literal,
    Union,
    Dict,
    Tuple,
    Iterable,
)

# UTILS ALIASES
DatetimeLike = Union[pd.Timestamp, np.datetime64, date, datetime, str]
"""Objects convertable to python datetimes"""

TimeLike = Union[time, str]
"""Objects convertable to python time objects"""

DateOrTime = Union[DatetimeLike, time]
"""Objects that are either DatetimeLike or TimeLike in nature"""


# DATATOOLS ALIASES
ValidData = Union[Path, str, AssetData, pd.DataFrame, np.ndarray, float]
"""Valid inputs for AssetData conversion"""


# COLLECTIONS ALIASES
TimeLike_T = TypeVar("TimeLike_T", bound=TimeLike)
"""A TimeLike TypeVar"""

Bindable = Union[Asset, Portfolio]
"""AG Objects that can be bound to environments"""

Trackable = Union[Iterable[Bindable], Bindable]
"""Objects acceptable for Environment's 'track' functionality"""

ValidFilter = Union[FilterExpression, dict[str, Stock], list[str], list[Stock]]
"""Objects that are valid filters for Universes"""


# PORTFOLIO ALIASES
Asset_T = TypeVar("Asset_T", bound=Asset, covariant=True)
"""A TypeVar for an Asset type"""


# STANDARD ALIASES
Expiry = Union[DatetimeLike, float]
"""Acceptable types for an option expiry"""


# ALGORITHM TYPES
Changes = Dict[str, tuple[float, float]]
"""Changes in a portfolio's positions across a time period"""

ChangesWithInfo = Tuple[Changes, pd.Timestamp, pd.Timestamp]
"""Changes, but with information about the relevant time period"""

Max = Literal["max"]
"""Shortcut alias for literal 'max'"""


# Getting this module as an object to assign values on
_self = sys.modules[__name__]


@dataclass
class errors:
    AssetDuplicationError = AssetDuplicationError


_alltypes: list[type] = [
    Performance,
    Backtest,
    Stats,
    AssetData,
    DataProtocol,
    PositionView,
    Position,
    Option,
]

_alltypes_organized: dict[str, list] = {
    "stats": [Performance, Backtest, Stats],
    "data": [AssetData, DataProtocol],
    "finance": [PositionView, Position, UniverseView],
    "standard": [Option],
    "typing": [
        DatetimeLike,
        TimeLike,
        DateOrTime,
        ValidData,
        TimeLike_T,
        Bindable,
        Trackable,
        ValidFilter,
        Asset_T,
        Expiry,
        Changes,
        ChangesWithInfo,
        Max,
    ],
}

_autodoc_aliases: list[str] = [
    "DatetimeLike",
    "TimeLike",
    "DateOrTime",
    "ValidData",
    "TimeLike_T",
    "Bindable",
    "Trackable",
    "ValidFilter",
    "Asset_T",
    "Expiry",
    "Changes",
    "ChangesWithInfo",
]

# These are the types that will show up in the documentation under dtypes
# Organization of objects in All determines order in the documentation.
# Please make sure they are ordered correctly when adding new items
__all__ = [
    # Errors and Exceptions
    "AssetDuplicationError",
    # Typing Types, Type Aliases
    "DatetimeLike",
    "TimeLike",
    "DateOrTime",
    "ValidData",
    "TimeLike_T",
    "Bindable",
    "Trackable",
    "ValidFilter",
    "Asset_T",
    "Expiry",
    "Changes",
    "ChangesWithInfo",
    "Max",
    # Actual Types
    "DataProtocol",
    "AssetData",
    "Backtest",
    "Option",
    "Performance",
    "Position",
    "Stats",
    "PositionView",
    # Collections
    "UniverseView",
    "Filter",
    "FilterExpression",
]
