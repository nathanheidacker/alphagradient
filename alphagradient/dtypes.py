# -*- coding: utf-8 -*-
"""Miscellaneous AlphaGradient types not otherwise accessible"""

# Standard imports
from dataclasses import dataclass
import sys

# Local imports
from ._algorithm import (
    Performance, 
    Backtest, 
    Stats,
)
from ._data._datatools import AssetData
from ._finance._asset import (
    AssetDuplicationError as _AssetDuplicationError, 
    DataProtocol,
)
from ._finance._portfolio import (
    View, 
    Position,
)
from ._finance._standard import Option


# Getting this module as an object to assign values on
_self = sys.modules[__name__]

@dataclass
class errors:
    AssetDuplicationError = _AssetDuplicationError

__all__ = [
    'Performance',
    'Backtest',
    'Stats',
    'AssetData',
    'View',
    'Position',
    'Option'
]