# -*- coding: utf-8 -*-
"""

Todo:
    * Type Hints
    * Implement other standard types at the bottom
"""

# Standard imports
from __future__ import annotations

from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from yaml import load, Loader
from numbers import Number
from random import random
from pathlib import Path
import math

# Third party imports
import yfinance as yf
import pandas as pd
import numpy as np

# Local imports
from ._asset import Asset, types
from .._data import AssetData
from .. import utils

# Typing
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    Iterable,
    Optional,
    Type,
    cast,
)

from ..utils import DatetimeLike, PyNumber

if TYPE_CHECKING:
    from ._portfolio import (
        Portfolio,
        Position,
    )

Expiry = Union[DatetimeLike, float]
"""Acceptable types for an option expiry"""

settings: dict
with open(Path(__file__).parent.joinpath("standard_asset_settings.yml")) as f:
    settings = load(f, Loader)
"""settings (dict): dictionary of settings for initializing standard asset subclasses"""


def _getinfo(base: str = "USD") -> pd.DataFrame:
    """think this function has been replaced by 'currency_info' in datatools, probably needs to be deleted"""

    def mapper(code):
        """TODO"""
        try:
            ticker = yf.Ticker(f"{code}{base}=X")
            history = ticker.history(period="1d", interval="1d")
            return history["Close"].item()
        except:
            return -1

    info = pd.read_pickle("AlphaGradient/finance/currency_info_old.p")
    info["VALUE"] = info["CODE"].map(mapper)
    info = info[info["VALUE"] > 0]

    with open("AlphaGradient/finance/currency_info2.p", "wb") as p:
        info.to_pickle(p)

    return info


class Currency(Asset, settings=settings["CURRENCY"]):
    """Asset object representing a currency

    Controls how the values of other assets correspond to one another
    if their bases differ.

    Attributes:
        base (str): A string representing the global base currency
        info (pd.DataFrame): Information about all supported currencies
        code (str): The currency code of this currency instance
        symbol (str): The symbol that represents quantities of this
            currency
        is_base (bool): Whether or not this currency instance is the
            base currency instance

    """

    base = "USD"
    info = pd.read_pickle(Path(__file__).parent.joinpath("currency_info.p"))

    def __new__(cls, *args: Any, **kwargs: Any) -> Currency:
        modified_args: Iterable
        if not args:
            args = (None,)
        if args[0] is None:
            modified_args = list(args)
            modified_args[0] = cls.base
            args = tuple(modified_args)
        return cast(Currency, super().__new__(cls, *args, **kwargs))

    def __init__(self, code: str, **kwargs: Any) -> None:
        self.validate_code(code, error=True)
        self.code = code
        self.symbol = self.info[self.info["CODE"] == self.code]["SYMBOL"].item()
        super().__init__(code)

    def __str__(self) -> str:
        return f"<{self.type} {self.symbol}{self.code}: {self.price}>"  # type: ignore[attr-defined]

    def valuate(self, *args: Any, **kwargs: Any) -> float:
        """This asset valuates itself by converting its value relative
        to USD to the base's value relative to USD"""
        return self.convert(self.code, self.base)

    def online_data(self, *args: Any, **kwargs: Any) -> Optional[AssetData]:
        """Online data is available for forex from yfinance

        TODO: I dont think this should accept any args
        """
        if self.is_base:
            return None
        else:
            data = yf.download(
                f"{self.code}{self.base}=X",
                auto_adjust=False,
                timeout=5,
                progress=False,
            )
            return AssetData(Currency, data)

    @classmethod
    def get_symbol(cls, base: str = None) -> str:
        """Returns the symbol corresponding to a currency, given that
        currency's international code"""
        base = cls.base if base is None else base
        return cls.info[cls.info["CODE"] == base]["SYMBOL"].item()

    @classmethod
    def validate_code(cls, code: str, error: bool = False) -> bool:
        """Validates a currency code by seeing if it is supported

        Args:
            code (str): The code to validate
            error (bool): Whether or not an error should be raised if
                an invalid code is enountered

        Returns:
            valid (bool): Whether the code is valid or not

        Raises:
            ValueError: raised when the code is invalid and error=True
        """
        if code in list(cls.info["CODE"]):
            return True
        if error:
            raise ValueError(f"{code} is not a recognized as a valid currency code")
        return False

    @classmethod
    def code_value(cls, code: str) -> bool:
        """Returns a currency's value relative to the USD from
        Currency.info

        Args:
            code (str): The code of the currency who's value to return

        Returns:
            value (Number): The value of the currency
        """
        return cls.info[cls.info["CODE"] == code]["VALUE"].item()

    def convert(self, target: str, base: str = None) -> float:
        """Converts target currency to base currency

        Returns the value of the target currency in units of the base currrency

        Returns:
            value (Number): target currency value in base currency"""
        base = self.code if base is None else base
        return self.code_value(target) / self.code_value(base)

    def rate(self, target: str, base: str = None) -> float:
        """Converts this currency to the target

        Returns the value of the base currency in units of the target currency

        Returns:
            value (Number): base currency value in target currency"""
        return 1 / self.convert(target, base)

    @property
    def is_base(self) -> bool:
        return self.code == self.base


class Stock(Asset, settings=settings["STOCK"]):
    """A financial asset representing stock in a publicly traded company

    An asset representing a single share of a publicly traded company.
    Supports online data from yahoo finance using yfinance.

    Attributes:
        ticker (str): This stock's ticker
    """

    def valuate(self, *args: Any, **kwargs: Any) -> float:
        return self.value

    def online_data(self) -> AssetData:
        data = yf.download(self.name, auto_adjust=False, timeout=3, progress=False)
        return AssetData(Stock, data)

    @property
    def ticker(self) -> str:
        """Alias for name, relevant to stocks"""
        return self.name.upper()

    def call(self, strike: float, expiry: Expiry) -> Call:
        """Creates a call from this stock given the strike and expiry

        Args:
            strike (Number): The call's strike
            expiry (datetime): The call's expiry

        Returns:
            call (Call): A call who's underlying is this stock
        """
        return Call(self, strike, expiry)

    def put(self, strike: float, expiry: Expiry) -> Put:
        """Creates a put from this stock given the strike and expiry

        Args:
            strike (Number): The put's strike
            expiry (datetime): The put's expiry

        Returns:
            put (Put): A put who's underlying is this stock
        """
        return Put(self, strike, expiry)


class Option(Asset, ABC, settings=settings["OPTION"]):
    """An asset representing a option contract for a stock.

    An abstract base class representing an option contract. Implements
    black scholes valuation

    Attributes:
        underlying (Stock): A stock that underlies the option contract
        strike (Number): The strike price of the contract
        expiry (datetime): The expiration date of the contract
        spot (Number): The value of the underlying asset
        ttm (timedelta): The amount of time left until the contract
            reaches maturity, or expires
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Option:

        # Unpacking args and kwargs for underlying, strike, and expiry
        underlying = args[0] if args else kwargs["underlying"]
        strike = args[1] if args and len(args) > 1 else kwargs["strike"]
        expiry = args[2] if args and len(args) > 2 else kwargs["expiry"]

        # Coercing expiry into a datetime object
        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry)
        elif isinstance(expiry, int):
            expiry = underlying.date + timedelta(days=expiry)
        elif isinstance(expiry, timedelta):
            expiry = underlying.date + expiry

        # Coercion failed
        if not isinstance(expiry, datetime):
            raise TypeError(
                f"Invalid input {expiry=} for "
                f"initialization of {underlying.name} "
                f"{cls.__name__}"
            )
        elif expiry.weekday() > 4:
            raise ValueError(
                f"Invalid expiry on {utils.get_weekday(expiry)} for {underlying.name} {cls.__name__}. Valid expiry days are Monday-Friday"
            )

        # The name of the contract
        key = f"{underlying.name}{strike}{cls.__name__[0]}{expiry.date().__str__()}"

        return cast(Option, super().__new__(cls, key))

    def __init__(
        self, underlying: Asset, strike: float, expiry: Expiry, **kwargs: Any
    ) -> None:

        # Ensuring that the strike price is always a reasonable value
        # Max of one sig fig
        if isinstance(strike, Number):
            self.strike = round(strike, 1)
        else:
            raise TypeError(f"Inavlid non-numerical input for {strike}")

        #  Coercing expiry into a datetime object
        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry)
        elif isinstance(expiry, int):
            expiry = underlying.date + timedelta(days=expiry)  # type: ignore[attr-defined]
        elif isinstance(expiry, timedelta):
            expiry = underlying.date + expiry  # type: ignore[attr-defined]

        # Coercion failed
        if not isinstance(expiry, datetime):
            raise TypeError(
                f"Invalid input {expiry=} for "
                f"initialization of {underlying.name} "
                f"{self.__class__.__name__}"
            )
        else:
            expiry = utils.set_time(expiry, underlying.market_close)  # type: ignore[attr-defined]

        # Option-specific Attribute initialization
        self._expiry = expiry
        self.underlying = underlying
        self._mature = False

        super().__init__(self.key, date=underlying.date, base=underlying.base, **kwargs)  # type: ignore[attr-defined]

    @property
    def key(self) -> str:
        return (
            f"{self.underlying.name}{self.strike}"
            f"{self.__class__.__name__[0]}{self.expiry.date().__str__()}"
        )

    @property
    def expiry(self) -> datetime:
        return self._expiry

    @staticmethod
    def cdf(x) -> float:
        """The standard cumulative distribution functon"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def _bsbase(
        spot: float, strike: float, rfr: float, dy: float, ttm: float, vol: float
    ) -> tuple[float, float]:
        """initialization of black scholes d1 and d2 for option valuation

        Returns d1 and d2 components for purposes of black scholes
        options valuation

        Args:
            spot (Number): The price of the underlying stock
            strike (Number): This contract's strike price
            rfr (Number): The global risk-free-rate
            dy (Number): The underlying stock's dividend yield
            ttm (Number): The time to maturity in years
            vol (Number): The underlying stock's historical volatility
        """

        # Calculation of d1, d2
        ttm = ttm if ttm > 0 else 1
        d = vol * math.sqrt(ttm)
        d1 = math.log(spot / strike) + ((rfr - dy + ((vol * vol) / 2)) * ttm)
        d1 /= d
        d2 = d1 - d

        # Calculation of d1, d2
        d1 = (math.log(spot / strike) + ((rfr - dy + ((vol * vol) / 2)) * ttm)) / (
            vol * math.sqrt(ttm)
        )
        d2 = d1 - (vol * math.sqrt(ttm))

        return d1, d2

    @property
    def expired(self) -> bool:
        return self.date >= self.expiry  # type: ignore[attr-defined]

    @staticmethod
    def exact_days(delta: timedelta) -> float:
        """Given a timedelta, returns the exact number of days as a float

        TODO: THis should probably be a utils function
        """
        return delta.total_seconds() / 86400

    @property
    def ttm(self) -> timedelta:
        return max(self.expiry - self.date, timedelta())  # type: ignore[attr-defined]

    @property
    def spot(self) -> float:
        return self.underlying.value

    def reset(self) -> None:
        """Resets the option so that it may be valuated again, even if
        past expiry"""
        self._mature = False

    @property
    def atm(self) -> bool:
        """Whether or not this call is 'At the money', or within 1% of the spot price"""
        return abs(self.strike - self.spot) < (self.spot * 0.01)


class Call(Option, settings=settings["CALL"]):
    """An Asset representing a call option contract

    Attributes:
        itm (bool): Whether or not this contract is currently in
            the money
    """

    def valuate(self, date: datetime, *args: Any, **kwargs: Any) -> float:
        """Call option valuation"""

        # If the option has already reached maturity, the valuation
        # should never change
        if self._mature:
            return self.value

        # The final valuation of the option upon expiration
        elif self.expired:

            # Ensure this is the last valuation
            self._mature = True

            # Only want to valuate if the option is not worthless
            if self.itm:
                return (self.underlying.value - self.strike) * 100
            else:
                return 0

        # Normal valuation, the option is not expired
        else:
            return self.black_scholes(date)

    def black_scholes(self, date: datetime) -> float:
        """Returns the black scholes option pricing output for this call"""
        ttm = self.exact_days(self.ttm) / 365
        div_yield = 0.01

        d1, d2 = self._bsbase(
            spot=self.spot,
            strike=self.strike,
            rfr=self.rfr,  # type: ignore[attr-defined]
            dy=div_yield,
            ttm=ttm,
            vol=self.underlying.vol(),
        )

        p1 = self.cdf(d1) * self.spot * math.pow(math.e, -1 * (div_yield * ttm))
        p2 = self.cdf(d2) * self.strike * math.pow(math.e, -1 * (self.rfr * ttm))  # type: ignore[attr-defined]
        call_premium = p1 - p2

        # Each contract is for 100 shares
        return call_premium * 100

    def expire(self, portfolio: Portfolio, position: Position[Call]) -> None:
        """Expiration behavior for call option positions"""

        # Expired call positions are only assigned if they are ITM
        if not position.asset.itm:
            return

        # Creating the new positions that the portfolio will receive
        # in the case of assignment or exercise
        underlying = position.asset.underlying
        key = f"{underlying.key}_LONG"
        quantity = position.quantity * 100
        cost = quantity * position.asset.strike
        underlying_pos = portfolio._positions.get(key, None)
        call_pos = position.__class__(underlying, quantity)  # type: ignore[arg-type]
        call_pos.cost = cost

        # Shorts are automatically assigned if itm
        if position.short:

            # Selling the underlying at the strike if we have it availalble
            if underlying_pos and underlying_pos.quantity >= quantity:
                portfolio._positions[key] -= call_pos
                portfolio.cash += cost

            # Otherwise cover the position
            else:
                portfolio.cash += position.value

        # Longs are automatically exercised
        else:

            # Buy the underlying at the strike price if we can afford it
            if portfolio.cash > call_pos.value:
                if underlying_pos:
                    portfolio._positions[key] += call_pos
                else:
                    portfolio._positions[key] = call_pos
                portfolio.cash -= cost

            # Otherwise liquidate the position
            else:
                portfolio.cash += position.value

    @property
    def itm(self) -> bool:
        """Whether or not this call is 'in the money'"""
        return self.strike < self.spot

    @property
    def otm(self) -> bool:
        """Whether or not this call is 'out of the money'"""
        return not self.itm


class Put(Option, settings=settings["OPTION"]):
    """An Asset representing a put option contract

    Attributes:
        itm (bool): Whether or not this contract is currently in
            the money
    """

    def valuate(self, date: datetime, *args: Any, **kwargs: Any) -> float:
        """Put option valuation"""

        # If the option has already reached maturity, the valuation
        # should never change
        if self._mature:
            return self.value

        # The final valuation of the option upon expiration
        elif self.expired:

            # Ensure that this is the final valuation
            self._mature = True

            # Only valuate if the option is not worthless
            if self.itm:
                return (self.strike - self.underlying.value) * 100
            else:
                return 0

        # Normal valuation, not expired
        else:
            return self.black_scholes(date)

    def black_scholes(self, date: datetime) -> float:
        """Returns the black scholes option pricing output for this put"""
        ttm = self.exact_days(self.ttm) / 365
        div_yield = 0.01

        d1, d2 = self._bsbase(
            spot=self.spot,
            strike=self.strike,
            rfr=self.rfr,  # type: ignore[attr-defined]
            dy=div_yield,
            ttm=ttm,
            vol=self.underlying.vol(),
        )

        p1 = self.cdf(-1 * d2) * self.strike * math.pow(math.e, -1 * (self.rfr * ttm))  # type: ignore[attr-defined]
        p2 = self.cdf(-1 * d1) * self.spot * math.pow(math.e, -1 * (div_yield * ttm))
        put_premium = p1 - p2

        # Each contract is for 100 shares
        return put_premium * 100

    def expire(self, portfolio: Portfolio, position: Position[Put]) -> None:
        """Expiration behavior for put option positions"""

        # Expired put positions are only assigned if they are ITM
        if not position.asset.itm:
            return None

        # Creating the new positions that the portfolio will receive
        # in the case of assignment or exercise
        underlying = position.asset.underlying
        key = f"{underlying.key}_LONG"
        quantity = position.quantity * 100
        cost = quantity * position.asset.strike
        underlying_pos = portfolio._positions.get(key, None)
        put_pos = position.__class__(underlying, quantity)  # type: ignore[arg-type]
        put_pos.cost = cost

        # Shorts are automatically assigned if itm
        if position.short:

            # Buying the underlying at the strike if we can afford it
            if underlying_pos:
                portfolio._positions[key] += put_pos
                portfolio.cash -= cost

            # Otherwise cover the position
            else:
                portfolio.cash += position.value

        # Longs are automatically exercised
        else:

            # Sell the underlying at the strike price
            if underlying_pos and underlying_pos.quantity >= quantity:
                portfolio._positions[key] -= put_pos
                portfolio.cash += cost

            # Otherwise liquidate the position
            else:
                portfolio.cash += position.value

    @property
    def itm(self) -> bool:
        return self.strike > self.underlying.value

    @property
    def otm(self) -> bool:
        return not self.itm


# BROWNIAN STOCK IMPLEMENTATION BELOW

"""
class BrownianStock(Asset, settings=settings["BROWNIANSTOCK"]):
    def __init__(self, ticker=None, date=None, resolution=timedelta(days=1), base=None):
        self.resolution = resolution
        if ticker is None:
            ticker = self._generate_ticker()
            while ticker in TYPES.BROWNIANSTOCK.instances:
                ticker = self._generate_ticker()

        super().__init__(ticker, date=date, base=base)

        self.rng = np.random.default_rng()
        self.price = self.rng.gamma(1.5, 100)


    def _generate_ticker(self, random=False, last=[None]):
        if random:
            return ''.join([chr(np.random.randint(65, 91))
                                 for _ in range(4)])
        else:
            name = None

            if last[0] is None:
                name = [65, 65, 65]

            else:
                name = [ord(char) for char in last[0]]
                for i, char in enumerate(name):
                    if char < 91:
                        name[i] += 1
                        break
                    else:
                        name[i] = 65
                else:
                    name =  [65 for _ in range((len(name) + 1))]

            name = ''.join([chr(char) for char in name])
            last[0] = name
            return name


    def valuate(self, date=None):
        date = self.normalize_date(date)
        if self.date + self.resolution > date:
            #do something
            return self.price

        return self.price
"""


"""
TO BE IMPLEMENTED IN THE FUTURE
-- TODO --

class Commodity(Asset):
    def __init__(self, item):
        super().__init__(item)

    def _valuate(self):
        return NotImplemented

class Future(Asset):
    def __init__(self):
        raise NotImplementedError

class RealEstate(Asset):
    def __init__(self):
        raise NotImplementedError

class Crypto(Asset):
    def __init__(self):
        raise NotImplementedError

class Virtual(Asset):
    def __init__(self):
        raise NotImplementedError

class Unique(Asset):
    def __init__(self):
        raise NotImplementedError

"""
