# -*- coding: utf-8 -*-
"""
Standard AlphaGradient Asset type implementations including Stocks, Currency,
Call and Put Option Contracts, etc.

TODO: Implement other standard asset types at the bottom
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
    """
    Asset object representing a currency

    Controls how the values of other assets correspond to one another
    if their bases differ. All Assets, Portfolios, and Environments utilize
    a base currency object that defines the default representation of prices.
    Assets and Portfolios may have explicitly specified currency bases that
    differ from their respective environment, but will default to that of their
    environment if none is specified during intantiation.

    For Assets, the base currency defines how Asset data is interpreted. For
    Portfolios, the base currency defines how Positions are created and
    represented. All financial positions in a portfolio are represented in that
    portfolio's base currency, even if the asset underlying the position uses
    a different base currency.

    Currencies are assets just like any other, so their relative values change
    across time just as they would in real life. The exchange between currencies
    is performed with the exchange rate the day of the transaction, rather
    than being constant across time.

    As an example, consider an asset with a base currency 'JPY', or japanese
    yen. When a Portfolio whose base is 'USD' purchases 100 units of this asset,
    the position will automatically be converted to USD using the rate **at the
    current date of the transaction**

    For the sake of clarity in the example below, assume the exchange rate
    between the Japanese Yen and the United States Dollar is 100:1 (100 Yen for
    1 USD)

    Examples:

        .. code:: python

            import alphagradient as ag

            class JapaneseAsset(ag.Asset):

                def valuate(date):
                    return self.value  # does nothing

            # Creating an intance of JapaneseAsset whose value is a constant
            # 35,000 Yen, which (for the sake of this example) is 350 USD
            jpyasset = JapaneseAsset(name="TEST", data=35000, base="JPY")
            portfolio = ag.Portfolio(100_000)
            portfolio.buy(jpyasset, 200)
            jpypos = portfolio.get_position(jpyasset)

        .. code:: pycon

            >>> jpyasset
            <JAPANESEASSET TEST: ¥35000 /unit>

            >>> jpypos
            <200 JAPANESEASSET_TEST units @ $350 /unit | VALUE: $70000.0 | RETURN: 0.0%>

            >>> jpypos.asset is jpyasset
            True

            >>> portfolio.cash
            <CASH: USD $30000>

    Parameters:
        code:
            The international currency code of the desired currency to instantiate

    Attributes:
        base (str):
            A string representing the global base currency

        info (pd.DataFrame):
            Information about all supported currencies

        code (str):
            The currency code of this currency instance

        symbol (str):
            The symbol that represents quantities of this currency

        is_base (bool):
            Whether or not this currency instance is the base currency instance
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

    @property
    def is_base(self) -> bool:
        return self.code == self.base

    def valuate(self, *args: Any, **kwargs: Any) -> float:
        """This asset valuates itself by converting its value relative
        to USD to the base's value relative to USD"""
        return self.convert(self.code, self.base)

    def online_data(self) -> Optional[AssetData]:
        """
        Online data is available for forex from yfinance

        TODO: I dont think this should accept any args, and in the future should
        provide a directly implementation so that we dont have to rely on
        yfinance.

        Returns:
            The appropriate forex data for this currency using yfinance
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
    def code_value(cls, code: str) -> bool:
        """
        Returns a currency's value relative to the USD from Currency.info

        Parameters:
            code:
                The international currency code of the currency who's value
                to return

        Returns:
            The value of the currency relative to USD
        """
        return cls.info[cls.info["CODE"] == code]["VALUE"].item()

    def convert(self, target: str, base: str = None) -> float:
        """
        Converts target currency to base currency

        Returns the value of the target currency in units of the base currrency

        Parameters:
            target:
                The target currency to convert to

            base:
                The base currency from which target is being evaluated

        Returns:
            The target currency's value in base currency
        """
        base = self.code if base is None else base
        return self.code_value(target) / self.code_value(base)

    @classmethod
    def get_symbol(cls, base: str = None) -> str:
        """Returns the symbol corresponding to a currency, given that
        currency's international code"""
        base = cls.base if base is None else base
        return cls.info[cls.info["CODE"] == base]["SYMBOL"].item()

    def rate(self, target: str, base: str = None) -> float:
        """
        Converts this currency to the target

        Returns the value of the base currency in units of the target currency

        Returns the inverse of the convert function.

        Parameters:
            target:
                The amount of this currency equal to a single unit of the base

            base:
                The currency whos going rate is desired


        Returns:
            The going rate for the base currency in the target currency
        """
        return 1 / self.convert(target, base)

    @classmethod
    def validate_code(cls, code: str, error: bool = False) -> bool:
        """
        Validates a currency code by seeing if it is supported

        Parameters:
            code:
                An international currency code; the code to validate

            error:
                Whether or not an error should be raised if an invalid code
                is enountered

        Returns:
            Whether the code is valid or not

        Raises:
            ValueError:
                When the code is invalid and error=True
        """
        if code in list(cls.info["CODE"]):
            return True
        if error:
            raise ValueError(f"{code} is not a recognized as a valid currency code")
        return False


class Stock(Asset, settings=settings["STOCK"]):
    """
    A financial asset representing stock in a publicly traded company

    An asset representing a single share of a publicly traded company.
    Supports online data from yahoo finance using yfinance.

    Parameters:
        ticker:
            A string of uppercase letters corresponding to the stock's ticker

        data (ValidData):
            The data input for the stock

    Attributes:
        ticker (str):
            This stock's ticker
    """

    @property
    def ticker(self) -> str:
        """Alias for name, relevant to stocks"""
        return self.name.upper()

    def valuate(self, *args: Any, **kwargs: Any) -> float:
        return self.value

    def online_data(self) -> AssetData:
        data = yf.download(self.name, auto_adjust=False, timeout=3, progress=False)
        return AssetData(Stock, data)

    def call(self, strike: float, expiry: Expiry) -> Call:
        """
        Creates a call from this stock given the strike and expiry

        Parameters:
            strike:
                The call's strike

            expiry (Expiry):
                The call's expiry

        Returns:
            A call who's underlying asset is this stock
        """
        return Call(self, strike, expiry)

    def put(self, strike: float, expiry: Expiry) -> Put:
        """
        Creates a put from this stock given the strike and expiry

        Parameters:
            strike:
                The put's strike

            expiry (Expiry):
                The put's expiry

        Returns:
            A put who's underlying asset is this stock
        """
        return Put(self, strike, expiry)


class Option(Asset, ABC, settings=settings["OPTION"]):
    """
    An asset representing a option contract for a stock.

    An abstract base class representing an option contract. Implements
    black scholes valuation

    Not directly instantiable. Used as a base class for both Call and Put
    implementations

    Attributes:
        underlying (Stock):
            A stock that underlies the option contract

        strike (float):
            The strike price of the contract

        expiry (datetime):
            The expiration date of the contract

        spot (float):
            The value of the underlying asset

        ttm (timedelta):
            The amount of time left until the contract reaches maturity, or
            expires
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
                f"Invalid expiry on {utils.get_weekday(expiry)} for "
                f"{underlying.name} {cls.__name__}. Valid expiry days are "
                "Monday-Friday"
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

        super().__init__(self.key, base=underlying.base, **kwargs)  # type: ignore[attr-defined]

    @property
    def atm(self) -> bool:
        """Whether or not this call is 'At the money', or within 1% of the spot price"""
        return abs(self.strike - self.spot) < (self.spot * 0.01)

    @property
    def expired(self) -> bool:
        return self.date >= self.expiry  # type: ignore[attr-defined]

    @property
    def expiry(self) -> datetime:
        return self._expiry

    @property
    def key(self) -> str:
        return (
            f"{self.underlying.name}{self.strike}"
            f"{self.__class__.__name__[0]}{self.expiry.date().__str__()}"
        )

    @property
    def spot(self) -> float:
        return self.underlying.value

    @property
    def ttm(self) -> timedelta:
        return max(self.expiry - self.date, timedelta())  # type: ignore[attr-defined]

    @staticmethod
    def _bsbase(
        spot: float, strike: float, rfr: float, dy: float, ttm: float, vol: float
    ) -> tuple[float, float]:
        """
        Initialization of black scholes d1 and d2 for option valuation

        Returns d1 and d2 components for purposes of black scholes
        options valuation

        Parameters:
            spot:
                The price of the underlying stock

            strike:
                This contract's strike price

            rfr:
                The global risk-free-rate

            dy:
                The underlying stock's dividend yield

            ttm:
                The time to maturity in years

            vol:
                The underlying stock's historical volatility

        Returns:
            black scholes d1 and d2 parameters
        """

        # NEW Calculation of d1, d2
        """
        ttm = ttm if ttm > 0 else 1
        d = vol * math.sqrt(ttm)
        d1 = math.log(spot / strike) + ((rfr - dy + ((vol * vol) / 2)) * ttm)
        d1 /= d
        d2 = d1 - d
        """

        # Calculation of d1, d2
        d1 = (math.log(spot / strike) + ((rfr - dy + ((vol * vol) / 2)) * ttm)) / (
            vol * math.sqrt(ttm)
        )
        d2 = d1 - (vol * math.sqrt(ttm))

        return d1, d2

    @staticmethod
    def cdf(x) -> float:
        """The standard cumulative distribution functon"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def exact_days(delta: timedelta) -> float:
        """Given a timedelta, returns the exact number of days as a float

        TODO: THis should probably be a utils function
        """
        return delta.total_seconds() / 86400

    def reset(self) -> None:
        """Resets the option so that it may be valuated again, even if
        past expiry"""
        self._mature = False


class Call(Option, settings=settings["CALL"]):
    """
    An Asset representing a call option contract

    Attributes:
        itm (bool):
            Whether or not this contract is currently in the money

        atm (bool):
            Whether or not this contract is currently at the money

        otm (bool):
            Whether or not this contract is currenct out of the money
    """

    @property
    def itm(self) -> bool:
        """Whether or not this call is 'in the money'"""
        return self.strike < self.spot

    @property
    def otm(self) -> bool:
        """Whether or not this call is 'out of the money'"""
        return not self.itm

    def valuate(self, date: datetime, *args: Any, **kwargs: Any) -> float:
        """
        Call option valuation using black scholes

        Parameters:
            date:
                The datetime to evaluate at

        Returns:
            The value of this call contract at the specified datetime, using
            the black scholes option pricing model
        """
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
        """
        Returns the black scholes option pricing output for this call

        Parameters:
            date:
                The date to evaluate at

        Returns:
            This call contract's value defined by the black scholes option
            pricing model on the given datetime
        """
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
        """
        Expiration behavior for call option positions

        This method is for interal use only and should not be called directly
        """
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


class Put(Option, settings=settings["OPTION"]):
    """
    An Asset representing a put option contract

    Attributes:
        itm (bool):
            Whether or not this contract is currently in the money

        atm (bool):
            Whether or not this contract is currently at the money

        otm (bool):
            Whether or not this contract is currenct out of the money
    """

    @property
    def itm(self) -> bool:
        return self.strike > self.underlying.value

    @property
    def otm(self) -> bool:
        return not self.itm

    def valuate(self, date: datetime, *args: Any, **kwargs: Any) -> float:
        """
        Put option valuation using black scholes

        Parameters:
            date:
                The datetime to evaluate at

        Returns:
            The value of this put contract at the specified datetime, using
            the black scholes option pricing model
        """
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
        """
        Returns the black scholes option pricing output for this put

        Parameters:
            date:
                The date to evaluate at

        Returns:
            This put contract's value defined by the black scholes option
            pricing model on the given datetime
        """
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
        """
        Expiration behavior for put option positions

        This method is for internal use only and should not be called directly.
        """

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
