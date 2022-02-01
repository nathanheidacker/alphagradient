from .asset import Asset, TYPES
from ..data import datatools
from abc import ABC, abstractmethod
import numpy as np
import math
from datetime import datetime, timedelta
import json

with open("AlphaGradient/finance/standard_asset_settings.json") as f:
	settings = json.load(f)

class Currency(Asset, settings=settings["CURRENCY"]):

	to_code = {
		'$': 'USD',
		'GBP': 'GBP',
		'YEN': 'YEN',
	}

	to_symbol = {v:k for k, v in to_code.items()}

	def __init__(self, identifier='USD'):

		if not isinstance(identifier, str):
			raise TypeError(
				f'''{identifier.__class__.__name__} {identifier} is not a valid currency identifier. 
				Please use a currency code or symbol as a string''')

		code = None
		symbol = None

		if len(identifier) == 1:
			code = self.to_code[identifier]
			symbol = identifier

		elif len(identifier) == 3:
			code = identifier
			symbol = self.to_symbol[identifier]

		else:
			raise ValueError(
				f'''{identifier} is not a valid currency identifier. 
				Please use a currency symbol or three digit currency code''')

		super().__init__(code, date=datetime.today(), require_data=False)

	def valuate(self):
		return NotImplemented


class Stock(Asset, settings=settings["STOCK"]):
	"""A financial asset representing stock in a publicly traded company

	More description here

	Attributes:
		attrs
	"""
	def valuate(self):
		return self.price

	def online_data(self):
		return datatools.from_yf(self.name)


class BrownianStock(Asset, settings=settings["BROWNIANSTOCK"]):
	def __init__(self, ticker=None, date=None):
		if ticker is None:
			ticker = 'AAAA'
			while ticker in TYPES.BROWNIANSTOCK.instances:
				ticker = ''.join([chr(np.random.randint(65, 91))
								 for _ in range(4)])

		super().__init__(ticker, date, require_data=False)

	def valuate(self):
		return NotImplemented


class Option(Asset, settings=settings["OPTION"]):

	def __init__(self, underlying, strike, expiry):
		super().__init__(underlying.name)

		if not isinstance(strike, (float, int)):
			try:
				strike = float(strike)
			except TypeError:
				raise f'''Invalid input type {strike=} 
				for initialization of {underlying.name} {self.__class__.__name__}'''
			except ValueError:
				raise f'''Unsuccessful conversion of {strike=} 
				to numeric type during initialization of {underlying.name} {self.type}'''
		self.strike = strike

		if isinstance(expiry, str):
			expiry = datetime.fromisoformat(expiry)
		elif isinstance(expiry, int):
			expiry = underlying.date + timedelta(days=expiry)
		elif isinstance(expiry, timedelta):
			expiry = underlying.date + expiry

		if not isinstance(expiry, datetime):
			raise TypeError(
				f'''Invalid input {expiry=} 
				for initialization of {underlying.name} {self.__class__.__name__}''')

		self.expiry = expiry

	def _black_scholes(self, spot, strike, rfr, dy, ttm, vol):
		'''initialization of black scholes d1 and d2 for option valuation'''

		# Standard cumulative distribution function
		def cdf(x):
			return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

		# Calculation of d1, d2
		d1 = (math.log(spot / strike) +
			  ((rfr - dy + ((vol * vol) / 2)) * ttm)) / (vol * math.sqrt(ttm))
		d2 = d1 - (vol * math.sqrt(ttm))

		return d1, d2


class Call(Option, settings=settings["CALL"]):
	def __init__(self, underlying, strike, expiry):
		super().__init__(underlying, strike, expiry)

	def valuate(self):
		return NotImplemented


class Put(Option, settings=settings["OPTION"]):
	def __init__(self):
		raise NotImplementedError

	def valuate(self):
		return NotImplemented


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
