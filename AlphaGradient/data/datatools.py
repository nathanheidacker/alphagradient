import pandas as pd
import numpy as np
from ..constants import VERBOSE, is_numeric
from typing import List
from aenum import Enum, NoAlias, auto, extend_enum, skip, unique
from datetime import datetime
import pickle
import weakref
from collections import namedtuple
from os import scandir

class Ledger(pd.DataFrame):

	instance = lambda: None

	def __new__(cls, *args):
		if cls.instance() is not None:
			return cls.instance()
		return super().__new__(cls)


	def __init__(self, data=None):
		# Try to access a ledger if one already exists
		try:
			data = data if data is not None else pd.read_pickle('AlphaGradient/data/ledger')

			if not isinstance(data, (Ledger, pd.DataFrame)):
				raise TypeError('Invalid input for Ledger')

			super().__init__(data)

		# Otherwise, create a new one
		except FileNotFoundError:
			data = pd.DataFrame(columns=['ID', 'TYPE', 'NAME', 'STATUS', 'DATE'])
			super().__init__(data)
			self.auto_update()

		Ledger.instance = lambda: self

	def append(self, *args):
		data = super().append(*args)
		return Ledger(data)

	def to_pickle(self):
		pd.to_pickle(self, 'AlphaGradient/data/ledger')

	def auto_update(self):

		for raw_file in scandir('AlphaGradient/data/raw'):
			name = raw_file.name.split('.')[0]
			data = self.loc[self['ID'] == name]
			if data.empty:
				asset_type, asset_name = self.id_info(name)
				entry = pd.DataFrame([[name, asset_type, asset_name, 1, datetime.today()]], columns=['ID', 'TYPE', 'NAME', 'STATUS', 'DATE'])
				self = self.append(entry)
			else:
				index = data.index.item()
				self.at[index, 'STATUS'] = 1
				self.at[index, 'DATE'] = datetime.today()

		for pickle in scandir('AlphaGradient/data/pickles'):
			data = self.loc[self['ID'] == pickle.name]
			if data.empty:
				asset_type, asset_name = self.id_info(pickle.name)
				entry = pd.DataFrame([[pickle.name, asset_type, asset_name, 2, datetime.today()]], columns=['ID', 'TYPE', 'NAME', 'STATUS', 'DATE'])
				self = self.append(entry)
			else:
				index = data.index.item()
				self.at[index, 'STATUS'] = 2
				self.at[index, 'DATE'] = datetime.today()

		self.to_pickle()

	def update(self, data):
		pass

	def update_entry(self):
		pass

	def add_entry(self):
		pass

	@staticmethod
	def id(asset_type, asset_name):
		return f'{asset_type}_{asset_name}'.strip().upper()

	@staticmethod
	def id_info(ledger_id):
		# Decompose the ID
		info = ledger_id.split('_', 1)

		# Split into relevant information
		asset_type = info[0]
		asset_name = info[1]

		return asset_type, asset_name

	def get_status(self, ledger_id, asset_name=None):
		if asset_name is not None:
			ledger_id = self.id(ledger_id, asset_name)

		entry = self.loc[self['ID'] == ledger_id]

		if not entry.empty:
			return entry['STATUS'].item()

		return 0


def from_pickle(asset_type, asset_name, ledger=None):
	# Check if an acceptable ledger is passed in
	if not isinstance(ledger, Ledger):
		ledger = Ledger()

	# Accessing the entry for this asset
	ledger_id = ledger.id(asset_type, asset_name)
	status = ledger.get_status(ledger_id)

	if status > 1:
		try:
			return pd.read_pickle(f'AlphaGradient/data/pickles/{ledger_id}')

		except FileNotFoundError:
			if status == 2:
				print('update ledger!')
			return None

	return None


def from_raw(asset_type, asset_name, ledger=None):
	# Check if an acceptable ledger is passed in
	if not isinstance(ledger, Ledger):
		ledger = Ledger()

	# Accessing the entry for this asset
	ledger_id = ledger.id(asset_type, asset_name)
	status = ledger.get_status(ledger_id)

	if status > 0:
		try:
			return AssetData(pd.read_csv(f'AlphaGradient/data/raw/{ledger_id}'))

		except FileNotFoundError:
			if status == 1:
				print('update ledger!')
			return None

	return None


def from_yf(tickers):
	if not isinstance(tickers, list):
		tickers = [tickers]
	for ticker in tickers:
		pass

	return None


def get_data(asset_type, asset_name, ledger=None):
	if not isinstance(ledger, Ledger):
		ledger = Ledger()

	ledger_id = ledger.id(asset_type, asset_name)
	status = ledger.get_status(ledger_id)

	data = None

	if status > 1:
		data = from_pickle(asset_type, asset_name)
		status = status - 1 if data is None else status

	if status < 1:
		data = from_raw(asset_type, asset_name)

	return data


# This is a little bit hacky, but this this needs to be defined outside of the scope of AssetData even though it is only intended to be used in that class. This is because the COLUMNS enum defined within will not allow the use of subclasses as values for enum members. By defining it outside, we can use Value within the COLUMNS enum scope, allowing us to bypass the requirement that all values be the same. Ordinarily, we could just use 'settings=NoAlias', but it imposes too many restrictions when loading saved asset datasets from pickles.
Value = namedtuple('Value', 'type name')

class AssetData(pd.DataFrame):

	class COLUMNS(Enum):

		DATE = Value('datetime64', 'DATE')
		OPEN = Value('float', 'OPEN')
		CLOSE = Value('float', 'CLOSE')

		def __str__(self):
			return self.name

		def __repr__(self):
			return f'\'{self.name}\''

		@property
		def type(self):
			return self.value[0]

	static = False

	def __init__(self, data=None, required=None, optional=None):

		# Check for column requirements
		required = required if required else {}
		optional = optional if optional else {}

		def column_format(column):
			return column.replace(' ', '_').upper()

		# Converting required and optional to dicts
		def to_dict(columns):
			if isinstance(columns, list):
				columns = [column_format(column) for column in columns]
				columns = {column:Value('float', column) for column in columns}

			elif isinstance(columns, dict):
				columns = {column_format(column):Value(columns[column], column_format(column)) for column in columns}

			return columns

		required = to_dict(required)
		optional = to_dict(optional)

		# Update COLUMNS enum to accomodate requirements
		columns = [self.DATE, self.OPEN, self.CLOSE]

		def extend_columns(columns):
			enums = [column.name for column in self.COLUMNS]
			columns = {k.replace(' ', '_').upper():v for k,v in columns.items()}
			for name in columns:
				if name not in enums:
					extend_enum(self.COLUMNS, name, columns[name])

		extend_columns(required)
		extend_columns(optional)

		# Updating columns to contain requirements
		columns += [self.COLUMNS[name] for name in required]

		# Handling AssetData inputs
		if isinstance(data, AssetData):
			if all([column in data.columns for column in required]):
				super().__init__(data)
				return
			else:
				raise ValueError('Dataset missing required columns')

		# Handling NoneType inputs
		data = 0 if data is None else data

		# Handling inputs that will result in single row dataframes
		if is_numeric(data):
			self.static = True
			data = pd.DataFrame([[datetime.today()] + [data] * (len(columns) - 1)], columns=columns, dtype=float)

		# Handling non DataFrame inputs, checking for required columns
		else:
			if isinstance(data, str):
				data = pd.read_csv(data)
			elif isinstance(data, np.ndarray):
				data = pd.DataFrame(data)

			# Converting all columns to enums, dropping all columns which do not exist in the enumeration
			to_convert = []
			to_drop = []
			data.columns = [column.replace(' ', '_').upper() for column in data.columns]
			available = [column.name for column in self.COLUMNS]
			for column in data.columns:
				if column in available:
					to_convert.append(self.COLUMNS[column])
				else:
					to_drop.append(column)
			data.drop(to_drop, axis=1, inplace=True)
			data.columns = to_convert

			# Verifying existence of all required columns
			if not all([column in data.columns for column in columns]):
				raise ValueError(f'Dataset missing required columns')

			# Verifying column-specific dtypes
			data = data.astype({column:column.type for column in data.columns})

		# Making the dataset datetime indexed, removing duplicate column
		if self.static or not data.empty:
			data = data.set_index(pd.DatetimeIndex(data[self.DATE]))
			data.drop(self.DATE, axis=1, inplace=True)

		super().__init__(data=data)

	# The limited scope of asset datasets allows the explicit definition of boolean conversions
	def __bool__(self):
		return not (self.empty or self.static)

	# Allows the user to get COLUMN enum members as though they were attributes of the dataset
	def __getattr__(self, attr):
		try:
			return self.COLUMNS[attr]
		except:
			return super().__getattr__(attr)

	# Allows the user to use strings to dynamically access the enum column names
	def __getitem__(self, item):
		try:
			item = self.COLUMNS[item]
		except:
			pass
		return super().__getitem__(item)

	# Allows the user to dynamically access column enum members
	def get_column(self, column):
		try:
			return self.COLUMNS[column.replace(' ', '_').upper()]
		except KeyError:
			return None
