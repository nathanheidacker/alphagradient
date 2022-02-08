# -*- coding: utf-8 -*-
"""AG module containing algorithm class

Todo:
	* Implement algorithms
"""

# Standard imports
from abc import ABC, abstractmethod

# Local imports


class Algorithm(ABC):

	def __init__(self):
		pass

	def __call__(self, *args, **kwargs):
		self.run(*args, **kwargs)

	@abstractmethod
	def run(*args, **kwargs):
		pass

	def intitialize_inputs(portfolio, start_date, end_date):
		pass
