import tensorflow as tf
import pdb
import math

from abc import ABC, abstractmethod


class ObjectiveCostBase(ABC):
	"""

	Base class for scalar cost functions
	with additive noise
	"""

	def __init__(self,dim_in,sigma_n):

		self.dim_in = dim_in
		self.sigma_n = sigma_n # Additive noise

	@abstractmethod
	def evaluate(self,X,add_noise=True):
		return NotImplementedError