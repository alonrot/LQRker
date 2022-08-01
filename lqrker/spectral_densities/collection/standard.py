import tensorflow as tf
import pdb
import math
from abc import ABC, abstractmethod
import numpy as np
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
from lqrker.spectral_densities.base import SpectralDensityBase
logger = get_logger(__name__)


class SquaredExponentialSpectralDensity(SpectralDensityBase):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		super().__init__(cfg_sampler,dim)

		# Parameters:
		self.ls = cfg.ls*1.0

		# Constant parameters:
		self.const = (2.*math.pi*self.ls**2)**(self.dim/2)

	def unnormalized_density(self,omega_in,log=False):
		"""
		in: omega_in [Nfeat,dim]
		return: S_vec [Nfeat,]

		Using now the N-dimensional formulation from Rasmussen

		Outputs a vector 

		Matern kernel spectral density
		"""

		S_vec = self.const * tf.math.exp(-2.*(math.pi*self.ls*omega_in)**2)

		# Multiply dimensions (i.e., assume that matern density factorizes w.r.t omega input dimensionality). Also, pump variance into it:
		S_vec = tf.reduce_prod(S_vec,axis=1)

		if log == True:
			return tf.math.log(S_vec)

		return S_vec, 0.0

class MaternSpectralDensity(SpectralDensityBase):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		super().__init__(cfg_sampler,dim)

		# Parameters:
		self.nu = cfg.nu*1.0
		assert self.nu > 2.0, "nu must be > 2"
		self.ls = cfg.ls*1.0

		# Constant parameters:
		self.lambda_val = tf.sqrt(2*self.nu)/self.ls
		self.const = ((2*tf.sqrt(math.pi))**self.dim)*tf.exp(tf.math.lgamma(self.nu+0.5*self.dim))*self.lambda_val**(2*self.nu) / tf.exp(tf.math.lgamma(self.nu))

	def unnormalized_density(self,omega_in,log=False):
		"""
		in: omega_in [Nfeat,dim]
		return: S_vec [Nfeat,]

		Using now the N-dimensional formulation from Rasmussen

		Outputs a vector 

		Matern kernel spectral density
		"""

		S_vec = self.const / ((self.lambda_val**2 + omega_in**2)**(self.nu+self.dim*0.5)) # Using omega directly (Sarkka) as opposed to 4pi*s (rasmsusen)

		# Multiply dimensions (i.e., assume that matern density factorizes w.r.t omega input dimensionality). Also, pump variance into it:
		S_vec = tf.math.reduce_prod(S_vec,axis=1,keepdims=True)

		# Return copies of S(w), as many as input dimensions:
		S_vec = tf.concat([S_vec]*self.dim,axis=1)

		if log == True:
			return tf.math.log(S_vec)

		return S_vec, 0.0

	def _nonlinear_system_fun(self,x):
		return tf.zeros(x.shape)
