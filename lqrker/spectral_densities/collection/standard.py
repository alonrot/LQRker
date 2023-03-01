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

		# Compatibility with nonlinearsys
		self.xdata = -1.0

	def unnormalized_density(self,omega_in,log=False):
		"""
		in: omega_in [Nfeat,dim]
		return: S_vec [Nfeat,]

		# This density corresponds to a stationary kernel. Hence, it depends on the L2 norm of the frequencies, i.e.,
		# S(omega) = S(||omega||^2)

		Using formulation [1], which depends on omega directly as opposed to 2*pi*f, presented in [2].


		[1] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.
		[2] Williams, C.K. and Rasmussen, C.E., 2006. Gaussian processes for machine learning (Vol. 2, No. 3, p. 4). Cambridge, MA: MIT press.
		"""

		# This density corresponds to a stationary kernel. Hence, its input is the L2 norm of the frequencies:
		omega_in_L2_squared = tf.math.reduce_sum(omega_in**2,axis=1,keepdims=True) # L2 norm, squared, [Nfeat,1]
		S_vec = self.const * tf.math.exp(-2.*(math.pi*self.ls)**2 * omega_in_L2_squared) # [Nfeat,1]

		# When modeling N-dimensional process, we get one spectral density per channel. However, because matern kernels depend
		# only on the L2 norm of the vector of frequencies, each channel has the same spectral density. Hence, we return copies of S(w), as many as input dimensions:
		S_vec = tf.concat([S_vec]*self.dim,axis=1)

		if log == True:
			return tf.math.log(S_vec)

		return S_vec, tf.zeros((1,self.dim))

	def _nonlinear_system_fun(self,x):
		return tf.zeros(x.shape)

	def update_dX_voxels(self,dX_new):
		# Compatibility with nonlinearsys
		return None


class MaternSpectralDensity(SpectralDensityBase):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		super().__init__(cfg_sampler,dim)

		# Parameters:
		self.nu = cfg.nu*1.0
		assert self.nu > 2.0, "nu must be > 2"
		self.ls = cfg.ls*1.0

		# Constant parameters:
		self.lambda_val = tf.sqrt(2*self.nu)/self.ls

		# self.const = ((2*tf.sqrt(math.pi))**self.dim)*tf.exp(tf.math.lgamma(self.nu+0.5*self.dim))*self.lambda_val**(2*self.nu) / tf.exp(tf.math.lgamma(self.nu))

		self.log_const = self.dim * tf.math.log(2*tf.sqrt(math.pi)) + tf.math.lgamma(self.nu+0.5*self.dim) + (2*self.nu)*self.lambda_val - tf.math.lgamma(self.nu)

	def unnormalized_density(self,omega_in,log=False):
		"""
		in: omega_in [Nfeat,dim]
		return: S_vec [Nfeat,]

		# This density corresponds to a stationary kernel. Hence, it depends on the L2 norm of the frequencies, i.e.,
		# S(omega) = S(||omega||^2)

		Using formulation [1], which depends on omega directly as opposed to 2*pi*f, presented in [2].


		[1] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.
		[2] Williams, C.K. and Rasmussen, C.E., 2006. Gaussian processes for machine learning (Vol. 2, No. 3, p. 4). Cambridge, MA: MIT press.
		"""


		# # NON-log form (doesn't work when self.dim is very large, i.e., self.dim > 50)

		# # This density corresponds to a stationary kernel. Hence, its input is the L2 norm of the frequencies:
		omega_in_L2_squared = tf.math.reduce_sum(omega_in**2,axis=1,keepdims=True) # L2 norm, squared, [Nfeat,1]
		# S_vec = self.const / ((self.lambda_val**2 + omega_in_L2_squared)**(self.nu+self.dim*0.5)) # Using omega directly (Sarkka) as opposed to 4pi*s (rasmsusen)


		# log form:
		log_S_vec = self.log_const - (self.nu+self.dim*0.5)*tf.math.log(self.lambda_val**2 + omega_in_L2_squared)
		

		if log == True:
			# log_S_vec = tf.math.log(S_vec)
			log_S_vec = tf.concat([log_S_vec]*self.dim,axis=1)
			return log_S_vec, tf.zeros((1,self.dim))
		else:
			# When modeling N-dimensional process, we get one spectral density per channel. However, because matern kernels depend
			# only on the L2 norm of the vector of frequencies, each channel has the same spectral density. Hence, we return copies of S(w), as many as input dimensions:
			S_vec = tf.math.exp(log_S_vec)
			S_vec = tf.concat([S_vec]*self.dim,axis=1)


		return S_vec, tf.zeros((1,self.dim))

	def _nonlinear_system_fun(self,x):
		return tf.zeros(x.shape)
