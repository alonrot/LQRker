import tensorflow as tf
import pdb
import math
from abc import ABC, abstractmethod
import numpy as np
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
from lqrker.spectral_densities.spectral_density_base import SpectralDensityBase
logger = get_logger(__name__)


class SquaredExponentialSpectralDensity(SpectralDensityBase):

	def __init__(self,cfg: dict, dim: int):
		super().__init__(cfg.samplerHMC,dim)

		# Parameters:
		self.ls = cfg.ls*1.0
		self.prior_var = cfg.prior_var*1.0

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
		S_vec = tf.reduce_prod(S_vec,axis=1)*self.prior_var

		if log == True:
			return tf.math.log(S_vec)

		return S_vec, 0.0

	def update_pars(self,args):
		pass
		# raise NotImplementedError("Child Class MaternSpectralDensity")

	def normalized_density(self):
		raise NotImplementedError("Child Class MaternSpectralDensity")

	def log_normalized_density(self):
		raise NotImplementedError("Child Class MaternSpectralDensity")


class MaternSpectralDensity(SpectralDensityBase):

	def __init__(self,cfg: dict, dim: int):
		super().__init__(cfg.samplerHMC,dim)

		# Parameters:
		self.nu = cfg.nu*1.0
		assert self.nu > 2.0, "nu must be > 2"
		self.ls = cfg.ls*1.0
		self.prior_var = cfg.prior_var*1.0

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
		S_vec = tf.reduce_prod(S_vec,axis=1)*self.prior_var

		if log == True:
			return tf.math.log(S_vec)

		return S_vec, 0.0

	def update_pars(self,args):
		pass
		# raise NotImplementedError("Child Class MaternSpectralDensity")

	def normalized_density(self):
		raise NotImplementedError("Child Class MaternSpectralDensity")

	def log_normalized_density(self):
		raise NotImplementedError("Child Class MaternSpectralDensity")

class KinkSpectralDensity(SpectralDensityBase):
# class KinkSpectralDensity(tfp.distributions.distribution.AutoCompositeTensorDistribution,SpectralDensityBase):

	def __init__(self, cfg, dim):
		super().__init__(cfg.samplerHMC,dim)

		assert cfg.prior_var > 0.0
		assert cfg.x_lim_max > cfg.x_lim_min
		assert cfg.Nsteps_integration > 3

		# Parameters:
		self.prior_var = cfg.prior_var*1.0
		x_lim_min = cfg.x_lim_min # -5.0
		x_lim_max = cfg.x_lim_max # +2.0
		Nsteps_integration = cfg.Nsteps_integration # 401

		# pdb.set_trace()
		xdata = tf.linspace(x_lim_min,x_lim_max,Nsteps_integration)
		self.dX = xdata[1] - xdata[0]
		self.xdata = tf.reshape(xdata,(-1,self.dim)) # [Nsteps,dim]
		self.fdata = self._kink_fun(self.xdata) # [Nsteps,1] (since this is a scalar system, we only have one channel)

	def unnormalized_density(self,omega_in,log=False):
		"""
		
		in: omega_in [Nfeat,dim]
		return: S_vec [Nfeat,]
		"""

		Sw_vec, phiw_vec = self._MVFourierTransform(omega_in) # [Nfeat,]

		if log == True:
			return tf.math.log(Sw_vec)

		return Sw_vec, phiw_vec

	def update_pars(self,args):
		pass
		# raise NotImplementedError("Child Class KinkSpectralDensity")

	def normalized_density(self):
		raise NotImplementedError("Child Class KinkSpectralDensity")

	def log_normalized_density(self):
		raise NotImplementedError("Child Class KinkSpectralDensity")


	def _MVFourierTransform(self,omega_vec):
		"""

		omega_vec: [Npoints,dim]
		return:
			Sw: [Npoints,]
			phiw: [Npoints,]
		"""
		
		omega_times_X = omega_vec @ tf.transpose(self.xdata) # [Npoints,Nsteps]
		part_real = tfp.math.trapz(y=tf.math.cos(omega_times_X)*tf.transpose(self.fdata),dx=self.dX,axis=1) # [Npoints,]
		part_imag = tfp.math.trapz(y=tf.math.sin(omega_times_X)*tf.transpose(self.fdata),dx=self.dX,axis=1) # [Npoints,]

		# Modulus (spectral density):
		Sw = tf.math.sqrt(part_real**2 + part_imag**2) # [Npoints]

		# Argument:
		phiw = tf.math.atan2(y=-part_imag,x=part_real) # [Npoints]

		return Sw, phiw

	def _kink_fun(self,x):
		return 0.8 + (x + 0.2)*(1. - 5./(1 + tf.math.exp(-2.*x)) )


	# def _log_prob(self,value):
	# 	"""
	# 	value: float or double Tensor
	# 	return: a Tensor of shape sample_shape(x) + self.batch_shape with values of type self.dtype
	# 	"""

	# 	out = self.unnormalized_density(omega_in=value,log=True)
	# 	pdb.set_trace()

	# 	return out

	# def _sample_n(self, n, seed=None):
	# 	samples = get_samples(n)

	# 	pdb.set_trace()

	# 	return samples
