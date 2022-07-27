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

class MaternSpectralDensity(SpectralDensityBase):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		super().__init__(cfg_sampler,dim)

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
		S_vec = tf.math.reduce_prod(S_vec,axis=1,keepdims=True)*self.prior_var

		# Return copies of S(w), as many as input dimensions:
		S_vec = tf.concat([S_vec]*self.dim,axis=1)

		if log == True:
			return tf.math.log(S_vec)

		return S_vec, 0.0

	def _nonlinear_system_fun(self,x):
		return tf.zeros(x.shape)

class NonLinearSystemSpectralDensity(SpectralDensityBase):
# class NonLinearSystemSpectralDensity(tfp.distributions.distribution.AutoCompositeTensorDistribution,SpectralDensityBase):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		super().__init__(cfg_sampler,dim)

		assert cfg.prior_var > 0.0
		assert cfg.x_lim_max > cfg.x_lim_min
		assert cfg.Nsteps_integration > 3

		# Parameters:
		self.prior_var = cfg.prior_var*1.0
		x_lim_min = cfg.x_lim_min # -5.0
		x_lim_max = cfg.x_lim_max # +2.0
		Nsteps_integration = cfg.Nsteps_integration # 401

		self.volume_x = (x_lim_max - x_lim_min)**self.dim # Assume hypercube

		# pdb.set_trace()
		xdata_aux = tf.linspace(x_lim_min,x_lim_max,Nsteps_integration)
		xgrid_data = tf.meshgrid(*([xdata_aux]*self.dim))
		self.xdata = tf.reshape(xgrid_data,(-1,self.dim)) # [Nsteps,dim]
		self.dX = (x_lim_max - x_lim_min) / Nsteps_integration
		self.fdata = self._nonlinear_system_fun(self.xdata) # [Nsteps,dim]


	def unnormalized_density(self,omega_in,log=False):
		"""
		
		in: omega_in [Nfeat,dim]
		return: S_vec [Nfeat,]
		"""

		Sw_vec, phiw_vec = self._MVFourierTransform(omega_in) # [Nfeat,dim]

		if log == True:
			return tf.math.log(Sw_vec)

		return Sw_vec, phiw_vec

	def _MVFourierTransform(self,omega_vec):
		"""

		omega_vec: [Npoints,dim]
		return:
			Sw: [Npoints,dim]
			phiw: [Npoints,dim]
		"""

		if omega_vec.shape[1] == 1:
			omega_times_X = omega_vec @ tf.transpose(self.xdata) # [Npoints,Nsteps]
			part_real = tfp.math.trapz(y=tf.math.cos(omega_times_X)*tf.transpose(self.fdata),dx=self.dX,axis=1) / self.volume_x # [Npoints,]
			part_imag = tfp.math.trapz(y=tf.math.sin(omega_times_X)*tf.transpose(self.fdata),dx=self.dX,axis=1) / self.volume_x # [Npoints,]
		else:

			omega_times_X = omega_vec @ tf.transpose(self.xdata) # [Npoints,Nsteps]
			M = omega_times_X.shape[1]
			part_real = tf.math.cos(omega_times_X) @ self.fdata / M # Monte Carlo sum, [Npoints,dim]
			part_imag = tf.math.sin(omega_times_X) @ self.fdata / M # Monte Carlo sum, [Npoints,dim]

		# Modulus (spectral density):
		Sw = tf.math.sqrt(part_real**2 + part_imag**2) # [Npoints,dim]

		# Argument:
		phiw = tf.math.atan2(y=-part_imag,x=part_real) # [Npoints,dim]

		return Sw, phiw

	@abstractmethod
	def _nonlinear_system_fun(self,x):
		raise NotImplementedError

	# def _monte_carlo_sum_approx_single_point(self,omega_in,Xdata,fXdata):
	# 	"""

	# 	omega_in: [1,dim]
	# 	Xdata: [Npoints,dim]
	# 	fXdata: [Npoints]

	# 	"""

	# 	M = Xdata.shape[0]
	# 	sum_all = tf.math.sum(fXdata**2)
	# 	for ii in range(M):
	# 		for jj in range(ii+1,M):
	# 			sum_all += 2.*fXdata[ii]*fXdata[jj]*tf.math.cos(omega_in @ (Xdata[ii:ii+1,:] - Xdata[jj:jj+1,:]))

	# 	sum_all = sum_all / M**2

	# 	return sum_all



class KinkSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		assert dim == 1
		super().__init__(cfg,cfg_sampler,dim)

	def _nonlinear_system_fun(self,x):
		"""

		Kink function, as described in [1]

		[1] Ialongo, A.D., Van Der Wilk, M., Hensman, J. and Rasmussen, C.E., 2019, May. 
		Overcoming mean-field approximations in recurrent Gaussian process models. In International Conference on Machine Learning (pp. 2931-2940). PMLR.
		"""
		return 0.8 + (x + 0.2)*(1. - 5./(1 + tf.math.exp(-2.*x)) )


class ParaboloidSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		super().__init__(cfg,cfg_sampler,dim)

	def _nonlinear_system_fun(self,x):
		"""
		x: [Npoints,self.dim]
		"""
		return tf.math.reduce_sum(x**2,axis=1)

class NoNameSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		assert dim == 1
		super().__init__(cfg,cfg_sampler,dim)

	def _nonlinear_system_fun(self,x,u=0.):

		"""

		Toy example unction, as described in [1, Sec. 6]

		[1] Frigola, R., Lindsten, F., Schön, T.B. and Rasmussen, C.E., 2013. Bayesian inference and learning in Gaussian process state-space models with particle MCMC. Advances in neural information processing systems, 26.
		"""

		self.a = 0.5
		self.b = 25.
		self.c = 8.
		# self.q = 10.

		return self.a*x + self.b*x / (1. + x**2) + self.c*u


class KinkSharpSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		assert dim == 1
		super().__init__(cfg,cfg_sampler,dim)

	def _nonlinear_system_fun(self,x):

		"""

		Toy example unction, as described in [1, Sec. 6]

		[1] Frigola, R., Chen, Y. and Rasmussen, C.E., 2014. Variational Gaussian process state-space models. Advances in neural information processing systems, 27.
		"""

		ind_smaller = x < 4
		out = x.numpy()
		out[ind_smaller] += 1.
		out[~ind_smaller] = -4.*out[~ind_smaller] + 21.

		return tf.convert_to_tensor(out,dtype=tf.float32)

		# if x < 4:
		# 	return x + 1
		# else:
		# 	return -4.*x + 21.

		# return self.a*x + self.b*x / (1. + x**2) + self.c*u


class VanDerPolSpectralDensity(NonLinearSystemSpectralDensity):

	def __init__(self, cfg: dict, cfg_sampler: dict, dim: int):
		
		assert dim == 2
		self.deltaT = cfg.deltaT
		super().__init__(cfg,cfg_sampler,dim)


	def _nonlinear_system_fun(self,x):
		"""
		x: [Npoints,self.dim]
		"""

		return self._controlled_dynamics(x=x[:,0:1],y=x[:,1::],u1=0.,u2=0.)

	def _controlled_dynamics(self,x, y, u1, u2):
	    
	    x_dot =-y + u1
	    y_dot = x - y + (x**2)*y + u2

	    x_next = x_dot*self.deltaT + x
	    y_next = y_dot*self.deltaT + y
	    xy_next = tf.concat([x_next,y_next],axis=1)

	    return xy_next


	# def _log_prob(self,value):
	# 	"""
		# See how the uniform distribution is implemented in tensorflow
	# 	value: float or double Tensor
	# 	return: a Tensor of shape sample_shape(x) + self.batch_shape with values of type self.dtype
	# 	"""

	# 	out = self.unnormalized_density(omega_in=value,log=True)
	# 	pdb.set_trace()

	# 	return out

	# def _sample_n(self, n, seed=None):
		# See how the uniform distribution is implemented in tensorflow
	# 	samples = get_samples(n)

	# 	pdb.set_trace()

	# 	return samples
