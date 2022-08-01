from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
import math
import pdb
import numpy as np

from lqrker.models import ReducedRankStudentTProcessBase
from lqrker.spectral_densities.base import SpectralDensityBase
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


class RRTPRegularFourierFeatures(ReducedRankStudentTProcessBase):
	"""

	
	This model assumes a dim-dimensional input and a scalar output.


	
	As described in [1, Sec. 2.3.3], which is analogous to [2].

	[1] Hensman, J., Durrande, N. and Solin, A., 2017. Variational Fourier Features for Gaussian Processes. J. Mach. Learn. Res., 18(1), pp.5537-5588.
	[2] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.


	TODO:
	3) Add other hyperparameters as trainable variables to the optimization
	4) Refactor all this in different files
	5) How can we infer the dominant frquencies from data? Can we compute S(w|Data) ?
	"""

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_int=0):

		super().__init__(dim,cfg,spectral_density,dim_out_int)

		# assert cfg.hyperpars.prior_var_factor > 0 and cfg.hyperpars.prior_var_factor <= 1.0
		self.prior_var_factor = cfg.hyperpars.prior_var_factor

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		# pdb.set_trace()
		WX = X @ tf.transpose(self.W_samples) # [Npoints, Nfeat]
		harmonics_vec = tf.math.cos(WX + tf.transpose(self.phi_samples_vec)) # [Npoints, Nfeat]
		harmonics_vec_scaled = harmonics_vec * tf.reshape(self.S_samples_vec,(1,-1)) # [Npoints, Nfeat]

		return harmonics_vec_scaled

	def get_cholesky_of_cov_of_prior_beta(self):
		return tf.linalg.diag(tf.math.sqrt(tf.reshape(self.S_samples_vec*self.prior_var_factor,(-1)) + self.get_noise_var()))
		
	def get_Sigma_weights_inv_times_noise_var(self):
		return self.get_noise_var() * tf.linalg.diag(1./tf.reshape(self.S_samples_vec*self.prior_var_factor,(-1)))

	def get_logdetSigma_weights(self):
		return tf.math.reduce_sum(tf.math.log(self.S_samples_vec*self.prior_var_factor))


class RRTPRandomFourierFeatures(ReducedRankStudentTProcessBase):

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_int=0):

		super().__init__(dim,cfg,spectral_density,dim_out_int)

		assert cfg.prior_var_factor > 0 and cfg.prior_var_factor <= 1.0
		self.prior_var = cfg.prior_var # Here, the prior variance is user-specified. In RRTPRegularFourierFeatures is given by the spectral density, so therein we use the factor self.prior_var_factor; here it's not necessary

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		u_samples = tfp.distributions.Uniform(low=0.0, high=2.*math.pi).sample(sample_shape=(1,self.Nfeat))
		WX = tf.transpose(self.W_samples @ tf.transpose(X)) # [Npoints, Nfeat]
		harmonics_vec = tf.math.cos(WX + u_samples) # [Npoints, Nfeat]

		return harmonics_vec

	def get_cholesky_of_cov_of_prior_beta(self):
		return tf.eye(self.Nfeat)*tf.math.sqrt((self.prior_var/self.Nfeat + self.get_noise_var()))

	def get_Sigma_weights_inv_times_noise_var(self):
		return self.get_noise_var() * (self.Nfeat/self.prior_var) * tf.eye(self.Nfeat)

	def get_logdetSigma_weights(self):
		return self.Nfeat*tf.math.log(self.prior_var)

class RRTPSarkkaFeatures(ReducedRankStudentTProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRTPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

		raise NotImplementedError("Needs refactoring following RRTPRandomFourierFeatures")

		super().__init__(dim,cfg)

		self.Nfeat = cfg.hyperpars.weights_features.Nfeat
		self.L = cfg.hyperpars.L
		self.jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]

		# Spectral density to be used:
		# self.spectral_density = MaternSpectralDensity(cfg.spectral_density,dim)

	def _get_eigenvalues(self):
		"""

		Eigenvalues of the laplace operator
		"""

		Lstack = tf.stack([self.L]*self.Nfeat) # [Nfeat, dim]
		jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1)) # [Nfeat, 1]
		# pdb.set_trace()
		Ljj = (math.pi * jj / (2.*Lstack))**2 # [Nfeat, dim]

		return tf.reduce_sum(Ljj,axis=1) # [Nfeat,]

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""
		
		xx = tf.stack([X]*self.Nfeat) # [Nfeat, Npoints, dim]
		# jj = tf.reshape(tf.range(self.Nfeat,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]
		# pdb.set_trace()
		feat = 1/tf.sqrt(self.L) * tf.sin( math.pi*self.jj*(xx + self.L)/(2.*self.L) ) # [Nfeat, Npoints, dim]
		return tf.transpose(tf.reduce_prod(feat,axis=-1)) # [Npoints, Nfeat]

	def get_Sigma_weights_inv_times_noise_var(self):
		omega_in = tf.sqrt(self._get_eigenvalues())
		S_vec = self.spectral_density.unnormalized_density(omega_in)
		ret = self.get_noise_var() * tf.linalg.diag(1./S_vec)

		# if tf.math.reduce_any(tf.math.is_nan(ret)):
		# 	pdb.set_trace()

		return ret

	def get_logdetSigma_weights(self):
		omega_in = tf.sqrt(self._get_eigenvalues())
		S_vec = self.spectral_density.unnormalized_density(omega_in)
		return tf.reduce_sum(tf.math.log(S_vec))

class RRTPQuadraticFeatures(ReducedRankStudentTProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRTPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

		raise NotImplementedError("Needs refactoring following RRTPRandomFourierFeatures")

		super().__init__(dim,cfg)

		# Elements of the diagonal matrix Lambda:
		# TODO: Test the line below
		self.log_diag_vals = self.add_weight(shape=(Nfeat,), initializer=tf.keras.initializers.Zeros(), trainable=True,name="log_diag_vars")

	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		Npoints = X.shape[0]
		SQRT2 = math.sqrt(2)

		if self.dim == 1:
			PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , X**2 ],axis=1)
		else:
			PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , SQRT2*tf.math.reduce_prod(X,axis=1,keepdims=True) , X**2 ],axis=1)

		assert cfg.weights_features.Nfeat == PhiX.shape[1], "Basically, for quadratic features the number of features is given a priori; the user cannot choose"

		return PhiX # [Npoints, Nfeat]

	def get_Sigma_weights_inv_times_noise_var(self):
		"""
		The Lambda matrix depends on the choice of features

		TODO: Test this function
		"""
		return self.get_noise_var()*tf.linalg.diag(tf.exp(-self.log_diag_vals))

	def get_logdetSigma_weights(self):
		# TODO: Test this function
		return tf.reduce_sum(self.log_diag_vals)

class RRTPLQRfeatures(ReducedRankStudentTProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRTPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

		raise NotImplementedError("Needs refactoring following RRTPRandomFourierFeatures")

		super().__init__(dim,cfg)

		# Get parameters:
		nu = cfg.hyperpars.nu
		Nsys = cfg.hyperpars.weights_features.Nfeat # Use as many systems as number of features

		# TODO: Test the line below
		self.log_diag_vals = self.add_weight(shape=(Nsys,), initializer=tf.keras.initializers.Zeros(), trainable=True,name="log_diag_vars")
		
		from lqrker.objectives.lqr_cost_chi2 import LQRCostChiSquared
		self.lqr_cost = LQRCostChiSquared(dim_in=dim,cfg=cfg,Nsys=Nsys)
		print("Make sure we're NOT using noise in the config file...")
		pdb.set_trace()

	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		cost_values_all = self.lqr_cost.evaluate(X,add_noise=False,verbo=True)

		return cost_values_all

	def get_Sigma_weights_inv_times_noise_var(self):
		# TODO: Test this function
		return self.get_noise_var()*tf.linalg.diag(tf.exp(-self.log_diag_vals))

	def get_logdetSigma_weights(self):
		# TODO: Test this function
		return tf.reduce_sum(self.log_diag_vals)

