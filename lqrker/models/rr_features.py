from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
import math
import pdb

from lqrker.models.rrtp import ReducedRankStudentTProcessBase
from lqrker.spectral_densities.base import SpectralDensityBase
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


class RRTPRegularFourierFeatures(ReducedRankStudentTProcessBase):
	"""
	
	As described in [1, Sec. 2.3.3], which is analogous to [2].

	[1] Hensman, J., Durrande, N. and Solin, A., 2017. Variational Fourier Features for Gaussian Processes. J. Mach. Learn. Res., 18(1), pp.5537-5588.
	[2] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.


	TODO:
	1) Maybe sample just a single vector, like in the jmlr paper
	2) Think about optimizing weights somehow. Prior on spectral density with model?
	3) Add other hyperparameters as trainable variables to the optimization
	4) Refactor all this in different files
	5) How can we infer the dominant frquencies from data? Can we compute S(w|Data) ?
	"""

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase):

		super().__init__(dim,cfg)

		self.Nfeat = cfg.hyperpars.weights_features.Nfeat

		# Spectral density to be used:
		self.spectral_density = spectral_density

	def update_spectral_density(self,args,state_ind):

		# Get density and angle:
		omega_min = -5.
		omega_max = +5.
		Ndiv = self.Nfeat
		omegapred = tf.linspace(omega_min,omega_max,Ndiv)
		omegapred = tf.reshape(omegapred,(-1,self.dim))
		S_samples_vec, phi_samples_vec = self.spectral_density.unnormalized_density(omegapred) # [Nsamples,1,dim], [Nsamples,]
		self.W_samples = omegapred

		self.normalization_constant_kernel = self.spectral_density.get_normalization_constant_numerical(omegapred)
		logger.info("normalization_constant_kernel: "+str(self.normalization_constant_kernel))

		self.S_samples_vec = S_samples_vec / self.normalization_constant_kernel
		self.phi_samples_vec = phi_samples_vec
		self.u_samples = tfp.distributions.Uniform(low=0.0, high=2.*math.pi).sample(sample_shape=(1,self.Nfeat))

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		# pdb.set_trace()
		WX = tf.transpose(self.W_samples @ tf.transpose(X)) # [Npoints, Nfeat]

		# self.phi_samples_vec = 0.0
		# harmonics_vec = tf.math.cos(WX + self.u_samples) # [Npoints, Nfeat], with random phases
		harmonics_vec = tf.math.cos(WX + self.phi_samples_vec) # [Npoints, Nfeat]
		harmonics_vec_scaled = harmonics_vec * tf.reshape(self.S_samples_vec,(1,-1)) # [Npoints, Nfeat]

		return harmonics_vec_scaled

	def get_cholesky_of_cov_of_prior_beta(self):
		return tf.linalg.diag(tf.math.sqrt(self.S_samples_vec)) + tf.eye(self.Nfeat)*tf.math.sqrt(self.get_noise_var())
		
	def get_Sigma_weights_inv_times_noise_var(self):
		return self.get_noise_var() * tf.linalg.diag(1./self.S_samples_vec)

	def get_logdetSigma_weights(self):
		return tf.math.reduce_sum(tf.math.log(self.S_samples_vec))


class RRTPRandomFourierFeatures(RRTPRegularFourierFeatures):

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase):

		super().__init__(dim,cfg)

		self.Nfeat = cfg.hyperpars.weights_features.Nfeat

		# Spectral density to be used:
		self.spectral_density = spectral_density

	def update_spectral_density(self,args,state_ind):

		W_samples_vec, Sw_vec_nor, phiw_vec = self.spectral_density.get_samples()

		self.W_samples = W_samples_vec
		self.S_samples_vec = Sw_vec_nor
		self.phi_samples_vec = phiw_vec
		self.u_samples = tfp.distributions.Uniform(low=0.0, high=2.*math.pi).sample(sample_shape=(1,self.Nfeat))

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		WX = tf.transpose(self.W_samples @ tf.transpose(X)) # [Npoints, Nfeat]

		# self.phi_samples_vec = 0.0
		# harmonics_vec = tf.math.cos(WX + self.u_samples) # [Npoints, Nfeat], with random phases
		harmonics_vec = tf.math.cos(WX + self.phi_samples_vec) # [Npoints, Nfeat]

		return harmonics_vec

	def get_Sigma_weights_inv_times_noise_var(self):
		return self.get_noise_var() * (self.Nfeat/self.prior_var) * tf.eye(self.Nfeat)

	def get_cholesky_of_cov_of_prior_beta(self):
		return tf.eye(self.Nfeat)*tf.math.sqrt((self.prior_var/self.Nfeat + self.get_noise_var()))

	def get_logdetSigma_weights(self):
		return self.Nfeat*tf.math.log(self.prior_var)

class RRTPSarkkaFeatures(ReducedRankStudentTProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRTPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

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

