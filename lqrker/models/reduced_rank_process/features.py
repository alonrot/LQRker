from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow_probability as tfp
import math
import pdb
# import numpy as np

from lqrker.models import ReducedRankProcessBase
from lqrker.spectral_densities.base import SpectralDensityBase
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

# sess = tf.compat.v1.Session()


class RRPLinearFeatures(ReducedRankProcessBase):
	"""

	"""

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_ind=0):

		super().__init__(dim,cfg,spectral_density,dim_out_ind)

		# self.Nfeat = self.W_samples.shape[0]
		# assert cfg.hyperpars.prior_variance > 0
		# self.prior_var = cfg.hyperpars.prior_variance

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""
		return X

	def get_prior_mean(self):
		return tf.zeros((self.W_samples.shape[0],1)) # [Npoints,1]

	def get_cholesky_of_cov_of_prior_beta(self):
		prior_var_factor = self.get_prior_variance()
		return tf.linalg.diag(tf.math.sqrt(tf.reshape(self.S_samples_vec*prior_var_factor,(-1)))) # T-Student's process, function prediction f(x)
		
	def get_Sigma_weights_inv_times_noise_var(self):
		prior_var_factor = self.get_prior_variance()
		return self.get_noise_var() * tf.linalg.diag(1./tf.reshape(self.S_samples_vec*prior_var_factor,(-1)))

class RRPDiscreteCosineFeatures(ReducedRankProcessBase):
	"""

	Inspired by [1] and [2].

	[1] Hensman, J., Durrande, N. and Solin, A., 2017. Variational Fourier Features for Gaussian Processes. J. Mach. Learn. Res., 18(1), pp.5537-5588.
	[2] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.


	TODO:
	3) Add other hyperparameters as trainable variables to the optimization
	4) Refactor all this in different files
	5) How can we infer the dominant frquencies from data? Can we compute S(w|Data) ?
	"""

	# @tf.function
	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_ind=0):

		super().__init__(dim,cfg,spectral_density,dim_out_ind)

		# self.Nfeat = self.W_samples.shape[0]
		self.Dw = (self.W_samples[1,-1] - self.W_samples[0,-1])**self.dim # Equivalent to math.pi/L for self.spectral_density.get_Wpoints_discrete()

		self.dbg_flag = False

	# @tf.function
	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		# pdb.set_trace()
		WX = X @ tf.transpose(self.W_samples) # [Npoints, Nfeat]
		dbg_phase = 0.0
		if self.dbg_flag:
			dbg_phase = math.pi/32.0
		harmonics_vec = tf.math.cos(WX + tf.transpose(self.phi_samples_vec) + dbg_phase) # [Npoints, Nfeat]

		return harmonics_vec

	# @tf.function
	def get_prior_mean(self):
		prior_mean_factor = self.get_prior_mean_factor()
		return self.Dw*self.S_samples_vec*prior_mean_factor # [Npoints,1]

	# @tf.function
	def get_cholesky_of_cov_of_prior_beta(self):
		raise ValueError("Get rid of this self.Zs !!!!!!!")
		prior_var_factor = self.Dw / self.Zs * self.get_prior_variance()
		# return tf.linalg.diag(tf.math.sqrt(tf.reshape(self.S_samples_vec*prior_var_factor,(-1)) + self.get_noise_var())) # T-Student's process, observation prediction y
		return tf.linalg.diag(tf.math.sqrt(tf.reshape(self.S_samples_vec*prior_var_factor,(-1)))) # T-Student's process, function prediction f(x)
		
	# @tf.function
	def get_Sigma_weights_inv_times_noise_var(self):

		# S_samples_vec_local = np.clip(self.S_samples_vec,1e-10,np.inf)
		S_samples_vec_local = tf.clip_by_value(t=self.S_samples_vec,clip_value_min=1e-10,clip_value_max=float("Inf"))

		raise ValueError("Get rid of this self.Zs !!!!!!!")
		prior_var_factor = self.Dw / self.Zs * self.get_prior_variance()
		aux = self.get_noise_var() * tf.linalg.diag(1./tf.reshape(S_samples_vec_local*prior_var_factor,[-1]))
		# aux = self.get_noise_var() * tf.linalg.diag(1./tf.reshape(self.S_samples_vec*prior_var_factor,(-1))) # Doesn't work when building a graph
		if tf.math.reduce_any(tf.math.is_inf(aux)):
			print(aux)
			print(self.get_noise_var())
			print(S_samples_vec_local)
			print(prior_var_factor)
			tf.autograph.trace(aux)
			tf.debugging.check_numerics(aux, message='aux has infs or nans')

			# print(sess.run(c)) # https://www.activestate.com/resources/quick-reads/how-to-debug-tensorflow/

			# pdb.set_trace()
			# raise NotImplementedError("@RRPDiscreteCosineFeatures.get_Sigma_weights_inv_times_noise_var()")

		return aux

	# def get_logdetSigma_weights(self):
	# 	return tf.math.reduce_sum(tf.math.log(self.S_samples_vec*self.prior_var_factor))

	# def get_cholesky_of_cov_of_prior_beta(self):
	# 	# return tf.eye(self.Nfeat)*tf.math.sqrt((self.prior_var/self.Nfeat + self.get_noise_var())) # T-Student's process, observation prediction y
	# 	return tf.eye(self.Nfeat)*tf.math.sqrt((self.prior_var/self.Nfeat)) # T-Student's process, function f(x) prediction

	# def get_Sigma_weights_inv_times_noise_var(self):
	# 	return self.get_noise_var() * (self.Nfeat/self.prior_var) * tf.eye(self.Nfeat)

	# def get_logdetSigma_weights(self):
	# 	return self.Nfeat*tf.math.log(self.prior_var)
	# 	
	# 	



class RRPDiscreteCosineFeaturesVariableIntegrationStep(ReducedRankProcessBase):

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_ind=0):

		super().__init__(dim,cfg,spectral_density,dim_out_ind)

		# # self.Nfeat = self.W_samples.shape[0]
		# self.Dw = (self.W_samples[1,-1] - self.W_samples[0,-1])**self.dim # Equivalent to math.pi/L for self.spectral_density.get_Wpoints_discrete()


		self.dbg_phase_added_to_features = 0.0

		self.hack_constant_variance = False


	def get_features_mat(self,X):
		"""

		X: [Nxpoints, dim_in]
		return: PhiX: [Nxpoints, Nomegas]
		"""

		WX = X @ tf.transpose(self.W_samples) # [Nxpoints, Nomegas]
		harmonics_vec = tf.math.cos(WX + tf.transpose(self.phi_samples_vec) + self.dbg_phase_added_to_features) # [Nxpoints, Nomegas]

		return harmonics_vec

	def get_prior_mean(self):
		prior_mean_factor = self.get_prior_mean_factor()
		return self.dw_vec * self.S_samples_vec * prior_mean_factor # [Nomegas,1]

	def get_cholesky_of_cov_of_prior_beta(self):
		diag_els = tf.squeeze(self.S_samples_vec * self.dw_vec * self.get_prior_variance()) # [Nomegas,]

		if self.hack_constant_variance:
			diag_els = self.get_prior_variance() * tf.ones(self.S_samples_vec.shape[0]) # [Nomegas,]

		return tf.linalg.diag(tf.math.sqrt(diag_els)) # [Nomegas,Nomegas]
		
	def get_Sigma_weights_inv_times_noise_var(self):

		S_samples_vec_local = tf.clip_by_value(t=self.S_samples_vec,clip_value_min=1e-10,clip_value_max=float("Inf")) # [Nomegas,1]
		diag_els = 1. / tf.squeeze(S_samples_vec_local * self.dw_vec * self.get_prior_variance()) # [Nomegas,]

		if self.hack_constant_variance:
			diag_els = self.get_prior_variance() * tf.ones(self.S_samples_vec.shape[0]) # [Nomegas,]

		out_mat = self.get_noise_var() * tf.linalg.diag(diag_els) # [Nomegas,Nomegas]
		if tf.math.reduce_any(tf.math.is_inf(out_mat)):
			print(out_mat)
			pdb.set_trace()
		return out_mat


class RRPRegularFourierFeatures(ReducedRankProcessBase):
	"""

	
	This model assumes a dim-dimensional input and a 1-dimensional output.

	As described in [1, Sec. 2.3.3], which is analogous to [2].

	[1] Hensman, J., Durrande, N. and Solin, A., 2017. Variational Fourier Features for Gaussian Processes. J. Mach. Learn. Res., 18(1), pp.5537-5588.
	[2] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.


	TODO:
	3) Add other hyperparameters as trainable variables to the optimization
	4) Refactor all this in different files
	5) How can we infer the dominant frquencies from data? Can we compute S(w|Data) ?
	"""

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_ind=0):

		super().__init__(dim,cfg,spectral_density,dim_out_ind)

		self.Dw = (self.W_samples[1,-1] - self.W_samples[0,-1])**self.dim # Equivalent to math.pi/L for self.spectral_density.get_Wpoints_discrete()
		# assert cfg.hyperpars.prior_variance > 0
		# self.prior_var_factor = self.Dw / self.Zs * cfg.hyperpars.prior_variance

		assert self.dim_out_ind == 0, "This model assumes a dim-dimensional input and a 1-dimensional output."


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

	def get_prior_mean(self):
		return tf.zeros((self.W_samples.shape[0],1))

	def get_cholesky_of_cov_of_prior_beta(self):
		prior_var_factor = self.Dw / self.Zs * self.get_prior_variance()
		# return tf.linalg.diag(tf.math.sqrt(tf.reshape(self.S_samples_vec*prior_var_factor,(-1)) + self.get_noise_var())) # T-Student's process, observation prediction y
		return tf.linalg.diag(tf.math.sqrt(tf.reshape(self.S_samples_vec*prior_var_factor,(-1)))) # T-Student's process, function prediction f(x)
		
	def get_Sigma_weights_inv_times_noise_var(self):
		prior_var_factor = self.Dw / self.Zs * self.get_prior_variance()
		return self.get_noise_var() * tf.linalg.diag(1./tf.reshape(self.S_samples_vec*prior_var_factor,(-1)))

	# def get_logdetSigma_weights(self):
	# 	return tf.math.reduce_sum(tf.math.log(self.S_samples_vec*prior_var_factor))


class RRPRandomFourierFeatures(ReducedRankProcessBase):

	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_ind=0):

		super().__init__(dim,cfg,spectral_density,dim_out_ind)

		# assert cfg.prior_var_factor > 0 and cfg.prior_var_factor <= 1.0
		# NOTE: Here, the prior variance is user-specified. In RRPRegularFourierFeatures is given by the spectral density, so therein we use the factor self.prior_var_factor; here it's not necessary
		self.Nfeat = self.W_samples.shape[0]

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		u_samples = tfp.distributions.Uniform(low=0.0, high=2.*math.pi).sample(sample_shape=(1,self.Nfeat))
		WX = tf.transpose(self.W_samples @ tf.transpose(X)) # [Npoints, Nfeat]
		harmonics_vec = tf.math.cos(WX + u_samples) # [Npoints, Nfeat]

		return harmonics_vec

	def get_prior_mean(self):
		return tf.zeros((self.W_samples.shape[0],1))

	def get_cholesky_of_cov_of_prior_beta(self):
		# return tf.eye(self.Nfeat)*tf.math.sqrt((self.get_prior_variance()/self.Nfeat + self.get_noise_var())) # T-Student's process, observation prediction y

		return tf.eye(self.Nfeat)*tf.math.sqrt((self.get_prior_variance()/self.Nfeat)) # T-Student's process, function f(x) prediction

	def get_Sigma_weights_inv_times_noise_var(self):
		return self.get_noise_var() * (self.Nfeat/self.get_prior_variance()) * tf.eye(self.Nfeat)

	# def get_logdetSigma_weights(self):
	# 	return self.Nfeat*tf.math.log(self.prior_var)

class RRPSarkkaFeatures(ReducedRankProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

		raise NotImplementedError("Needs refactoring following RRPRandomFourierFeatures")

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

class RRPQuadraticFeatures(ReducedRankProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

		raise NotImplementedError("Needs refactoring following RRPRandomFourierFeatures")

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

class RRPLQRfeatures(ReducedRankProcessBase):
	"""

	TODO: Incompatible with current implementation of SpectralDensityBase.
	Needs refactoring following RRPRandomFourierFeatures
	"""

	def __init__(self, dim: int, cfg: dict):

		raise NotImplementedError("Needs refactoring following RRPRandomFourierFeatures")

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

