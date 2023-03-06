import tensorflow as tf
import pdb
import math
from abc import ABC, abstractmethod
# import numpy as np
# import tensorflow.experimental.numpy as tnp # https://www.tensorflow.org/guide/tf_numpy

import numpy as np
# from lqrker.objectives.lqr_cost_chi2 import LQRCostChiSquared
import tensorflow_probability as tfp
from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

from lqrker.spectral_densities.base import SpectralDensityBase
from lqrker.utils.common import CommonUtils

# import warnings
# warnings.filterwarnings("error")

import matplotlib.pyplot as plt
import matplotlib
markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 25
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels+2)


class ReducedRankProcessBase(ABC,tf.keras.layers.Layer):
	"""

	Reduced-Rank Student-t Process
	==============================
	We implement the Student-t process presented in [1]. However, instead of using
	a kernel function, we use the weight-space view from [1, Sec. 2.1.2]
	in order to reduce computational speed by using a finite set of features.
	See also [3].

	We assume herein non-zero mean.

	We use a Bayesian linear model:


	[1] Shah, A., Wilson, A. and Ghahramani, Z., 2014, April. Student-t processes as alternatives to Gaussian processes. In Artificial intelligence and statistics (pp. 877-885). PMLR.

	[2] Rasmussen, C.E. and Nickisch, H., 2010. Gaussian processes for machine
	learning (GPML) toolbox. The Journal of Machine Learning Research, 11,
	pp.3011-3015.

	[3] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank
	Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.

	

	"""
	# def __init__(self, dim, Nfeat, sigma_n, nu, **kwargs):
	# @tf.function
	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_ind=0, **kwargs):
		"""
		
		dim: Dimensionality of the input space
		Nfeat: Number of features
		L: Half Length of the hypercube. Each dimension has length [-L, L]
		"""

		super().__init__(**kwargs)


		assert cfg.which_process in ["gaussian","student-t"]
		self.which_process = cfg.which_process
		if self.which_process == "student-t":
			assert cfg.hyperpars.nu_init > 2, "Requirement: nu > 2"
			nu_init = cfg.hyperpars.nu_init
			self.log_nu = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(tf.math.log(nu_init-2.0)), trainable=True,name="log_nu")

		self.dim = dim

		# This model assumes a dim-dimensional input and a scalar output.
		# We need to select the output we care about for the spectral density points:
		# self.select_output_dimension(dim_out_ind)
		assert dim_out_ind >= 0 and dim_out_ind <= self.dim
		self.dim_out_ind = dim_out_ind

		# Specify weights:
		# self.Nfeat = cfg.hyperpars.weights_features.Nfeat
		# self.log_diag_vals = self.add_weight(shape=(self.Nfeat,), initializer=tf.keras.initializers.Zeros(), trainable=True,name="log_diag_vars")
		self.log_noise_std = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(tf.math.log(cfg.hyperpars.noise_std_process)), trainable=True,name="log_noise_std_dim{0:d}".format(self.dim_out_ind))
		# self.log_noise_std = tf.math.log(cfg.hyperpars.noise_std_process)
		# self.log_L = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(tf.math.log(cfg.hyperpars.L_init)), trainable=True,name="log_L")

		assert cfg.hyperpars.prior_variance > 0
		self.log_prior_variance = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(tf.math.log(cfg.hyperpars.prior_variance)), trainable=True,name="log_prior_variance_dim{0:d}".format(self.dim_out_ind))

		# self.log_prior_mean_factor = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(tf.math.log(cfg.hyperpars.prior_mean_factor)), trainable=True,name="log_prior_mean_factor_dim{0:d}".format(self.dim_out_ind))
		self.log_prior_mean_factor = tf.math.log(cfg.hyperpars.prior_mean_factor)


		# Learning parameters:
		self.learning_rate = cfg.learning.learning_rate
		self.epochs = cfg.learning.epochs
		self.stop_loss_val = cfg.learning.stopping_condition.loss_val

		# No data case:
		self.X = None
		self.Y = None

		self.prior_beta_already_computed = False
		self.predictive_beta_already_computed = False

		self.mean_beta_predictive = None
		self.chol_cov_beta_predictive = None
		self.mean_beta_prior = None
		self.chol_cov_beta_prior = None

		# ----------------------------------------------------------------------------------------------------------
		# Parameters only relevant to child classes (old version)
		# ----------------------------------------------------------------------------------------------------------



		# # Spectral density to be used:
		# self.spectral_density = spectral_density

		# self.S_samples_vec = self.spectral_density.Sw_points[:,self.dim_out_ind:self.dim_out_ind+1] # [Npoints,1]
		# self.phi_samples_vec = self.spectral_density.phiw_points[:,self.dim_out_ind:self.dim_out_ind+1] # [Npoints,1]
		# self.W_samples = self.spectral_density.W_points # [Npoints,self.dim]
		
		# Zs = self.spectral_density.get_normalization_constant_numerical(self.W_samples) # [self.dim,]
		# self.Zs = Zs[self.dim_out_ind:self.dim_out_ind+1]
		
		# # Process specific things:
		# if self.which_process == "student-t":
		# 	nu = self.get_nu()
		# 	self.Zs = self.Zs * (nu/(nu - 2.))


		# # Convert to tensors:
		# self.S_samples_vec = tf.convert_to_tensor(self.S_samples_vec,tf.float32)
		# self.phi_samples_vec = tf.convert_to_tensor(self.phi_samples_vec,tf.float32)
		# self.W_samples = tf.convert_to_tensor(self.W_samples,tf.float32)
		# self.Zs = tf.convert_to_tensor(self.Zs,tf.float32)


		# ----------------------------------------------------------------------------------------------------------
		# ----------------------------------------------------------------------------------------------------------



		# ----------------------------------------------------------------------------------------------------------
		# Parameters only relevant to child classes (new, after training omegas with NN)
		# ----------------------------------------------------------------------------------------------------------

		# Spectral density to be used:
		
		if spectral_density.Sw_points.ndim == 3:
			assert spectral_density.phi_samples_vec.ndim == 3
			assert spectral_density.W_samples.ndim == 3
			assert spectral_density.dw_vec.ndim == 3
			self.S_samples_vec = spectral_density.Sw_points[self.dim_out_ind,...] # [Nomegas,1]
			self.phi_samples_vec = spectral_density.phiw_points[self.dim_out_ind,...] # [Nomegas,1]
			self.W_samples = spectral_density.W_points[self.dim_out_ind,...] # [Nomegas,self.dim]
			self.dw_vec = spectral_density.dw_vec[self.dim_out_ind,...] # [Nomegas,1]
			# spectral_density.dX_vec[self.dim_out_ind,...] # [Nxpoints,1] # Not needed!
		else:
			self.S_samples_vec = spectral_density.Sw_points
			self.phi_samples_vec = spectral_density.phiw_points
			self.W_samples = spectral_density.W_points
			self.dw_vec = spectral_density.dw_vec

		self.Zs = np.array([1.])

		# Process specific things:
		if self.which_process == "student-t":
			nu = self.get_nu()
			self.Zs = self.Zs * (nu/(nu - 2.))

		# Convert to tensors:
		self.S_samples_vec = tf.convert_to_tensor(self.S_samples_vec,tf.float32)
		self.phi_samples_vec = tf.convert_to_tensor(self.phi_samples_vec,tf.float32)
		self.W_samples = tf.convert_to_tensor(self.W_samples,tf.float32)
		self.dw_vec = tf.convert_to_tensor(self.dw_vec,tf.float32)
		self.Zs = tf.convert_to_tensor(self.Zs,tf.float32)

		# # ----------------------------------------------------------------------------------------------------------
		# # ----------------------------------------------------------------------------------------------------------

		
		# Make it stationary
		# ==================
		
		make_it_stationary = False
		if make_it_stationary:

			Sw_vec_np = self.S_samples_vec.numpy()[0:-1]
			ind_mid = Sw_vec_np.shape[0]//2
			Sw_vec_np[ind_mid::,0] = Sw_vec_np[0:ind_mid,0]

			if tf.math.reduce_all(self.phi_samples_vec == 0.0):
				phiw_vec_np = np.zeros((self.S_samples_vec.shape[0]-1,1),dtype=np.float32)
			else:
				phiw_vec_np = self.phi_samples_vec.numpy()[0:-1]
			phiw_vec_np[0:ind_mid,0] = 0.0
			phiw_vec_np[ind_mid::,0] = -math.pi/2.

			omegapred_np = self.W_samples.numpy()[0:-1]
			omegapred_np[ind_mid::,0] = omegapred_np[0:ind_mid,0]

			self.S_samples_vec = tf.convert_to_tensor(Sw_vec_np,tf.float32)
			self.phi_samples_vec = tf.convert_to_tensor(phiw_vec_np,tf.float32)
			self.W_samples = tf.convert_to_tensor(omegapred_np,tf.float32)

			


	@abstractmethod
	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""
		raise NotImplementedError

	@abstractmethod
	def get_Sigma_weights_inv_times_noise_var(self):
		"""
		X: None
		return: None
		"""
		raise NotImplementedError

	@abstractmethod
	def get_cholesky_of_cov_of_prior_beta(self):
		raise NotImplementedError

	@abstractmethod
	def get_prior_mean(self):
		"""
		We use [1, Sec 9.3.3] to compute the posterior of a Bayesian linear model with non-zero prior mean

		[1] Deisenroth, M.P., Faisal, A.A. and Ong, C.S., 2020. Mathematics for machine learning. Cambridge University Press.
		"""
		raise NotImplementedError

	def get_logdetSigma_weights(self):
		raise NotImplementedError

	def add2dataset(self,xnew,ynew):
		raise NotImplementedError


	# def select_output_dimension(self,dim_out_ind):
	# 	assert dim_out_ind >= 0 and dim_out_ind <= self.dim
	# 	self.dim_out_ind = dim_out_ind

	# @tf.function
	def get_noise_var(self):
		"""

		TODO: Think about maybe using the softplus transform log(1 + exp(x))
		https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus
		"""
		return tf.exp(2.0*tf.squeeze(self.log_noise_std))

	# @tf.function
	def get_log_noise_std(self):
		return self.log_noise_std

	# @tf.function
	# def get_L(self):
	# 	return tf.exp(self.log_L)

	# @tf.function
	def get_prior_variance(self):
		return tf.exp(self.log_prior_variance)

	# @tf.function
	def get_prior_mean_factor(self):
		return tf.exp(self.log_prior_mean_factor)

	# @tf.function
	def get_nu(self):
		assert self.which_process == "student-t"
		return tf.exp(tf.squeeze(self.log_nu)) + 2.0

	# @tf.function
	def print_weights_info(self):
		
		logger.info("Trained weights:")
		for ii in range(len(self.trainable_weights)):

			if "log_nu" in self.trainable_weights[ii].name:
				str_info = " ** nu: " + str(self.get_nu())
			elif "log_noise_std" in self.trainable_weights[ii].name:
				str_info = " ** noise_std: " + str(tf.sqrt(self.get_noise_var()).numpy())

			logger.info(str_info)

	# @tf.function
	def get_prior_cov_inverse(self):
		"""

		This is needed for (a) predictive distributions and (b) loss function.
		
		cov(f(x)) = cov(phi(x)*beta) = phi(x)^T @ Sigma0 @ phi(x) + sigma_n^2
		Computing the inverse of cov(f(x)) scales O(N^3) where N is the number of datapoints. We wanna change that to O(M^3N),
		where M is the number of features. To that end, we use the matrix inversion lemma, and obtain:

		Kinv = cov(f(x))^{-1} = sigma_n^{-2} * ( I - phi(x)^T @ A^{-1} @ phi(x) ), with A = L.L^T and L being computed at self._update_features()
		"""
		Kinv = 1/self.get_noise_var()*( tf.eye(self.X.shape[0]) - self.PhiX @ tf.linalg.cholesky_solve(self.Lchol, tf.transpose(self.PhiX)) ) # [Npoints,Npoints]
		return Kinv

	# @tf.function
	def get_log_det_prior_cov_inverse(self,Kinv=None):
		"""

		See the explanation in self.get_prior_cov_inverse()

		This is an implementation of log(det(Kinv)), which is numerically more stable than the direct approach

		"""

		if Kinv is None:
			Kinv = self.get_prior_cov_inverse()

		Kinv_no_noise = self.get_noise_var() * Kinv # [Npoints,Npoints]
		log_det_Kinv_no_noise = tf.linalg.logdet(Kinv_no_noise)

		return log_det_Kinv_no_noise - 2.*Kinv_no_noise.shape[0]*self.get_log_noise_std()

	# @tf.function
	def get_MLII_loss_student_t(self):
		"""

		Compute the negative log evidence for a multivariate Student-t distribution as in [*].

		[*] Shah, A., Wilson, A. and Ghahramani, Z., 2014, April. Student-t processes as alternatives to Gaussian processes. In *Artificial intelligence and statistics* (pp. 877-885). PMLR.
		"""

		# Compute relevant variables without updating the global self.Lchol, self.PhiX yet
		# Lchol, PhiX = self._update_features() # chol(PhiXTPhiX + Sigma_weights_inv_times_noise_var) [Nfeat,Nfeat] ; PhiX [Npoints, Nfeat]
		self._update_features() # chol(PhiXTPhiX + Sigma_weights_inv_times_noise_var) [Nfeat,Nfeat] ; PhiX [Npoints, Nfeat]

		nu = self.get_nu()

		# Compute data fit:
		Kinv = self.get_prior_cov_inverse()
		mean_prior = self.PhiX @ self.get_prior_mean() # [Npoints,1]
		term_data_fit = tf.transpose(self.Y - mean_prior) @ (Kinv @ (self.Y - mean_prior))
		assert tf.squeeze(term_data_fit < 0.0) == False
		# term_data_fit_clipped = tf.clip_by_value(term_data_fit,clip_value_min=0.0,clip_value_max=float("Inf"))
		data_fit = -0.5*(nu + self.X.shape[0])*tf.math.log1p( term_data_fit / (nu-2.) )

		# Compute model complexity:
		"""
		-0.5*log(det(K)) = -0.5*log(1/det(Kinv)) = 0.5*log(det(Kinv))
		"""
		model_complexity = 0.5*self.get_log_det_prior_cov_inverse(Kinv)

		# Compute constant terms:
		const = tf.math.lgamma(0.5*(nu + self.X.shape[0])) - 0.5*self.X.shape[0]*tf.math.log(math.pi*(nu-2.)) - tf.math.lgamma(0.5*nu)

		# Compute loss as -log(p(y))
		loss_val = -data_fit - model_complexity - const

		if tf.math.is_nan(loss_val) or tf.math.is_inf(loss_val):
			pdb.set_trace()

		return loss_val

	# @tf.function
	def get_MLII_loss_gaussian(self):
		"""

		TODO: Do not update features unless the hyperparameters have changed. Have a way to detect that.

		TODO:
		1) We're mixing the prior mean with the posterior covariance. See Rasmussen


		"""


		raise NotImplementedError("Here we need the inverse of the prior; but the equations look like the predictive distribution; double check we're doing the right thing....")

		# Compute relevant variables without updating the global self.Lchol, self.PhiX yet
		# Lchol, PhiX = self._update_features() # chol(PhiXTPhiX + Sigma_weights_inv_times_noise_var) [Nfeat,Nfeat] ; PhiX [Npoints, Nfeat]
		self._update_features(verbosity=True) # chol(PhiXTPhiX + Sigma_weights_inv_times_noise_var) [Nfeat,Nfeat] ; PhiX [Npoints, Nfeat]

		# Compute data fit:
		logger.info("Computing data fit term...")
		Kinv = self.get_prior_cov_inverse()
		mean_prior = self.PhiX @ self.get_prior_mean() # [Npoints,1]
		data_fit = -0.5*tf.transpose(self.Y - mean_prior) @ (Kinv @ (self.Y - mean_prior))

		# Compute model complexity:
		logger.info("Computing model complexity term...")
		"""
		-0.5*log(det(K)) = -0.5*log(1/det(Kinv)) = 0.5*log(det(Kinv))
		"""
		# model_complexity = 0.5*tf.linalg.logdet(Kinv)
		model_complexity = 0.5*self.get_log_det_prior_cov_inverse(Kinv) # This operation is O(N^3), where N is the number of datapoints

		# Compute loss as -log(p(y))
		logger.info("Done! Returning loss...")
		loss_val = -data_fit - model_complexity

		if tf.math.is_nan(loss_val) or tf.math.is_inf(loss_val):
			pdb.set_trace()

		return loss_val

	# @tf.function
	def get_MLII_loss_gaussian_predictive(self,xpred):
		"""

		xpred: [Npoints,dim_in]

		TODO:
		1) We're mixing the prior mean with the posterior covariance. See Rasmussen
		2) Can we get the loss using only the cholesky?
		3) We need to add the noise to the covariance



		TODO: Do not update features unless the hyperparameters have changed. Have a way to detect that.
		"""

		raise NotImplementedError

		# Compute relevant variables without updating the global self.Lchol, self.PhiX yet
		# Lchol, PhiX = self._update_features() # chol(PhiXTPhiX + Sigma_weights_inv_times_noise_var) [Nfeat,Nfeat] ; PhiX [Npoints, Nfeat]
		self._update_features(verbosity=True) # chol(PhiXTPhiX + Sigma_weights_inv_times_noise_var) [Nfeat,Nfeat] ; PhiX [Npoints, Nfeat]

		mean_beta, cov_beta_chol = self.predict_beta(from_prior=False)

		logger.info("Computing matrix of features Phi(xpred) ...")
		Phi_pred = self.get_features_mat(xpred) # [Npoints, Nfeat]

		# pdb.set_trace()
		mean_pred = Phi_pred @ mean_beta # [Npoints,1]
		cov_pred_chol = Phi_pred @ cov_beta_chol # [Npoints, Nfeat]
		# They should both be: [Npoints,]

		cov_pred = cov_pred_chol @ tf.transpose(cov_pred_chol)

		# pdb.set_trace()
		# data_fit = -0.5*((self.Y - mean_pred)/cov_pred_chol)**2

		# aux = (self.Y - mean_pred) @ cov_pred_chol

		# model_complexity = -0.5*tf.math.log(2.*math.pi) * cov_pred_chol

		# loss_val = 

		return loss_val

	# @tf.function
	def get_MLII_loss(self,which_process):
		if which_process == "gaussian":
			return self.get_MLII_loss_gaussian()
		elif which_process == "student-t":
			return self.get_MLII_loss_student_t()

	# @tf.function
	def train_model(self,verbosity=False):
		"""
		TODO: Speed up the training by:
		1) Achieving O(N) complexity on the get_MLII_loss_gaussian() loss
		2) Exploring the options mentioned in the links below
		for speeding up TF2
		"""
		# https://github.com/tensorflow/tensorflow/issues/30596
		# https://stackoverflow.com/a/61349421
		tf.function(self._train_model(verbosity))

	# @tf.function
	def _train_model(self,verbosity=False):
		"""

		"""

		logger.info("Training the model...")

		# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		epoch = 0
		done = False
		loss_value_curr = float("Inf")
		trainable_weights_best = self.get_weights()
		while epoch < self.epochs and not done:

			with tf.GradientTape() as tape:
				loss_value = self.get_MLII_loss(which_process=self.which_process)

			logger.info("Training the model... 1")
			grads = tape.gradient(loss_value, self.trainable_weights)
			logger.info("Training the model... 2")
			optimizer.apply_gradients(zip(grads, self.trainable_weights))
			logger.info("Training the model... 3")

			if (epoch+1) % 10 == 0:
				logger.info("Training loss at epoch %d / %d: %.4f" % (epoch+1, self.epochs, float(loss_value)))

			if loss_value <= self.stop_loss_val:
				done = True
			
			if loss_value < loss_value_curr:
				trainable_weights_best = self.get_weights()
				loss_value_curr = loss_value
			
			epoch += 1
			logger.info("Training the model... 4")

		if done == True:
			logger.info("Training finished because loss_value = {0:f} (<= {1:f})".format(float(loss_value),float(self.stop_loss_val)))

		self.set_weights(weights=trainable_weights_best)

		if verbosity:
			self.print_weights_info()

	# @tf.function
	def update_model(self,X,Y):

		self._update_dataset(X,Y)
		self._update_features()

		self.prior_beta_already_computed = False
		self.predictive_beta_already_computed = False
		# self.acquired_sample_mv0_for_sample_path_callable = False

	# @tf.function
	def _update_dataset(self,X,Y):
		self.X = X

		if len(Y.shape) == 1:
			assert Y.shape[0] > 0
			self.Y = tf.reshape(Y,(-1,1))
		elif len(Y.shape) == 2:
			self.Y = Y
		else:
			raise ValueError("Y size is not correct")

	# @tf.function
	def _update_features(self,verbosity=True):
		"""

		Compute some relevant quantities that will be later used for prediction:
		PhiX, 
		"""

		if verbosity: logger.info("Computing matrix of features PhiX ...")
		PhiX = self.get_features_mat(self.X) # [Npoints,Nfeat]
		# PhiX = self.get_features_mat(self.X) / tf.math.sqrt(float(self.W_samples.shape[0])) # [Npoints,Nfeat]

		PhiXTPhiX = tf.transpose(PhiX) @ PhiX
		Sigma_weights_inv_times_noise_var = self.get_Sigma_weights_inv_times_noise_var()
		# pdb.set_trace()

		AA_sym = PhiXTPhiX + Sigma_weights_inv_times_noise_var
		BB_sym = 0.5*(AA_sym + tf.transpose(AA_sym)) # Ensure symmetry. Due to numerical imprecisions, the matrix might not always be completely symmetric
		# pdb.set_trace()

		if tf.math.reduce_any(tf.math.is_inf(BB_sym)):
			logger.info("infs in BB_sym"); pdb.set_trace()

		if tf.math.reduce_any(tf.math.is_nan(BB_sym)):
			logger.info("nans in BB_sym"); pdb.set_trace()

		if verbosity: logger.info("    Computing cholesky decomposition of {0:d} x {1:d} matrix ...".format(PhiXTPhiX.shape[0],PhiXTPhiX.shape[1]))

		CC_sym = CommonUtils.fix_eigvals(BB_sym,verbosity=verbosity)
		# BB_sym = CommonUtils.fix_eigvals_other_way(BB_sym,verbosity=verbosity)

		DD_sym = 0.5*(CC_sym + tf.transpose(CC_sym)) # Ensure symmetry. Due to numerical imprecisions, the matrix might not always be completely symmetric

		Lchol = tf.linalg.cholesky(DD_sym) # Lower triangular A = L.L^T
		self.Lchol = Lchol
		self.PhiX = PhiX
		# if update_global_vars:
		# 	return None
		# else:
		return self.Lchol, self.PhiX

		# AA_sym_inv = tf.linalg.inv(AA_sym)
		# # AA_sym_inv = CommonUtils.fix_eigvals(AA_sym_inv)
		# self.Lchol_of_inv = tf.linalg.cholesky(AA_sym_inv)

	# @tf.function
	def get_predictive_beta_distribution(self):
		"""
		We use [1, Sec 9.3.3] to compute the posterior of a Bayesian linear model with non-zero prior mean

		[1] Deisenroth, M.P., Faisal, A.A. and Ong, C.S., 2020. Mathematics for machine learning. Cambridge University Press.
		"""

		if self.predictive_beta_already_computed:
			# logger.info("predictive_beta_distribution doesn't need to be recomputed because the dataset was never updated")
			return self.mean_beta_predictive, self.chol_cov_beta_predictive

		# Get mean:
		PhiXY_plus_mean_term = tf.transpose(self.PhiX) @ self.Y + self.get_Sigma_weights_inv_times_noise_var() @ self.get_prior_mean()
		# mean_beta = tf.linalg.cholesky_solve(self.Lchol, PhiXY_plus_mean_term) / np.sqrt(1000.)
		mean_beta = tf.linalg.cholesky_solve(self.Lchol, PhiXY_plus_mean_term)
		# mean_beta = (self.Lchol_of_inv @ tf.transpose(self.Lchol_of_inv)) @ PhiXY_plus_mean_term


		if self.which_process == "student-t":
			var_noise = self.get_noise_var()
			nu = self.get_nu()

			# Update parameters from the Student-t distribution:
			nu_pred = nu + self.X.shape[0]

			K11_inv = self.get_prior_cov_inverse()
			beta1 = tf.transpose(self.Y) @ ( K11_inv @ self.Y )
			cov_beta_factor_sqrt = tf.sqrt( (nu + beta1 - 2) / (nu_pred-2) * self.get_noise_var() )

		elif self.which_process == "gaussian":
			cov_beta_factor_sqrt = tf.sqrt(self.get_noise_var())


		"""
		theta ~ N(mu,Sigma)
		Sigma = L.L^T
		We need L
		
		Sigma = Sigma_tilde * var_noise    (var_noise = self.get_noise_var())
		Sigma_tilde = (Lchol @ Lchol^T)^{-1}
		Sigma = (Lchol @ Lchol^T)^{-1} * var_noise = ( (Lchol^T)^{-1} @ Lchol^{-1} ) * var_noise = (Lchol^T)^{-1}*std_noise @ Lchol^{-1}*std_noise
		Hence, L = (Lchol^T)^{-1}*std_noise

		For theta ~ t(mu,Sigma), we just need to multiply by tf.sqrt( (nu + beta1 - 2) / (nu_pred-2) )
		"""
		cov_beta_chol = tf.linalg.inv(tf.transpose(self.Lchol)) * cov_beta_factor_sqrt
		# cov_beta_chol = self.Lchol_of_inv * cov_beta_factor_sqrt

		# pdb.set_trace()

		self.predictive_beta_already_computed = True
		self.mean_beta_predictive = mean_beta
		self.chol_cov_beta_predictive = cov_beta_chol

		return mean_beta, cov_beta_chol

	# @tf.function
	def get_prior_beta_distribution(self):
		"""

		NOTE: We can't have an additive noise model here. We do as they do in [2, Fig. 3], i.e., we add the noise
		directly to the covariance matrix, but this doesn't correspond to th covariance of t1+t2.

		The covariance is taken from [1, Sec. 7.7]
	
		[1] Petersen, K.B. and Pedersen, M.S., 2008. The matrix cookbook. Technical University of Denmark, 7(15), p.510.
		[2] Shah, A., Wilson, A. and Ghahramani, Z., 2014, April. Student-t processes as alternatives to Gaussian processes. In *Artificial intelligence and statistics* (pp. 877-885). PMLR.

		"""

		if self.which_process == "student-t":
			nu = self.get_nu()
			fac_cov = tf.math.sqrt( nu / (nu - 2.) )
		elif self.which_process == "gaussian":
			fac_cov = 1.0

		if self.prior_beta_already_computed:
			# logger.info("prior_beta_distribution doesn't need to be recomputed because the dataset was never updated")
			return self.mean_beta_prior, self.chol_cov_beta_prior

		# Get prior cov:
		chol_cov_beta_prior = self.get_cholesky_of_cov_of_prior_beta() * fac_cov

		# Get prior mean:
		mean_beta_prior = self.get_prior_mean()

		self.prior_beta_already_computed = True
		self.mean_beta_prior = mean_beta_prior
		self.chol_cov_beta_prior = chol_cov_beta_prior

		return mean_beta_prior, chol_cov_beta_prior

	# @tf.function
	def predict_beta(self,from_prior=False):

		if self.X is None and self.Y is None or from_prior:
			mean_beta, cov_beta_chol = self.get_prior_beta_distribution()
		else:
			mean_beta, cov_beta_chol = self.get_predictive_beta_distribution()

		return mean_beta, cov_beta_chol

	# @tf.function
	def predict_at_locations(self,xpred,from_prior=False):
		"""

		xpred: [Npoints, dim]
		"""

		mean_beta, cov_beta_chol = self.predict_beta(from_prior)

		# logger.info("Computing matrix of features Phi(xpred) ...")
		Phi_pred = self.get_features_mat(xpred) # [Nxpoints, Nomegas]

		mean_pred = Phi_pred @ mean_beta
		cov_pred_chol = Phi_pred @ cov_beta_chol
		cov_pred = cov_pred_chol @ tf.transpose(cov_pred_chol) # L.L^T

		# DEBUG
		# if from_prior:
		# 	print("tf.sqrt(cov_pred[0,0]):",tf.sqrt(cov_pred[0,0]))
		# 	cov_beta = cov_beta_chol @ tf.transpose(cov_beta_chol)
		# 	print("cov_beta[0,0]:",cov_beta[0,0])
		# 	# self.S_samples_vec
		# 	pdb.set_trace()

		return tf.squeeze(mean_pred), cov_pred

	# @tf.function
	def get_sample_path_callable(self,Nsamples,from_prior=False):

		# logger.info("Predicting; from_prior = {0:s} || self.dim_out_ind = {1:d} ...".format(str(from_prior),self.dim_out_ind))
		mean_beta, cov_beta_chol = self.predict_beta(from_prior) # Computationally expensive call

		# logger.info("Sampling from standard MVN || self.dim_out_ind = {0:d} ...".format(self.dim_out_ind))
		
		# """
		# Using self.acquired_sample_mv0_for_sample_path_callable
		# """
		# if self.acquired_sample_mv0_for_sample_path_callable:
		# 	sample_mv0 = self.sample_mv0_for_sample_path_callable
		# else:
		# 	Nfeat = cov_beta_chol.shape[0]
		# 	sample_mv0 = self.get_sample_multivariate_standard_prior(Nfeat,Nsamples) # [Nfeat,Nsamples]
		# 	self.sample_mv0_for_sample_path_callable = sample_mv0
		# 	# self.acquired_sample_mv0_for_sample_path_callable = True
		# 	self.acquired_sample_mv0_for_sample_path_callable = False # We purposedly want this to be False because we're setting Nsamples=1 and calling this function only ONCE every rollout. However, refactoring all this would be good
		

		"""
		Not using self.acquired_sample_mv0_for_sample_path_callable
		"""
		Nfeat = cov_beta_chol.shape[0]
		sample_mv0 = self.get_sample_multivariate_standard_prior(Nfeat,Nsamples) # [Nfeat,Nsamples]

		aux = tf.reshape(mean_beta,(-1,1)) + cov_beta_chol @ sample_mv0 # [Nfeat,1] + [Nfeat,Nfeat] @ [Nfeat,Nsamples]

		# logger.info("self.dim_out_ind = {0:d} ...".format(self.dim_out_ind))
		# print("aux: ",aux[0:5,0:5])
		# print("mean_beta: ",mean_beta[0:5,0:5])
		# print("cov_beta_chol: ",cov_beta_chol[0:5,0:5])

		# @tf.function
		def nonlinfun_sampled_callable(x):
			"""
			x: [Npoints, self.dim_in]
			"""

			# logger.info("self.dim_out_ind = {0:d} ...".format(self.dim_out_ind))
			# print("aux: ",aux[0:5,0:5])

			return self.get_features_mat(x) @ aux # [Npoints, Nfeat] @ [Nfeat,Nsamples] = [Npoints,Nsamples]

		return nonlinfun_sampled_callable

	# @tf.function
	def sample_path_from_predictive(self,xpred,Nsamples,from_prior=False):
		"""

		xpred: [Npoints,self.dim]
		sample_xpred: [Npoints,Nsamples]

		"""

		fx = self.get_sample_path_callable(Nsamples,from_prior)
		sample_xpred = fx(xpred)

		return sample_xpred

	# @tf.function
	def sample_state_space_from_prior_recursively(self,x0,x1,traj_length,sort=False,plotting=False):
		"""

		Pass two initial latent values
		x0: [1,self.dim]
		x1: [1,self.dim]

		The GP won't be training during sampling, i.e., we won't call self.train_model()
		"""

		raise NotImplementedError("This function is deprecated; check sample_state_space_from_prior_recursively() @ MOrrp.py ")

		xmin = -6.
		xmax = +3.
		Ndiv = 201
		xpred = tf.linspace(xmin,xmax,Ndiv)
		xpred = tf.reshape(xpred,(-1,1))

		fx = self.get_sample_path_callable(Nsamples=1)

		yplot_true_fun = self.spectral_density._nonlinear_system_fun(xpred)
		yplot_sampled_fun = fx(xpred)

		assert traj_length > 2

		Xtraining = tf.identity(self.X) # Copy tensor
		Ytraining = tf.identity(self.Y) # Copy tensor
		Xtraining_and_new = tf.concat([Xtraining,x0],axis=0)
		Ytraining_and_new = tf.concat([Ytraining,x1],axis=0)
		self.update_model(X=Xtraining_and_new,Y=Ytraining_and_new)

		if plotting:
			hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(12,8),sharex=True)
			hdl_fig.suptitle(r"Kink function simulation $x_{t+1} = f(x_t) + \varepsilon$"+", kernel: {0}".format("kink"),fontsize=fontsize_labels)

		xsamples = np.zeros((traj_length,self.dim),dtype=np.float32)
		xsamples[0,:] = x0
		xsamples[1,:] = x1
		resample_mvt0 = True
		for ii in range(1,traj_length-1):

			xsample_tp = tf.convert_to_tensor(value=xsamples[ii:ii+1,:],dtype=np.float32)

			if ii > 1: 
				resample_mvt0 = False

			# xsamples[ii+1,:] = self.sample_path_from_predictive(xpred=xsample_tp,Nsamples=1,resample_mvt0=resample_mvt0)
			xsamples[ii+1,:] = fx(xsample_tp)

			Xnew = tf.convert_to_tensor(value=xsamples[0:ii,:],dtype=np.float32)
			Ynew = tf.convert_to_tensor(value=xsamples[1:ii+1,:],dtype=np.float32)

			Xtraining_and_new = tf.concat([Xtraining,Xnew],axis=0)
			Ytraining_and_new = tf.concat([Ytraining,Ynew],axis=0)
			self.update_model(X=Xtraining_and_new,Y=Ytraining_and_new)

			if plotting:
				# Plot what's going on at each iteration here:
				MO_mean_pred, cov_pred = self.predict_at_locations(xpred)
				MO_std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))
				hdl_splots[0].cla()
				hdl_splots[0].plot(xpred,MO_mean_pred,linestyle="-",color="b",lw=3)
				hdl_splots[0].fill_between(xpred[:,0],MO_mean_pred - 2.*MO_std_pred,MO_mean_pred + 2.*MO_std_pred,color="cornflowerblue",alpha=0.5)
				hdl_splots[0].plot(Xnew[:,0],Ynew[:,0],marker=".",linestyle="--",color="gray",lw=0.5,markersize=5)
				hdl_splots[0].set_xlim([xmin,xmax])
				hdl_splots[0].set_ylabel(r"$x_{t+1}$",fontsize=fontsize_labels)
				hdl_splots[0].plot(xpred,yplot_true_fun,marker="None",linestyle="-",color="grey",lw=1)
				hdl_splots[0].plot(xpred,yplot_sampled_fun,marker="None",linestyle="--",color="r",lw=0.5)

				plt.pause(0.1)
				# input()

		xsamples_X = xsamples[0:-1,:]
		xsamples_Y = xsamples[1::,:]

		if sort:
			assert x0.shape[1] == 1, "This only makes sense in 1D"
			ind_sort = tf.argsort(xsamples_X[:,0],axis=0)
			# pdb.set_trace()
			xsamples_X = xsamples_X[ind_sort,:]
			xsamples_Y = xsamples_Y[ind_sort,:]

		# Go back to the dataset at it was:
		self.update_model(X=Xtraining,Y=Ytraining)

		return xsamples_X, xsamples_Y

	# @tf.function
	def get_sample_multivariate_standard_prior(self,Npred,Nsamples):

		if self.which_process == "student-t":
			return self.get_sample_mvt0(Npred,Nsamples)
		elif self.which_process == "gaussian":
			return self.get_sample_mvn0(Npred,Nsamples)

	# @tf.function
	def get_sample_mvn0(self,Npred,Nsamples):
		"""
		Sample a path from MVN(0,I)
		return: [Npred,Nsamples]

		"""
		samples = tf.random.normal(shape=(Npred,Nsamples),mean=0.0,stddev=1.0) # [Npred,Nsamples]
		return samples

	# @tf.function
	def get_sample_mvt0(self,Npred,Nsamples):
		"""
		Sample a path from MVT(nu,0,I)
		Using: (i) uniform sphere, (ii) inverse gamma, and (iii) Chi-squared

		return: [Npred,Nsamples]

		[1] Shah, A., Wilson, A. and Ghahramani, Z., 2014, April. Student-t processes
		as alternatives to Gaussian processes. In Artificial intelligence and
		statistics (pp. 877-885). PMLR.

		"""

		# Sample from unit sphere:
		dist_sphe = tfp.distributions.SphericalUniform(dimension=Npred)
		sample_sphe = dist_sphe.sample(sample_shape=(Nsamples,))

		nu = self.get_nu()

		# Sample from inverse Gamma:
		alpha = 0.5*nu
		beta = 0.5
		dist_ig = tfp.distributions.InverseGamma(concentration=alpha,scale=beta)
		sample_ig = dist_ig.sample(sample_shape=(Nsamples,1))

		# Sample from chi-squared:
		dist_chi2 = tfp.distributions.Chi2(df=Npred)
		sample_chi2 = dist_chi2.sample(sample_shape=(Nsamples,1))

		# Sample from MVT(nu,0,I):
		sample_mvt0 = tf.math.sqrt((nu-2) * sample_chi2 * sample_ig) * sample_sphe

		return tf.transpose(sample_mvt0) # [Npred,Nsamples]

	def get_predictive_entropy_of_truncated_dist(self):
		"""
		Moments of truncated Student-t distribution:
		https://link.springer.com/content/pdf/10.1016/j.jkss.2007.06.001.pdf

		"""
		raise NotImplementedError

	def get_predictive_entropy(self,cov_pred):
		"""

		See [1,2].
		In Sec. 2.4, H_{Z_0} is formulated, and its relation with Z is given.
		In Sec. 2.1, eq. (4), the entropy of Z ( H_{Z}) is given as a function of H_{Z_0}.
		This coincides with "Properties of differential entropy" [3].
		In such entropy, the only term that depends on the predictive location x* is
		the predictive covariance.
		Luckily, such predictive covariance also depends on the observations and on nu.
		Therefore, using such entropy for BO would be different from using the entropy
		of a Gaussian.
		
		In order to do BO, we are interested in the entropy at each location.
		Assuming that cov_pred is the predictive covariance at a set of locations xpred,
		we get the variance from the diagonal and compute the entropy for eaach element of the diagonal.

		[1] ARELLANO‐VALLE, R.B., CONTRERAS‐REYES, J.E. and Genton, M.G., 2013.
		Shannon Entropy and Mutual Information for Multivariate Skew‐Elliptical
		Distributions. Scandinavian Journal of Statistics, 40(1), pp.42-62.
		[2] https://www.jstor.org/stable/23357252?seq=5#metadata_info_tab_contents
		[3] https://en.wikipedia.org/wiki/Differential_entropy

		"""
		entropy = 0.5*tf.math.log( tf.linalg.diag_part(cov_pred) )
		raise NotImplementedError("Make sure we switch here between gaussian/student-t")


	def call(self, inputs):
		# y = tf.matmul(inputs, self.w) + self.b
		# return tf.math.cos(y)
		# logger.info("self.call(): <><><><>      This method should not be called yet... (!)      <><><><>")
		raise NotImplementedError


