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


class ReducedRankStudentTProcessBase(ABC,tf.keras.layers.Layer):
	"""

	Reduced-Rank Student-t Process
	==============================
	We implement the Student-t process presented in [1]. However, instead of using
	a kernel function, we use the weight-space view from [1, Sec. 2.1.2]
	in order to reduce computational speed by using a finite set of features.
	See also [3].

	We assume zero mean. Extending it non-zero mean is trivial.


	[1] Shah, A., Wilson, A. and Ghahramani, Z., 2014, April. Student-t processes as alternatives to Gaussian processes. In Artificial intelligence and statistics (pp. 877-885). PMLR.

	[2] Rasmussen, C.E. and Nickisch, H., 2010. Gaussian processes for machine
	learning (GPML) toolbox. The Journal of Machine Learning Research, 11,
	pp.3011-3015.

	[3] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank
	Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.
	"""
	# def __init__(self, dim, Nfeat, sigma_n, nu, **kwargs):
	def __init__(self, dim: int, cfg: dict, spectral_density: SpectralDensityBase, dim_out_int=0, **kwargs):
		"""
		
		dim: Dimensionality of the input space
		Nfeat: Number of features
		L: Half Length of the hypercube. Each dimension has length [-L, L]
		"""

		super().__init__(**kwargs)

		self.dim = dim
		assert cfg.hyperpars.nu > 2, "Requirement: nu > 2"
		self.nu = cfg.hyperpars.nu # Related to t-Student's distribution
		sigma_n_init = cfg.hyperpars.sigma_n.init

		self.Nfeat = cfg.hyperpars.weights_features.Nfeat
		
		# Specify weights:
		# self.log_diag_vals = self.add_weight(shape=(self.Nfeat,), initializer=tf.keras.initializers.Zeros(), trainable=True,name="log_diag_vars")
		self.log_noise_std = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(tf.math.log(sigma_n_init)), trainable=True,name="log_noise_std")

		# Learning parameters:
		self.learning_rate = cfg.learning.learning_rate
		self.epochs = cfg.learning.epochs
		self.stop_loss_val = cfg.learning.stopping_condition.loss_val

		# No data case:
		self.X = None
		self.Y = None





		# ----------------------------------------------------------------------------------------------------------
		# Parameters only relevant to child classes
		# ----------------------------------------------------------------------------------------------------------

		# Spectral density to be used:
		self.spectral_density = spectral_density

		# This model assumes a dim-dimensional input and a scalar output.
		# We need to select the output we care about for the spectral density points:
		self.select_output_dimension(dim_out_int)
		
		self.S_samples_vec = self.spectral_density.Sw_points[:,self.dim_out_ind:self.dim_out_ind+1] # [Npoints,1]
		self.phi_samples_vec = self.spectral_density.phiw_points[:,self.dim_out_ind:self.dim_out_ind+1] # [Npoints,1]
		self.W_samples = self.spectral_density.W_points # [Npoints,self.dim]

		# ----------------------------------------------------------------------------------------------------------
		# ----------------------------------------------------------------------------------------------------------


	def select_output_dimension(self,dim_out_ind):
		assert dim_out_ind >= 0 and dim_out_ind <= self.dim
		self.dim_out_ind = dim_out_ind

	def add2dataset(self,xnew,ynew):
		pass

	def get_noise_var(self):
		"""

		TODO: Think about maybe using the softplus transform log(1 + exp(x))
		https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus
		"""

		ret = tf.exp(2.0*self.log_noise_std)

		# if tf.math.reduce_any(tf.math.is_nan(ret)):
		# 	pdb.set_trace()

		return ret

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

	def get_logdetSigma_weights(self):
		raise NotImplementedError

	def get_MLII_loss_gaussian(self):
		"""

		TODO: Not used. Move to the corresponding class rrgp
		TODO: Make sure that we call this function self.get_MLII_loss() after calling self.update_model()
		TODO: Remove this method and place it in rrgp.py

		"""

		Lchol = tf.linalg.cholesky(tf.transpose(self.PhiX) @ self.PhiX + self.get_Sigma_weights_inv_times_noise_var() ) # Lower triangular A = L.L^T

		# Compute Ky_inv:
		K11_inv = 1/self.get_noise_var()*( tf.eye(self.X.shape[0]) - self.PhiX @ tf.linalg.cholesky_solve(Lchol, tf.transpose(self.PhiX)) )

		data_fit = -0.5*tf.transpose(self.Y - self.M) @ (K11_inv @ (self.Y-self.M))

		model_complexity = -0.5*self.get_logdetSigma_weights() - tf.reduce_sum( tf.math.log( tf.linalg.diag_part(Lchol) ) )

		return -data_fit - model_complexity

	def get_MLII_loss(self):
		"""

		Compute the negative log evidence for a multivariate Student-t distribution
		The terms that do not depend on the hyperprameters (defined with
		self.add_weight() in self.__init__()) have not been included

		NOTE: The model complexity term that depends on the hyperparameters and data
		is the same is in the Gaussian case, i.e., log(det(Ky)^{-0.5})
		
		TODO: Make sure that we call this function self.get_MLII_loss() after calling self.update_model()
		"""

		# logger.info("    Computing cholesky decomposition of {0:d} x {1:d} matrix ...".format(self.PhiX.shape[1],self.PhiX.shape[1]))

		Kmat = tf.transpose(self.PhiX) @ self.PhiX + self.get_Sigma_weights_inv_times_noise_var()

		try:
			Lchol = tf.linalg.cholesky(Kmat) # Lower triangular A = L.L^T
		except Exception as inst:
			logger.info("type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			logger.info("@get_MLII_loss: Failed to compute: chol( PhiX^T.PhiX + Diag_mat ) ...")

			logger.info("Modifying Sigma_weights by adding eigval: {0:f} ...".format(float(min_eigval)))
			# min_eigval_posi = self.fix_eigvals(self.PhiX)
			Kmat = self.fix_eigvals(Kmat)
			# return 10*min_eigval_posi
			# return 10.0*tf.reduce_max(self.PhiX)
			# Lchol = tf.linalg.cholesky(tf.transpose(self.PhiX) @ self.PhiX + self.get_Sigma_weights_inv_times_noise_var() + min_eigval*tf.eye(self.PhiX.shape[1]) ) # Lower triangular A = L.L^T
			Lchol = tf.linalg.cholesky(Kmat) # Lower triangular A = L.L^T

			# raise ValueError("Failed to compute: chol( PhiX^T.PhiX + Diag_mat )")
			# logger.info("@get_MLII_loss(): Returning Inf....")
			# return tf.constant([[float("Inf")]])

		# Compute Ky_inv:
		# pdb.set_trace()
		K11_inv = 1/self.get_noise_var()*( tf.eye(self.X.shape[0]) - self.PhiX @ tf.linalg.cholesky_solve(Lchol, tf.transpose(self.PhiX)) )

		# Compute data fit:
		term_data_fit = tf.clip_by_value(tf.transpose(self.Y - self.M) @ (K11_inv @ (self.Y-self.M)),clip_value_min=0.0,clip_value_max=float("Inf"))
		# term_data_fit = tf.transpose(self.Y - self.M) @ (K11_inv @ (self.Y-self.M))

		data_fit = -0.5*(self.nu + self.X.shape[0])*tf.math.log1p( term_data_fit / (self.nu-2.) )

		# Compute model complexity:
		# A = det(Lchol) = prod(diag_part(Lchol))
		# log(A) = sum(log(diag_part(Lchol)))
		model_complexity = -0.5*self.get_logdetSigma_weights() - tf.reduce_sum( tf.math.log( tf.linalg.diag_part(Lchol) ) )

		loss_val = -data_fit - model_complexity

		# pdb.set_trace()
		# if tf.math.is_nan(loss_val):
		# 	pdb.set_trace()
		# 	return tf.constant([[float("Inf")]])

		return loss_val

	def train_model(self):
		"""

		"""

		logger.info("Training the model...")

		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		# logger.info(self.trainable_weights[0][0:10])
		# logger.info(self.trainable_weights[1])

		epoch = 0
		done = False
		while epoch < self.epochs and not done:

			with tf.GradientTape() as tape:

				# pdb.set_trace()
				loss_value = self.get_MLII_loss()

			grads = tape.gradient(loss_value, self.trainable_weights)
			optimizer.apply_gradients(zip(grads, self.trainable_weights))

			if (epoch+1) % 10 == 0:
				logger.info("Training loss at epoch %d / %d: %.4f" % (epoch+1, self.epochs, float(loss_value)))

			# Stopping condition:
			if not tf.math.is_nan(loss_value):
				if loss_value <= self.stop_loss_val:
					done = True
			
			epoch += 1

		if done == True:
			logger.info("Training finished because loss_value = {0:f} (<= {1:f})".format(float(loss_value),float(self.stop_loss_val)))

		# logger.info(self.trainable_weights[0])
		# logger.info(self.trainable_weights[1])

	def update_model(self,X,Y):

		logger.info("Updating model...")

		self._update_dataset(X,Y)
		self._update_features()

	def _update_dataset(self,X,Y):
		self.X = X

		if Y.ndim == 1:
			self.Y = tf.reshape(Y,(-1,1))
		else:
			assert Y.ndim == 2
			self.Y = Y

	def _update_features(self):
		"""

		Compute some relevant quantities that will be later used for prediction:
		PhiX, 
		"""

		logger.info("Computing matrix of features PhiX ...")
		self.PhiX = self.get_features_mat(self.X)
		
		# pdb.set_trace()

		PhiXTPhiX = tf.transpose(self.PhiX) @ self.PhiX

		Sigma_weights_inv_times_noise_var = self.get_Sigma_weights_inv_times_noise_var() # A

		logger.info("    Computing cholesky decomposition of {0:d} x {1:d} matrix ...".format(PhiXTPhiX.shape[0],PhiXTPhiX.shape[1]))
		try:
			self.Lchol = tf.linalg.cholesky(PhiXTPhiX + Sigma_weights_inv_times_noise_var) # Lower triangular A = L.L^T
		except Exception as inst:
			
			logger.info("type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			logger.info("Failed to compute: chol( PhiX^T.PhiX + Diag_mat ) || Fixing it...")

			# Fix it by modifying the hyperparameters:
			# Extract the most negative eigenvalue:
			aux2 = tf.math.real(tf.eigvals(PhiXTPhiX))
			min_eigval_posi = tf.abs(tf.reduce_min(aux2))
			min_eigval_posi = 1.1*min_eigval_posi

			# Add such eigenvalue as jitter:
			if hasattr(self, 'log_diag_vals'):
				self.log_diag_vals.assign( -tf.math.log(tf.exp(-self.log_diag_vals) + min_eigval_posi*tf.exp(-2.0*self.log_noise_std)) )
				Sigma_weights_inv_times_noise_var = self.get_Sigma_weights_inv_times_noise_var()
			else:
				logger.info("Fixture only implemented for child classes where log_diag_vals is used.")
				logger.info("For other classes, this fuxture needs to be implemented...")

			logger.info("Modifying Sigma_weights by adding eigval: {0:f} ...".format(float(min_eigval_posi)))
			self.Lchol = tf.linalg.cholesky(PhiXTPhiX + Sigma_weights_inv_times_noise_var) # Lower triangular A = L.L^T
			
			# raise ValueError("Failed to compute: chol( PhiX^T.PhiX + Diag_mat )")
			# bbb = tf.transpose(self.PhiX) @ self.PhiX + Sigma_weights_inv_times_noise_var + min_eigval*tf.eye(self.PhiX.shape[1])
			# bbb2 = tf.math.real(tf.eigvals(bbb))
			# logger.info("bbb2:",bbb2)
			# pdb.set_trace()

		# Prior mean:
		self.M = tf.zeros((self.X.shape[0],1))

	@staticmethod
	def fix_eigvals(Kmat):
		"""

		Among the negative eigenvalues, get the 'most negative one'
		and return it with flipped sign
		"""

		
		Kmat_sol = tf.linalg.cholesky(Kmat)
		if tf.math.reduce_any(tf.math.is_nan(Kmat_sol)):
			logger.info("Kmat needs to be fixed...")
		else:
			logger.info("Kmat is PD; nothing to fix...")
			return Kmat

		try:
			eigvals, eigvect = tf.linalg.eigh(Kmat)
		except Exception as inst:
			logger.info("type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			logger.info("Failed to compute tf.linalg.eigh(Kmat) ...")
			pdb.set_trace()

		max_eigval = tf.reduce_max(tf.math.real(eigvals))
		min_eigval = tf.reduce_min(tf.math.real(eigvals))

		# Compte eps:
		# eps must be such that the condition number of the resulting matrix is not too large
		max_order_eigval = tf.math.ceil(tf.experimental.numpy.log10(max_eigval))
		eps = 10**(max_order_eigval-8) # We set a maximum condition number of 8

		# Fix eigenvalues:
		eigvals_fixed = eigvals + tf.abs(min_eigval) + eps

		# pdb.set_trace()
		logger.info(" Fixed by adding " + str(tf.abs(min_eigval).numpy()))
		logger.info(" and also by adding " + str(eps.numpy()))

		Kmat_fixed = eigvect @ ( tf.linalg.diag(eigvals_fixed) @ tf.transpose(eigvect) ) # tf.transpose(eigvect) is the same as tf.linalg.inv(eigvect) | checked

		# Kmat_fixed_sym = 0.5*(Kmat_fixed + tf.transpose(Kmat_fixed))

		try:
			tf.linalg.cholesky(Kmat_fixed)
		except:
			pdb.set_trace()

		# pdb.set_trace()

		return Kmat_fixed


	@abstractmethod
	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""
		raise NotImplementedError

	def get_predictive_beta_distribution(self):
		"""

		TODO: Make sure this is the right way of inclduing the prior mean
		"""

		# Get mean:
		PhiXY = tf.transpose(self.PhiX) @ (self.Y - self.M)
		mean_beta = tf.linalg.cholesky_solve(self.Lchol, PhiXY)

		# Adding non-zero mean. Check that self.M is also non-zero
		# mean_pred += tf.zeros((xpred.shape[0],1))

		var_noise = self.get_noise_var()

		# Update parameters from the Student-t distribution:
		nu_pred = self.nu + self.X.shape[0]

		K11_inv = 1/var_noise*( tf.eye(self.X.shape[0]) - self.PhiX @ tf.linalg.cholesky_solve(self.Lchol, tf.transpose(self.PhiX)) )
		beta1 = tf.transpose(self.Y - self.M) @ ( K11_inv @ (self.Y - self.M) )
		cov_beta_factor_sqrt = tf.sqrt( (self.nu + beta1 - 2) / (nu_pred-2) * var_noise )


		# wishart_cholesky_to_iw_cholesky = tfp.bijectors.CholeskyToInvCholesky()
		# Mchol = wishart_cholesky_to_iw_cholesky.forward(tf.transpose(self.Lchol))
		# Mchol = wishart_cholesky_to_iw_cholesky.forward(self.Lchol)
		aaa = tf.linalg.inv(tf.transpose(self.Lchol))
		# pdb.set_trace()

		cov_beta_chol = tf.linalg.inv(tf.transpose(self.Lchol)) * cov_beta_factor_sqrt
		return mean_beta, cov_beta_chol

	def get_prior_beta_distribution(self):
		"""

		NOTE: here we are adding noise, but in self.get_predictive_beta_distribution() we aren't
		NOTE: The prior mean and covariance are hardcoded to zero
		NOTE: We can't have an additive noise model here. We do as they do in [2, Fig. 3], i.e., we add the noise
		directly to the covariance matrix, but this doesn't correspond to th covariance of t1+t2.

		The covariance is taken from [1, Sec. 7.7]
	
		[1] Petersen, K.B. and Pedersen, M.S., 2008. The matrix cookbook. Technical University of Denmark, 7(15), p.510.
		[2] Shah, A., Wilson, A. and Ghahramani, Z., 2014, April. Student-t processes as alternatives to Gaussian processes. In *Artificial intelligence and statistics* (pp. 877-885). PMLR.

		"""
		
		# Get prior cov:
		chol_cov_beta_prior = self.get_cholesky_of_cov_of_prior_beta() * tf.math.sqrt( (self.nu) / (self.nu - 2.) )

		# Get prior mean:
		mean_beta_prior = tf.zeros((chol_cov_beta_prior.shape[0],1))

		return mean_beta_prior, chol_cov_beta_prior

	def predict_beta(self,from_prior=False):

		if self.X is None and self.Y is None or from_prior:
			mean_beta, cov_beta_chol = self.get_prior_beta_distribution()
		else:
			mean_beta, cov_beta_chol = self.get_predictive_beta_distribution()

		return mean_beta, cov_beta_chol

	def predict_at_locations(self,xpred,from_prior=False):
		"""

		Optimize this by returning the diagonal elements of the cholesky decomposition of cov_pred

		"""

		mean_beta, cov_beta_chol = self.predict_beta(from_prior)

		logger.info("Computing matrix of features Phi(xpred) ...")
		Phi_pred = self.get_features_mat(xpred)

		# pdb.set_trace()
		mean_pred = Phi_pred @ mean_beta
		cov_pred_chol = Phi_pred @ cov_beta_chol
		cov_pred = cov_pred_chol @ tf.transpose(cov_pred_chol) # L.L^T

		return tf.squeeze(mean_pred), cov_pred

	def get_sample_path_callable(self,Nsamples,from_prior=False):

		mean_beta, cov_beta_chol = self.predict_beta(from_prior)

		Nfeat = cov_beta_chol.shape[0]
		sample_mvt0 = self.get_sample_mvt0(Nfeat,Nsamples)
		aux = tf.reshape(mean_beta,(-1,1)) + cov_beta_chol @ sample_mvt0

		def nonlinfun_sampled_callable(x):
			return self.get_features_mat(x) @ aux # [Npoints,Nsamples]

		return nonlinfun_sampled_callable

	def sample_path_from_predictive(self,xpred,Nsamples,from_prior=False):
		"""

		xpred: [Npoints,self.dim]
		sample_xpred: [Npoints,Nsamples]

		"""

		fx = self.get_sample_path_callable(Nsamples,from_prior)
		sample_xpred = fx(xpred)

		return sample_xpred

	def sample_state_space_from_prior_recursively(self,x0,x1,traj_length,sort=False,plotting=False):
		"""

		Pass two initial latent values
		x0: [1,self.dim]
		x1: [1,self.dim]

		The GP won't be training during sampling, i.e., we won't call self.train_model()
		"""

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

	def get_predictive_entropy_of_truncated_dist(self):
		"""

		https://link.springer.com/content/pdf/10.1016/j.jkss.2007.06.001.pdf
		"""

		pass

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

		return entropy

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

		# Sample from inverse Gamma:
		alpha = 0.5*self.nu
		beta = 0.5
		dist_ig = tfp.distributions.InverseGamma(concentration=alpha,scale=beta)
		sample_ig = dist_ig.sample(sample_shape=(Nsamples,1))

		# Sample from chi-squared:
		dist_chi2 = tfp.distributions.Chi2(df=Npred)
		sample_chi2 = dist_chi2.sample(sample_shape=(Nsamples,1))

		# Sample from MVT(nu,0,I):
		sample_mvt0 = tf.math.sqrt((self.nu-2) * sample_chi2 * sample_ig) * sample_sphe

		return tf.transpose(sample_mvt0) # [Npred,Nsamples]


	def call(self, inputs):
		# y = tf.matmul(inputs, self.w) + self.b
		# return tf.math.cos(y)
		logger.info("self.call(): <><><><>      This method should not be called yet... (!)      <><><><>")
		pass


