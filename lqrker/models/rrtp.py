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
from lqrker.utils.spectral_densities import MaternSpectralDensity, CartPoleSpectralDensity
logger = get_logger(__name__)


"""
Refactor:

1) Make the self.log_diag_vals sub-class dependent. For Sarkka features we have one thing,
for Random F.F we have another thing
2) Study in rrblr.py the effect of an uncentered x in the domain. Also, study what happens if we use
random eigenvalues, or if we let the j's be picked randomly or inferred from data. If that doesn't improve the
predictions, it could be that the system dynamics are not anywhere close to a Matern process...?
3) Try to derive the kernel following Sarkka's approach for the inverted pendulum equations, by linearizing
"""



class ReducedRankStudentTProcessBase(ABC,tf.keras.layers.Layer):
	"""

	Reduced-Rank Student-t Process
	==============================
	We implement the Student-t process presented in [1]. However, instead of using
	a kernel function, we use the weight-space view [] weight-space view from [1,
	Sec. 2.1.2] in order to reduce computational speed by using a finite set of
	features.

	We assume zero mean. Extending it non-zero mean is trivial.


	[1] Rasmussen, C.E. and Nickisch, H., 2010. Gaussian processes for machine
	learning (GPML) toolbox. The Journal of Machine Learning Research, 11,
	pp.3011-3015.

	[2] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank
	Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.
	"""
	# def __init__(self, dim, Nfeat, sigma_n, nu, **kwargs):
	def __init__(self, dim: int, cfg: dict, **kwargs):
		"""
		
		dim: Dimensionality of the input space
		Nfeat: Number of features
		L: Half Length of the hypercube. Each dimension has length [-L, L]
		"""

		logger.info("Initializing ReducedRankStudentTProcessBase() class ...")

		super().__init__(**kwargs)

		self.dim = dim
		assert cfg.hyperpars.nu > 2, "Requirement: nu > 2"
		self.nu = cfg.hyperpars.nu # Related to t-Student's distribution
		Nfeat = cfg.hyperpars.weights_features.Nfeat
		sigma_n_init = cfg.hyperpars.sigma_n.init

		# Specify weights:
		# self.log_diag_vals = self.add_weight(shape=(Nfeat,), initializer=tf.keras.initializers.Zeros(), trainable=True,name="log_diag_vars")
		self.log_noise_std = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(tf.math.log(sigma_n_init)), trainable=True,name="log_noise_std")

		# Learning parameters:
		self.learning_rate = cfg.learning.learning_rate
		self.epochs = cfg.learning.epochs
		self.stop_loss_val = cfg.learning.stopping_condition.loss_val


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

		Sigma_weights_inv_times_noise_var = self.get_Sigma_weights_inv_times_noise_var()

		# self.DBG_eigvals = tf.eigvals(tf.transpose(self.PhiX) @ self.PhiX)

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

	def get_predictive_moments(self,xpred):
		
		logger.info("Computing matrix of features Phi(xpred) ...")
		Phi_pred = self.get_features_mat(xpred)
		
		# Get mean:
		PhiXY = tf.transpose(self.PhiX) @ (self.Y - self.M)
		mean_pred = Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, PhiXY)

		# Adding non-zero mean. Check that self.M is also non-zero
		# mean_pred += tf.zeros((xpred.shape[0],1))

		var_noise = self.get_noise_var()

		# Get covariance:
		K22 = var_noise * Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, tf.transpose(Phi_pred))

		# Update parameters from the Student-t distribution:
		nu_pred = self.nu + self.X.shape[0]

		# We copmute K11_inv using the matrix inversion lemma to avoid cubic complexity on the number of evaluations
		K11_inv = 1/var_noise*( tf.eye(self.X.shape[0]) - self.PhiX @ tf.linalg.cholesky_solve(self.Lchol, tf.transpose(self.PhiX)) )
		beta1 = tf.transpose(self.Y - self.M) @ ( K11_inv @ (self.Y - self.M) )
		cov_pred = (self.nu + beta1 - 2) / (nu_pred-2) * K22

		return tf.squeeze(mean_pred), cov_pred

	# def sample_prior_ssm(self,x0):

	# 	Phi0 = self.get_features_mat(x0)

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

	def sample_mvt0(self,Npred,Nsamples):
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

	def sample_path(self,mean_pred,cov_pred,Nsamples):
		"""

		Sample a path from the posterior MTV as:
		return: mean + Lchol @ MVT(nu,0,I) # [Npred, Nsamples]
		"""

		Npred = cov_pred.shape[0]
		Lchol_cov_pred = tf.linalg.cholesky(cov_pred + 5e-5*tf.eye(cov_pred.shape[0]))
		aux = tf.reshape(mean_pred,(-1,1)) + Lchol_cov_pred @ self.sample_mvt0(Npred,Nsamples)
		# pdb.set_trace()
		# aux = tf.reshape(mean_pred,(-1,1)) + Lchol_cov_pred @ tf.random.normal(shape=(cov_pred.shape[0],1), mean=0.0, stddev=1.0)

		return aux # [Npred,Nsamples]

	def get_predictive_entropy_of_truncated_dist(self):
		"""

		https://link.springer.com/content/pdf/10.1016/j.jkss.2007.06.001.pdf
		"""

		pass


	def call(self, inputs):
		# y = tf.matmul(inputs, self.w) + self.b
		# return tf.math.cos(y)
		logger.info("self.call(): <><><><>      This method should not be called yet... (!)      <><><><>")
		pass

class RRTPRandomFourierFeatures(ReducedRankStudentTProcessBase):
	"""
	
	TODO:
	1) Maybe sample just a single vector, like in the jmlr paper
	2) Think about optimizing weights somehow. Prior on spectral density with model?
	3) Add other hyperparameters as trainable variables to the optimization
	4) Refactor all this in different files
	5) How can we infer the dominant frquencies from data? Can we compute S(w|Data) ?
	"""

	def __init__(self, dim: int, cfg: dict, spectral_density):

		super().__init__(dim,cfg)

		self.Nfeat = cfg.hyperpars.weights_features.Nfeat

		# Spectral density to be used:
		self.spectral_density = spectral_density
		# self.spectral_density = CartPoleSpectralDensity(cfg.spectral_density_pars)
		# self.spectral_density = MaternSpectralDensity(cfg.spectral_density,dim)

	def update_spectral_density(self,args,state_ind):

		self.spectral_density.update_pars(args) # Left for compatibility with CartPoleSpectralDensity() and others

		# Sample from the density:
		W_samples_vec, S_samples_vec, phi_samples_vec = self.spectral_density.get_samples() # [Nsamples,1,dim], [Nsamples,]
		self.W_samples = tf.reshape(W_samples_vec,(self.Nfeat,self.dim))
		self.S_samples_vec = S_samples_vec
		self.phi_samples_vec = phi_samples_vec
		self.u_samples = tfp.distributions.Uniform(low=0.0, high=2.*math.pi).sample(sample_shape=(1,self.Nfeat))

		# print("self.W_samples:",self.W_samples)

	def get_features_mat(self,X):
		"""

		X: [Npoints, dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		WX = tf.transpose(self.W_samples @ tf.transpose(X)) # [Npoints, Nfeat]

		# self.phi_samples_vec = 0.0
		# harmonics_vec = tf.math.cos(WX + self.u_samples) # [Npoints, Nfeat], with random phases
		harmonics_vec = tf.math.cos(WX + self.phi_samples_vec) # [Npoints, Nfeat]
		harmonics_vec_scaled = harmonics_vec * tf.reshape(self.S_samples_vec,(1,-1)) # [Npoints, Nfeat]

		return harmonics_vec_scaled
		
	def get_Sigma_weights_inv_times_noise_var(self):
		return self.get_noise_var() * tf.eye(self.Nfeat)

	def get_logdetSigma_weights(self):
		return 0.0


class RRTPSarkkaFeatures(ReducedRankStudentTProcessBase):

	def __init__(self, dim: int, cfg: dict):

		super().__init__(dim,cfg)

		self.Nfeat = cfg.hyperpars.weights_features.Nfeat
		self.L = cfg.hyperpars.L
		self.jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]

		# Spectral density to be used:
		self.spectral_density = MaternSpectralDensity(cfg.spectral_density,dim)

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

	def __init__(self, dim: int, cfg: dict):
		super().__init__(dim,cfg)

		# Get parameters:
		nu = cfg.hyperpars.nu
		Nsys = cfg.hyperpars.weights_features.Nfeat # Use as many systems as number of features

		# TODO: Test the line below
		self.log_diag_vals = self.add_weight(shape=(Nsys,), initializer=tf.keras.initializers.Zeros(), trainable=True,name="log_diag_vars")

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

