import tensorflow as tf
import pdb
import math
# import numpy as np
# import tensorflow.experimental.numpy as tnp # https://www.tensorflow.org/guide/tf_numpy

import tensorflow_probability as tfp

class ReducedRankBayesianLinearRegression:
	"""
	"""
	def __init__(self, dim, Nfeat, L, sigma_n):
		"""
		
		dim: Dimensionality of the input space
		Nfeat: Number of features
		L: Half Length of the hypercube. Each dimension has length [-L, L]
		"""
		self.dim = dim
		self.Nfeat = Nfeat
		self.L = tf.constant([L]*dim)
		self.jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]
		self.eigvals = self._get_eigenvalues()
		self.sigma_n = sigma_n
		self.sigma2_n = sigma_n**2

		self.spectral_density = lambda w: self.spectral_density_matern(w)
		# self.spectral_density = lambda w: 1.0*self.spectral_density_SE(w)

	def update_dataset(self,X,Y):
		self.X = X

		if Y.ndim == 1:
			self.Y = tf.reshape(Y,(-1,1))
		else:
			assert Y.ndim == 2
			self.Y = Y

		self._update_features()

	def add2dataset(self,xnew,ynew):
		pass

	def spectral_density_matern(self,omega_in):
		"""

		The spectral density function maps R+ -> R+

		However, look at the following TODO:
		# TODO: Look at eq. (4.15) from Rasmussen. It actually depends on the dimensionality D, and it's not multidimensional!!! :(
				See also appendix from Sarkaa
				See also the Sarkka lecture. therein, the density function is R^D -> R, which will imply major modification in this entire class
		"""

		# Using now the N-dimensional formulation from Rasmussen

		# Tunable parameters:
		p = 2
		nu = p + 0.5
		ls = 0.1
		prior_var = 10.0

		lambda_val = tf.sqrt(2*nu)/ls
		# S_vec = ((2*tf.sqrt(math.pi)*tf.exp(tf.math.lgamma(nu+0.5))) / (tf.exp(tf.math.lgamma(nu)))) * lambda_val**(2*nu)/((lambda_val**2 + omega_in**2)**(nu+0.5))
		const = ((2*tf.sqrt(math.pi))**self.dim)*tf.exp(tf.math.lgamma(nu+0.5*self.dim))*lambda_val**(2*nu) / tf.exp(tf.math.lgamma(nu))
		S_vec = const / ((lambda_val**2 + omega_in**2)**(nu+self.dim*0.5)) # Using omega directly (Sarkka) as opposed to 4pi*s (rasmsusen)

		# Pump variance into it:
		S_vec = S_vec*prior_var

		# print("S_vec:",S_vec)

		# from scipy.special import gamma
		# import numpy as np
		# S_vec = self.sigma2_n * ((2*np.sqrt(np.pi)*gamma(nu+0.5)) / (gamma(nu))) * lambda_val**(2*nu)/((lambda_val**2 + omega_in**2)**(nu+0.5))
		# print("S_vec np:",S_vec)

		# pdb.set_trace()

		return S_vec


	def spectral_density_SE(self,omega_in):

		ls = 0.1

		const = (tf.sqrt(2*math.pi)*ls)**self.dim
		S_vec = const * tf.exp( -2*math.pi**2 * ls**2 * omega_in**2 )

		return S_vec

	def _update_features(self):
		"""

		Cache the expensive operation
		"""

		self.PhiX = self.get_features_mat(self.X)

		# TODO: See if we can 'cache' the line below:
		Lambda_inv_times_noise_var = tf.linalg.diag( self.sigma2_n * 1./self.spectral_density(tf.sqrt(self.eigvals)) ) # original
		# Lambda_inv_times_noise_var = self.sigma2_n*tf.eye(self.Nfeat) # [DBG] qudratic features
		
		self.Lchol = tf.linalg.cholesky(tf.transpose(self.PhiX) @ self.PhiX + Lambda_inv_times_noise_var ) # Lower triangular L.L^T

		self.PhiXY = tf.transpose(self.PhiX) @ self.Y


		# Distribution over the weigths th, such that: th_i ~ N(Ainv @ PhiXT @ Y , sigma2_n*Ainv ), for i=[1,..,n_samples]
		# To be used in
		self.mean_th_post = tf.linalg.cholesky_solve(self.Lchol, self.PhiXY)
		self.chol_cov_th_post = tf.math.sqrt(self.sigma2_n) * tf.transpose(tf.linalg.inv(self.Lchol))


	def _get_eigenvalues(self):
		"""

		Eigenvalues of the laplace operator
		"""

		Lstack = tf.stack([self.L]*self.Nfeat) # [Nfeat, dim]
		jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1)) # [Nfeat, 1]
		# pdb.set_trace()
		Ljj = (math.pi * jj / (2.*Lstack))**2 # [Nfeat, dim]

		return tf.reduce_sum(Ljj,axis=1) # [Nfeat,]

	def get_features_mat(self,x):
		"""

		x: [Npoints, dim]
		return: [Npoints, Nfeat]
		"""
		
		xx = tf.stack([x]*self.Nfeat) # [Nfeat, Npoints, dim]
		# jj = tf.reshape(tf.range(self.Nfeat,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]
		# pdb.set_trace()
		feat = 1/tf.sqrt(self.L) * tf.sin( math.pi*self.jj*(xx + self.L)/(2.*self.L) ) # [Nfeat, Npoints, dim]
		return tf.transpose(tf.reduce_prod(feat,axis=-1)) # [Npoints, Nfeat]

	# def get_features_mat(self,X):
	# 	"""
	# 	[DBG] qudratic features

	# 	X: [Npoints, in_dim]
	# 	return: PhiX: [Npoints, Nfeat]
	# 	"""

	# 	Npoints = X.shape[0]
	# 	SQRT2 = math.sqrt(2)

	# 	if self.dim == 1:
	# 		PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , X**2 ],axis=1)
	# 	elif self.dim == 2:
	# 		PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , SQRT2*tf.math.reduce_prod(X,axis=1,keepdims=True) , X**2 ],axis=1)
	# 	else:
	# 		raise NotImplementedError

	# 	return PhiX # [Npoints, Nfeat]

	def get_features_mat_grad(self,x):
		"""
		
		We compute the features of the gradients of the input x by piling them up as [ (dim dim ... dim) , Nfeat]
		"""

		xx = tf.stack([x]*self.Nfeat) # [Nfeat, Npoints, dim]
		Nfeat = xx.shape[0]
		Npoints = xx.shape[1]
		# feat_grad = np.zeros((self.dim,Npoints,Nfeat)) # [dim, Npoints, Nfeat]
		feat_grad_list = [None]*self.dim

		for ii in range(self.dim):

			# To create phase_der:
				# Tensorflow doesn't uspport item assignment in a slicing operation:
				# phase_der = tf.zeros(self.dim)
				# phase_der[ii] = math.pi/2.0 # Breaks

				# Masking: https://github.com/tensorflow/tensorflow/issues/14132#issuecomment-483002522
				# new = original * mask + other * (1 - mask)

				# We just use numpy because tensor + ndarray = tensor -> Yes, but there's an easier way that dowsn't involve importing numpy
				# phase_der = tnp.zeros(self.dim)
				# phase_der[ii] = math.pi/2.0 

			# Easier way:
			phase_der = tf.constant([0.0]*ii + [math.pi/2.0] + [0.0]*(self.dim - ii - 1)) #[0, 0, ..., pi/2, ..., 0, 0], where pi/2 is placed at index ii

			# No problem with this:
			# Notice the term (math.pi*self.jj/(2.*self.L))**(1/self.dim). It comes
			# from derivating the sin() function. It needs to be exponentiated to
			# **(1/self.dim) because later the prod() across dimensions will be taken.
			# TODO: Figure out a way of not having to exponentiate with **(1/self.dim)
			# TODO: Make sure the 1/tf.sqrt(self.L) isn't redundant with the new termn. Where does 1/tf.sqrt(self.L) come from???
			feat_der = (math.pi*self.jj/(2.*self.L))**(1/self.dim) * 1/tf.sqrt(self.L) * tf.sin( math.pi*self.jj*(xx + self.L)/(2.*self.L) + phase_der) # [Nfeat, Npoints, dim]

			# To create feat_grad with dimensions [dim, Npoints, Nfeat]:
				# This involves item assignment, which is forbidden in tensorflow. We could define feat_grad using numpy to do the item assignment. But then, we depend on numpy. tensorflow.experimental.numpy also doesn't support item assignment.
				# feat_grad[ii,:,:] = tf.transpose(tf.reduce_prod(feat_der,axis=-1)) # [ii, Npoints, Nfeat] <- [Npoints, Nfeat]

			# Instead, we create a list. Later on we use stack
			feat_grad_list[ii] = tf.transpose(tf.reduce_prod(feat_der,axis=-1)) # [ii] <- [Npoints, Nfeat]

		feat_grad = tf.stack(feat_grad_list) # [dim, Npoints, Nfeat]

		# We pile up the gradients as [ (dim dim ... dim) , Nfeat], where dim is repeated Npoints times in (dim dim ... dim) 
		feat_grad2 = tf.transpose(feat_grad,perm=[1,0,2]) # The returned tensor's dimension i will correspond to the input dimension perm[i]. So, we permute the first two.
		feat_grad3 = tf.reshape(feat_grad2,(Npoints*self.dim,Nfeat)) # Now, by reshaping, we get what we were looking for

		return feat_grad3 # [Npoints*dim, Nfeat] 


	def get_features_mat_points_grad(self,x,x4grad):
		"""

		Npoints = x.shape[0]
		Npoints4grad = x4grad.shape[0]

		We want to return [ (Npoints dim dim ... dim) , Nfeat], where dim is repeated Npoints4grad times in (Npoints dim dim ... dim) 
		The returned matrix is [ Npoints + dim*Npoints4grad , Nfeat ]
		
		If x = x4grad, the computations should be simpler

		"""
		pass

	def get_predictive_moments_grad(self,xpred):
		"""

		TODO: In the future, return a matrix cov_pred_der: [Npoints, dim, dim]
		Essentially, for each of the Npoints gradients, the covariance matrix of the gradient with itself needs to be computed across all dimensions, and hence, it is a tensor [dim x dim].
		Hence, for each of the Npoints gradients we have one [dim x dim] matrix.
		We might need this. This can be returned as a [Npoints x dim x dim] tensor, or as a block-diagonal matrix [Npoints*dim x Npoints*dim]
		
		Instead, what we are returning right now is only the diagonal elements of such covariance matrix, which results in a matrix [Npoints, dim]
		"""

		Phi_pred = self.get_features_mat_grad(xpred)
		
		# Get mean:
		
		mean_pred = Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, PhiXY) # In the Sarkka paper Phi_pred is transposed, but it should be wrong...

		# Get covariance:
		cov_pred = self.sigma2_n * Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, tf.transpose(Phi_pred))

		# Reshape into matrices [Npoints, dim] by piling up the gradient of each point.
		# mean_pred is [Npoints*dim] as [(dim dim ... dim)] repeated Npoints times
		# std_pred_der is [Npoints*dim] as [(dim dim ... dim)] repeated Npoints times
		Npoints = xpred.shape[0]

		mean_pred_rs = tf.reshape(mean_pred,(Npoints,self.dim))
		
		std_pred_der = tf.sqrt(tf.linalg.diag_part(cov_pred))
		std_pred_der_rs = tf.reshape(std_pred_der,(Npoints,self.dim))

		# pdb.set_trace()
		return mean_pred_rs, std_pred_der_rs

	def get_predictive_moments(self,xpred):
		
		# MATLAB code from Särkkä:
		# % Eigenfunctions
		# Phit   = eigenfun(NN,xt);
		# Phi    = eigenfun(NN,x); 
		# PhiPhi = Phi'*Phi;        % O(nm^2)
		# Phiy   = Phi'*y;
		# lambda = eigenval(NN)';

		#   % Solve GP with optimized hyperparameters and 
		# % return predictive mean and variance 
		# k = S(sqrt(lambda),lengthScale,magnSigma2);
		# L = chol(PhiPhi + diag(sigma2./k),'lower'); 
		# Eft = Phit*(L'\(L\Phiy));
		# Varft = sigma2*sum((Phit/L').^2,2); 

		# % Notice boundaries
		# Eft(abs(xt) > Lt) = 0; Varft(abs(xt) > Lt) = 0;

		Phi_pred = self.get_features_mat(xpred)
		
		# Get mean:
		mean_pred = Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, self.PhiXY) # In the Sarkka paper Phi_pred is transposed, but it should be wrong...

		# Get covariance:
		cov_pred = self.sigma2_n * Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, tf.transpose(Phi_pred))

	
		return tf.squeeze(mean_pred), cov_pred


	def sample_from_predictive(self,xpred,n_samples):
		"""

		NOTE: This is rather inefficient. Use self.get_posterior_weights_samples_for_callable_from_predictive()

		"""
		
		mean, cov = self.get_predictive_moments(xpred)

		mvn = tfp.distributions.MultivariateNormalTriL(loc=mean,scale_tril=tf.linalg.cholesky(cov))

		samples = mvn.sample(sample_shape=(n_samples))

		return samples



	def get_posterior_weights_samples_for_callable_from_predictive(self,n_samples):
		"""

		Return samples th_i ~ N(Ainv @ PhiXT @ Y , sigma2_n*Ainv ), for i=[1,..,n_samples]

		Then, a callabale can be constructed as f(x) ~= phi(x)^T @ th_t

		"""

		th_samples = self.mean_th_post + self.chol_cov_th_post @ tf.random.normal(shape=(self.mean_th_post.shape[0],n_samples))

		# # Perturb e.g. the mean slightly; the predictive power gets completely lost. It seems tha this solution is powerful, but very non-robust and prone to unstability
		# th_samples = self.mean_th_post + 1e-2*tf.ones(self.mean_th_post.shape) + self.chol_cov_th_post @ tf.random.normal(shape=(self.mean_th_post.shape[0],n_samples))
		# So, it seems that there's an optimal distribution of weights th_i ~ N(m,V) that make the model predict with accuracy, while with any other comnation of m and V,
		# we'd lose all predictive power.

		return th_samples

