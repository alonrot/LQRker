import tensorflow as tf
import pdb
import math
# import numpy as np
# import tensorflow.experimental.numpy as tnp # https://www.tensorflow.org/guide/tf_numpy

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

	def spectral_density_matern(self,omega_vec):
		# TODO: Make sure that here we are required to enter a vetor, not a scalar (see appendix from Sarkaa)
		# TODO: Look at eq. (4.15) from Rasmussen. It actually depends on the dimensionality D, and it's not multidimensional!!! :(

		# Using now the N-dimensional formulation from Rasmussen

		p = 3
		nu = p + 0.5
		ls = 0.5
		lambda_val = tf.sqrt(2*nu)/ls

		# S_vec = ((2*tf.sqrt(math.pi)*tf.exp(tf.math.lgamma(nu+0.5))) / (tf.exp(tf.math.lgamma(nu)))) * lambda_val**(2*nu)/((lambda_val**2 + omega_vec**2)**(nu+0.5))
		const = ((2*tf.sqrt(math.pi))**self.dim)*tf.exp(tf.math.lgamma(nu+0.5*self.dim))*lambda_val**(2*nu) / tf.exp(tf.math.lgamma(nu))
		S_vec = const / ((lambda_val**2 + omega_vec**2)**(nu+self.dim*0.5)) # Using omega directly (Sarkka) as opposed to 4pi*s (rasmsusen)

		# print("S_vec:",S_vec)

		# from scipy.special import gamma
		# import numpy as np
		# S_vec = self.sigma2_n * ((2*np.sqrt(np.pi)*gamma(nu+0.5)) / (gamma(nu))) * lambda_val**(2*nu)/((lambda_val**2 + omega_vec**2)**(nu+0.5))
		# print("S_vec np:",S_vec)

		# pdb.set_trace()

		return S_vec


	def spectral_density_SE(self,omega_vec):

		ls = 0.1

		const = (tf.sqrt(2*math.pi)*ls)**self.dim
		S_vec = const * tf.exp( -2*math.pi**2 * ls**2 * omega_vec**2 )

		return S_vec

	def _update_features(self):
		"""

		Cache the expensive operation
		"""

		self.PhiX = self.get_features_mat(self.X)
		Lambda_inv_times_noise_var = tf.linalg.diag( self.sigma2_n * 1./self.spectral_density(tf.sqrt(self.eigvals)) )
		self.Lchol = tf.linalg.cholesky(tf.transpose(self.PhiX) @ self.PhiX + Lambda_inv_times_noise_var ) # Lower triangular

	def _get_eigenvalues(self):

		Lstack = tf.stack([self.L]*self.Nfeat) # [Nfeat, dim]
		jj = tf.reshape(tf.range(1,self.Nfeat+1,dtype=tf.float32),(-1,1)) # [Nfeat, 1]
		Ljj = (math.pi * jj / (2.*Lstack))**2 # [Nfeat, dim]

		return tf.reduce_sum(Ljj,axis=1) # [Nfeat,]

	def get_features_mat(self,x):
		
		xx = tf.stack([x]*self.Nfeat) # [Nfeat, Npoints, dim]
		# jj = tf.reshape(tf.range(self.Nfeat,dtype=tf.float32),(-1,1,1)) # [Nfeat, 1, 1]
		# pdb.set_trace()
		feat = 1/tf.sqrt(self.L) * tf.sin( math.pi*self.jj*(xx + self.L)/(2.*self.L) ) # [Nfeat, Npoints, dim]
		return tf.transpose(tf.reduce_prod(feat,axis=-1)) # [Npoints, Nfeat]

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
			feat_der = math.pi*self.jj/(2.*self.L) * 1/tf.sqrt(self.L) * tf.sin( math.pi*self.jj*(xx + self.L)/(2.*self.L) + phase_der) # [Nfeat, Npoints, dim]

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
		Essentially, for each of the Npoints gradients, the covariance matrix of the gradient with itself needs to be computed across all dimensions, and hence, it is a tensor.
		We might need this.
		Instead, what we are returning right now is only the diagonal elements of such covariance matrix, which results in a matrix [Npoints, dim]
		"""

		Phi_pred = self.get_features_mat_grad(xpred)
		
		# Get mean:
		PhiXY = tf.transpose(self.PhiX) @ self.Y
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
		PhiXY = tf.transpose(self.PhiX) @ self.Y
		mean_pred = Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, PhiXY) # In the Sarkka paper Phi_pred is transposed, but it should be wrong...

		# Get covariance:
		cov_pred = self.sigma2_n * Phi_pred @ tf.linalg.cholesky_solve(self.Lchol, tf.transpose(Phi_pred))

	
		return tf.squeeze(mean_pred), cov_pred




