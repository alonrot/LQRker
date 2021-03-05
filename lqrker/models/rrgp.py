import tensorflow as tf
import pdb
import math
from abc import ABC, abstractmethod
# import numpy as np
# import tensorflow.experimental.numpy as tnp # https://www.tensorflow.org/guide/tf_numpy

class ReducedRankGaussianProcessBase(ABC):
	"""

	Reduced-Rank Gaussian Process
	=============================
	Here we use the weight-space view from [1, Sec. 2.1.2] in order to reduce computational
	speed by using a finite set of features. The equations correspond to the equivalent formulation
	in [2].

	We assume zero mean. Extending it non-zero mean is trivial.


	[1] Rasmussen, C.E. and Nickisch, H., 2010. Gaussian processes for machine
	learning (GPML) toolbox. The Journal of Machine Learning Research, 11,
	pp.3011-3015.

	[2] Solin, A. and Särkkä, S., 2020. Hilbert space methods for reduced-rank
	Gaussian process regression. Statistics and Computing, 30(2), pp.419-446.
	"""
	def __init__(self, dim, Nfeat, sigma_n):
		"""
		
		dim: Dimensionality of the input space
		Nfeat: Number of features
		L: Half Length of the hypercube. Each dimension has length [-L, L]
		"""
		self.dim = dim
		self.Nfeat = Nfeat
		self.sigma2_n = sigma_n**2

	def add2dataset(self,xnew,ynew):
		pass

	def update_model(self,X,Y):

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

		Cache the expensive operation
		"""

		self.PhiX = self.get_features_mat(self.X)

		# TODO: See if we can 'cache' the line below:
		Lambda_inv_times_noise_var = self.sigma2_n*tf.eye(self.Nfeat) # [DBG] qudratic features
		
		self.Lchol = tf.linalg.cholesky(tf.transpose(self.PhiX) @ self.PhiX + Lambda_inv_times_noise_var ) # Lower triangular

	@abstractmethod
	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""
		raise NotImplementedError

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


class RRGPQuadraticFeatures(ReducedRankGaussianProcessBase):

	def __init__(self, dim, Nfeat, sigma_n):

		assert dim <= 2

		super().__init__(dim, Nfeat, sigma_n)

	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		Npoints = X.shape[0]
		SQRT2 = math.sqrt(2)

		if self.dim == 1:
			PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , X**2 ],axis=1)
		elif self.dim == 2:
			PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , SQRT2*tf.math.reduce_prod(X,axis=1,keepdims=True) , X**2 ],axis=1)
		else:
			raise NotImplementedError

		return PhiX # [Npoints, Nfeat]


