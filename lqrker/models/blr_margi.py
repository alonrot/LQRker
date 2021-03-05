import tensorflow as tf
import math
import pdb
from abc import ABC, abstractmethod

class BayesianLinearRegressionMarginalizedWeights(ABC):
	"""

	Bayesian Linear Regression with marginalized weights where the features are abstract methods.
	This class assumes a linear observation model where y is scalar.

	y = w^T.Phi(x) + eps,      eps ~ N(0,v)

	y: scalar
	w: [Nfeat, 1]
	Phi: [Nfeat, 1]

	p(w,v) ~ NIW(M,K,S0,N0)

	We follow the derivation from [1].

	NOTE: The predictive distribution of this class is very similar to that of
	linear Gaussian models, low-rank GPs, BLR where w is marginalized out, BLR where w is modeled
	using normal-inverse-wishart and sampled, etc.

	[1] Minka, T., 2000. Bayesian linear regression. Technical report, MIT.
	"""
	def __init__(self, in_dim, num_features, sigma2_n):
		self.in_dim = in_dim
		self.num_features = num_features
		self.sigma2_n = sigma2_n

		self.K = tf.eye(self.num_features) # [Nfeat, Nfeat]
		# self.K = tf.linalg.diag([0.01,0.1,1])
		# TODO: Diagonal matrix whose elements are the value of the pdf corresponding to
		# the sample of the linear system generated by that pdf value.

		self.M = tf.zeros((self.num_features,1))

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

		
		"""

		self.PhiX = self.get_features_mat(self.X) # [Npoints, Nfeat]
				
		# Compule chol(Sxx), where Sxx = PhiX^T.PhiX + K and Sxx: [Nfeat, Nfeat]
		# Sxx defined in eq.(15)
		# pdb.set_trace()
		self.Lchol = tf.linalg.cholesky(tf.transpose(self.PhiX) @ self.PhiX + self.K ) # Lower triangular of Sxx

		# Compute Syx = Y^T.PhiX [Nfeat,]
		# Syx defined in eq.(16)
		# pdb.set_trace()
		self.Syx = tf.transpose(self.PhiX) @ self.Y + self.K @ self.M

	@abstractmethod
	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""
		raise NotImplementedError
		# print("abstractmethod")

		# Npoints = X.shape[0]
		# PhiX = tf.random.uniform(shape=(Npoints,self.num_features), minval=0.0, maxval=1.0)

		# return PhiX # [Npoints, Nfeat]

	def get_predictive_moments(self,xpred):
		"""

		xpred: [Npred, in_dim]
		"""

		Phi_pred = self.get_features_mat(xpred) # [Npred, Nfeat]
		
		# Get mean: vec(Syx.Sxx^{-1}PhiX)
		# Mean defined in eq. (46) and (57)
		Sxx_inv_times_xpred = tf.linalg.cholesky_solve(self.Lchol,tf.transpose(Phi_pred))
		mean_pred = tf.transpose(self.Syx) @ Sxx_inv_times_xpred

		# Get covariance:
		# Cov defined in eq. (46) and (57), where C is defined in eq. (53)
		# Although (57) is a matrix-normal distribution, in our case Y' is a vector, not a matrix, so 
		# vec(mean) = mean and things simplify. See https://en.wikipedia.org/wiki/Matrix_normal_distribution
		Cinv = tf.eye(xpred.shape[0]) + Phi_pred @ Sxx_inv_times_xpred
		cov_pred =  tf.linalg.inv(Cinv)*self.sigma2_n
		# pdb.set_trace()

		return tf.squeeze(mean_pred), cov_pred


class BLRQuadraticFeatures(BayesianLinearRegressionMarginalizedWeights):

	def __init__(self, in_dim, num_features, sigma2_n):

		assert in_dim <= 2

		super().__init__(in_dim, num_features, sigma2_n)

	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		Npoints = X.shape[0]
		SQRT2 = math.sqrt(2)

		if self.in_dim == 1:
			PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , X**2 ],axis=1)
		elif self.in_dim == 2:
			PhiX = tf.concat([ tf.ones((Npoints,1)) , SQRT2*X , SQRT2*tf.math.reduce_prod(X,axis=1,keepdims=True) , X**2 ],axis=1)
		else:
			raise NotImplementedError

		return PhiX # [Npoints, Nfeat]

	def update_K(self):
		self.K = tf.eye(self.num_features) # [Nfeat, Nfeat]




