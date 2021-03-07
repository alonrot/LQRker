import tensorflow as tf
import pdb
import math
from abc import ABC, abstractmethod
# import numpy as np
# import tensorflow.experimental.numpy as tnp # https://www.tensorflow.org/guide/tf_numpy

import numpy as np
from lqrker.solve_lqr import GenerateLQRData

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
	def __init__(self, dim, Nfeat, sigma_n, nu, **kwargs):
		"""
		
		dim: Dimensionality of the input space
		Nfeat: Number of features
		L: Half Length of the hypercube. Each dimension has length [-L, L]
		"""

		super().__init__(**kwargs)

		self.dim = dim
		self.Nfeat = Nfeat
		assert nu > 2
		self.nu = nu

		# Specify weights:
		self.log_diag_vals = self.add_weight(shape=(Nfeat,), initializer=tf.keras.initializers.Ones(), trainable=True)
		self.log_noise_std = self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(sigma_n**2), trainable=True)

	def add2dataset(self,xnew,ynew):
		pass

	def get_noise_var(self):
		"""

		TODO: Think about maybe using the softplus transform log(1 + exp(x))
		https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus
		"""

		return tf.exp(2.0*self.log_noise_std)

	def get_Sigma_weights_inv_times_noise_var(self):
		return self.get_noise_var()*tf.linalg.diag(tf.exp(-self.log_diag_vals))

	def train_model(self):

		# loss_kl_div = LossKLDiv(Sigma_noise)

		# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

		# for epoch in range(epochs):

		# 	with tf.GradientTape() as tape:

		# 		loss_value = 0
		# 		for jj in range(Ninstances-1):

		# 			X = Xtrain[ jj*Npred:(jj+1)*Npred , : ]
		# 			Y = Ytrain[ jj*Npred:(jj+1)*Npred ]

		# 			# pdb.set_trace()
		# 			x_new = tf.reshape(X[-1,:],[-1,in_dim])
		# 			y_new = tf.reshape(Y[-1],[-1,1])

		# 			X = X[0:-1,:]
		# 			Y = Y[0:-1]

		# 			# pdb.set_trace()

		# 			mean, cov = blr.q_predictive_gaussian(X, Y, x_new)

		# 			# pdb.set_trace()
		# 			loss_value += loss_kl_div.get(mean_pred=mean,cov_pred=cov,y_new=y_new)

		# 		loss_value = loss_value / Ninstances


		# 	grads = tape.gradient(loss_value, blr.model_features.trainable_weights)
		# 	optimizer.apply_gradients(zip(grads, blr.model_features.trainable_weights))


			
		# 	if epoch % 10 == 0:
		# 		print("Training loss (for one epoch) at epoch %d: %.4f" % (epoch, float(loss_value)))
		# 	# print("Seen so far: %d samples" % ((Ninstances*(T-1) ) )
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
		
		# pdb.set_trace()

		Sigma_weights_inv_times_noise_var = self.get_Sigma_weights_inv_times_noise_var()

		self.Lchol = tf.linalg.cholesky(tf.transpose(self.PhiX) @ self.PhiX + Sigma_weights_inv_times_noise_var ) # Lower triangular

		self.M = tf.zeros((self.X.shape[0],1))

	@abstractmethod
	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""
		raise NotImplementedError

	def get_predictive_moments(self,xpred):
		
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

	def call(self, inputs):
		# y = tf.matmul(inputs, self.w) + self.b
		# return tf.math.cos(y)
		pass


class RRTPQuadraticFeatures(ReducedRankStudentTProcessBase):

	def __init__(self, dim, Nfeat, sigma_n, nu):

		assert dim <= 2

		super().__init__(dim, Nfeat, sigma_n, nu)

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


class RRTPLQRfeatures(ReducedRankStudentTProcessBase):

	def __init__(self, dim, Nfeat, sigma_n, nu):
		super().__init__(dim, Nfeat, sigma_n, nu)

		# Parameters:
		Q_emp = np.array([[1.0]])
		R_emp = np.array([[0.1]])
		dim_state = Q_emp.shape[0]
		dim_control = R_emp.shape[1]		
		mu0 = np.zeros((dim_state,1))
		Sigma0 = np.eye(dim_state)
		Nsys = Nfeat
		Ncon = 1 # Not needed

		# Generate systems:
		self.lqr_data = GenerateLQRData(Q_emp,R_emp,mu0,Sigma0,Nsys,Ncon,check_controllability=True)
		self.A_samples, self.B_samples = self.lqr_data._sample_systems(Nsamples=Nfeat)

		for ii in range(Nfeat):
			self.lqr_data._check_controllability(self.A_samples[ii,:,:], self.B_samples[ii,:,:])

	def get_features_mat(self,X):
		"""
		X: [Npoints, in_dim]
		return: PhiX: [Npoints, Nfeat]
		"""

		assert self.dim == 1

		Npoints = X.shape[0]
		cost_values_all = np.zeros((Npoints,self.Nfeat))
		for ii in range(Npoints):

			Q_des = tf.expand_dims(X[ii,:],axis=1)
			R_des = np.array([[0.1]])
			
			for jj in range(self.Nfeat):

				cost_values_all[ii,jj] = self.lqr_data.solve_lqr.forward_simulation(self.A_samples[jj,:,:], self.B_samples[jj,:,:], Q_des, R_des)


		return tf.convert_to_tensor(cost_values_all,dtype=tf.float32) # [Npoints, Nfeat]







