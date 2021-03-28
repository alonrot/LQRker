import tensorflow as tf
import gpflow
from lqrker.models.lqr_kernel_gpflow import LQRkernel, LQRMean
import pdb

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

class LQRkernelTransformed(gpflow.kernels.Kernel):
	"""

	We assume that J(xi) follows a log-Normal distribution as 

	J(xi) = exp(f(xi) + sigma_n^2),

	where f is a latent function modeled as a Gaussian process. With thr above
	transformation, we carry all the inference in f. However, the LQR kernel and
	LQR mean functions where derived for J(xi). Hence, in this class, we obtain
	the mean and kernel functions for f(xi) by transforming the LQR kernel and
	mean functions of J(xi).

	TODO: Double-check the noise
	
	References:
	Halliwell, L.J., 2015. The lognormal random multivariate. In Casualty
	Actuarial Society E-Forum, Spring (Vol. 5).
	"""
	def __init__(self, cfg, dim, A_samples, B_samples):
		super().__init__(active_dims=[0])

		self.lqr_ker = LQRkernel(cfg,dim,A_samples,B_samples)
		self.lqr_mean = LQRMean(cfg,dim,A_samples,B_samples)

	def K(self,X,X2=None):
		"""

		Gram Matrix

		X: [Npoints, dim_in]
		X2: [Npoints2, dim_in]

		return:
		Kmat: [Npoints,Npoints2]

		TODO: If not needed, remove the 'add_noise' part

		"""

		add_noise = False
		if X2 is None:
			X2 = X
			add_noise = True

		mX = self.lqr_mean(X)
		mX2 = self.lqr_mean(X2)

		KXX2 = self.lqr_ker.K(X,X2)

		KXX2 = tf.convert_to_tensor(KXX2,dtype=tf.float64)

		mX_times_mX2 = mX @ tf.transpose(mX2) # outer product mX.mX^T

		Kmat = tf.math.log( KXX2 / mX_times_mX2 + 1.0)

		if add_noise:
			# pdb.set_trace()
			# Kmat += 1e-1*tf.eye(Kmat.shape[0],dtype=tf.float64)
			pass

		# pdb.set_trace()

		return Kmat

	def K_diag(self,X):
		"""
		It’s simply the diagonal of the K function, in the case where X2 is None.
		It must return a one-dimensional vector.

		X: [Npoints, dim_in]

		return:
		Kmat: [Npoints,]
		"""

		KX_vec = self.lqr_ker.K_diag(X)
		mX = self.lqr_mean(X)

		Kvec = tf.math.log(KX_vec / tf.squeeze(mX)**2 + 1.0)

		# pdb.set_trace()

		return Kvec

	def update_system_samples_and_weights(self,A_samples, B_samples):
		self.lqr_ker.update_system_samples_and_weights(A_samples,B_samples)
		self.lqr_mean.update_system_samples_and_weights(A_samples,B_samples)


class LQRMeanTransformed(gpflow.mean_functions.MeanFunction):
	"""

	We assume that J(xi) follows a log-Normal distribution as 

	J(xi) = exp(f(xi) + sigma_n^2),

	where f is a latent function modeled as a Gaussian process. With thr above
	transformation, we carry all the inference in f. However, the LQR kernel and
	LQR mean functions where derived for J(xi). Hence, in this class, we obtain
	the mean and kernel functions for f(xi) by transforming the LQR kernel and
	mean functions of J(xi).

	TODO: Double-check the noise

	References:
	Halliwell, L.J., 2015. The lognormal random multivariate. In Casualty
	Actuarial Society E-Forum, Spring (Vol. 5).
	"""

	def __init__(self,cfg, dim, A_samples, B_samples):
		self.lqr_ker = LQRkernel(cfg,dim,A_samples,B_samples)
		self.lqr_mean = LQRMean(cfg,dim,A_samples,B_samples)

	def update_system_samples_and_weights(self,A_samples, B_samples):
		self.lqr_ker.update_system_samples_and_weights(A_samples,B_samples)
		self.lqr_mean.update_system_samples_and_weights(A_samples,B_samples)

	def __call__(self,X):

		mX = self.lqr_mean(X)

		Kvec = self.lqr_ker.K_diag(X)

		# mean_vec = tf.math.log(mX) - 0.5 * tf.reshape(Kvec,(-1,1)) # Wrong

		mean_vec = 2.*tf.math.log(mX) - 0.5 * tf.math.log( mX**2 + tf.reshape(Kvec,(-1,1)) )

		# pdb.set_trace()

		return mean_vec
