import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures

import numpy as np
from lqrker.solve_lqr import GenerateLQRData

import gpflow

import tensorflow_probability as tfp

def sample_v1(nu,dim,Nsamples):
	"""
	Using a Multivariate Gaussian and the squared root of an inverse Gamma

	"""

	# Sample from inverse Gamma:
	alpha = 0.5*nu
	beta = 0.5
	dist_ig = tfp.distributions.InverseGamma(concentration=alpha,scale=beta)
	sample_ig = dist_ig.sample(sample_shape=(Nsamples,1))


	dist_nor = tfp.distributions.Normal(loc = 0.0, scale = tf.math.sqrt(nu - 2.))	
	sample_nor = dist_nor.sample(sample_shape=(Nsamples,dim))

	# Sample from MVT(nu,0,I):
	sample_mvt0 = sample_nor * tf.math.sqrt(sample_ig)

	K = tf.constant([[1.0,0.2],[0.2,1.0]]) + 0.5*tf.eye(dim)
	sample_mvt = sample_mvt0 @ tf.linalg.cholesky(K)

	return sample_mvt

def sample_v2(nu,dim,Nsamples):
	"""

	Using: (i) uniform sphere, (ii) inverse gamma, and (iii) Chi-squared
	"""

	dist_sphe = tfp.distributions.SphericalUniform(dimension=dim)
	sample_sphe = dist_sphe.sample(sample_shape=(Nsamples,))

	# Sample from inverse Gamma:
	alpha = 0.5*nu
	beta = 0.5
	dist_ig = tfp.distributions.InverseGamma(concentration=alpha,scale=beta)
	sample_ig = dist_ig.sample(sample_shape=(Nsamples,1))

	# Sample from chi-squared:
	dist_chi2 = tfp.distributions.Chi2(df=dim)
	sample_chi2 = dist_chi2.sample(sample_shape=(Nsamples,1))

	# Sample from MVT(nu,0,I):
	sample_mvt0 = tf.math.sqrt((nu-2) * sample_chi2 * sample_ig) * sample_sphe

	K = tf.constant([[1.0,0.2],[0.2,1.0]]) + 0.5*tf.eye(dim)
	sample_mvt = sample_mvt0 @ tf.linalg.cholesky(K)

	return sample_mvt


if __name__ == "__main__":

	"""
	Trying to reproduce Fig. 3 from [1]

	[1] Shah, A., Wilson, A. and Ghahramani, Z., 2014, April. Student-t processes
	as alternatives to Gaussian processes. In Artificial intelligence and
	statistics (pp. 877-885). PMLR.
	"""
	
	dim = 2
	nu = 2.1
	Nsamples = 1e5
	M = tf.zeros(2)

	my_samples1 = sample_v1(nu,dim,Nsamples)
	my_samples2 = sample_v2(nu,dim,Nsamples)

	psd_kernels = tfp.math.psd_kernels
	kernel = psd_kernels.ExponentiatedQuadratic(amplitude=1.0,length_scale=0.5)

	# index_points = tf.constant([10,2,2,2])
	index_points = tf.constant([[-1.0],[1.0]])
	# index_points = tf.constant([[1.0],[0.5]])
	# index_points = np.expand_dims(np.linspace(-1., 1., 2), -1)
	# pdb.set_trace()
	mvt = tfp.distributions.StudentTProcess(df=nu, kernel=kernel, index_points=index_points, mean_fn=None)

	samples = mvt.sample(Nsamples)

	# pdb.set_trace()

	hdl_fig, hdl_splots = plt.subplots(3,1,figsize=(14,10),sharex=True)
	# hdl_fig.suptitle("Reduced-rank Student-t process")
	hdl_splots[0].plot(samples[:,0],samples[:,1],marker="o",color="blue",linestyle="None")

	hdl_splots[1].plot(my_samples1[:,0],my_samples1[:,1],marker="o",color="blue",linestyle="None")

	hdl_splots[2].plot(my_samples2[:,0],my_samples2[:,1],marker="o",color="blue",linestyle="None")

	plt.show(block=True)



