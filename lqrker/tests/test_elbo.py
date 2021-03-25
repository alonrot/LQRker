import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
import gpflow
import hydra
import numpy as np

from lqrker.experiments.generate_dataset import generate_dataset
from lqrker.experiments.validate_model import split_dataset

from lqrker.models.lqr_kernel_gpflow import LQRkernel, LQRMean
from lqrker.models.lqr_kernel_trans_gpflow import LQRkernelTransformed, LQRMeanTransformed

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

from lqrker.utils.generate_linear_systems import GenerateLinearSystems


class LossElboLQR(tf.keras.layers.Layer):

	def __init__(self,cfg,dim,Xtrain,Ytrain,**kwargs):

		super().__init__(**kwargs)

		Nsys = 10

		self.sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
		self.gen_linsys = GenerateLinearSystems(dim_state=cfg.RRTPLQRfeatures.dim_state,
												dim_control=cfg.RRTPLQRfeatures.dim_control,
												Nsys=Nsys,
												check_controllability=cfg.RRTPLQRfeatures.check_controllability,
												prior="MNIW")

		A_samples, B_samples = self.gen_linsys()
		A = A_samples[0,:,:].reshape(1,cfg.RRTPLQRfeatures.dim_state,cfg.RRTPLQRfeatures.dim_state)
		B = B_samples[0,:,:].reshape(1,cfg.RRTPLQRfeatures.dim_state,cfg.RRTPLQRfeatures.dim_state)

		# TODO: Here we might need need the transformed versions!!!
		lqr_ker = LQRkernel(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)
		lqr_mean = LQRMean(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)


		XX = tf.cast(Xtrain,dtype=tf.float64)
		YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)
		Npred = 40
		xlim = eval(cfg.dataset.xlims)
		xpred = 10**tf.reshape(tf.linspace(xlim[0],xlim[1],Npred),(-1,1))


		# lqr_ker = gpflow.kernels.Matern52()
		mod = gpflow.models.GPR(data=(XX,YY), kernel=lqr_ker, mean_function=lqr_mean)
		sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
		mod.likelihood.variance.assign(sigma_n**2)
		xxpred = tf.cast(xpred,dtype=tf.float64)
		# gpflow.utilities.print_summary(mod)
		mean_pred_gpflow, cov_pred_gpflow = mod.predict_f(xxpred,full_cov=True)



		# Compute necessary moments:
		K_AB_inv_mean = 0
		m_AB_times_K_AB_inv_mean = 0
		for ii in range(Nsys):

			K_AB = lqr_ker.K(Xtrain)
			m_AB = lqr_mean(Xtrain)

			# # mean_pred_gpflow, cov_pred_gpflow = mod.predict_f(xxpred,full_cov=True)
			# m_AB, K_AB = mod.predict_f(xxpred,full_cov=True)

			K_AB_inv_mean += tf.linalg.inv(K_AB)

			m_AB_times_K_AB_inv_mean += tf.transpose(m_AB) @ K_AB_inv_mean

			A = A_samples[ii,:,:].reshape(1,cfg.RRTPLQRfeatures.dim_state,cfg.RRTPLQRfeatures.dim_state)
			B = B_samples[ii,:,:].reshape(1,cfg.RRTPLQRfeatures.dim_state,cfg.RRTPLQRfeatures.dim_state)
			lqr_ker.update_system_samples_and_weights(A,B)
			lqr_mean.update_system_samples_and_weights(A,B)


		self.K_AB_inv_mean = K_AB_inv_mean / Nsys
		self.m_AB_times_K_AB_inv_mean = m_AB_times_K_AB_inv_mean / Nsys

		# Data:
		self.Ytrain = Ytrain
		self.Xtrain = Xtrain

		# Specify weights:
		self.Nevals = self.Xtrain.shape[0]
		NLvec = (self.Nevals+1)*self.Nevals//2
		self.Lvec = self.add_weight(shape=(NLvec,), initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None), trainable=True, name="Lvec",dtype=tf.float64)
		# self.m = self.add_weight(shape=(self.Nevals,), initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None), trainable=True,name="m",dtype=tf.float64)
		self.m = self.add_weight(shape=(self.Nevals,), initializer=tf.keras.initializers.Constant(self.Ytrain), trainable=True,name="m",dtype=tf.float64)

	
	def _make_lower_triangular(self):
		"""
		Create an (NxN) lower triangular matrix from a vector of dimension N(N+1)/2

		return:
		L: [self.Nevals, self.Nevals] (lower triangular matrix)
		"""

		# Create list of column tensors with padded zeroes at the beginning:
		mlist = []
		c = 0
		for ii in range(self.Nevals):

			mlist.append( tf.reshape(self.Lvec[c:c+self.Nevals-ii],(-1,1)) )
			c += self.Nevals-ii

			# Pad necessary zeroes:
			mlist[ii] = tf.concat( [tf.zeros((ii,1),dtype=tf.float64),mlist[ii]] , axis=0)

		# Concatenate all column vectors to obtain the lower triangular matrix
		L = tf.concat(mlist,axis=1)

		return L

	def loss(self):

		L = self._make_lower_triangular()
		Sigma = L @ tf.transpose(L)

		part1 = 1./self.sigma_n**2 * tf.reduce_sum(self.Ytrain*self.m) - 0.5*(1./self.sigma_n**2) * ( tf.linalg.trace(Sigma) + tf.reduce_sum(self.m**2) )
		part2 = -0.5*tf.linalg.trace(Sigma @ self.K_AB_inv_mean) - 0.5*( tf.reshape(self.m,(1,-1)) @ self.K_AB_inv_mean @ tf.reshape(self.m,(-1,1)) ) + self.m_AB_times_K_AB_inv_mean @ tf.reshape(self.m,(-1,1))
		part3 = 0.5*tf.linalg.logdet(Sigma)

		return -tf.squeeze(part1 + part2 + part3) # Flip sign because tf optimizers minimize by default

	def __call__(self):
		return self.loss()

@hydra.main(config_path="../experiments/",config_name="config.yaml")
def main(cfg: dict) -> None:
	"""

	LQR - Inifinite horizon case
	No process noise, i.e., v_k = 0
	E[x0] = 0

	Use GPflow and a tailored kernel
	"""

	my_seed = 1
	np.random.seed(my_seed)
	tf.random.set_seed(my_seed)

	X,Y,A,B = generate_dataset(cfg)

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,
												perc_training=cfg.validation.perc_training,
												Ncut=cfg.validation.Ncut)

	if isinstance(cfg.dataset.dim,str):
		dim = eval(cfg.dataset.dim)
	else:
		dim = cfg.dataset.dim

	loss_elbo_lqr = LossElboLQR(cfg,dim,Xtrain,Ytrain)

	# ll = loss_elbo_lqr()
	# print("ll:",ll)

	# Adam optimizer:
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

	# logger.info(self.trainable_weights[0][0:10])
	# logger.info(self.trainable_weights[1])

	epoch = 0
	Nepochs = 1000
	done = False
	while epoch < Nepochs and not done:

		pdb.set_trace()

		with tf.GradientTape() as tape:

			# pdb.set_trace()
			loss_value = loss_elbo_lqr()

		grads = tape.gradient(loss_value, loss_elbo_lqr.trainable_weights)
		optimizer.apply_gradients(zip(grads, loss_elbo_lqr.trainable_weights))

		if (epoch+1) % 10 == 0:
			logger.info("Training loss at epoch %d / %d: %.4f" % (epoch+1, Nepochs, float(loss_value)))

		# # Stopping condition:
		# if not tf.math.is_nan(loss_value):
		# 	if loss_value <= self.stop_loss_val:
		# 		done = True
		
		epoch += 1

	# if done == True:
	# 	logger.info("Training finished because loss_value = {0:f} (<= {1:f})".format(float(loss_value),float(self.stop_loss_val)))

	# logger.info(self.trainable_weights[0][0:10])
	# logger.info(self.trainable_weights[1])

	# ll = loss_elbo_lqr()
	# print("ll:",ll)


	mean = loss_elbo_lqr.m
	L = loss_elbo_lqr._make_lower_triangular()
	Sigma = L @ tf.transpose(L)
	print("mean:",mean)
	print("Sigma:",Sigma)
	pdb.set_trace()


	# # Use gpflow to do predictions:
	# XX = tf.cast(Xtrain,dtype=tf.float64)
	# YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)

	# # Manipulate kernels:
	# # ker_stat = gpflow.kernels.Matern52(lengthscales=2.0,variance=1.0)
	# # ker_tot = lqr_ker * ker_stat # This works bad as the lengthscale affects
	# # equally all regions of the space. Note that if we use this option, we must
	# # switch off the Sigma(th,th') in kernel_gpflow.

	# A_samples, B_samples = self.gen_linsys()
	# A = A_samples[0,:,:].reshape(1,cfg.RRTPLQRfeatures.dim_state,cfg.RRTPLQRfeatures.dim_state)
	# B = B_samples[0,:,:].reshape(1,cfg.RRTPLQRfeatures.dim_state,cfg.RRTPLQRfeatures.dim_state)
	# lqr_ker = LQRkernel(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)
	# lqr_mean = LQRMean(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)



	# gpflow.conditionals.base_conditional()

	# f_mean_zero, f_var = gpflow.conditionals.base_conditional(	kmn,
	# 															kmm + s,
	# 															knn,
	# 															err,
	# 															full_cov=full_cov,
	# 															white=False)



 #        X_data, Y_data = self.data
 #        err = Y_data - self.mean_function(X_data)

 #        kmm = self.kernel(X_data)
 #        knn = self.kernel(Xnew, full_cov=full_cov)
 #        kmn = self.kernel(X_data, Xnew)

 #        num_data = X_data.shape[0]
 #        s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))

	# # lqr_ker = gpflow.kernels.Matern52()
	# mod = gpflow.models.GPR(data=(XX,YY), kernel=lqr_ker, mean_function=lqr_mean)
	# sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
	# mod.likelihood.variance.assign(sigma_n**2)
	# xxpred = tf.cast(xpred,dtype=tf.float64)

	# # opt = gpflow.optimizers.Scipy()
	# # opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=300))

	# # mod.kernel.lengthscales.assign(cfg.GaussianProcess.hyperpars.ls.init)
	# # mod.kernel.variance.assign(cfg.GaussianProcess.hyperpars.prior_var.init)


	# gpflow.utilities.print_summary(mod)
	# mean_pred_gpflow, var_pred_gpflow = mod.predict_f(xxpred)

	# mean_vec = mean_pred_gpflow
	
	# std_pred_gpflow = np.sqrt(var_pred_gpflow)
	# fpred_quan_minus = mean_pred_gpflow - 2.*std_pred_gpflow
	# fpred_quan_plus = mean_pred_gpflow + 2.*std_pred_gpflow


	"""
	TODO list
	=========
	NOTE: This is the posterior for the Gaussian process for f, with J = exp(f) and p(A,B) distributed as MNIW

	0) Because we are doing inference in f, we need to (i) work with log(Ytrain) and (ii) use LQRkernelTransformed, LQRMeanTransformed.
		0.1) Before doing this, check what's wrong in test_lqr_kernel_logGP_analysis.py
	1) Re-derive all equations and possibly double-check with colleagues
		1.1) Is it really ok to assume that q(f|A,B) = q(f) and just igonre all the q(A,B) aprt by assuming p(A,B) = q(A,B) ???
	2) Double-check the implementation
	3) Refactor into classes
	4) Construct the LQR Gaussian process
	"""


if __name__ == "__main__":

	main()


