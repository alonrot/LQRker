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

from lqrker.losses.loss_elbo_qAB import LossElboLQR_MatrixNormalWishart


class GPLQR(tf.keras.layers.Layer):
	"""

	This model uses a variational distribution over the system matrices q(A,B) to
	compute p(f* | Y) = ∫ p(f* | A,B,Y) q(A,B), which we approximate via
	sampling. This gives us a mixture of Gaussians, which moments can be computed
	analytically.

	"""

	def __init__(self,cfg,dim,**kwargs):

		super().__init__(**kwargs)

		self.Nsys = 5

		self.dim_state = cfg.RRTPLQRfeatures.dim_state
		self.dim_control = cfg.RRTPLQRfeatures.dim_control

		self.sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
		self.gen_linsys = GenerateLinearSystems(dim_state=self.dim_state,
												dim_control=cfg.RRTPLQRfeatures.dim_control,
												Nsys=self.Nsys,
												check_controllability=cfg.RRTPLQRfeatures.check_controllability,
												prior="MNIW")

		# Pre-sample the systems fromt he variational distribution q(A,B):
		self.A_samples, self.B_samples = self.gen_linsys()

		# Initialize kernel and mean functions:
		A_curr = self.A_samples[0,:,:].reshape(1,self.dim_state,self.dim_state)
		B_curr = self.B_samples[0,:,:].reshape(1,self.dim_state,self.dim_control)
		
		self.lqr_ker = LQRkernelTransformed(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A_curr,B_samples=B_curr)
		self.lqr_mean = LQRMeanTransformed(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A_curr,B_samples=B_curr)

		# self.lqr_ker = LQRkernel(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A_curr,B_samples=B_curr)
		# self.lqr_mean = LQRMean(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A_curr,B_samples=B_curr)

		self.loss_class = LossElboLQR_MatrixNormalWishart(cfg,dim,Xtrain=None,Ytrain=None)

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)


	def _find_qAB_via_ELBO(self):		

		# logger.info(self.trainable_weights[0][0:10])
		# logger.info(self.trainable_weights[1])

		epoch = 0
		Nepochs = 2
		done = False
		while epoch < Nepochs and not done:

			# pdb.set_trace()

			with tf.GradientTape() as tape:

				loss_value = self.loss_class()

			grads = tape.gradient(loss_value, self.loss_class.trainable_weights)
			try:
				self.optimizer.apply_gradients(zip(grads, self.loss_class.trainable_weights))
			except:
				print("No!")
				pdb.set_trace()

			if (epoch+1) % 1 == 0:
				logger.info("Training loss at epoch %d / %d: %.4f" % (epoch+1, Nepochs, float(loss_value)))

			# # Stopping condition:
			# if not tf.math.is_nan(loss_value):
			# 	if loss_value <= self.stop_loss_val:
			# 		done = True
			
			epoch += 1

		
		return self.loss_class.get_variational_parameters(get_dict=True)

	def _get_predictive_moments_for_system(self,xpred):

		err = self.Y - self.lqr_mean(self.X)

		kmm = self.lqr_ker.K(self.X)
		knn = self.lqr_ker.K_diag(xpred)
		kmn = self.lqr_ker.K(self.X, xpred)

		s = self.sigma_n**2 * tf.eye(self.X.shape[0],dtype=tf.float64)


		f_mean_zero, f_var = gpflow.conditionals.base_conditional(	kmn,
																	kmm + s,
																	knn,
																	err,
																	full_cov=False,
																	white=False)
		f_mean = f_mean_zero + self.lqr_mean(xpred)

		return f_mean, f_var


	def get_predictive_moments(self,xpred):
		"""

		The predictive distribution p(f* | Y) ~= ∫ p(f* | A,B,Y) q(A,B) ~= 1/M sum_j( N(f*;mu*_j,sigma2*_j) )
		can be approximated as a mixture of Gaussians.

		Herein, we compute the first and second moments of such mixture of Gaussians.
		"""

		# Generate new system samples with updated variational parameters:
		self.A_samples, self.B_samples = self.gen_linsys()
		
		# Compute necessary moments:
		f_mean_avg = 0.
		f_var_avg = 0.
		f_mean_vec = np.zeros((xpred.shape[0],self.Nsys))
		for ii in range(self.Nsys):

			A = self.A_samples[ii,:,:].reshape(1,self.dim_state,self.dim_state)
			B = self.B_samples[ii,:,:].reshape(1,self.dim_state,self.dim_control)
			self.lqr_ker.update_system_samples_and_weights(A,B)
			self.lqr_mean.update_system_samples_and_weights(A,B)

			f_mean, f_var = self._get_predictive_moments_for_system(xpred)

			f_mean_vec[:,ii] = f_mean[:,0]

			f_mean_avg += f_mean
			f_var_avg += f_var

		# Compute expectation and variance of a mixture of Gaussians:
		f_mean_avg = f_mean_avg / self.Nsys
		f_var_avg = f_var_avg / self.Nsys
		f_var_avg += tf.reshape(tf.math.reduce_variance(f_mean_vec,axis=1),(-1,1)) 

		# NOTE: We're returning the variance, not the covariance of f*

		return f_mean_avg, f_var_avg

	def update_model(self,X,Y):

		logger.info("Updating dataset in GPLQR...")
		self._update_dataset(X,Y)

		logger.info("Updating dataset inside the loss class...")
		self.loss_class.update_dataset(X,Y)

		# Run ELBO:
		logger.info("Maximizing ELBO to find q(A,B)...")
		self.var_pars = self._find_qAB_via_ELBO()
		M_q = self.var_pars["M_q"]
		v_q = self.var_pars["v_q"]
		nu_q = self.var_pars["nu_q"]
		w_q = self.var_pars["w_q"]

		# Update generator of linear systems:
		self.gen_linsys.update_parameters_matrix_normal_inverse_Wishart(M_q.numpy(),
																		v_q.numpy(),
																		nu_q[0].numpy(),
																		w_q.numpy())

	def _update_dataset(self,X,Y):
		self.X = tf.cast(X,dtype=tf.float64)

		if Y.ndim == 1:
			self.Y = tf.reshape(Y,(-1,1))
		else:
			assert Y.ndim == 2
			self.Y = Y
