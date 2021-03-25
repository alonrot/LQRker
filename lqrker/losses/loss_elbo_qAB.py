import tensorflow as tf

import pdb
import matplotlib.pyplot as plt
import gpflow
import hydra
import numpy as np

from lqrker.models.lqr_kernel_gpflow import LQRkernel, LQRMean
from lqrker.models.lqr_kernel_trans_gpflow import LQRkernelTransformed, LQRMeanTransformed

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

from lqrker.utils.generate_linear_systems import GenerateLinearSystems


class LossElboLQR_MatrixNormalWishart(tf.keras.layers.Layer):
	"""
	Parameters of variational distribution q(A,B) = q(B|A)q(A)
	==========================================================
	q(B|A) = MatrixNormal(Mq,A,Vq), where Mq: [nx,nu], Vq = diag(vq_1,vq_2,...,vq_nu), with vq_i > 0
	q(A) = InverseWishart(nu_q,Omega_q), where nu_q >= nx, Omega_q = diag(wq_1,wq_2,...,wq_nx), with wq_i > 0
	
	The variational distribution approximates the posterior p(A,B|Y) ≈ q(A,B)

	Parameters of prior distribution p(A,B)
	=======================================
	p(B|A) = MatrixNormal(Mp,A,Vp), where Mp = 0, Vp = diag(vp_1,vp_2,...,vp_nu)
	p(A) = InverseWishart(nu_p,Omega_p), where nu_p >= nx, Omega_p = diag(wp_1,wp_2,...,wp_nx)

	"""

	def __init__(self,cfg,dim,Xtrain,Ytrain,**kwargs):

		super().__init__(**kwargs)


		self.Xtrain = Xtrain
		self.Ytrain = Ytrain

		self.Nsys = 2
		self.dim_state = cfg.RRTPLQRfeatures.dim_state
		self.dim_control = cfg.RRTPLQRfeatures.dim_control

		self.sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
		self.gen_linsys = GenerateLinearSystems(dim_state=self.dim_state,
												dim_control=self.dim_control,
												Nsys=self.Nsys,
												check_controllability=cfg.RRTPLQRfeatures.check_controllability,
												prior="MNIW")

		A_samples, B_samples = self.gen_linsys()
		A = A_samples[0,:,:].reshape(1,self.dim_state,self.dim_state)
		B = B_samples[0,:,:].reshape(1,self.dim_state,self.dim_state)

		# TODO: Here we might need need the transformed versions!!!
		self.lqr_ker = LQRkernel(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)
		self.lqr_mean = LQRMean(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)

		# Variational parameters:
		self.M_q = self.add_weight(shape=(self.dim_state,self.dim_control), initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None), trainable=True, name="M_q",dtype=tf.float32)
		self.log_v_q = self.add_weight(shape=(self.dim_control,), initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None), trainable=True, name="log_v_q",dtype=tf.float32)
		self.log_w_q = self.add_weight(shape=(self.dim_state,), initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None), trainable=True, name="log_w_q",dtype=tf.float32)

		# NOTE: log_nu_q_minus_dim_state = log(nu_q - dim_state)
		self.log_nu_q_minus_dim_state = self.add_weight(shape=(1,), initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None), constraint=tf.keras.constraints.NonNeg(), trainable=True, name="log_nu_q_minus_dim_state",dtype=tf.float32)

		# Prior parameters:
		self.v_p = tf.ones(self.dim_control,dtype=tf.float32)
		self.w_p = tf.ones(self.dim_state,dtype=tf.float32)
		self.nu_p = self.dim_state + 1

	def update_dataset(self,X,Y):
		self.Xtrain = tf.cast(X,dtype=tf.float64)

		if Y.ndim == 1:
			self.Ytrain = tf.reshape(Y,(-1,1))
		else:
			assert Y.ndim == 2
			self.Ytrain = Y

	def get_variational_parameters(self,get_dict=False):

		if get_dict:
			var_pars = dict(M_q=self.M_q,
							v_q=tf.math.exp(self.log_v_q),
							nu_q=tf.math.exp(self.log_nu_q_minus_dim_state) + self.dim_state,
							w_q=tf.math.exp(self.log_w_q))
			return var_pars
		else:
			return self.M_q, tf.math.exp(self.log_v_q), tf.math.exp(self.log_nu_q_minus_dim_state) + self.dim_state, tf.math.exp(self.log_w_q)

	def DKL_matrix_normal(self,A):
		"""

		DKL( q(B|A) || p(B|A) )

		Both, q and p are matrix-normal distributions.

		We ignore all the terms that don't depend on the variational parameters

		"""

		Achol = tf.linalg.cholesky(A)

		part1 = -tf.reduce_sum(self.log_v_q) + self.dim_state * tf.reduce_sum( tf.math.exp(self.log_v_q) / self.v_p )
		part2 = tf.linalg.trace(tf.linalg.diag(1/self.v_p) @ tf.transpose(self.M_q) @ tf.linalg.cholesky_solve(tf.cast(Achol,dtype=tf.float32),self.M_q))

		# pdb.set_trace()

		return 0.5*(part1 + part2)

	def DKL_inverse_wishart(self):
		"""

		DKL( q(A) || p(A) )

		Both, q and p are inverse Wishart distributions.

		We ignore all the terms that don't depend on the variational parameters

		"""

		nu_q = tf.math.exp(self.log_nu_q_minus_dim_state) + self.dim_state
		w_q = tf.exp(self.log_w_q)

		gamma_arg = 0.5*nu_q + 0.5*(1 - tf.range(1,self.dim_state+1,dtype=tf.float32))
		log_gamma = tf.math.lgamma(gamma_arg)

		digamma_arg = tf.math.digamma( 0.5*(nu_q - self.dim_state + tf.range(1,self.dim_state+1,dtype=tf.float32)) )

		part1 = -tf.reduce_sum(log_gamma) + 0.5*nu_q*tf.reduce_sum( self.w_p / w_q ) -0.5*self.dim_state*nu_q
		part2 = -0.5*self.nu_p*( tf.reduce_sum(tf.math.log(self.w_p)) - tf.reduce_sum(self.log_w_q)) -0.5*(self.nu_p - nu_q) * tf.reduce_sum(digamma_arg)

		return part1 + part2


	def log_evidence_given_AB(self):

		mLQR = self.lqr_mean(self.Xtrain)
		KLQR = self.lqr_ker.K(self.Xtrain)

		KLQR_plus_noise = KLQR + self.sigma_n**2*tf.eye(self.Xtrain.shape[0],dtype=tf.float64)

		err = tf.reshape(self.Ytrain,(-1,1)) - mLQR

		try:
			KLQR_plus_noise_chol = tf.linalg.cholesky(KLQR_plus_noise)
		except:
			pdb.set_trace()

		# pdb.set_trace()

		part1 = -0.5*tf.transpose(err) @ tf.linalg.cholesky_solve(KLQR_plus_noise_chol,err) 
		part2 = -0.5*tf.linalg.logdet(KLQR_plus_noise)

		# pdb.set_trace()

		part12 = tf.cast(part1 + part2,dtype=tf.float32)

		return part12


	def loss(self):

		M_q, v_q, nu_q, w_q = self.get_variational_parameters()

		print("M_q:",M_q)
		print("v_q:",v_q)
		print("nu_q:",nu_q)
		print("w_q:",w_q)

		# pdb.set_trace()

		# Generate systems (A,B) with the current values of the variational parameters:
		self.gen_linsys.update_parameters_matrix_normal_inverse_Wishart(M_q.numpy(),
																		v_q.numpy(),
																		nu_q[0].numpy(),
																		w_q.numpy())
		A_samples, B_samples = self.gen_linsys()

		# Compute averages of terms log_evidence_given_AB() and DKL_matrix_normal(),
		# both of which depend on choices of the system (A,B):
		part1_avg = 0
		part2_avg = 0
		for ii in range(self.Nsys):

			A = A_samples[ii,:,:].reshape(1,self.dim_state,self.dim_state)
			B = B_samples[ii,:,:].reshape(1,self.dim_state,self.dim_state)

			self.lqr_ker.update_system_samples_and_weights(A,B)
			self.lqr_mean.update_system_samples_and_weights(A,B)

			part1 = self.log_evidence_given_AB()
			part1_avg += part1

			part2 = -self.DKL_matrix_normal(A_samples[ii,:,:])
			# pdb.set_trace()
			part2_avg += part2


		part1_avg = part1_avg / self.Nsys
		part2_avg = part2_avg / self.Nsys

		part3 = -self.DKL_inverse_wishart()

		# pdb.set_trace()

		loss_val = -(part1_avg + part2_avg + part3) # Flip sign because ELBO asks for maximization and this is a loss

		return loss_val



	def __call__(self):
		return self.loss()