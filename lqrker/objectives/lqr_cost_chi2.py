import tensorflow as tf
import tensorflow_probability as tfp
import pdb
import numpy as np

from lqrker.objectives.objective_cost import ObjectiveCostBase
from lqrker.utils.solve_lqr import SolveLQR
from lqrker.utils.generate_linear_systems import GenerateLinearSystems

import time

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


class LQRCostChiSquared(ObjectiveCostBase):
	"""
	

	This class generates samples from the LQR cost by letting the initial
	condition x0 ~ N(mu0,Sigma0) be a random sample.
	A collection of system samples (A_samples,B_samples) can be passed.
	Otherwise, a number of random controllable system samples will be generated.

	"""

	def __init__(self,dim_in,cfg,Nsys=1,A_samples=None,B_samples=None):
		super().__init__(dim_in,sigma_n=0.0)

		self.dim_state = cfg.dim_state
		self.dim_control = cfg.dim_control
		self.dim_in = dim_in
		if self.dim_in != 1: # If dim_in == 1, special protocol
			assert self.dim_in == self.dim_state + self.dim_control

		mu0 = eval(cfg.initial_state_distribution.mu0)
		Sigma0 = eval(cfg.initial_state_distribution.Sigma0)

		self.Nsys = Nsys

		# Generate single system:
		if A_samples is None and B_samples is None:
			generate_linear_systems = GenerateLinearSystems(self.dim_state,self.dim_control,Nsys,check_controllability=cfg.check_controllability)
			self.A_samples, self.B_samples = generate_linear_systems()
		else:
			self.A_samples = A_samples
			self.B_samples = B_samples
		
		Q_emp = eval(cfg.empirical_weights.Q_emp)
		R_emp = eval(cfg.empirical_weights.R_emp)
		self.solve_lqr = SolveLQR(Q_emp,R_emp,mu0,Sigma0)

	def evaluate(self,X,with_noise=True,verbo=False):
		"""
		
		X: [Npoints, self.dim_in]

		"""

		# Verbosity:
		# time_elapsed = np.zeros(Npoints)
		Nskip = 1000

		Npoints = X.shape[0]
		cost_values_all = np.zeros((Npoints,self.Nsys))
		for ii in range(Npoints):

			if verbo and (ii+1) % Nskip == 1:
				start = time.time()

			if self.dim_in > 1:
				Q_des = tf.linalg.diag(X[ii,0:self.dim_state])
				R_des = tf.linalg.diag(X[ii,self.dim_state::])
			else:
				Q_des_diag = tf.concat([X[ii],tf.ones(self.dim_state-1,dtype=X.dtype)],axis=0)
				Q_des = tf.linalg.diag(Q_des_diag)
				R_des = tf.linalg.eye(self.dim_control)

			# pdb.set_trace()

			# logger.info("Computing cost for point nr. {0:d} / {1:d}".format(ii+1,Npoints))
			for jj in range(self.Nsys):

				# logger.info("Computing cost for system {0:d} / {1:d}, point nr. {2:d} / {3:d}".format(jj+1,self.Nsys,ii+1,Npoints))
				if with_noise:
					cost_values_all[ii,jj] = self.solve_lqr.forward_simulation_with_random_initial_condition(self.A_samples[jj,:,:], self.B_samples[jj,:,:], Q_des, R_des)
				else:
					cost_values_all[ii,jj] = self.solve_lqr.forward_simulation_expected_value(self.A_samples[jj,:,:], self.B_samples[jj,:,:], Q_des, R_des)

			if verbo and (ii+1) % Nskip == 0:
				# time_elapsed[ii] = time.time() - start
				time_elapsed = time.time() - start
				logger.info("Point {0:d} / {1:d} per point with {2:d} features".format(ii+1,Npoints,self.Nsys))
				logger.info("Took {0:f} [sec] to compute {1:d} points with {2:d} features".format(time_elapsed,Nskip,self.Nsys))

		# logger.info("{0:f} [sec] on average per point with {1:d} features".format(np.mean(time_elapsed),self.Nsys))
		# logger.info("{0:f} [sec] in total with {1:d} features".format(np.sum(time_elapsed),self.Nsys))

		if self.Nsys == 1:
			cost_values_all = tf.squeeze(cost_values_all)
		else:
			cost_values_all = tf.convert_to_tensor(cost_values_all,dtype=tf.float32)

		return cost_values_all # [Npoints, self.Nsys]


