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


class LQRCostStudent(ObjectiveCostBase):
	"""

	Deprecated. Will remove. From now on using LQRCostChiSquared()

	"""

	def __init__(self,dim_in,sigma_n,nu,cfg,Nsys=1,A=None,B=None):
		super().__init__(dim_in,sigma_n)

		Q_emp = eval(cfg.empirical_weights.Q_emp)
		R_emp = eval(cfg.empirical_weights.R_emp)

		self.dim_state = Q_emp.shape[0]
		self.dim_control = R_emp.shape[0]
		self.dim_in = dim_in
		if self.dim_in != 1: # If dim_in == 1, special protocol
			assert self.dim_in == self.dim_state + self.dim_control

		# Parameters:
		if cfg.initial_state_distribution.mu0 == "zeros":
			mu0 = np.zeros((self.dim_state,1))
		elif cfg.initial_state_distribution.mu0 == "random":
			pass
		else:
			raise ValueError

		if cfg.initial_state_distribution.Sigma0 == "identity":
			Sigma0 = np.eye(self.dim_state)
		else:
			raise ValueError

		self.Nsys = Nsys # By default, we use only one system (the "real" one)
		Ncon = 1 # Irrelevant

		# Generate single system:
		if A is None and B is None:
			generate_linear_systems = GenerateLinearSystems(self.dim_state,self.dim_control,Nsys,check_controllability=cfg.check_controllability)
			self.A_samples, self.B_samples = generate_linear_systems()

		else:
			self.A_samples = A
			self.B_samples = B

		self.solve_lqr = SolveLQR(Q_emp,R_emp,mu0,Sigma0)
		self.dist_student_t = tfp.distributions.StudentT(df=nu,loc=0.0,scale=self.sigma_n)

	def evaluate(self,X,with_noise=True,verbo=False):
		"""
		
		X: [Npoints, self.dim_in]

		"""

		Npoints = X.shape[0]
		cost_values_all = np.zeros((Npoints,self.Nsys))
		# time_elapsed = np.zeros(Npoints)
		Nskip = 1000
		for ii in range(Npoints):

			if verbo and (ii+1) % Nskip == 1:
				start = time.time()

			if self.dim_in > 1:
				Q_des = tf.linalg.diag(X[ii,0:self.dim_state])
				R_des = tf.linalg.diag(X[ii,self.dim_state::])
			else:
				Q_des = tf.linalg.diag(X[ii])
				R_des = tf.constant([[1]])

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

		# Sample noise from independent Student's-t distributions:
		if with_noise:
			samples_noise = self.dist_student_t.sample(sample_shape=(Npoints,self.Nsys))
			cost_values_all += samples_noise

		if self.Nsys == 1:
			cost_values_all = tf.squeeze(cost_values_all)
		else:
			cost_values_all = tf.convert_to_tensor(cost_values_all,dtype=tf.float32)

		return cost_values_all # [Npoints, self.Nsys]

		print("Not rescaling anymore!!!!")
		# # Rescaling cost to avoid numerical unstability:
		# cost_values_all = cost_values_all / 10**(0.15*(self.dim_state + self.dim_control))

		return cost_values_all # [Npoints, self.Nsys]


