import tensorflow as tf
import tensorflow_probability as tfp
import pdb
import numpy as np

from lqrker.objectives.objective_cost import ObjectiveCostBase
from lqrker.solve_lqr import GenerateLQRData

import time

class LQRCostStudent(ObjectiveCostBase):
	"""

	TODO: Have another LQR base class that this one inherits from,
	and leave here only the Student's-t related things (just the noise)
	"""

	def __init__(self,dim_in,sigma_n,nu,cfg,Nsys=1):
		super().__init__(dim_in,sigma_n)

		Q_emp = eval(cfg.empirical_weights.Q_emp)
		R_emp = eval(cfg.empirical_weights.R_emp)

		self.dim_state = Q_emp.shape[0]
		self.dim_control = R_emp.shape[0]
		assert dim_in == self.dim_state + self.dim_control

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
		self.lqr_data = GenerateLQRData(Q_emp,R_emp,mu0,Sigma0,Nsys,Ncon,check_controllability=cfg.check_controllability)
		self.A_samples, self.B_samples = self.lqr_data._sample_systems(Nsamples=self.Nsys)

		self.dist_student_t = tfp.distributions.StudentT(df=nu,loc=0.0,scale=self.sigma_n)

	def evaluate(self,X,add_noise=True,verbo=False):
		"""
		
		X: [Npoints, self.dim_in]

		"""

		Npoints = X.shape[0]
		cost_values_all = np.zeros((Npoints,self.Nsys))
		# time_elapsed = np.zeros(Npoints)
		Nskip = 10
		for ii in range(Npoints):

			if verbo and (ii+1) % Nskip == 1:
				start = time.time()

			Q_des = tf.linalg.diag(X[ii,0:self.dim_state])
			R_des = tf.linalg.diag(X[ii,self.dim_state::])

			# print("Computing cost for point nr. {0:d} / {1:d}".format(ii+1,Npoints))
			for jj in range(self.Nsys):

				# print("Computing cost for system {0:d} / {1:d}, point nr. {2:d} / {3:d}".format(jj+1,self.Nsys,ii+1,Npoints))
				cost_values_all[ii,jj] = self.lqr_data.solve_lqr.forward_simulation(self.A_samples[jj,:,:], self.B_samples[jj,:,:], Q_des, R_des)

			if verbo and (ii+1) % Nskip == 0:
				# time_elapsed[ii] = time.time() - start
				time_elapsed = time.time() - start
				print("Point {0:d} / {1:d} per point with {2:d} features".format(ii+1,Npoints,self.Nsys))
				print("Took {0:f} [sec] to compute {1:d} points with {2:d} features".format(time_elapsed,Nskip,self.Nsys))

		# print("{0:f} [sec] on average per point with {1:d} features".format(np.mean(time_elapsed),self.Nsys))
		# print("{0:f} [sec] in total with {1:d} features".format(np.sum(time_elapsed),self.Nsys))

		# Sample noise from independent Student's-t distributions:
		if add_noise:
			samples_noise = self.dist_student_t.sample(sample_shape=(Npoints,self.Nsys))
			cost_values_all += samples_noise

		if self.Nsys == 1:
			cost_values_all = tf.squeeze(cost_values_all)

		# Rescaling cost to avoid numerical unstability:
		cost_values_all = cost_values_all / 10**(0.15*(self.dim_state + self.dim_control))

		return tf.convert_to_tensor(cost_values_all,dtype=tf.float32) # [Npoints, self.Nsys]


