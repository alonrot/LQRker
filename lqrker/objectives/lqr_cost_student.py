import tensorflow as tf
import tensorflow_probability as tfp
import pdb
import numpy as np

from lqrker.objectives.objective_cost import ObjectiveCostBase
from lqrker.solve_lqr import GenerateLQRData

class LQRCostStudent(ObjectiveCostBase):

	def __init__(self,dim_in,sigma_n,nu):
		super().__init__(dim_in,sigma_n)

		# Parameters:
		Q_emp = np.array([[1.0]])
		R_emp = np.array([[0.1]])
		dim_state = Q_emp.shape[0]
		dim_control = R_emp.shape[1]		
		mu0 = np.zeros((dim_state,1))
		Sigma0 = np.eye(dim_state)
		Nsys = 1 # We use only one system (the real one)
		Ncon = 1 # Irrelevant

		# Generate single system:
		self.lqr_data = GenerateLQRData(Q_emp,R_emp,mu0,Sigma0,Nsys,Ncon,check_controllability=True)
		self.A_samples, self.B_samples = self.lqr_data._sample_systems(Nsamples=Nsys)
		self.dist_student_t = tfp.distributions.StudentT(df=nu,loc=0.0,scale=self.sigma_n)

	def evaluate(self,X,add_noise=True):
		"""
		X: [Npoints, self.dim_in]

		"""

		Npoints = X.shape[0]
		cost_values_all = np.zeros(Npoints)
		for ii in range(Npoints):

			Q_des = tf.expand_dims(X[ii,:],axis=1)
			R_des = np.array([[0.1]])
			
			cost_values_all[ii] = self.lqr_data.solve_lqr.forward_simulation(self.A_samples[0,:,:], self.B_samples[0,:,:], Q_des, R_des)

		# Sample noise from independent Student's-t distributions:
		if add_noise:
			samples_noise = self.dist_student_t.sample(sample_shape=(Npoints))
			cost_values_all += samples_noise

		return tf.convert_to_tensor(cost_values_all,dtype=tf.float32)