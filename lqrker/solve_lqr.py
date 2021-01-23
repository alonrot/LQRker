import scipy
import pdb
import numpy as np


class SolveLQR:
	"""

	This class simulates forward experiments of a linear time-invariant system,
	when the cost is quadratic, parametrized with (Q_emp,R_emp).

	For each feedback controller K, the function get_cost() returns the corresponding
	cost.

	Each feedback controller K is computed based on an underlying design cost
	parametrized with (Q_des,R_des). It is computed by solving the Riccatti equation:
	K = DLQR(A,B,Q_des,R_des)
	"""

	def __init__(self,Q_emp,R_emp):

		self.Q_emp = Q_emp
		self.R_emp = R_emp

	def _get_controller(self, A, B, Q_des, R_des):

		K = scipy.linalg.solve_discrete_are(A, B, self.Q_emp, self.R_emp)

		return K

	def forward_simulation(self, A, B, Q_des, R_des):

		K = self._get_controller(A, B, Q_des, R_des)

		# J = 
		# return J


class GenerateLQRData():
	"""

	Given a matrix-normal distribution over the linear model matrices (A,B)
	and a distribution over possible controller designs theta = (Q_des,R_des),
	this class generates forward simulations of the sampled systems (A_j,B_j) with 
	controller designs theta_j. For each forward simulation, we compute the
	resulting quadratic cost, according to the empirical weights (Q_emp,R_emp)
	"""

	def __init__(self,Q_emp,R_emp):

		self.dim_state = Q_emp.shape[0]
		self.dim_control = R_emp.shape[0]

		self.solve_lqr = SolveLQR(Q_emp,R_emp)

	def _sample_systems(self,Nsamples):

		M = np.zeros((self.dim_state,self.dim_state))
		V = np.eye(self.dim_state)
		U = 2.0*np.eye(self.dim_state)
		A_samples = self._construct_matrix_normal_distribution(M,V,U,Nsamples)

		M = np.zeros((self.dim_state,self.dim_control))
		V = np.eye(self.dim_control)
		U = 2.0*np.eye(self.dim_state)
		B_samples = self._construct_matrix_normal_distribution(M,V,U,Nsamples)

		return A_samples, B_samples

	def _sample_controller_design_parameters(self,Nsamples_controller):

		# https://en.wikipedia.org/wiki/Gamma_distribution
		alpha = 1.0 # shape
		beta = 1.0 # rate
		theta_pars = scipy.stats.gamma.rvs(a=alpha,loc=0,scale=1/beta,size=(self.dim_state + self.dim_control,Nsamples_controller))

		Q_des_samples = np.zeros((self.dim_state,self.dim_state,Nsamples_controller))
		R_des_samples = np.zeros((self.dim_control,self.dim_control,Nsamples_controller))

		for jj in range(Nsamples_controller):

			Q_des_samples[:,:,jj] = np.eye(theta_pars[0:self.dim_state])
			R_des_samples[:,:,jj] = np.eye(theta_pars[self.dim_control::])

		return Q_des_samples, R_des_samples, theta_pars

	def _sample_matrix_normal_distribution(self,M,V,U,Nsamples):
		U = 2.0*np.eye(self.dim_state)
		V = np.eye(self.dim_state)
		M = np.zeros((self.dim_state,self.dim_state))

		mat_samples = numpy.random.multivariate_normal(mean=M.ravel(), cov=np.kron(V, U),size=Nsamples).reshape(M.shape)

		return mat_samples

	def compute_cost_for_each_controller(self,):

		Nsamples_system = 10
		Nsamples_controller = 10
		cost_values_all = np.zeros((Nsamples_system,Nsamples_controller))
		theta_pars_all = np.zeros((self.dim_state + self.dim_control,Nsamples_controller,Nsamples_system))
		
		A_samples, B_samples = self._sample_systems(Nsamples=Nsamples_system)
		for ii in range(Nsamples_system):

			Q_des_samples, R_des_samples, theta_pars = self._sample_controller_design_parameters(Nsamples_controller)
			theta_pars_all[:,:,ii] = theta_pars[:,:]
			for jj in range(Nsamples_controller):


				Q_des = Q_des_samples[:,:,jj]
				R_des = R_des_samples[:,:,jj]

				cost_values_all[ii,jj] = self.solve_lqr.forward_simulation(A_samples[:,:,ii], B_samples[:,:,ii], Q_des, R_des)


		return cost_values_all, theta_pars_all





