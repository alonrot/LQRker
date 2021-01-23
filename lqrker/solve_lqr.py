from scipy import linalg as la
from scipy import stats as sts
import pdb
import numpy as np
import control


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

	def __init__(self,Q_emp,R_emp,mu0,Sigma0):

		assert Q_emp.ndim == 2
		assert R_emp.ndim == 2
		assert mu0.ndim == 2, "Pass mu0 as a 2D column vector"
		assert mu0.shape[1] == 1
		assert Sigma0.ndim == 2

		self.Q_emp = Q_emp
		self.R_emp = R_emp
		self.mu0 = mu0
		self.Sigma0 = Sigma0


	def _get_controller(self, A, B, Q_des, R_des):
		P, eig, K = control.dare(A, B, Q_des, R_des)
		# P: solution to the Ricatti equation for discrete time invariant linear systems
		# eig: eigenvalues of the closed loop A - BK
		# K: feedabck gain assuming u = -Kx

		return K

	def forward_simulation(self, A, B, Q_des, R_des):
		"""

		We consider here an infinite horizon LQR with stochastic initial condition
		TODO: consider the LQG case
		"""

		K = self._get_controller(A, B, Q_des, R_des)

		A_tilde = A + np.matmul(B,-K) # Flip sign of the controller, as we assume u = Kx
		eig = la.eigvals(A_tilde)
		assert np.all(np.absolute(eig) <= 1.0), "The eigenvalues must be inside the unit circle"

		A_tilde_inv = np.linalg.inv(A_tilde)
		Q_tilde = self.Q_emp + np.matmul(K.T,np.matmul(self.R_emp,K))

		A_syl = A_tilde.T
		B_syl = -A_tilde_inv
		Q_syl = -np.matmul(Q_tilde,A_tilde_inv)

		P = la.solve_sylvester(A_syl,B_syl,Q_syl)

		# Q_new = Q_tilde
		# R_new = np.zeros((2,2))
		# pdb.set_trace()
		# Plib,_,_ = control.dare(A_tilde, np.zeros((2,2)), Q_new, R_new)


		J = np.trace(np.matmul(P,self.Sigma0)) + np.matmul(self.mu0.T,np.matmul(P,self.mu0))
		assert J.shape == (1,1)
		
		return J[0,0]
		
class GenerateLQRData():
	"""

	Given a matrix-normal distribution over the linear model matrices (A,B)
	and a distribution over possible controller designs theta = (Q_des,R_des),
	this class generates forward simulations of the sampled systems (A_j,B_j) with 
	controller designs theta_j. For each forward simulation, we compute the
	resulting quadratic cost, according to the empirical weights (Q_emp,R_emp)
	"""

	def __init__(self,Q_emp,R_emp,mu0,Sigma0,check_controllability=True):

		self.dim_state = Q_emp.shape[0]
		self.dim_control = R_emp.shape[0]

		self.solve_lqr = SolveLQR(Q_emp,R_emp,mu0,Sigma0)

		self.check_controllability = check_controllability

	def _sample_systems(self,Nsamples):

		M = np.zeros((self.dim_state,self.dim_state))
		V = np.eye(self.dim_state)
		U = 2.0*np.eye(self.dim_state)
		A_samples = self._sample_matrix_normal_distribution(M,V,U,Nsamples)

		M = np.zeros((self.dim_state,self.dim_control))
		V = np.eye(self.dim_control)
		U = 2.0*np.eye(self.dim_state)
		B_samples = self._sample_matrix_normal_distribution(M,V,U,Nsamples)

		return A_samples, B_samples


	def _check_controllability(self,A,B):
		
		assert A.shape[0] > 1, "This function is not designed for scalar systems"

		# Controlability:
		ctrb = B
		AB_mult = B
		for ii in range(1,self.dim_state-1):
			AB_mult = np.matmul(A,AB_mult)
			ctrb = np.hstack((ctrb,AB_mult))

		rank = np.linalg.matrix_rank(ctrb)


		# assert rank == A.shape[0], "The generated system is not controllable"
		# if rank != A.shape[0]:
		# 	pdb.set_trace()


	def _sample_controller_design_parameters(self,Nsamples_controller):

		# https://en.wikipedia.org/wiki/Gamma_distribution
		alpha = 1.0 # shape
		beta = 1.0 # rate
		theta_pars = sts.gamma.rvs(a=alpha,loc=0,scale=1/beta,size=(self.dim_state + self.dim_control,Nsamples_controller))

		Q_des_samples = np.zeros((self.dim_state,self.dim_state,Nsamples_controller))
		R_des_samples = np.zeros((self.dim_control,self.dim_control,Nsamples_controller))

		# pdb.set_trace()
		for jj in range(Nsamples_controller):

			Q_des_samples[:,:,jj] = np.diag(theta_pars[0:self.dim_state,jj])
			R_des_samples[:,:,jj] = np.diag(theta_pars[self.dim_control::,jj])

		return Q_des_samples, R_des_samples, theta_pars

	def _sample_matrix_normal_distribution(self,M,V,U,Nsamples):
		mat_samples = np.random.multivariate_normal(mean=M.ravel(), cov=np.kron(V, U),size=Nsamples).reshape( [-1,M.shape[0],M.shape[1]] )
		return mat_samples

	def compute_cost_for_each_controller(self,):
		"""
		We sample Nsys linear time-invariant systems living within the system uncertainty and, 
		for each system, we sample Ncon controller designs (Q_des,R_des)
		For each system, and each design, an optimal infinite-horizon LQR controller is computed.
		This controller is executed on the simulated system, and the quadratic cost J, computed with
		the empirical weights (Q_emp,R_emp) is returned.

		returns:
		cost_values_all: [Nsys x Ncon] # One cost value for each system and each controller
		theta_pars_all: [(dim_state + dim_control) x  ]
		"""

		Nsys = 10
		Ncon = 15
		cost_values_all = np.zeros((Nsys,Ncon))
		theta_pars_all = np.zeros((self.dim_state + self.dim_control,Nsys,Ncon))
		
		A_samples, B_samples = self._sample_systems(Nsamples=Nsys)
		for ii in range(Nsys):

			Q_des_samples, R_des_samples, theta_pars = self._sample_controller_design_parameters(Ncon)
			theta_pars_all[:,ii,:] = theta_pars[:,:]
			for jj in range(Ncon):


				Q_des = Q_des_samples[:,:,jj]
				R_des = R_des_samples[:,:,jj]

				# pdb.set_trace()
				if self.check_controllability == True:
					self._check_controllability(A_samples[ii,:,:], B_samples[ii,:,:])

				cost_values_all[ii,jj] = self.solve_lqr.forward_simulation(A_samples[ii,:,:], B_samples[ii,:,:], Q_des, R_des)

		# pdb.set_trace()

		return cost_values_all, theta_pars_all





