import tensorflow as tf
import pdb
import gpflow
import pdb
import numpy as np
from lqrker.utils.solve_lqr import SolveLQR

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

class LQRkernel(gpflow.kernels.Kernel):
	"""

	This is the LQR kernel for a LTI system, which LQR controller is parametrized
	through the Q and R design cost matrices. The empirical cost is quadratic and
	fixed. Note that this class does not include process noise, as the kernel
	derivations are slightly more complicated.

	The kernel expresses Cov(J(xi),J(xj)), where xi is the parametrization of the
	LQR controller and J(xi) is the cost value obtained from running a forward
	simulation on the LTI system with an empirical cost. J(xi) is a random
	variate with unknown density, but that can be expressed as a linear
	combination of chi-squared independent random variates.

	TODO: Incorporate process noise (LQG) and obtain the LQG kernel.
	TODO: Remove the numpy library
	"""
	def __init__(self,cfg, dim, A_samples, B_samples):
		super().__init__(active_dims=[0])


		self.dim_state = cfg.dim_state
		self.dim_in = dim

		mu0 = eval(cfg.initial_state_distribution.mu0)
		self.Sigma0 = eval(cfg.initial_state_distribution.Sigma0)
		# self.Sigma0 = np.eye(self.dim_state)
		# print("Not using the original Sigma0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

		Q_emp = eval(cfg.empirical_weights.Q_emp)
		R_emp = eval(cfg.empirical_weights.R_emp)
		self.solve_lqr = SolveLQR(Q_emp,R_emp,mu0,self.Sigma0)

		self.A_samples = A_samples
		self.B_samples = B_samples

		# Number of systems:
		self.M = self.A_samples.shape[0]

		# Weights:
		self.w = (1./self.M)*tf.ones(self.M)

		# Parameter of the distance function:
		self.eta = 5.0

	def _LQR_kernel(self,SP1,SP2=None):

		if SP2 is None:
			ker_val = np.trace(SP1)**2 + 2.0*np.linalg.matrix_power(SP1,2)
		else:
			ker_val = np.trace(SP1) * np.trace(SP2) + 2.0*np.trace(SP1 @ SP2)

		return ker_val

	def _get_Lyapunov_solution(self,theta_vec):

		if self.dim_in > 1:
			Q_des = tf.linalg.diag(X[0:self.dim_state])
			R_des = tf.linalg.diag(X[self.dim_state::])
		else:
			Q_des = tf.linalg.diag(theta_vec)
			R_des = tf.constant([[1]])

		P_list = []
		for j in range(self.M):
			A = self.A_samples[j,:,:]
			B = self.B_samples[j,:,:]
			P_list.append( self.solve_lqr.get_Lyapunov_solution(A, B, Q_des, R_des) )

		return P_list

	def Sigma0_dist(self,x_vec, y_vec):
		return self.Sigma0 * tf.exp(-tf.math.reduce_euclidean_norm(x_vec - y_vec) / self.eta)

	def K(self,X,X2=None):
		"""

		Gram Matrix

		X: [Npoints, dim_in]
		X2: [Npoints2, dim_in]

		return:
		Kmat: [Npoints,Npoints2]

		TODO: Vectorize the call to _get_laypunov_sol() and _LQR_kernel()

		"""

		# Incorporate distance in the Sigma itself (experimental):
		# Sigma0_dist(self.Sigma0,X[ii,:],X2[jj,:])

		if X2 is None:
			X2 = X
		
		Kmat = np.zeros((X.shape[0],X2.shape[0]))
		for ii in range(X.shape[0]):
			for jj in range(X2.shape[0]):

				P_X_ii_list = self._get_Lyapunov_solution(X[ii,:])
				P_X_jj_list = self._get_Lyapunov_solution(X2[jj,:])

				Sigma0_dist = self.Sigma0_dist(X[ii,:],X2[jj,:])

				k_rc = 0
				for sysj_r in range(self.M):
					for sysj_c in range(sysj_r,self.M):

						P_X_ii = Sigma0_dist @ P_X_ii_list[sysj_r]
						P_X_jj = Sigma0_dist @ P_X_jj_list[sysj_c]

						if sysj_r == sysj_c:
							k_rc += self.w[sysj_c]**2 * self._LQR_kernel(P_X_ii,P_X_jj)
						else:
							k_rc += 2.*self.w[sysj_r]*self.w[sysj_c]*self._LQR_kernel(P_X_ii,P_X_jj)

				Kmat[ii,jj] = k_rc

				if ii == 0 and jj == 19:
					# pdb.set_trace()
					pass


		# pdb.set_trace()
		# print("@K(): Kmat", Kmat)

		# If square matrix, fix noise:
		if Kmat.shape[0] == Kmat.shape[1]:
			
			# TODO: When self.M > 1, cholesky fails. This, however, doesn't solve it...
			# Solve this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

			# print("eigvals:",np.linalg.eigvals(Kmat))
			# Kmat += 1e-2*np.eye(Kmat.shape[0])
			# pdb.set_trace()
			pass

		return Kmat

	def K_diag(self,X):
		"""
		It’s simply the diagonal of the K function, in the case where X2 is None.
		It must return a one-dimensional vector.

		X: [Npoints, dim_in]

		return:
		Kmat: [Npoints,]
		"""

		Kmat_diag = np.zeros(X.shape[0])
		for ii in range(X.shape[0]):
			P_X_ii_list = self._get_Lyapunov_solution(X[ii,:])

			k_r = 0
			for sysj_r in range(self.M):
				P_X_ii = self.Sigma0 @ P_X_ii_list[sysj_r]
				k_r += self.w[sysj_r]**2*self._LQR_kernel(P_X_ii)
			Kmat_diag[ii] = k_r

		# print("@K(): Kmat_diag", Kmat_diag)
		# pdb.set_trace()

		return Kmat_diag


class LQRMean(gpflow.mean_functions.MeanFunction):
	"""

	This is the expectation of the LQR cost for a LTI system, which LQR controller is parametrized
	through the Q and R design cost matrices. The empirical cost is quadratic and
	fixed. Note that this class does not include process noise.

	The mean expresses E[J(xi)], where xi is the parametrization of the
	LQR controller and J(xi) is the cost value obtained from running a forward
	simulation on the LTI system with an empirical cost. J(xi) is a random
	variate with unknown density, but that can be expressed as a linear
	combination of chi-squared independent random variates.

	TODO: Incorporate process noise (LQG) and obtain the LQG mean.
	"""
	def __init__(self, cfg, dim, A_samples=None, B_samples=None):

		Q_emp = eval(cfg.empirical_weights.Q_emp)
		R_emp = eval(cfg.empirical_weights.R_emp)

		self.dim_state = Q_emp.shape[0]
		self.dim_in = dim

		mu0 = eval(cfg.initial_state_distribution.mu0)
		self.Sigma0 = eval(cfg.initial_state_distribution.Sigma0)
		# self.Sigma0 = np.eye(self.dim_state)
		# print("Not using the original Sigma0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


		self.solve_lqr = SolveLQR(Q_emp,R_emp,mu0,self.Sigma0)

		self.A_samples = A_samples
		self.B_samples = B_samples

		# Number of systems:
		self.M = self.A_samples.shape[0]

		# Weights:
		self.w = (1./self.M)*tf.ones(self.M)

	def _get_Lyapunov_solution(self,theta_vec):

		if self.dim_in > 1:
			Q_des = tf.linalg.diag(X[0:self.dim_state])
			R_des = tf.linalg.diag(X[self.dim_state::])
		else:
			Q_des = tf.linalg.diag(theta_vec)
			R_des = tf.constant([[1]])

		P_list = []
		for j in range(self.M):
			A = self.A_samples[j,:,:]
			B = self.B_samples[j,:,:]
			P_list.append( self.solve_lqr.get_Lyapunov_solution(A, B, Q_des, R_des) )

		return P_list

	def _mean_fun(self,SP):
		"""

		Herein we are assuming zero mean for the initial condition, i.e., mu0 = 0.
		"""

		return np.trace(SP)

	def __call__(self,X):
		"""

		X: [Npoints, dim]

		return: [Npoints,1]
		"""

		mean_vec = np.zeros((X.shape[0],1))
		for ii in range(X.shape[0]):

			P_X_ii_list = self._get_Lyapunov_solution(X[ii,:])

			mean_r = 0
			for sysj_r in range(self.M):
				P_X_ii = self.Sigma0 @ P_X_ii_list[sysj_r]
				mean_r += self.w[sysj_r]*self._mean_fun(P_X_ii)
			mean_vec[ii,0] = mean_r

		# print("@K(): mean_vec", mean_vec)

		return mean_vec