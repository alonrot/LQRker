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

		self.Q_emp = eval(cfg.empirical_weights.Q_emp)
		self.R_emp = eval(cfg.empirical_weights.R_emp)

		self.dim_state = self.Q_emp.shape[0]
		self.dim_in = dim

		# Parameters:
		if cfg.initial_state_distribution.mu0 == "zeros":
			mu0 = np.zeros((self.dim_state,1))
		elif cfg.initial_state_distribution.mu0 == "random":
			pass
		else:
			raise ValueError

		if cfg.initial_state_distribution.Sigma0 == "identity":
			self.Sigma0 = np.eye(self.dim_state)
		else:
			raise ValueError

		self.solve_lqr = SolveLQR(self.Q_emp,self.R_emp,mu0,self.Sigma0)

		self.A_samples = A_samples
		self.B_samples = B_samples

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

		# TODO: Make this dependent on all the systems (linear combination of P):
		A = self.A_samples[0,:,:]
		B = self.B_samples[0,:,:]

		P = self.solve_lqr.get_Lyapunov_solution(A, B, Q_des, R_des)

		return P

	def K(self,X,X2=None):
		"""

		Gram Matrix

		X: [Npoints, dim_in]
		X2: [Npoints2, dim_in]

		return:
		Kmat: [Npoints,Npoints2]

		TODO: Vectorize the call to _get_laypunov_sol() and _LQR_kernel()

		"""

		if X2 is None:
			X2 = X
		
		Kmat = np.zeros((X.shape[0],X2.shape[0]))
		for ii in range(X.shape[0]):
			for jj in range(X2.shape[0]):

				P_X_ii = self._get_Lyapunov_solution(X[ii,:])
				P_X_jj = self._get_Lyapunov_solution(X2[jj,:])

				P_X_ii = self.Sigma0 @ P_X_ii
				P_X_jj = self.Sigma0 @ P_X_jj

				Kmat[ii,jj] = self._LQR_kernel(P_X_ii,P_X_jj)

		# print("@K(): Kmat", Kmat)

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
			P_X_ii = self._get_Lyapunov_solution(X[ii,:])
			P_X_ii = self.Sigma0 @ P_X_ii
			Kmat_diag[ii] = self._LQR_kernel(P_X_ii)

		# print("@K(): Kmat_diag", Kmat_diag)

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

		self.Q_emp = eval(cfg.empirical_weights.Q_emp)
		self.R_emp = eval(cfg.empirical_weights.R_emp)

		self.dim_state = self.Q_emp.shape[0]
		self.dim_in = dim

		# Parameters:
		if cfg.initial_state_distribution.mu0 == "zeros":
			mu0 = np.zeros((self.dim_state,1))
		elif cfg.initial_state_distribution.mu0 == "random":
			pass
		else:
			raise ValueError

		if cfg.initial_state_distribution.Sigma0 == "identity":
			self.Sigma0 = np.eye(self.dim_state)
		else:
			raise ValueError

		self.solve_lqr = SolveLQR(self.Q_emp,self.R_emp,mu0,self.Sigma0)

		self.A_samples = A_samples
		self.B_samples = B_samples

	def _get_Lyapunov_solution(self,theta_vec):

		if self.dim_in > 1:
			Q_des = tf.linalg.diag(X[0:self.dim_state])
			R_des = tf.linalg.diag(X[self.dim_state::])
		else:
			Q_des = tf.linalg.diag(theta_vec)
			R_des = tf.constant([[1]])

		# TODO: Make this dependent on all the systems (linear combination of P):
		A = self.A_samples[0,:,:]
		B = self.B_samples[0,:,:]

		P = self.solve_lqr.get_Lyapunov_solution(A, B, Q_des, R_des)

		return P

	def _mean_fun(self,SP):

		return np.trace(SP)

	def __call__(self,X):
		"""

		X: [Npoints, dim]

		return: [Npoints,1]
		"""

		mean_vec = np.zeros((X.shape[0],1))
		for ii in range(X.shape[0]):
			P_X_ii = self._get_Lyapunov_solution(X[ii,:])
			P_X_ii = self.Sigma0 @ P_X_ii
			mean_vec[ii,0] = self._mean_fun(P_X_ii)

		# print("@K(): mean_vec", mean_vec)

		return mean_vec