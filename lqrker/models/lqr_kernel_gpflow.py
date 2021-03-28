import tensorflow as tf
import pdb
import gpflow
import pdb
import numpy as np
from lqrker.utils.solve_lqr import SolveLQR

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

class LQRmomentsCommon():
	"""

	Collection of methods that are common to both, LQRkernel and LQRmean.
	"""

	@staticmethod
	def _get_Lyapunov_solution(theta_vec,dim_in,dim_state,dim_control,M,A_samples,B_samples,solve_lqr):

		if dim_in > 1:
			Q_des = tf.linalg.diag(theta_vec[0:dim_state])
			R_des = tf.linalg.diag(theta_vec[dim_state::])
		else:
			Q_des_diag =  tf.concat([theta_vec,tf.ones(dim_state-1,dtype=tf.float64)],axis=0)
			Q_des = tf.linalg.diag(Q_des_diag)
			R_des = tf.linalg.eye(dim_control)

		P_list = []
		for j in range(M):
			A = A_samples[j,:,:]
			B = B_samples[j,:,:]
			P_list.append( solve_lqr.get_Lyapunov_solution(A, B, Q_des, R_des) )

		return P_list

	@staticmethod
	def _define_weights(M):
		# The GP predictions are sensitive to this. If self.w = (1./self.M)*tf.ones(self.M), the variance is too small. HOwever, if in the future we need to fix this (to have coherency in the theory), we can simply pump in prior variance and treat it as a hyperparameter.
		# pdb.set_trace()
		# w = (1./tf.sqrt(1.*M))*tf.ones(M)
		w = (1./M)*tf.ones(M)
		# w = tf.ones(M)

		return w

class LQRkernel(gpflow.kernels.Kernel,LQRmomentsCommon):
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
		self.dim_control = cfg.dim_control
		self.dim_in = dim

		mu0 = eval(cfg.initial_state_distribution.mu0)
		self.Sigma0 = eval(cfg.initial_state_distribution.Sigma0)
		# self.Sigma0 = np.eye(self.dim_state)
		# print("Not using the original Sigma0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

		Q_emp = eval(cfg.empirical_weights.Q_emp)
		R_emp = eval(cfg.empirical_weights.R_emp)
		self.solve_lqr = SolveLQR(Q_emp,R_emp,mu0,self.Sigma0)

		self.update_system_samples_and_weights(A_samples,B_samples)

		# Parameter of the distance function:
		# self.eta = 0.1 # Plots something for Nsys=4
		self.eta = 1.0

		# # Prior variance:
		# self.var_prior = 10.0

	def _LQR_kernel(self,SP1,SP2=None):

		if SP2 is None:
			ker_val = np.trace(SP1)**2 + 2.0*np.trace(np.linalg.matrix_power(SP1,2))
		else:
			ker_val = np.trace(SP1) * np.trace(SP2) + 2.0*np.trace(SP1 @ SP2)

		return ker_val

	def update_system_samples_and_weights(self,A_samples, B_samples):
		self.A_samples = A_samples
		self.B_samples = B_samples
		self.M = self.A_samples.shape[0]
		self.w = self._define_weights(self.M)

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

				P_X_ii_list = self._get_Lyapunov_solution(X[ii,:],self.dim_in,self.dim_state,
													self.dim_control,self.M,self.A_samples,
													self.B_samples,self.solve_lqr)
				P_X_jj_list = self._get_Lyapunov_solution(X2[jj,:],self.dim_in,self.dim_state,
													self.dim_control,self.M,self.A_samples,
													self.B_samples,self.solve_lqr)
				# P_X_ii_list = self._get_Lyapunov_solution(X[ii,:])
				# P_X_jj_list = self._get_Lyapunov_solution(X2[jj,:])

				Sigma0_dist = self.Sigma0_dist(X[ii,:],X2[jj,:])

				k_rc = 0
				for sysj_r in range(self.M):
					for sysj_c in range(sysj_r,self.M):

						P_X_ii = Sigma0_dist @ P_X_ii_list[sysj_r]
						P_X_jj = Sigma0_dist @ P_X_jj_list[sysj_c]

						if sysj_r == sysj_c:
							k_rc += self.w[sysj_c]**2 * self._LQR_kernel(P_X_ii,P_X_jj)
						else:
							pass # [DBG]: Trying to see what happens if we just do a linear combination of kernels
							# k_rc += 2.*self.w[sysj_r]*self.w[sysj_c]*self._LQR_kernel(P_X_ii,P_X_jj)

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

		# Convert to tensor, just in case:
		Kmat = tf.convert_to_tensor(Kmat, dtype=tf.float64)

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
			# P_X_ii_list = self._get_Lyapunov_solution(X[ii,:])
			P_X_ii_list = self._get_Lyapunov_solution(X[ii,:],self.dim_in,self.dim_state,
												self.dim_control,self.M,self.A_samples,
												self.B_samples,self.solve_lqr)


			k_r = 0
			for sysj_r in range(self.M):
				P_X_ii = self.Sigma0 @ P_X_ii_list[sysj_r]
				k_r += self.w[sysj_r]**2*self._LQR_kernel(P_X_ii)

			Kmat_diag[ii] = k_r

		# print("@K(): Kmat_diag", Kmat_diag)
		# pdb.set_trace()

		# Convert to tensor, just in case:
		Kmat_diag = tf.convert_to_tensor(Kmat_diag, dtype=tf.float64)

		return Kmat_diag


class LQRMean(gpflow.mean_functions.MeanFunction,LQRmomentsCommon):
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

		self.dim_state = cfg.dim_state
		self.dim_control = cfg.dim_control
		self.dim_in = dim

		mu0 = eval(cfg.initial_state_distribution.mu0)
		self.Sigma0 = eval(cfg.initial_state_distribution.Sigma0)
		# self.Sigma0 = np.eye(self.dim_state)
		# print("Not using the original Sigma0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


		self.solve_lqr = SolveLQR(Q_emp,R_emp,mu0,self.Sigma0)

		self.update_system_samples_and_weights(A_samples,B_samples)

	def update_system_samples_and_weights(self,A_samples, B_samples):
		self.A_samples = A_samples
		self.B_samples = B_samples
		self.M = self.A_samples.shape[0]
		self.w = self._define_weights(self.M)
	
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

			# P_X_ii_list = self._get_Lyapunov_solution(X[ii,:])
			P_X_ii_list = self._get_Lyapunov_solution(X[ii,:],self.dim_in,self.dim_state,
												self.dim_control,self.M,self.A_samples,
												self.B_samples,self.solve_lqr)

			mean_r = 0
			for sysj_r in range(self.M):
				P_X_ii = self.Sigma0 @ P_X_ii_list[sysj_r]
				mean_r += self.w[sysj_r]*self._mean_fun(P_X_ii)
			mean_vec[ii,0] = mean_r

		# print("@K(): mean_vec", mean_vec)

		mean_vec = tf.convert_to_tensor(mean_vec, dtype=tf.float64)

		return mean_vec


