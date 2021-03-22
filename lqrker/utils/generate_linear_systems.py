from scipy import linalg as la
from scipy import stats as sts
import pdb
import numpy as np
import control

class GenerateLinearSystems():
	"""

	This class provides helper functions to generate a collection of controllable
	linear systems from a given prior distribution on the systems.

	TODO: Pass this to tensorflow
	"""

	def __init__(self,dim_state,dim_control,Nsys,check_controllability=True):

		# Number of systems to sample:
		self.Nsamples = Nsys

		# Dimensionality:
		self.dim_state = dim_state
		self.dim_control = dim_control

		self.check_controllability = check_controllability

	def _sample_systems(self,Nsamples):
		"""

		We have assumed that the matrices are sampled following independent
		matrix-normal distributions.
		
		TODO: Propose an inverse-wishart-normal prior, as we'll be allowed to choose
		the prior eigenvalues for A.
		"""

		M = np.zeros((self.dim_state,self.dim_state))
		V = np.eye(self.dim_state)
		U = 0.5*np.eye(self.dim_state)
		A_samples = self._sample_matrix_normal_distribution(M,V,U,Nsamples)

		M = np.zeros((self.dim_state,self.dim_control))
		V = np.eye(self.dim_control)
		U = 0.5*np.eye(self.dim_state)
		B_samples = self._sample_matrix_normal_distribution(M,V,U,Nsamples)

		if self.check_controllability:
			for ii in range(Nsamples):
				self._check_controllability(A_samples[ii,:,:], B_samples[ii,:,:])

		return A_samples, B_samples

	def _check_controllability(self,A,B):
		
		# Controlability:
		ctrb = B
		AB_mult = B
		for ii in range(1,self.dim_state):
			AB_mult = A @ AB_mult
			ctrb = np.hstack((ctrb,AB_mult))

		rank = np.linalg.matrix_rank(ctrb)

		assert rank == A.shape[0], "The generated system is not controllable"
		if rank != A.shape[0]:
			pdb.set_trace()

	def _sample_matrix_normal_distribution(self,M,V,U,Nsamples):
		mat_samples = np.random.multivariate_normal(mean=M.ravel(), cov=np.kron(V, U),size=Nsamples).reshape( [-1,M.shape[0],M.shape[1]] )
		return mat_samples

	def __call__(self):
		return self._sample_systems(self.Nsamples)


