from scipy import linalg as la
from scipy import stats as sts
import pdb
import numpy as np
import control
from scipy.stats import invwishart, matrix_normal

class GenerateLinearSystems():
	"""

	This class provides helper functions to generate a collection of controllable
	linear systems from a given prior distribution on the systems.

	TODO: Pass this to tensorflow
	"""

	def __init__(self,dim_state,dim_control,Nsys,check_controllability=True,prior="MNIW"):

		# Number of systems to sample:
		self.Nsamples = Nsys

		# Dimensionality:
		self.dim_state = dim_state
		self.dim_control = dim_control

		self.check_controllability = check_controllability

		self.prior = prior

	def _sample_systems_MNIW(self,Nsamples):
		"""

		We have assumed that the matrices are sampled following independent
		matrix-normal distributions.
		
		TODO: Propose an inverse-wishart-normal prior, as we'll be allowed to choose
		the prior eigenvalues for A.
		"""

		# scipy.stats.invwishart
		# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invwishart.html
		# https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf

		# scipy.stats.matrix_normal
		# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.matrix_normal.html

		# Tensorflow: The matrix Wishart distribution parameterized with Cholesky factors.
		# https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/WishartTriL


		# Distribution over A, inverse Wishart:
		S0 = 0.5*np.eye(self.dim_state)
		df = self.dim_state + 1

		# Distribution over B, matrix-normal distribution:
		M = np.zeros((self.dim_state,self.dim_control))
		# U = np.eye(self.dim_state) -> A | rowcov
		V = 2.0*np.eye(self.dim_control) # -> colcov

		A_samples = invwishart.rvs(df=df,scale=S0,size=Nsamples)
		A_samples = np.reshape(A_samples,(Nsamples,self.dim_state,self.dim_state))
		B_samples = np.zeros((Nsamples,self.dim_state,self.dim_control))
		for ii in range(Nsamples):
			B_samples[ii,:,:] = matrix_normal.rvs(mean=M,rowcov=A_samples[ii,:,:],colcov=V,size=1)

		if self.check_controllability:
			for ii in range(Nsamples):
				self._check_controllability(A_samples[ii,:,:], B_samples[ii,:,:])

		return A_samples, B_samples


	def _sample_systems_MN(self,Nsamples):

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
		# if rank != A.shape[0]:
		# 	pdb.set_trace()

	def _sample_matrix_normal_distribution(self,M,V,U,Nsamples):
		mat_samples = np.random.multivariate_normal(mean=M.ravel(), cov=np.kron(V, U),size=Nsamples).reshape( [-1,M.shape[0],M.shape[1]] )
		return mat_samples

	def __call__(self):

		if self.prior == "MN":
			return self._sample_systems_MN(self.Nsamples)
		elif self.prior == "MNIW":
			return self._sample_systems_MNIW(self.Nsamples)
		else:
			ValueError("Wrong prior")



if __name__ == "__main__":


	gen = GenerateLinearSystems(dim_state=3,dim_control=2,Nsys=10,check_controllability=True)

	A_samples, B_samples = gen()

	print(A_samples)
	print(B_samples)


