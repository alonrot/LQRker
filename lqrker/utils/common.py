from abc import ABC, abstractmethod
import tensorflow as tf
import math
import pdb
import numpy as np
import scipy
from scipy import stats

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


class CommonUtils():

	def __init__(self):
		pass

	@staticmethod
	def create_Ndim_grid(xmin,xmax,Ndiv,dim):
		"""
		
		Create a regular grid on a hypercube [xmin,xmax]**dim
		and "vectorize" it as a matrix with Ndiv**dim rows and dim columns,
		such that each point can be accessed as vectorized_grid[i,:]

		return:
			vectorized_grid: [Ndiv**dim,dim]

		"""

		xpred = tf.linspace(xmin,xmax,Ndiv)
		xpred_data = tf.meshgrid(*([xpred]*dim),indexing="ij")
		vectorized_grid = tf.concat([tf.reshape(xpred_data_el,(-1,1)) for xpred_data_el in xpred_data],axis=1)

		return vectorized_grid

	@staticmethod
	def fix_eigvals_other_way(Kmat,verbosity=False):

		# Work with numpy because tf.linalg.eig and tf.linalg.eigvals return COMPLEX eigenvalues due to numerical precision issues.
		# This shouldn't be the case because BB_sym is symmetric and real.
		if tf.is_tensor(Kmat):
			Kmat = Kmat.numpy()

		if np.any(np.isnan(Kmat)):
			logger.info("The input matrix contains nans...")
			pdb.set_trace()

		assert np.all(Kmat == Kmat.T), "Matrix must be symmetric!"

		try:
			Kmat_sol = scipy.linalg.cholesky_banded(Kmat)
		except:
			if verbosity: logger.info("Kmat needs to be fixed...")
		else:
			if verbosity: logger.info("Kmat is PD; nothing to fix...")
			return Kmat_sol


		# Get the lowest eigenvalue in absolute value:
		eig_min = np.amin(np.linalg.eigvalsh(Kmat))
		assert eig_min < 0.0
		eig_min_abs = abs(eig_min)

		# Compute its log:
		log_eig_min_abs = np.math.log10(eig_min_abs)

		Nfac = np.ceil(10**(-(-7-log_eig_min_abs)))

		AAt_sym_corrected = AA_sym / Nfac + 1e-6*tf.eye(AA_sym.shape[0])
		Lchol = np.linalg.cholesky(AAt_sym_corrected) * np.sqrt(Nfac)

		return Lchol

	@staticmethod
	def fix_eigvals(Kmat,verbosity=False):
		"""

		Among the negative eigenvalues, get the 'most negative one'
		and return it with flipped sign
		"""

		if tf.math.reduce_any(tf.math.is_nan(Kmat)):
			logger.info("The input matrix contains nans...")
			pdb.set_trace()

		Kmat_sol = tf.linalg.cholesky(Kmat)
		# Kmat_sym = 0.5*(Kmat + tf.transpose(Kmat))
		# Kmat_sol = tf.linalg.cholesky(Kmat_sym)
		if tf.math.reduce_any(tf.math.is_nan(Kmat_sol)):
			if verbosity: logger.info("Kmat needs to be fixed...")
		else:
			if verbosity: logger.info("Kmat is PD; nothing to fix...")
			return Kmat

		try:
			eigvals, eigvect = tf.linalg.eigh(Kmat)
			# eigvals, eigvect = tf.linalg.eigh(Kmat_sym)
		except Exception as inst:
			logger.info("type: {0:s} | args: {1:s}".format(str(type(inst)),str(inst.args)))
			logger.info("Failed to compute tf.linalg.eigh(Kmat) ...")
			pdb.set_trace()

		max_eigval = tf.reduce_max(tf.math.real(eigvals))
		min_eigval = tf.reduce_min(tf.math.real(eigvals))

		# Compte eps:
		# eps must be such that the condition number of the resulting matrix is not too large
		max_order_eigval = tf.math.ceil(tf.experimental.numpy.log10(max_eigval))
		eps = 10**(max_order_eigval-8) # We set a maximum condition number of 8

		# Fix eigenvalues:
		eigvals_fixed = eigvals + tf.abs(min_eigval) + eps

		# pdb.set_trace()
		if verbosity: logger.info(" Fixed by adding " + str(tf.abs(min_eigval).numpy()))
		if verbosity: logger.info(" and also by adding " + str(eps.numpy()))

		Kmat_fixed = eigvect @ ( tf.linalg.diag(eigvals_fixed) @ tf.transpose(eigvect) ) # tf.transpose(eigvect) is the same as tf.linalg.inv(eigvect) | checked

		# Kmat_fixed_sym = 0.5*(Kmat_fixed + tf.transpose(Kmat_fixed))

		try:
			tf.linalg.cholesky(Kmat_fixed)
		except:
			pdb.set_trace()

		# pdb.set_trace()

		return Kmat_fixed


	@staticmethod
	def sample_standard_multivariate_normal_inside_confidence_set(Nsamples,Nels,min_prob_chi2):

		level_val = stats.chi2.ppf(q=min_prob_chi2,df=Nels)
		noise_vec = np.ndarray.astype(np.random.randn(Nsamples,Nels),dtype=np.float32)
		noise_squared_vals = np.sum(noise_vec*noise_vec,axis=1)
		ind_inside = noise_squared_vals <= level_val

		if np.all(ind_inside):
			return noise_vec

		c = 0
		c_max = 100
		all_inside = np.all(ind_inside)
		ind_inside_new = ind_inside
		while not all_inside and c < c_max:

			Nsamples_new = sum(~ind_inside_new)
			noise_vec_new = np.ndarray.astype(np.random.randn(Nsamples_new,Nels),dtype=np.float32)
			noise_squared_vals_new = np.sum(noise_vec_new*noise_vec_new,axis=1)
			ind_inside_new = noise_squared_vals_new <= level_val

			# Replace samples that were outside:
			noise_vec[~ind_inside] = noise_vec_new

			# Check again which samples need to be fixed:
			noise_squared_vals = np.sum(noise_vec*noise_vec,axis=1)
			ind_inside = noise_squared_vals <= level_val

			all_inside = np.all(ind_inside)

			c += 1

		if not all_inside:
			noise_vec = noise_vec[ind_inside]
		
		print("Returning {0:d} / {1:d} samples".format(np.sum(ind_inside),Nsamples))

		return noise_vec