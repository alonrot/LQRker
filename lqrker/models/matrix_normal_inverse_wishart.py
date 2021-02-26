import tensorflow as tf
import pdb
import math

class MatrixNormalInverseWishart:
	"""
	Once we have the posterior, the idea is to sample
	"""
	def __init__(self,dim_row,dim_col):
		self.dim_row = dim_row
		self.dim_col = dim_col

	def get_inverse_wishart():

		if self.dim_row == 1:
			"""
			This is a specialized univariate case of the Inverse Wishart.
			In this case, we use the inverse-gamma distribution
			See https://en.wikipedia.org/wiki/Inverse-Wishart_distribution
			and
			https://en.wikipedia.org/wiki/Inverse-gamma_distribution
			"""

			alpha = 2.0
			beta = 1.0

			mean = beta / (alpha - 1)

			return mean

		else:
			pass

	def get_matrix_normal():

		if self.dim_row == 1:
			"""
			This is a specialized case
			"""


		else:
			pass


