import tensorflow as tf
from abc import ABC, abstractmethod
import tensorflow_probability as tfp

import pdb
import matplotlib.pyplot as plt
import gpflow
import hydra
import numpy as np

from lqrker.models.lqr_kernel_gpflow import LQRkernel, LQRMean
from lqrker.models.lqr_kernel_trans_gpflow import LQRkernelTransformed, LQRMeanTransformed

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

from lqrker.utils.generate_linear_systems import GenerateLinearSystems

class LossBase(ABC):

	def __init__(self,mean_pred,var_pred):
		self.mean_pred = tf.cast(tf.squeeze(mean_pred),dtype=tf.float32)
		self.var_pred = tf.cast(tf.squeeze(var_pred),dtype=tf.float32)

	def SMSE(self,cost_vals):
		"""
		Standarized mean squared error (SMSE)
		"""

		cost_vals = tf.cast(tf.squeeze(cost_vals),dtype=tf.float32)
		return tf.reduce_mean( (self.mean_pred-cost_vals)**2/self.var_pred )

	@abstractmethod
	def MSLL(self,nu,cost_vals):
		"""
		Mean standardized log loss (MSLL)
		"""
		raise NotImplementedError

class LossStudentT(LossBase):

	def __init__(self,mean_pred,var_pred,nu):
		super().__init__(mean_pred,var_pred)
		self.nu = nu

	def MSLL(self,cost_vals):
		"""
		Mean standardized log loss (MSLL)
		"""

		dist_studentT = tfp.distributions.StudentT(df=self.nu,loc=self.mean_pred,scale=tf.sqrt(self.var_pred))
		return tf.reduce_mean(-dist_studentT.log_prob(cost_vals))

class LossGaussian(LossBase):

	def __init__(self,mean_pred,var_pred):
		super().__init__(mean_pred,var_pred)

	def MSLL(self,cost_vals):
		"""
		Mean standardized log loss (MSLL)
		"""

		dist_normal = tfp.distributions.Normal(loc=self.mean_pred,scale=tf.sqrt(self.var_pred))
		return tf.reduce_mean(-dist_normal.log_prob(cost_vals))


class LossKLDiv():
	def __init__(self,Sigma_noise):
		self.Sigma_noise = Sigma_noise

	def get(self,mean_pred,cov_pred,y_new):
		return tf.math.log(1 + cov_pred/self.Sigma_noise) + (y_new - mean_pred)**2 / cov_pred

