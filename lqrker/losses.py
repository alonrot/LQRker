import tensorflow as tf

class LossKLDiv():
	def __init__(self,Sigma_noise):
		self.Sigma_noise = Sigma_noise

	def get(self,mean_pred,cov_pred,y_new):
		return tf.math.log(1 + cov_pred/self.Sigma_noise) + (y_new - mean_pred)**2 / cov_pred

class LossMultivariateStudentT_MLII():

	@staticmethod
	def get(Y,PhiX,Sigma_weights,)