import tensorflow as tf
import pdb

class MyDenseReLuLayer(tf.keras.layers.Layer):
	# Adding **kwargs to support base Keras layer arguments
	def __init__(self, in_dim, out_dim, name="MyDenseReLuLayer", **kwargs):
		super().__init__(**kwargs)

		self.w = self.add_weight(shape=(in_dim, out_dim), initializer="random_normal", trainable=True)
		self.b = self.add_weight(shape=(out_dim,), initializer="zeros", trainable=True)

	def call(self, inputs):
		y = tf.matmul(inputs, self.w) + self.b
		return tf.nn.relu(y)


class ModelFeatures(tf.keras.Model):
	"""

	This is the NN model representing the Phi(x) part of the Bayesian linear model, i.e.
	J(x) ~ Phi(x) * beta

	"""

	def __init__(self, in_dim, num_features_out, name="ModelFeatures", **kwargs):
		super().__init__(**kwargs)

		self.dense = MyDenseReLuLayer(in_dim=in_dim, out_dim=num_features_out)

		# Adding weights:
		self.L0 = self.add_weight(shape=(num_features_out, num_features_out), initializer="random_normal", trainable=True)
		self.beta0 = self.add_weight(shape=(num_features_out, 1), initializer="random_normal", trainable=True)

	def call(self, x):
		# x = self.dense(x)
		# return self.blr(x)
		return self.dense(x)

class BayesianLinearRegression:
	"""

	Wrapper class that includes the Neural Network that models
	the features Phi(x), as well as the functions to compute the
	predictive distribution of beta, used to eventually
	compute the predictive q(y|x) distribution
	"""
	def __init__(self, in_dim, num_features_out, Sigma_noise):

		self.model_features = ModelFeatures(in_dim=in_dim, num_features_out=num_features_out)
		self.Sigma_noise = Sigma_noise

	def _matrix_normal_posterior(self, L0, beta0, Y, X, Phi):
		"""

		Matrix normal posterior conditioned on a "trajectory" of datapoints { x_j, y_j }_j=1:T

		T: number of data samples

		X: [in_dim x T] (TODO: check this dimensionality!)
		Y: [T x 1]
		L0: [n_features x n_features]
		beta0: [n_features x 1]
		Phi: [T x n_features]
		"""

		n_features = L0.shape[0]
		Lambda0 = tf.matmul(L0,tf.transpose(L0)) + 1e-6*tf.eye(n_features)
		Lambda_post = tf.matmul(tf.transpose(Phi),Phi) + Lambda0

		# pdb.set_trace()
		alpha = tf.matmul(tf.transpose(Phi),Y) + tf.matmul(Lambda0,beta0)
		Lambda_post_inv = tf.linalg.inv(Lambda_post) # [n_features x n_features]

		beta_post = tf.matmul( Lambda_post_inv , alpha ) # [n_features x 1]

		return beta_post, Lambda_post_inv

	def get_PhiX(self,X):
		"""
		X: [batch x in_dim]
		"""
		return self.model_features(X)

	def q_predictive_gaussian(self, X, Y, x_new):

		# Phi = model(X)
		Phi = self.get_PhiX(X)
		beta_post, Lambda_post_inv = self._matrix_normal_posterior(L0=self.model_features.L0, beta0=self.model_features.beta0, Y=Y, X=X, Phi=Phi)


		"""
		x_new: [in_dim x 1] (TODO: check this dimensionality!)
		beta_post: [n_features x 1]
		Lambda_post_inv: [n_features x n_features]
		Sigma_noise: [1 x 1]
		"""

		# Gaussian mean:
		Phi_x_new = self.model_features(x_new)
		# mean = tf.matmul( tf.transpose(beta_post) , Phi_x_new ) # dot product beta^T * Phi(x)
		mean = tf.matmul( Phi_x_new, beta_post )  # dot product beta^T * Phi(x)

		# pdb.set_trace()
		cov = ( 1 + tf.matmul( tf.matmul( Phi_x_new, Lambda_post_inv) , tf.transpose(Phi_x_new) ) ) * self.Sigma_noise # [1 x 1]

		# mean, cov = self.q_predictive_gaussian(model=self.model_features, x_new=x_new, beta_post=beta_post, Lambda_post_inv=Lambda_post_inv)

		return mean, cov


	def q_learned_prior(self, x_test):

		Lambda0 = tf.matmul(self.model_features.L0,tf.transpose(self.model_features.L0)) + 1e-6*tf.eye(self.model_features.L0.shape[0])
		Lambda0_inv = tf.linalg.inv(Lambda0)
		beta0 = self.model_features.beta0

		# Mean:
		Phi_x_test = self.model_features(x_test)
		mean = tf.matmul( Phi_x_test, beta0 )  # dot product beta^T * Phi(x)

		# Covariance matrix:
		cov = ( 1 + tf.matmul( tf.matmul( Phi_x_test, Lambda0_inv) , tf.transpose(Phi_x_test) ) ) * self.Sigma_noise # [1 x 1]

		return mean, cov


