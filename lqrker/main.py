import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
import numpy as np
from lqrker.losses import LossKLDiv
from lqrker.model_blr import BayesianLinearRegression, ModelFeatures

def custom_training_loop():
	"""
	Based on https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/#low-level_handling_of_metrics
	"""

	in_dim = 1
	num_features_out = 16
	T = 101
	Sigma_noise_std = 0.1
	Sigma_noise = Sigma_noise_std**2

	Ninstances = 20

	# Inputs:
	Xtrain = tf.random.uniform(shape=(Ninstances*T,in_dim), minval=-2.0, maxval=2.0, dtype=tf.dtypes.float32, seed=None, name=None)

	# Output function (LQR cost):
	# pdb.set_trace()
	y_true = lambda Xtrain: tf.reduce_sum(Xtrain**2,axis=1) + Sigma_noise_std * tf.random.normal(shape=(Xtrain.shape[0],))

	# if in_dim == 1:
	# 	y_true = lambda Xtrain: 5.0*tf.math.sin(Xtrain[:,0]) + Sigma_noise_std * tf.random.normal(shape=(Xtrain.shape[0],))

	Ytrain = y_true(Xtrain)

	# pdb.set_trace()

	Ytrain = tf.reshape(Ytrain,[-1,1])

	# model = ModelFeatures(in_dim=in_dim, num_features_out=num_features_out)
	blr = BayesianLinearRegression(in_dim,num_features_out,Sigma_noise)
	loss_kl_div = LossKLDiv(Sigma_noise)

	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

	epochs = 30
	for epoch in range(epochs):

		with tf.GradientTape() as tape:

			loss_value = 0
			for jj in range(Ninstances-1):

				X = Xtrain[ jj*T:(jj+1)*T , : ]
				Y = Ytrain[ jj*T:(jj+1)*T ]

				x_new = tf.reshape(X[-1,:],[-1,in_dim])
				y_new = tf.reshape(Y[-1],[-1,1])

				X = X[0:-1,:]
				Y = Y[0:-1]

				# pdb.set_trace()

				mean, cov = blr.q_predictive_gaussian(X, Y, x_new)

				# pdb.set_trace()
				loss_value += loss_kl_div.get(mean_pred=mean,cov_pred=cov,y_new=y_new)

			loss_value = loss_value / Ninstances


		grads = tape.gradient(loss_value, blr.model_features.trainable_weights)
		optimizer.apply_gradients(zip(grads, blr.model_features.trainable_weights))


		# Log every 200 batches.
		if epoch % 2 == 0:
			print("Training loss (for one epoch) at epoch %d: %.4f" % (epoch, float(loss_value)))
		# print("Seen so far: %d samples" % ((Ninstances*(T-1) ) )


	# Plotting prior:
	# ==============
	if in_dim == 1:
		x_test = tf.reshape(tf.linspace(-2,+2,101),[-1,1])
		mean, cov = blr.q_learned_prior(x_test)
		std = tf.sqrt(tf.linalg.diag_part(cov))
		
		# pdb.set_trace()

		hdl_fig, hdl_plot = plt.subplots(1,1)
		x_test_np = x_test.numpy()[:,0]
		mean_np = mean.numpy()[:,0]
		std_np = std.numpy()
		hdl_plot.plot(x_test_np,mean_np,color="blue",linestyle="-",linewidth=3)
		hdl_plot.fill(np.concatenate([x_test_np, x_test_np[::-1]]),np.concatenate([mean_np-std_np,(mean_np+std_np)[::-1]]),\
			# alpha=.2, fc=color_var, ec='None', label='95% confidence interval')
			alpha=.2, fc="blue", ec='None')

		plt.show(block=True)


	# Using T test samples, to validate the posterior
	# ===============================================
	#
	# x_test = tf.linspace(-2,+2,101)
	x_test = tf.random.uniform(shape=(T,in_dim), minval=-2.0, maxval=2.0, dtype=tf.dtypes.float32, seed=None, name=None)
	y_test = y_true(x_test)
	y_test = tf.reshape(y_test,[-1,1])

	X = x_test[0:-1,:]
	Y = y_test[0:-1]

	x_new = tf.reshape(x_test[-1,:],[-1,in_dim])
	y_new = tf.reshape(y_test[-1],[-1,1])

	# pdb.set_trace()

	mean, cov = blr.q_predictive_gaussian(X, Y, x_new)

	print("")
	print("mean:",mean)
	print("cov:",cov)



	# Things to try:
	# 1) Reduce/remove the eye noise added to Lambda0, to make sure it doesn't have an influence
	# 2) Contrast the plotted prior with the predictive conditioned on part of the test data
	# 3) Why for some parameters we get a different prior than for others?
	# 4) 


if __name__ == "__main__":

	custom_training_loop()








