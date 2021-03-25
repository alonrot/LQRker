import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
import numpy as np
from lqrker.losses.loss_collection import LossKLDiv
from lqrker.model_blr import BayesianLinearRegression, ModelFeatures

def custom_training_loop():
	"""
	Based on https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/#low-level_handling_of_metrics
	"""

	in_dim = 1
	num_features_out = 128
	T = 11
	Sigma_noise_std = 0.1
	Sigma_noise = Sigma_noise_std**2
	xlim = [-2.,2.]

	Ninstances = 20 # Average over "trajectories"

	# Inputs:
	Xtrain = tf.random.uniform(shape=(Ninstances*T,in_dim), minval=xlim[0], maxval=xlim[1])

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

	epochs = 50
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


	# Plotting prior and predictive
	# ==============
	if in_dim == 1:
		
		# Grid:
		x_test = tf.reshape(tf.linspace(-4,+4,101),[-1,1])

		# True function:
		x_test = tf.cast(x_test,dtype=tf.float32)
		y_true_grid = y_true(x_test)

		# Prior:
		mean_prior, cov_prior = blr.q_learned_prior(x_test)
		std_prior = tf.sqrt(tf.linalg.diag_part(cov_prior))

		# Predictive:
		# Ntest = T - 1 # Same as the number of points we train with
		Ntest = 2
		Xtest = tf.random.uniform(shape=(Ntest,in_dim), minval=xlim[0], maxval=xlim[1])
		Ytest = y_true(Xtest)
		Ytest = tf.reshape(Ytest,[-1,1])
		mean_post, cov_post = blr.q_predictive_gaussian(X=Xtest, Y=Ytest, x_new=x_test)
		std_post = tf.sqrt(tf.linalg.diag_part(cov_post))

		# Plotting:
		hdl_fig, hdl_plot = plt.subplots(2,1,figsize=(6,6))
		hdl_prior = hdl_plot[0]
		hdl_post = hdl_plot[1]

		# Plot prior:
		x_test_np = x_test.numpy()[:,0]
		mean_prior_np = mean_prior.numpy()[:,0]
		std_prior_np = std_prior.numpy()
		hdl_prior.plot(x_test_np,mean_prior_np,color="blue",linestyle="-",linewidth=3)
		hdl_prior.fill(np.concatenate([x_test_np, x_test_np[::-1]]),np.concatenate([mean_prior_np-std_prior_np,(mean_prior_np+std_prior_np)[::-1]]),\
			# alpha=.2, fc=color_var, ec='None', label='95% confidence interval')
			alpha=.2, fc="blue", ec='None')
		xlim_plot = [xlim[0]-2, xlim[1]+2]
		hdl_prior.set_xlim(xlim_plot)
		hdl_prior.plot(x_test_np,y_true_grid.numpy(),linestyle="--",color="grey")

		# Plot predictive:
		x_test_np = x_test.numpy()[:,0]
		mean_post_np = mean_post.numpy()[:,0]
		std_post_np = std_post.numpy()
		hdl_post.plot(x_test_np,mean_post_np,color="blue",linestyle="-",linewidth=3)
		hdl_post.fill(np.concatenate([x_test_np, x_test_np[::-1]]),np.concatenate([mean_post_np-std_post_np,(mean_post_np+std_post_np)[::-1]]),\
			# alpha=.2, fc=color_var, ec='None', label='95% confidence interval')
			alpha=.2, fc="blue", ec='None')
		xlim_plot = [xlim[0]-2, xlim[1]+2]
		hdl_post.set_xlim(xlim_plot)
		hdl_post.plot(Xtest.numpy()[:,0],Ytest.numpy(),marker="o",linestyle="None",color="black",markersize=5)
		hdl_post.plot(x_test_np,y_true_grid.numpy(),linestyle="--",color="grey")

		plt.show(block=True)

if __name__ == "__main__":

	custom_training_loop()








