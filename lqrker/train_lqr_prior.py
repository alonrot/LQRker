import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
import numpy as np
from lqrker.losses import LossKLDiv
from lqrker.model_blr import BayesianLinearRegression, ModelFeatures
from lqrker.solve_lqr import GenerateLQRData, SolveLQR

# def plan_prediction_horizon(Ntot,Nhor):

# 	assert Nhor > 1
# 	ll = Ntot
# 	indices = []
# 	ind_min = 1
# 	ind_max = Nhor
# 	while ll > 0:

# 		ind_max_new = np.random.randint(low=ind_min,high=min(ind_max,Ntot))
# 		indices.append(ind_max)

# 		ind_min = ind_max



def custom_training_loop():
	"""
	Based on https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch/#low-level_handling_of_metrics
	"""


	# Q_emp = np.array([[1.0,0.0],[0.0,1.0]])
	# R_emp = np.array([[0.1,0.0],[0.0,0.1]])

	Q_emp = np.array([[1.0]])
	R_emp = np.array([[0.1]])

	dim_state = Q_emp.shape[0]
	dim_control = R_emp.shape[1]
	
	# Distribution of the initial condition:
	# The choice of the initial condition does not affect the final cost value
	# in LTI systems controlled with infinite horizon LQR
	# However, the parameters of the distribution affect the final solution
	mu0 = np.zeros((dim_state,1))
	Sigma0 = np.eye(dim_state)

	# Number of systems to sample:
	Nsys = 10

	# Number of controller designs to sample, for each samples system:
	Ncon = 20

	learning_rate = 1e-3
	Npred = 20
	Ninstances = 10
	# assert Nsys*Ncon % Npred*Ninstances == 0
	# Ninstances = Nsys*Ncon // Npred # Average over "trajectories"
	# pdb.set_trace()
	epochs = 300

	# Better reshape here

	generate_lqr_data = GenerateLQRData(Q_emp,R_emp,mu0,Sigma0,Nsys,Ncon)

	cost_values_all, theta_pars_all = generate_lqr_data.compute_cost_for_each_controller()


	# Input data:
	in_dim = dim_state + dim_control
	Xtrain = theta_pars_all.reshape(in_dim,-1).T
	Ytrain = cost_values_all.reshape(-1,1)
	Xtrain = tf.cast(Xtrain,dtype=tf.float32)
	Ytrain = tf.cast(Ytrain,dtype=tf.float32)

	num_features_out = 128
	Sigma_noise_std = 0.01
	Sigma_noise = Sigma_noise_std**2
	xlim = [-2.,2.]

	# pdb.set_trace()

	# model = ModelFeatures(in_dim=in_dim, num_features_out=num_features_out)
	blr = BayesianLinearRegression(in_dim,num_features_out,Sigma_noise)
	loss_kl_div = LossKLDiv(Sigma_noise)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	for epoch in range(epochs):

		with tf.GradientTape() as tape:

			loss_value = 0
			for jj in range(Ninstances-1):

				X = Xtrain[ jj*Npred:(jj+1)*Npred , : ]
				Y = Ytrain[ jj*Npred:(jj+1)*Npred ]

				# pdb.set_trace()
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


		
		if epoch % 10 == 0:
			print("Training loss (for one epoch) at epoch %d: %.4f" % (epoch, float(loss_value)))
		# print("Seen so far: %d samples" % ((Ninstances*(T-1) ) )


	# Plotting prior and predictive
	# ==============
	if in_dim == 1:
		
		# Grid:
		x_test = tf.reshape(tf.linspace(-4,+4,101),[-1,1])

		# Prior:
		mean_prior, cov_prior = blr.q_learned_prior(x_test)
		std_prior = tf.sqrt(tf.linalg.diag_part(cov_prior))

		# Predictive:
		Ntest = Npred - 1 # Same as the number of points we train with
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

		plt.show(block=True)


	if in_dim == 2:

		# Grid:
		Ngrid = 71
		x_min = 0.1
		x_max = 2.0
		x_vec = tf.linspace(x_min,x_max,Ngrid)
		xx1, xx2 = tf.meshgrid(x_vec,x_vec)
		x_test = tf.concat([tf.reshape(xx1,(-1,1)),tf.reshape(xx2,(-1,1))],axis=1)

		# Prior:
		mean_prior, cov_prior = blr.q_learned_prior(x_test)
		std_prior = tf.sqrt(tf.linalg.diag_part(cov_prior))

		# Predictive:
		Ntest = Npred - 1 # Same as the number of points we train with
		Xtest = tf.random.uniform(shape=(Ntest,in_dim), minval=x_min, maxval=x_max)

		A_true, B_true = generate_lqr_data._sample_systems(1)
		A_true = np.squeeze(A_true,axis=0)
		B_true = np.squeeze(B_true,axis=0)

		Ytest = np.zeros((Ntest,1))
		for jj in range(Ntest):

			Q_des = np.array([[Xtest[jj,0].numpy()]])
			R_des = np.array([[Xtest[jj,1].numpy()]])
			# pdb.set_trace()
			Ytest[jj,0] = generate_lqr_data.solve_lqr.forward_simulation(A_true, B_true, Q_des, R_des)

		mean_post, cov_post = blr.q_predictive_gaussian(X=Xtest, Y=Ytest, x_new=x_test)
		std_post = tf.sqrt(tf.linalg.diag_part(cov_post))

		hdl_fig, hdl_plot = plt.subplots(2,2,figsize=(10,10))
		hdl_prior_mean = hdl_plot[0,0]
		hdl_prior_std = hdl_plot[0,1]
		hdl_post_mean = hdl_plot[1,0]
		hdl_post_std = hdl_plot[1,1]

		# Plot prior:
		mean_prior_np = np.reshape(mean_prior,(Ngrid,Ngrid))
		std_prior_np = np.reshape(std_prior,(Ngrid,Ngrid))
		
		im_mean = hdl_prior_mean.imshow(mean_prior_np)
		hdl_prior_mean.set_xlim([x_min,x_max])
		hdl_prior_mean.set_ylim([x_min,x_max])
		hdl_prior_mean.set_xlabel("Q_des")
		hdl_prior_mean.set_ylabel("R_des")
		hdl_prior_mean.plot(Xtest[:,0],Xtest[:,1],marker="o",color="black",markersize=7,linestyle="None")
		hdl_fig.colorbar(im_mean, ax=hdl_prior_mean)
		# ax_cb.yaxis.tick_right()
		# ax_cb.yaxis.set_tick_params(labelright=False)

		im_std = hdl_prior_std.imshow(std_prior_np)
		hdl_prior_std.set_xlim([x_min,x_max])
		hdl_prior_std.set_ylim([x_min,x_max])
		hdl_prior_std.set_xlabel("Q_des")
		hdl_prior_std.set_ylabel("R_des")
		hdl_prior_std.plot(Xtest[:,0],Xtest[:,1],marker="o",color="black",markersize=7,linestyle="None")
		hdl_fig.colorbar(im_mean, ax=hdl_prior_std)


		# Plot posterior:
		mean_post_np = np.reshape(mean_post,(Ngrid,Ngrid))
		std_post_np = np.reshape(std_post,(Ngrid,Ngrid))

		im_mean = hdl_post_mean.imshow(mean_post_np)
		hdl_post_mean.set_xlim([x_min,x_max])
		hdl_post_mean.set_ylim([x_min,x_max])
		hdl_post_mean.set_xlabel("Q_des")
		hdl_post_mean.set_ylabel("R_des")
		hdl_post_mean.plot(Xtest[:,0],Xtest[:,1],marker="o",color="black",markersize=7,linestyle="None")
		hdl_fig.colorbar(im_mean, ax=hdl_post_mean)
		# ax_cb.yaxis.tick_right()
		# ax_cb.yaxis.set_tick_params(labelright=False)

		im_std = hdl_post_std.imshow(std_post_np)
		hdl_post_std.set_xlim([x_min,x_max])
		hdl_post_std.set_ylim([x_min,x_max])
		hdl_post_std.set_xlabel("Q_des")
		hdl_post_std.set_ylabel("R_des")
		hdl_post_std.plot(Xtest[:,0],Xtest[:,1],marker="o",color="black",markersize=7,linestyle="None")
		hdl_fig.colorbar(im_mean, ax=hdl_post_std)

		plt.show(block=True)





	# Things to try:
	# 1) Reduce/remove the eye noise added to Lambda0, to make sure it doesn't have an influence
	# 2) Contrast the plotted prior with the predictive conditioned on part of the test data
	# 3) Why the prior distribution is not always the same, i.e., depending on the configuration of the NN, it changes?
	# 4) In alpaca, they use tanh as activation function "due to favorable
	# properties in terms of variance smoothness and the behavior of the variance
	# far from the observed data."
	# 5) In Alpaca, the meta-learning part is about training the model to condition on data. 
	# Training without the meta-learning is training without conditioning on data. Then, the prior over
	# beta is adjusted to fir the A,B models and the controller parameters, but not to condition.
	# 6) # Alpaca: nn_layers: [128,128, 32] inp = tf.layers.dense(inputs=inp, units=units,activation=activation) with tanh
	# Consider sampling only controllable systems by using their canonical controlable realization:
	# https://en.wikipedia.org/wiki/State-space_representation#Canonical_realizations
	# Controllability: https://en.wikipedia.org/wiki/Controllability#Discrete_linear_time-invariant_(LTI)_systems
	# In the Zi Wang paper "Regret bounds for meta Bayesian optimization with an
	# unknown Gaussian process prior", they use a 1-hidden-layer neural network
	# with cosine activation function and a linear output layer with
	# function-specific weights Wi
	# https://ziw.mit.edu/meta_bo/

	# Normal-Wishart distribution -> Models mean and covariance of a MVN
	# Inverse-Wishart distribution -> Models covariance of a MVN. Its conjugate posterior is a Student's t-distribution

	# from keras.layers import Input, Dense, Lambda, Concatenate
	# from keras.models import Model
	# from keras.callbacks import *
	# from keras import optimizers
	# import keras.backend as K
    # def make_network(self):
    #     self.dense_num = 2048
    #     # input data is of the shape
    #     # (data_idx,fcn_idx,dim_x)

    #     x_input = Input(shape=(self.dim_x,), name='x', dtype='float32')
    #     phi = Dense(self.dense_num, activation=cosine_activation, use_bias=True)(x_input)
    #     last_layers = []
    #     for i in range(self.n_fcns):
    #         last_layers.append(Dense(1, activation='linear', name='W_' + str(i), use_bias=False)(phi))
    #     outputs = Concatenate(axis=-1)(last_layers)

    #     self.score_fcn = Model(input=[x_input], output=outputs)
    #     self.feat_fcn = Model(input=[x_input], output=phi)

    #     adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #     self.score_fcn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    # Concatenate layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/concatenate


if __name__ == "__main__":

	custom_training_loop()








