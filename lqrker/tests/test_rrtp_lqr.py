import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures

import numpy as np
from lqrker.solve_lqr import GenerateLQRData

import gpflow

def cost(X,sigma_n,A_samples=None,B_samples=None):

	assert X.shape[1] == 1, "Cost tailored for 1-dim, for now"

	# Parameters:
	Q_emp = np.array([[1.0]])
	R_emp = np.array([[0.1]])
	dim_state = Q_emp.shape[0]
	dim_control = R_emp.shape[1]		
	mu0 = np.zeros((dim_state,1))
	Sigma0 = np.eye(dim_state)
	Nsys = 1 # We use only one system (the real one)
	Ncon = 1 # Irrelevant

	# Generate systems:
	lqr_data = GenerateLQRData(Q_emp,R_emp,mu0,Sigma0,Nsys,Ncon,check_controllability=True)
	if A_samples is None and B_samples is None:
		A_samples, B_samples = lqr_data._sample_systems(Nsamples=Nsys)

	Npoints = X.shape[0]
	cost_values_all = np.zeros(Npoints)
	for ii in range(Npoints):

		Q_des = tf.expand_dims(X[ii,:],axis=1)
		R_des = np.array([[0.1]])
		
		cost_values_all[ii] = lqr_data.solve_lqr.forward_simulation(A_samples[0,:,:], B_samples[0,:,:], Q_des, R_des)


	cost_values_all += tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

	# pdb.set_trace()

	return tf.convert_to_tensor(cost_values_all,dtype=tf.float32), A_samples, B_samples # [Npoints, Nfeat], __, __


if __name__ == "__main__":
	
	dim = 1
	Nfeat = 200
	sigma_n = 0.5 # The cholesky decomposition is sensitive to this number. If too small, it fails
	nu = 2.1
	rrtp_lqr = RRTPLQRfeatures(dim=dim,
									Nfeat=Nfeat,
									sigma_n=sigma_n,
									nu=nu)
	Xlim = 2.0

	# Evaluate:
	Nevals = 15
	X = 10**tf.random.uniform(shape=(Nevals,dim),minval=-Xlim,maxval=Xlim)
	Yex, A_samples, B_samples = cost(X,0.05*sigma_n)

	# # pdb.set_trace()
	# Y = Yex + tf.constant([5.0]+[0.0]*(Yex.shape[0]-1))
	Y = Yex

	rrtp_lqr.update_model(X,Y)

	# Implement here a model optimization step, by optimizing self.Sigma_weights_inv_times_noise_var
	rrtp_lqr.train_model()

	# Prediction/test locations:
	Npred = 150
	if dim == 1:
		xpred = 10**tf.reshape(tf.linspace(-Xlim,Xlim,Npred),(-1,1))
	else:
		xpred = 10**tf.random.uniform(shape=(20,dim),minval=-Xlim,maxval=Xlim)

	# Compute predictive moments:
	mean_pred, cov_pred = rrtp_lqr.get_predictive_moments(xpred)
	std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))

	# Sample paths:
	# sample_paths = rrtp_lqr.sample_path(mean_pred=mean_pred,cov_pred=cov_pred,Nsamples=2)

	entropy_pred = rrtp_lqr.get_predictive_entropy(cov_pred)

	# Regression with gpflow:
	# pdb.set_trace()
	ker = gpflow.kernels.Matern52()
	XX = tf.cast(X,dtype=tf.float64)
	YY = tf.cast(tf.reshape(Y,(-1,1)),dtype=tf.float64)
	mod = gpflow.models.GPR(data=(XX,YY), kernel=ker, mean_function=None)
	mod.likelihood.variance.assign(sigma_n**2)
	mod.kernel.lengthscales.assign(10)
	mod.kernel.variance.assign(5.0)
	xxpred = tf.cast(xpred,dtype=tf.float64)
	mean_pred_gpflow, var_pred_gpflow = mod.predict_f(xxpred)
	opt = gpflow.optimizers.Scipy()
	opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=300))
	gpflow.utilities.print_summary(mod)

	# pdb.set_trace()

	# Calculate true cost:
	f_cost,_,_ = cost(xpred,0.0,A_samples,B_samples)

	if dim == 1:

		# xpred = tf.convert_to_tensor(tf.experimental.numpy.log10(xpred))

		hdl_fig, hdl_splots = plt.subplots(4,1,figsize=(14,10),sharex=True)
		hdl_fig.suptitle("Reduced-rank Student-t process")
		hdl_splots[0].plot(xpred,mean_pred)
		fpred_quan_plus = mean_pred + std_pred
		fpred_quan_minus = mean_pred - std_pred
		hdl_splots[0].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
			alpha=.2, fc="blue", ec='None')
		hdl_splots[0].plot(X,Y,color="black",linestyle="None",markersize=5,marker="o")
		hdl_splots[0].set_xlim([xpred[0,0],xpred[-1,0]])
		hdl_splots[0].plot(xpred,f_cost,linestyle="--",marker=None,color="black")

		# hdl_splots[0].plot(xpred,sample_paths,linestyle="-",marker=None,color="red")

		hdl_splots[1].plot(xpred,entropy_pred)


		hdl_splots[2].plot(xpred,mean_pred_gpflow)
		std_pred_gpflow = tf.sqrt(var_pred_gpflow)
		fpred_quan_plus = mean_pred_gpflow + std_pred_gpflow
		fpred_quan_minus = mean_pred_gpflow - std_pred_gpflow
		hdl_splots[2].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
			alpha=.2, fc="blue", ec='None')
		hdl_splots[2].plot(X,Y,color="black",linestyle="None",markersize=5,marker="o")
		hdl_splots[2].set_xlim([xpred[0,0],xpred[-1,0]])
		hdl_splots[2].plot(xpred,f_cost,linestyle="--",marker=None,color="black")

		entropy_pred_gpflow = 0.5*tf.math.log(var_pred_gpflow)
		hdl_splots[3].plot(xpred,entropy_pred_gpflow)


		plt.show(block=True)


	"""
	TODO: 
	1) Compare this with a standard reduced-rank GP with a Matern kernel from Sarkka.
	2) Compute the entropy from the student-t distribution -> see how it depends from Sigma.
		https://math.stackexchange.com/questions/2272184/differential-entropy-of-the-multivariate-student-t-distribution
	3) Multivariate chi-squared distribution?

	4) What are we gonna use this for in iLQG ???
		5.1) Are there (Q,R) matrices naturally in iLQG or are they coming from differentiating the cost?

	5) Study the benefits of the Student's-t process

	6) Maybe stop here and read the papers from Deisenroth
	7) For the iLQR thing:
		7.1) Define features mapping: FROM parameters of terminal cost TO a set of
		cost-to-go values, each one depending on a different nonlinear model,
		obtained by sampling a different set of "simulator" parameters
		7.2) Use such feature to do BO with log(cost) and RRTP
		7.3) If the features are too expensive to calculate, they can be
		pre-calculated in a meta-learning fashion...

	"""






