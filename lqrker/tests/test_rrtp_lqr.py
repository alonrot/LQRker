import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures

import numpy as np
from lqrker.solve_lqr import GenerateLQRData

def cost(X,sigma_n):

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
	A_samples, B_samples = lqr_data._sample_systems(Nsamples=Nsys)

	Npoints = X.shape[0]
	cost_values_all = np.zeros(Npoints)
	for ii in range(Npoints):

		Q_des = tf.expand_dims(X[ii,:],axis=1)
		R_des = np.array([[0.1]])
		
		cost_values_all[ii] = lqr_data.solve_lqr.forward_simulation(A_samples[0,:,:], B_samples[0,:,:], Q_des, R_des)


	cost_values_all += tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

	# pdb.set_trace()

	return tf.convert_to_tensor(cost_values_all,dtype=tf.float32) # [Npoints, Nfeat]


if __name__ == "__main__":
	
	dim = 1
	Nfeat = 200
	sigma_n = 0.5
	nu = 2.5
	rrtp_lqr = RRTPLQRfeatures(dim=dim,
									Nfeat=Nfeat,
									sigma_n=sigma_n,
									nu=nu)
	Xlim = 2.0

	# Evaluate:
	Nevals = 5
	X = 10**tf.random.uniform(shape=(Nevals,dim),minval=-Xlim,maxval=Xlim)
	Yex = cost(X,0.1*sigma_n)

	# # pdb.set_trace()
	# Y = Yex + tf.constant([5.0]+[0.0]*(Yex.shape[0]-1))
	Y = Yex

	rrtp_lqr.update_model(X,Y)

	# Prediction/test locations:
	Npred = 100
	if dim == 1:
		xpred = 10**tf.reshape(tf.linspace(-Xlim,Xlim,Npred),(-1,1))
	else:
		xpred = 10**tf.random.uniform(shape=(20,dim),minval=-Xlim,maxval=Xlim)

	# Compute predictive moments:
	mean_pred, cov_pred = rrtp_lqr.get_predictive_moments(xpred)
	std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))
	# pdb.set_trace()

	if dim == 1:

		# xpred = tf.convert_to_tensor(tf.experimental.numpy.log10(xpred))

		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(14,10),sharex=True)
		hdl_fig.suptitle("Reduced-rank Student-t process")
		hdl_splots.plot(xpred,mean_pred)
		fpred_quan_plus = mean_pred + std_pred
		fpred_quan_minus = mean_pred - std_pred
		hdl_splots.fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
			alpha=.2, fc="blue", ec='None')
		hdl_splots.plot(X,Y,color="black",linestyle="None",markersize=5,marker="o")
		hdl_splots.set_xlim([xpred[0,0],xpred[-1,0]])


		# hdl_splots[1].plot(xpred,mean_pred_der)
		# fpred_quan_plus = mean_pred_der + std_pred_der
		# fpred_quan_minus = mean_pred_der - std_pred_der
		# hdl_splots[1].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
		# 	alpha=.2, fc="blue", ec='None')


		plt.show(block=True)



	"""
	TODO: 
	1) Compare this with a standard reduced-rank GP with a Matern kernel from Sarkka.
	2) Plot the true cost (we need to sample the llinear system only once)
	"""






