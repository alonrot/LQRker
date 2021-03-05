import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.blr_margi import BLRQuadraticFeatures

def cost_parabola(X,sigma_n):
	return tf.reduce_sum(X**2,axis=1) + tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

def cost_linear(X,sigma_n):
	return tf.reduce_sum(X,axis=1) + tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

if __name__ == "__main__":
	
	dim = 2
	Nfeat = dim*(dim+1)//2 + dim + 1
	sigma_n = 1.0
	sigma2_n = sigma_n**2
	blrmargi = BLRQuadraticFeatures(in_dim=dim,
									num_features=Nfeat,
									sigma2_n=sigma2_n)
	Xlim = 5.0

	# Evaluate:
	Nevals = 5
	X = tf.random.uniform(shape=(Nevals,dim),minval=-Xlim,maxval=Xlim)
	# Y = cost_linear(X,sigma_n)
	Y = cost_parabola(X,sigma_n)
	
	blrmargi.update_model(X,Y)

	# Prediction/test locations:
	Npred = 200
	if dim == 1:
		xpred = tf.reshape(tf.linspace(-Xlim,Xlim,Npred),(-1,1))
	else:
		xpred = tf.random.uniform(shape=(20,dim),minval=-Xlim,maxval=Xlim)

	# Compute predictive moments:
	mean_pred, cov_pred = blrmargi.get_predictive_moments(xpred)
	std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))
	# pdb.set_trace()

	if dim == 1:
		hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(14,10),sharex=True)
		hdl_fig.suptitle("Reduced-rank GP (Särkkä)")
		hdl_splots[0].plot(xpred,mean_pred)
		fpred_quan_plus = mean_pred + std_pred
		fpred_quan_minus = mean_pred - std_pred
		hdl_splots[0].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
			alpha=.2, fc="blue", ec='None')
		hdl_splots[0].plot(X,Y,color="black",linestyle="None",markersize=5,marker="o")
		hdl_splots[0].set_xlim([xpred[0,0],xpred[-1,0]])


		# hdl_splots[1].plot(xpred,mean_pred_der)
		# fpred_quan_plus = mean_pred_der + std_pred_der
		# fpred_quan_minus = mean_pred_der - std_pred_der
		# hdl_splots[1].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
		# 	alpha=.2, fc="blue", ec='None')


		plt.show(block=True)

