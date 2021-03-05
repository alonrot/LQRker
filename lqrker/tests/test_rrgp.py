import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrgp import RRGPQuadraticFeatures

def cost_parabola(X,sigma_n):
	return tf.reduce_sum(X**2,axis=1) + tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

def cost_linear(X,sigma_n):
	return tf.reduce_sum(X,axis=1) + tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

if __name__ == "__main__":
	
	dim = 1
	Nfeat = dim*(dim+1)//2 + dim + 1
	sigma_n = 1.0
	rrgp = RRGPQuadraticFeatures(dim=dim,
									Nfeat=Nfeat,
									sigma_n=sigma_n)
	Xlim = 5.0

	# Evaluate:
	Nevals = 10
	X = tf.random.uniform(shape=(Nevals,dim),minval=-Xlim,maxval=Xlim)
	# Y = cost_linear(X,sigma_n)
	Yex = cost_parabola(X,0.1*sigma_n)

	# Y = Yex + tf.constant([5.0]+[0.0]*(Yex.shape[0]-1))
	Y = Yex
	
	rrgp.update_model(X,Y)

	# Prediction/test locations:
	Npred = 200
	if dim == 1:
		xpred = tf.reshape(tf.linspace(-Xlim,Xlim,Npred),(-1,1))
	else:
		xpred = tf.random.uniform(shape=(20,dim),minval=-Xlim,maxval=Xlim)

	# Compute predictive moments:
	mean_pred, cov_pred = rrgp.get_predictive_moments(xpred)
	std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))
	# pdb.set_trace()

	if dim == 1:
		hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(14,10),sharex=True)
		hdl_fig.suptitle("Reduced-rank GP (Särkkä)")
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

