import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrblr import ReducedRankBayesianLinearRegression

def cost_parabola(X,sigma_n):
	return tf.reduce_sum(X**2,axis=1) + tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

def cost_linear(X,sigma_n):
	return tf.reduce_sum(X,axis=1) + tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

if __name__ == "__main__":
	
	dim = 1

	# Hypercube domain:
	L = 4.0
	Lred = 0.5*L # Training data should be in a reduced domain due to the Dirichlet boundary conditions
	# Nfeat = 180 # Number of features
	Nfeat = dim*(dim+1)//2 + dim + 1 # quadratic features
	sigma_n = 0.1

	# R-R-BLR:
	rrblr = ReducedRankBayesianLinearRegression(dim=dim,Nfeat=Nfeat,L=4.0,sigma_n=sigma_n)

	# Evaluate:
	Nevals = 3
	X = tf.random.uniform(shape=(Nevals,dim),minval=-Lred,maxval=Lred)
	# Y = cost_linear(X,sigma_n=sigma_n)
	Y = cost_parabola(X,sigma_n)
	
	rrblr.update_dataset(X,Y)

	# Prediction/test locations:
	Npred = 200
	if dim == 1:
		xpred = tf.reshape(tf.linspace(-L,L,Npred),(-1,1))
	else:
		xpred = tf.random.uniform(shape=(20,dim),minval=-L,maxval=L)

	# Compute derivative:
	mean_pred_der, std_pred_der = rrblr.get_predictive_moments_grad(xpred)
	# pdb.set_trace()

	# Compute predictive moments:
	mean_pred, cov_pred = rrblr.get_predictive_moments(xpred)
	std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))

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


		hdl_splots[1].plot(xpred,mean_pred_der)
		fpred_quan_plus = mean_pred_der + std_pred_der
		fpred_quan_minus = mean_pred_der - std_pred_der
		hdl_splots[1].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
			alpha=.2, fc="blue", ec='None')


		plt.show(block=True)

