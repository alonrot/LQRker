import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrblr import ReducedRankBayesianLinearRegression

def cost(X):
	return tf.reduce_sum( X**2 , axis = 1)

if __name__ == "__main__":
	
	dim = 1

	# Hypercube domain:
	L = 4.0
	Lred = 0.5*L # Training data should be in a reduced domain due to the Dirichlet boundary conditions
	Nfeat = 200 # Number of features

	# R-R-BLR:
	rrblr = ReducedRankBayesianLinearRegression(dim=dim,Nfeat=Nfeat,L=4.0,sigma_n=0.1)

	# Evaluate:
	Nevals = 10
	X = tf.random.uniform(shape=(Nevals,dim),minval=-Lred,maxval=Lred)
	Y = cost(X)
	
	rrblr.update_dataset(X,Y)

	# Prediction/test locations:
	Npred = 200
	if dim == 1:
		xpred = tf.reshape(tf.linspace(-L,L,Npred),(-1,1))
	else:
		xpred = tf.random.uniform(shape=(20,dim),minval=-L,maxval=L)

	# Compute predictive moments:
	mean_pred, cov_pred = rrblr.get_predictive_moments(xpred)
	std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))

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
		plt.show(block=True)



