import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrblr import ReducedRankBayesianLinearRegression

fontsize = 17

def cost_parabola(X,sigma_n):
	return tf.reduce_sum(0.1*X**2,axis=1) + tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

def cost_linear(X,sigma_n):
	return tf.reduce_sum(X,axis=1) + tf.random.normal(shape=(X.shape[0],), mean=0.0, stddev=sigma_n)

def test():
	
	dim = 1

	# Hypercube domain:
	L = 4.0
	Lred = L # Training data should be in a reduced domain due to the Dirichlet boundary conditions
	Nfeat = 180 # Number of features
	# Nfeat = dim*(dim+1)//2 + dim + 1 # quadratic features
	sigma_n = 0.1

	# R-R-BLR:
	rrblr = ReducedRankBayesianLinearRegression(dim=dim,Nfeat=Nfeat,L=4.0,sigma_n=sigma_n)

	# Evaluate:
	Nevals = 5
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


def test_presentation_litreview():

	dim = 1

	# Hypercube domain:
	L = 4.0
	Lred = L # Training data should be in a reduced domain due to the Dirichlet boundary conditions
	# Nfeat = 180 # Number of features
	Nfeat = 2
	# Nfeat = dim*(dim+1)//2 + dim + 1 # quadratic features
	sigma_n = 0.1


	# Evaluate:
	Nevals = 5
	# X = tf.random.uniform(shape=(Nevals,dim),minval=-Lred,maxval=Lred)
	import numpy as np
	# Xnp = -L + 2*L*np.array([0.2,0.4,0.6,0.7])
	Xnp = -L + 2*L*np.array([0.01,0.2,0.4,0.6,0.7,0.99])
	X = tf.convert_to_tensor(Xnp[:,None],dtype=tf.float32)
	# Y = cost_linear(X,sigma_n=sigma_n)
	Y = cost_parabola(X,sigma_n)

	# Prediction/test locations:
	Npred = 200
	if dim == 1:
		xpred = tf.reshape(tf.linspace(-L,L,Npred),(-1,1))
	else:
		xpred = tf.random.uniform(shape=(20,dim),minval=-L,maxval=L)

	ii = 0
	hdl_fig, hdl_splots = None, None
	for n_feats in [2,5,16]:


		# R-R-BLR:
		rrblr = ReducedRankBayesianLinearRegression(dim=dim,Nfeat=n_feats,L=4.0,sigma_n=sigma_n)
		rrblr.update_dataset(X,Y)

		# Compute predictive moments:
		mean_pred, cov_pred = rrblr.get_predictive_moments(xpred)
		std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))

		if dim == 1:
			hdl_fig, hdl_splots = plotting(ii,n_feats,xpred,mean_pred,std_pred,X,Y,hdl_fig=hdl_fig,hdl_splots=hdl_splots)

		ii += 1

	# Add a standard GP regression with matern kernel:
	# ii = 3
	import gpflow
	k = gpflow.kernels.Matern52()
	m = gpflow.models.GPR(data=(tf.cast(X,dtype=tf.float64).numpy(), tf.cast(tf.reshape(Y,(-1,1)),dtype=tf.float64).numpy()), kernel=k, mean_function=None)
	# pdb.set_trace()
	m.likelihood.variance.assign(sigma_n**2)
	m.kernel.lengthscales.assign(0.5)
	xpred = np.reshape(np.linspace(-L,L,Npred,dtype=np.float64),(-1,1))
	mean_pred, var_pred = m.predict_f(xpred)
	var_pred = var_pred[:,-1][:,None]
	mean_pred = mean_pred[:,-1][:,None]
	std_pred = np.sqrt(var_pred)
	fpred_quan_plus = mean_pred + std_pred
	fpred_quan_minus = mean_pred - std_pred
	hdl_splots[ii].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[ii].plot(xpred,mean_pred)
	hdl_splots[ii].plot(X,Y,color="black",linestyle="None",markersize=5,marker="o")
	hdl_splots[ii].set_xlim([xpred[0,0],xpred[-1,0]])
	hdl_splots[ii].set_title("Matern5/2",fontsize=fontsize)
	hdl_splots[ii].set_xlabel("x",fontsize=fontsize)
	hdl_splots[ii].set_ylabel("f(x)",fontsize=fontsize)
	# hdl_splots[ii].set_ylim([-1.3,1.3])

	savefig = False
	# name_file = "/features_dem"
	name_file = "/features_dem_bound"
	if savefig:
		hdl_fig.savefig('./pics4presentation'+ name_file,bbox_inches='tight',dpi=300,transparent=True)
	else:
		plt.show(block=True)



def plotting(ii,n_feats,xpred,mean_pred,std_pred,X,Y,hdl_fig=None,hdl_splots=None):

		if hdl_fig is None and hdl_splots is None:
			hdl_fig, hdl_splots = plt.subplots(4,1,figsize=(14,10),sharex=True)
		hdl_fig.suptitle("Reduced-rank GP | Matern5/2")
		hdl_splots[ii].plot(xpred,mean_pred)
		fpred_quan_plus = mean_pred + std_pred
		fpred_quan_minus = mean_pred - std_pred
		hdl_splots[ii].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
			alpha=.2, fc="blue", ec='None')
		hdl_splots[ii].plot(X,Y,color="black",linestyle="None",markersize=5,marker="o")
		hdl_splots[ii].set_xlim([xpred[0,0],xpred[-1,0]])
		# hdl_splots[ii].set_ylim([-1.3,1.3])
		hdl_splots[ii].set_title("m = "+str(n_feats),fontsize=fontsize)
		hdl_splots[ii].set_ylabel("f(x)",fontsize=fontsize)

		return hdl_fig, hdl_splots



if __name__ == "__main__":

	# test()

	test_presentation_litreview()

