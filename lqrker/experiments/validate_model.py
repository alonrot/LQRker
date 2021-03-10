import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures

from lqrker.objectives.lqr_cost_student import LQRCostStudent
from lqrker.losses import LossStudentT, LossGaussian

import gpflow

import pickle

import hydra

import numpy as np

def load_dataset(num_fun):

	# Load dataset:
	path2file = "./../../2021-03-10/19-55-02/LQRcost_dataset_{0:d}.pickle".format(num_fun)
	fid = open(path2file, "rb")
	XY_dataset = pickle.load(fid)
	X = XY_dataset["X"]
	Y = XY_dataset["Y"]

	return X,Y

def split_dataset(X,Y,perc_training,Ncut=None):

	assert perc_training > 0 and perc_training < 100

	if Ncut is not None:
		assert Ncut <= X.shape[0]
		X = X[0:Ncut,:]
		Y = Y[0:Ncut]

	Ntrain = round(X.shape[0] * perc_training/100)

	Xtrain = X[0:Ntrain,:]
	Ytrain = Y[0:Ntrain]

	Xtest = X[Ntrain::,:]
	Ytest = Y[Ntrain::]

	print("Splitting the dataset in {0:d} datapoints for training and {1:d} for testing".format(Ntrain,X.shape[0]-Ntrain))

	return Xtrain, Ytrain, Xtest, Ytest

@hydra.main(config_path=".",config_name="config.yaml")
def validate_rrtp(cfg: dict) -> None:

	X,Y = load_dataset(num_fun=0)

	dim = eval(cfg.dataset.dim)
	assert X.shape[1] == dim

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,perc_training=20,Ncut=100)

	# Model:
	rrtp_lqr = RRTPLQRfeatures(dim=dim,cfg=cfg.RRTPLQRfeatures)
	rrtp_lqr.update_model(Xtrain,Ytrain)
	rrtp_lqr.train_model()

	# Compute predictive moments:
	mean_pred, cov_pred = rrtp_lqr.get_predictive_moments(Xtest)
	std_pred = tf.sqrt(tf.linalg.diag_part(cov_pred))


	# Validate:
	loss_rrtp = LossStudentT(mean_pred=mean_pred,var_pred=tf.linalg.diag_part(cov_pred),\
							nu=cfg.RRTPLQRfeatures.hyperpars.nu)
	
	smse_rrtp = loss_rrtp.SMSE(Ytest)
	print("smse_rrtp:",smse_rrtp)

	msll_rrtp = loss_rrtp.MSLL(Ytest)
	print("msll_rrtp:",msll_rrtp)

@hydra.main(config_path=".",config_name="config.yaml")
def validate_gpflow(cfg: dict) -> None:

	X,Y = load_dataset(num_fun=0)

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,perc_training=20,Ncut=100)

	# Build model:
	ker = gpflow.kernels.Matern52()
	XX = tf.cast(Xtrain,dtype=tf.float64)
	YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)
	mod = gpflow.models.GPR(data=(XX,YY), kernel=ker, mean_function=gpflow.mean_functions.Constant())
	mod.likelihood.variance.assign(1.0)
	mod.kernel.lengthscales.assign(1.0)
	mod.kernel.variance.assign(10.0)

	xxpred = tf.cast(Xtest,dtype=tf.float64)
	mean_pred_gpflow, var_pred_gpflow = mod.predict_f(xxpred)
	opt = gpflow.optimizers.Scipy()
	maxiter = cfg.RRTPLQRfeatures.learning.epochs
	opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=10*maxiter))
	gpflow.utilities.print_summary(mod)

	# Validate:
	loss_gpflow = LossGaussian(mean_pred=mean_pred_gpflow,var_pred=var_pred_gpflow)
	
	smse_gp = loss_gpflow.SMSE(Ytest)
	print("smse_gp:",smse_gp)
	msll_gp = loss_gpflow.MSLL(Ytest)
	print("msll_gp:",msll_gp)

if __name__ == "__main__":

	validate_gpflow()
	validate_rrtp()





