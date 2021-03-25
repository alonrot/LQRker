import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures

from lqrker.losses.loss_collection import LossStudentT, LossGaussian

import gpflow
import pickle
import hydra
import numpy as np

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

def load_dataset(num_fun):

	path_append = "/../../"
	# path_append = "outputs"

	# Load dataset:
	# path2file = "./../../2021-03-10/19-55-02/LQRcost_dataset_{0:d}.pickle".format(num_fun) # Random sequence xlims: "[-4,4]"; Nevals: 10000; Nobj_functions: 100
	# path2file = "./../../2021-03-11/16-22-26/LQRcost_dataset_{0:d}.pickle".format(num_fun) # Sobol sequence xlims: "[-2,2]"; Nevals: 100; Nobj_functions: 10
	# path2file = "./../../2021-03-11/17-07-33/LQRcost_dataset_{0:d}.pickle".format(num_fun) # Same as above, no noise
	path2file = "./"+path_append+"/2021-03-11/20-08-59/LQRcost_dataset_{0:d}.pickle".format(num_fun) # Same as above, Nevals: 10000; Nobj_functions: 100
	fid = open(path2file, "rb")
	XY_dataset = pickle.load(fid)
	X = XY_dataset["X"]
	Y = XY_dataset["Y"]

	return X,Y

def split_dataset(X,Y,perc_training,Ncut=None):

	assert perc_training > 0 and perc_training <= 100

	if Ncut is not None and Ncut != "None":
		assert Ncut <= X.shape[0]
		X = X[0:Ncut,:]
		Y = Y[0:Ncut]

	Ntrain = round(X.shape[0] * perc_training/100)

	Xtrain = X[0:Ntrain,:]
	Ytrain = Y[0:Ntrain]

	Xtest = X[Ntrain::,:]
	Ytest = Y[Ntrain::]

	logger.info("Splitting the dataset in {0:d} datapoints for training and {1:d} for testing".format(Ntrain,X.shape[0]-Ntrain))

	return Xtrain, Ytrain, Xtest, Ytest


def validate_rrtp_for_func(cfg,ii):

	Nfuns = cfg.validation.Nfuns
	logger.info("Validating RRTP model with function {0:d} / {1:d}".format(ii+1,Nfuns))

	X,Y = load_dataset(num_fun=ii)

	dim = eval(cfg.dataset.dim)
	assert X.shape[1] == dim

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,
												perc_training=cfg.validation.perc_training,
												Ncut=cfg.validation.Ncut)

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

	return float(loss_rrtp.SMSE(Ytest)), float(loss_rrtp.MSLL(Ytest))


def validate_gpflow_for_func(cfg,ii):

	Nfuns = cfg.validation.Nfuns
	logger.info("Validating GPFLOW model with function {0:d} / {1:d}".format(ii+1,Nfuns))

	X,Y = load_dataset(num_fun=ii)

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,
												perc_training=cfg.validation.perc_training,
												Ncut=cfg.validation.Ncut)

	# Build model:
	ker = gpflow.kernels.Matern52()
	XX = tf.cast(Xtrain,dtype=tf.float64)
	YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)
	mod = gpflow.models.GPR(data=(XX,YY), kernel=ker, mean_function=gpflow.mean_functions.Constant())
	mod.likelihood.variance.assign(cfg.GaussianProcess.hyperpars.sigma_n.init**2)
	mod.kernel.lengthscales.assign(cfg.GaussianProcess.hyperpars.ls.init)
	mod.kernel.variance.assign(cfg.GaussianProcess.hyperpars.prior_var.init)
	mod.mean_function.c.assign(tf.constant([cfg.GaussianProcess.hyperpars.mean.init]))

	xxpred = tf.cast(Xtest,dtype=tf.float64)
	opt = gpflow.optimizers.Scipy()
	maxiter = cfg.GaussianProcess.learning.epochs
	opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=maxiter))
	gpflow.utilities.print_summary(mod)
	mean_pred_gpflow, var_pred_gpflow = mod.predict_f(xxpred)

	# Validate:
	loss_gpflow = LossGaussian(mean_pred=mean_pred_gpflow,var_pred=var_pred_gpflow)
	
	return float(loss_gpflow.SMSE(Ytest)), float(loss_gpflow.MSLL(Ytest))


@hydra.main(config_path=".",config_name="config.yaml")
def validate_rrtp(cfg: dict) -> None:

	Nfuns = cfg.validation.Nfuns
	smse_rrtp_vec = np.zeros(Nfuns)
	msll_rrtp_vec = np.zeros(Nfuns)
	for ii  in range(Nfuns):

		smse_rrtp_vec[ii], msll_rrtp_vec[ii] = validate_rrtp_for_func(cfg,ii)

	mean_smse_rrtp = np.mean(smse_rrtp_vec)
	std_smse_rrtp = np.std(smse_rrtp_vec)	
	mean_msll_rrtp = np.mean(msll_rrtp_vec)
	std_msll_rrtp = np.std(msll_rrtp_vec)

	logger.info("smse_rrtp: {0:f} ({1:f})".format(mean_smse_rrtp,std_smse_rrtp))
	logger.info("msll_rrtp: {0:f} ({1:f})".format(mean_msll_rrtp,std_msll_rrtp))



@hydra.main(config_path=".",config_name="config.yaml")
def validate_gpflow(cfg: dict) -> None:

	Nfuns = cfg.validation.Nfuns
	smse_gp_vec = np.zeros(Nfuns)
	msll_gp_vec = np.zeros(Nfuns)
	for ii  in range(Nfuns):

		smse_gp_vec[ii], msll_gp_vec[ii] = validate_gpflow_for_func(cfg,ii)

	mean_smse_gp = np.mean(smse_gp_vec)
	std_smse_gp = np.std(smse_gp_vec)
	mean_msll_gp = np.mean(msll_gp_vec)
	std_msll_gp = np.std(msll_gp_vec)

	logger.info("smse_gp: {0:f} ({1:f})".format(mean_smse_gp,std_smse_gp))
	logger.info("msll_gp: {0:f} ({1:f})".format(mean_msll_gp,std_msll_gp))


if __name__ == "__main__":

	validate_rrtp()
	validate_gpflow()



