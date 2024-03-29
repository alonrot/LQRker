import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures

from lqrker.objectives.lqr_cost_student import LQRCostStudent
from lqrker.losses.loss_collection import LossStudentT, LossGaussian

import gpflow

import hydra

import numpy as np

from generate_dataset import generate_dataset
from validate_model import split_dataset

@hydra.main(config_path=".",config_name="config.yaml")
def main(cfg):

	my_seed = 1
	np.random.seed(my_seed)
	tf.random.set_seed(my_seed)

	X,Y = generate_dataset(cfg)

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,
												perc_training=cfg.validation.perc_training,
												Ncut=cfg.validation.Ncut)
	# Model:
	dim = eval(cfg.dataset.dim)
	rrtp_lqr = RRTPLQRfeatures(dim=dim,cfg=cfg.RRTPLQRfeatures)
	rrtp_lqr.update_model(Xtrain,Ytrain)
	rrtp_lqr.train_model()
	
	xlim = eval(cfg.dataset.xlims)
	
	# Prediction/test locations:
	Npred = 200
	if dim == 1:
		xpred = 10**tf.reshape(tf.linspace(xlim[0],xlim[1],Npred),(-1,1))
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
	XX = tf.cast(Xtrain,dtype=tf.float64)
	YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)
	mod = gpflow.models.GPR(data=(XX,YY), kernel=ker, mean_function=None)
	sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
	mod.likelihood.variance.assign(sigma_n**2)
	mod.kernel.lengthscales.assign(10)
	mod.kernel.variance.assign(5.0)
	xxpred = tf.cast(xpred,dtype=tf.float64)
	opt = gpflow.optimizers.Scipy()
	opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=300))
	gpflow.utilities.print_summary(mod)
	mean_pred_gpflow, var_pred_gpflow = mod.predict_f(xxpred)

	# # Calculate true cost:
	# f_cost = lqr_cost_student.evaluate(xpred,add_noise=False)

	# # Validate:
	# loss_rrtp = LossStudentT(mean_pred=mean_pred,var_pred=tf.linalg.diag_part(cov_pred),nu=nu)
	# loss_gpflow = LossGaussian(mean_pred=mean_pred_gpflow,var_pred=var_pred_gpflow)
	
	# smse_rrtp = loss_rrtp.SMSE(f_cost)
	# smse_gp = loss_gpflow.SMSE(f_cost)
	# print("smse_rrtp:",smse_rrtp)
	# print("smse_gp:",smse_gp)

	# msll_rrtp = loss_rrtp.MSLL(f_cost)
	# msll_gp = loss_gpflow.MSLL(f_cost)
	# print("msll_rrtp:",msll_rrtp)
	# print("msll_gp:",msll_gp)

	# pdb.set_trace()


	if dim == 1:

		# xpred = tf.convert_to_tensor(tf.experimental.numpy.log10(xpred))

		hdl_fig, hdl_splots = plt.subplots(4,1,figsize=(14,10),sharex=True)
		hdl_fig.suptitle("Reduced-rank Student-t process")
		hdl_splots[0].plot(xpred,mean_pred)
		fpred_quan_plus = mean_pred + std_pred
		fpred_quan_minus = mean_pred - std_pred
		hdl_splots[0].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
			alpha=.2, fc="blue", ec='None')
		hdl_splots[0].plot(Xtrain,Ytrain,color="black",linestyle="None",markersize=5,marker="o")
		hdl_splots[0].set_xlim([xpred[0,0],xpred[-1,0]])
		# hdl_splots[0].plot(xpred,f_cost,linestyle="--",marker=None,color="black")

		# hdl_splots[0].plot(xpred,sample_paths,linestyle="-",marker=None,color="red")

		hdl_splots[1].plot(xpred,entropy_pred)


		hdl_splots[2].plot(xpred,mean_pred_gpflow)
		std_pred_gpflow = tf.sqrt(var_pred_gpflow)
		fpred_quan_plus = mean_pred_gpflow + std_pred_gpflow
		fpred_quan_minus = mean_pred_gpflow - std_pred_gpflow
		hdl_splots[2].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
			alpha=.2, fc="blue", ec='None')
		hdl_splots[2].plot(Xtrain,Ytrain,color="black",linestyle="None",markersize=5,marker="o")
		hdl_splots[2].set_xlim([xpred[0,0],xpred[-1,0]])
		# hdl_splots[2].plot(xpred,f_cost,linestyle="--",marker=None,color="black")

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

if __name__ == "__main__":

	main()





