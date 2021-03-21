import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
import gpflow
import hydra
import numpy as np

from lqrker.experiments.generate_dataset import generate_dataset
from lqrker.experiments.validate_model import split_dataset

from lqrker.models.lqr_kernel_gpflow import LQRkernel, LQRMean
from lqrker.models.lqr_kernel_trans_gpflow import LQRkernelTransformed, LQRMeanTransformed

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

def model_LQRcost_as_GP(cfg,X,Y,A,B,xpred):

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,
												perc_training=cfg.validation.perc_training,
												Ncut=cfg.validation.Ncut)

	if isinstance(cfg.dataset.dim,str):
		dim = eval(cfg.dataset.dim)
	else:
		dim = cfg.dataset.dim

	lqr_ker = LQRkernel(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)
	lqr_mean = LQRMean(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)

	XX = tf.cast(Xtrain,dtype=tf.float64)
	YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)
	mod = gpflow.models.GPR(data=(XX,YY), kernel=lqr_ker, mean_function=lqr_mean)
	sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
	mod.likelihood.variance.assign(sigma_n**2)
	xxpred = tf.cast(xpred,dtype=tf.float64)
	# opt = gpflow.optimizers.Scipy()
	# opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=300))
	gpflow.utilities.print_summary(mod)
	mean_pred_gpflow, var_pred_gpflow = mod.predict_f(xxpred)

	mean_vec = mean_pred_gpflow
	
	std_pred_gpflow = np.sqrt(var_pred_gpflow)
	fpred_quan_minus = mean_pred_gpflow - 2.*std_pred_gpflow
	fpred_quan_plus = mean_pred_gpflow + 2.*std_pred_gpflow

	return mean_vec, fpred_quan_minus, fpred_quan_plus, Xtrain, Ytrain

def model_LQRcost_as_logGP(cfg,X,Y,A,B,xpred):

	Y = tf.math.log(Y)

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,
												perc_training=cfg.validation.perc_training,
												Ncut=cfg.validation.Ncut)

	if isinstance(cfg.dataset.dim,str):
		dim = eval(cfg.dataset.dim)
	else:
		dim = cfg.dataset.dim

	lqr_ker = LQRkernelTransformed(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)
	lqr_mean = LQRMeanTransformed(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)

	XX = tf.cast(Xtrain,dtype=tf.float64)
	YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)
	mod = gpflow.models.GPR(data=(XX,YY), kernel=lqr_ker, mean_function=lqr_mean)
	sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
	mod.likelihood.variance.assign(sigma_n**2)
	xxpred = tf.cast(xpred,dtype=tf.float64)
	# opt = gpflow.optimizers.Scipy()
	# opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=300))
	gpflow.utilities.print_summary(mod)
	mean_pred_gpflow, var_pred_gpflow = mod.predict_f(xxpred)

	transform_moments_back = True
	# transform_moments_back = False
	if transform_moments_back:

		# We transform the moments of p(f* | D) back to the moments of Y:
		# mean_vec = tf.exp( mean_pred_gpflow + 0.5 * var_pred_gpflow ) # Mean
		# mean_vec = tf.exp( mean_pred_gpflow - var_pred_gpflow ) # Mode
		mean_vec = tf.exp( mean_pred_gpflow ) # Median

		fpred_quan_plus = tf.exp( mean_pred_gpflow  + tf.sqrt(2.0*var_pred_gpflow) * tf.cast(tf.math.erfinv(2.*0.95 - 1.),dtype=tf.float64) )
		fpred_quan_minus = tf.exp( mean_pred_gpflow  + tf.sqrt(2.0*var_pred_gpflow) * tf.cast(tf.math.erfinv(2.*0.05 - 1.),dtype=tf.float64) )

		# pdb.set_trace()
		Ytrain = tf.exp(Ytrain)

	else:

		# We provide the moments of p(f* | D):
		mean_vec = mean_pred_gpflow	
		std_pred_gpflow = np.sqrt(var_pred_gpflow)
		fpred_quan_plus = mean_pred_gpflow + 2.*std_pred_gpflow
		fpred_quan_minus = mean_pred_gpflow - 2.*std_pred_gpflow

	return mean_vec, fpred_quan_minus, fpred_quan_plus, Xtrain, Ytrain

@hydra.main(config_path="../experiments/",config_name="config.yaml")
def main(cfg: dict) -> None:
	"""

	LQR - Inifinite horizon case
	No process noise, i.e., v_k = 0
	E[x0] = 0

	Use GPflow and a tailored kernel
	"""
	
	my_seed = 3
	np.random.seed(my_seed)
	tf.random.set_seed(my_seed)

	# activate_log_process = False
	activate_log_process = True

	xlim = eval(cfg.dataset.xlims)

	Npred = 60
	xpred = 10**tf.reshape(tf.linspace(xlim[0],xlim[1],Npred),(-1,1))

	X,Y,A,B = generate_dataset(cfg)

	mean_vec_logGP, fpred_quan_minus_logGP, fpred_quan_plus_logGP, Xtrain, Ytrain_logGP = model_LQRcost_as_logGP(cfg,X,Y,A,B,xpred)
	mean_vec, fpred_quan_minus, fpred_quan_plus, Xtrain, Ytrain = model_LQRcost_as_GP(cfg,X,Y,A,B,xpred)


	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(14,10),sharex=True)
	hdl_splots[0].plot(xpred,mean_vec_logGP)
	hdl_splots[0].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus_logGP,(fpred_quan_plus_logGP)[::-1]],axis=0),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[0].set_title("logGP with LQR kernel. We do GP regression on f")
	hdl_splots[0].set_xlabel("x")
	hdl_splots[0].set_xlim(xpred[0,0],xpred[-1,0])
	hdl_splots[0].plot(Xtrain,Ytrain_logGP,marker="o",color="black",linestyle="None")


	hdl_splots[1].plot(xpred,mean_vec)
	hdl_splots[1].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[1].set_title("Standard GP with LQR kernel. We do GP regression on J")
	hdl_splots[1].set_xlabel("x")
	hdl_splots[1].set_xlim(xpred[0,0],xpred[-1,0])
	hdl_splots[1].plot(Xtrain,Ytrain,marker="o",color="black",linestyle="None")
	hdl_splots[1].set_xlabel("x")

	plt.show(block=True)


	"""
	TODO
	====

	0) tests/test_lqr_kernel_logGP_analysis.py -> the Gram Matrix returns almost
	the same values everywhere....
	1) Optimization parameters in the logGP and GP processes. While Sigma0 is a
	user choice, we can play around with sigma_n. We can't optimize if the kernel
	isn't fully implemented in tensorflow.
	2) We could also see Sigma0 as a parameter, even though it's fixed for sampling x0 to get samples of the LQR cost
	3) Extend the kernel for multiple systems


	"""



if __name__ == "__main__":

	main()


