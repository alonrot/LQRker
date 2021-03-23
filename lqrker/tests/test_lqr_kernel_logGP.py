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

from lqrker.utils.generate_linear_systems import GenerateLinearSystems

def model_LQRcost_as_GP(cfg,X,Y,A,B,xpred):

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,
												perc_training=cfg.validation.perc_training,
												Ncut=cfg.validation.Ncut)

	if isinstance(cfg.dataset.dim,str):
		dim = eval(cfg.dataset.dim)
	else:
		dim = cfg.dataset.dim


	# Generate new system samples for the kernel:
	use_systems_from_cost = False
	if not use_systems_from_cost:
		generate_linear_systems = GenerateLinearSystems(dim_state=cfg.RRTPLQRfeatures.dim_state,
														dim_control=cfg.RRTPLQRfeatures.dim_control,
														Nsys=10,
														check_controllability=cfg.RRTPLQRfeatures.check_controllability)
		A, B = generate_linear_systems()

	lqr_ker = LQRkernel(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)
	lqr_mean = LQRMean(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)

	XX = tf.cast(Xtrain,dtype=tf.float64)
	YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)

	# Manipulate kernels:
	# ker_stat = gpflow.kernels.Matern52(lengthscales=2.0,variance=1.0)
	# ker_tot = lqr_ker * ker_stat # This works bad as the lengthscale affects
	# equally all regions of the space. Note that if we use this option, we must
	# switch off the Sigma(th,th') in kernel_gpflow.

	# lqr_ker = gpflow.kernels.Matern52()
	mod = gpflow.models.GPR(data=(XX,YY), kernel=lqr_ker, mean_function=lqr_mean)
	sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
	mod.likelihood.variance.assign(sigma_n**2)
	xxpred = tf.cast(xpred,dtype=tf.float64)

	# opt = gpflow.optimizers.Scipy()
	# opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=300))

	# mod.kernel.lengthscales.assign(cfg.GaussianProcess.hyperpars.ls.init)
	# mod.kernel.variance.assign(cfg.GaussianProcess.hyperpars.prior_var.init)


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

	# Generate new system samples for the kernel:
	use_systems_from_cost = False
	if not use_systems_from_cost:
		generate_linear_systems = GenerateLinearSystems(dim_state=cfg.RRTPLQRfeatures.dim_state,
														dim_control=cfg.RRTPLQRfeatures.dim_control,
														Nsys=10,
														check_controllability=cfg.RRTPLQRfeatures.check_controllability)
		A, B = generate_linear_systems()

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
		mean_vec = tf.exp( mean_pred_gpflow + 0.5 * var_pred_gpflow ) # Mean
		# mean_vec = tf.exp( mean_pred_gpflow - var_pred_gpflow ) # Mode
		# mean_vec = tf.exp( mean_pred_gpflow ) # Median

		fpred_quan_plus = tf.exp( mean_pred_gpflow  + tf.sqrt(2.0*var_pred_gpflow) * tf.cast(tf.math.erfinv(2.*0.95 - 1.),dtype=tf.float64) )
		fpred_quan_minus = tf.exp( mean_pred_gpflow  + tf.sqrt(2.0*var_pred_gpflow) * tf.cast(tf.math.erfinv(2.*0.05 - 1.),dtype=tf.float64) )

		# pdb.set_trace()
		Ytrain = tf.exp(Ytrain)

		# Entropy:
		entropy_vec = tf.math.log(var_pred_gpflow) + mean_pred_gpflow

	else:

		# We provide the moments of p(f* | D):
		mean_vec = mean_pred_gpflow	
		std_pred_gpflow = np.sqrt(var_pred_gpflow)
		fpred_quan_plus = mean_pred_gpflow + 2.*std_pred_gpflow
		fpred_quan_minus = mean_pred_gpflow - 2.*std_pred_gpflow

	return mean_vec, fpred_quan_minus, fpred_quan_plus, Xtrain, Ytrain, entropy_vec


def model_LQRcost_as_mixture_of_Gaussians(cfg,X,Y,A,B,xpred):
	pass


@hydra.main(config_path="../experiments/",config_name="config.yaml")
def main(cfg: dict) -> None:
	"""

	LQR - Inifinite horizon case
	No process noise, i.e., v_k = 0
	E[x0] = 0

	Use GPflow and a tailored kernel
	"""
	
	my_seed = 1
	np.random.seed(my_seed)
	tf.random.set_seed(my_seed)

	# activate_log_process = False
	activate_log_process = True

	xlim = eval(cfg.dataset.xlims)

	Npred = 100
	xpred = 10**tf.reshape(tf.linspace(xlim[0],xlim[1],Npred),(-1,1))

	X,Y,A,B = generate_dataset(cfg)

	mean_vec, fpred_quan_minus, fpred_quan_plus, Xtrain, Ytrain = model_LQRcost_as_GP(cfg,X,Y,A,B,xpred)
	mean_vec_logGP, fpred_quan_minus_logGP, fpred_quan_plus_logGP, Xtrain, Ytrain_logGP, entropy_vec = model_LQRcost_as_logGP(cfg,X,Y,A,B,xpred)


	hdl_fig, hdl_splots = plt.subplots(3,1,figsize=(14,10),sharex=True)
	hdl_splots[0].plot(xpred,mean_vec_logGP)
	hdl_splots[0].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus_logGP,(fpred_quan_plus_logGP)[::-1]],axis=0),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[0].set_title("logGP with LQR kernel. We do GP regression on f")
	hdl_splots[0].set_xlabel("x")
	hdl_splots[0].set_xlim(xpred[0,0],xpred[-1,0])
	hdl_splots[0].plot(Xtrain,Ytrain_logGP,marker="o",color="black",linestyle="None")

	hdl_splots[1].plot(xpred,entropy_vec)
	hdl_splots[1].set_xlim(xpred[0,0],xpred[-1,0])
	hdl_splots[1].set_title("Entropy of logGP")


	hdl_splots[2].plot(xpred,mean_vec)
	hdl_splots[2].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[2].set_title("Standard GP with LQR kernel. We do GP regression on J")
	hdl_splots[2].set_xlabel("x")
	hdl_splots[2].set_xlim(xpred[0,0],xpred[-1,0])
	hdl_splots[2].plot(Xtrain,Ytrain,marker="o",color="black",linestyle="None")
	hdl_splots[2].set_xlabel("x")

	plt.show(block=True)


	"""
	TODO
	====

	0) tests/test_lqr_kernel_logGP_analysis.py -> the Gram Matrix returns almost
	the same values everywhere....
	1) Change the system sampling by a normal-Wishart distribution (look at
	Schon) and weight the kernel parameters accordingly. then, introduce
	var_prior as an extra parameter to compensate.
	2) Try to make the parameters we have right now, i.e., l and var_prior
	trainable. Consider doing ARD for Sigma0(th,th').
	3) If we're not using cij, refactor the cost and maybe create a git tag to
	come back to it if necessary.
	4) Try the mixture of Gaussians. It's quite easy.

	"""



if __name__ == "__main__":

	main()


