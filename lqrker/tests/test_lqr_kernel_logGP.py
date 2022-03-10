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
	use_systems_from_cost = True
	if not use_systems_from_cost:
		generate_linear_systems = GenerateLinearSystems(dim_state=cfg.RRTPLQRfeatures.dim_state,
												dim_control=cfg.RRTPLQRfeatures.dim_control,
												Nsys=1,
												check_controllability=cfg.RRTPLQRfeatures.check_controllability,
												prior="MNIW")

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
												Nsys=1,
												check_controllability=cfg.RRTPLQRfeatures.check_controllability,
												prior="MNIW")
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
	# activate_log_process = True

	xlim = eval(cfg.dataset.xlims)

	Npred = 100
	# xpred = 10**tf.reshape(tf.linspace(xlim[0],xlim[1],Npred),(-1,1))
	xpred = tf.reshape(tf.linspace(0.0,5.0,Npred),(-1,1))

	X,Y,A,B = generate_dataset(cfg)

	mean_vec, fpred_quan_minus, fpred_quan_plus, Xtrain, Ytrain = model_LQRcost_as_GP(cfg,X,Y,A,B,xpred)
	# mean_vec_logGP, fpred_quan_minus_logGP, fpred_quan_plus_logGP, Xtrain, Ytrain_logGP, entropy_vec = model_LQRcost_as_logGP(cfg,X,Y,A,B,xpred)


	hdl_fig, hdl_splots = plt.subplots(3,1,figsize=(14,10),sharex=True)
	# hdl_splots[0].plot(xpred,mean_vec_logGP)
	# hdl_splots[0].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus_logGP,(fpred_quan_plus_logGP)[::-1]],axis=0),\
	# 	alpha=.2, fc="blue", ec='None')
	# hdl_splots[0].set_title("logGP with LQR kernel. We do GP regression on f")
	# hdl_splots[0].set_xlabel("x")
	# hdl_splots[0].set_xlim(xpred[0,0],xpred[-1,0])
	# hdl_splots[0].plot(Xtrain,Ytrain_logGP,marker="o",color="black",linestyle="None")

	# hdl_splots[1].plot(xpred,entropy_vec)
	# hdl_splots[1].set_xlim(xpred[0,0],xpred[-1,0])
	# hdl_splots[1].set_title("Entropy of logGP")


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
	1) Weight the sum of kernels using the normalized density values p(A_j,B_j)/(sum_j p(A_j,B_j))
	Introduce var_prior as an extra parameter to compensate.
	2) Try to make the parameters we have right now, i.e., l and var_prior
	trainable. Consider doing ARD for Sigma0(th,th').
	3) If we're not using cij, refactor the cost and maybe create a git tag to
	come back to it if necessary.

	5) [Not needed] In test_elbo, LossElboLQR() class, we should use the transformed mean and kernel!!!
	6) Same thing for thr new GPLQR (with the Gaussian Mixture Model)


	/After coming to Berkeley/
	SOLVED 1. In model_LQRcost_as_GP(), we can see that if
	we use a Matern kernel with large lengthscales, the model doesn't react
	to the data, same as it happens with our LQR kernel. Conversely, if we
	use the Matern kernel with small lengthscales (lines 64,65), the model
	actually reacts to the data. This can only mean that somehow the LQR
	kernel hyperparameters are just very stiff. Maybe, try changing
	somthing, but what? Q_emp, R_emp are user choices...
	SOLUTION: The kernel does react to data. The reason it seemed it didn't was because
	the noise parameter of the likelihood of the GP model was too large compared to the posterior variance of the kernel
	itself, which is very small. By simply reducing the std of the likelihood noise to 0.01, we can
	see the GP shrinking around the evaluations. Importantly, if the variance of the initial condition (i.e., Sigma0 in x0 ~ N(mu0,Sigma0))
	is too large, then the cost observations are too scatered in the Y axis. For the sake of presentation, we should have a smaller Sigma0.
	In addition, we should find a way to increase the prior variance of the kernel itself while keeping a small amount of likelihod noise.
	
	2. Changing V_noise
	in SolveLQR.get_stationary_covariance_between_two_systems() and
	SolveLQR.get_stationary_variance() only influences the total magnitude
	of the variance, but not the lengthscales themselves...
	
	3. So far, the kernel is stiff. This may be because the model (A,B) we're using is fixed. 
		Maybe, to make the regression flexible, we need to add more model flexibility to the kernel, as we did in the CDC paper.
		NO! Because the data is generated using the excat same model... 
	4. Maybe the data we're using varies just too crazy. Use the mean plus noise...

	5. The V is HARDCODED!!! in solve_lqr.get_stationary_covariance_between_two_systems()
	6. The observations are being generated in LQRCostChiSquared() using solve_lqr.forward_simulation_with_random_initial_condition() OR
		solve_lqr.forward_simulation_expected_value(), both of which DO NOT have the v_k terms into account, while they should actually have to,
		for consistency. 
		For solve_lqr.forward_simulation_with_random_initial_condition(), the easiest would be to roll out the model.
		For solve_lqr.forward_simulation_expected_value(), there is an analytical expression, which can be extracted from the first 20 equations
		in the paper. I wrote down the solution, but it depends linearly on N, so when N goes to infty, we have problems... This result (i.e., the
		expected cost of an LQG problem) should be well known. Look for it in a book.


	"""



if __name__ == "__main__":

	main()


