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

	dim = eval(cfg.dataset.dim)

	X,Y,A,B = generate_dataset(cfg)

	if activate_log_process:
		Y = tf.math.log(Y)

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,
												perc_training=cfg.validation.perc_training,
												Ncut=cfg.validation.Ncut)

	xlim = eval(cfg.dataset.xlims)

	Npred = 60
	xpred = 10**tf.reshape(tf.linspace(xlim[0],xlim[1],Npred),(-1,1))

	# Regression with gpflow:
	if activate_log_process:
		lqr_ker = LQRkernelTransformed(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)
		lqr_mean = LQRMeanTransformed(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)
	else:
		lqr_ker = LQRkernel(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)
		lqr_mean = LQRMean(cfg=cfg.RRTPLQRfeatures,dim=dim,A_samples=A,B_samples=B)

	# pdb.set_trace()

	XX = tf.cast(Xtrain,dtype=tf.float64)
	YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)
	mod = gpflow.models.GPR(data=(XX,YY), kernel=lqr_ker, mean_function=lqr_mean)
	# sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
	sigma_n = 1.0
	mod.likelihood.variance.assign(sigma_n**2)
	xxpred = tf.cast(xpred,dtype=tf.float64)
	# opt = gpflow.optimizers.Scipy()
	# opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=300))
	gpflow.utilities.print_summary(mod)
	mean_pred_gpflow, var_pred_gpflow = mod.predict_f(xxpred)

	if activate_log_process:

		# mean_vec = tf.exp( mean_pred_gpflow + 0.5 * var_pred_gpflow ) # Mean
		# mean_vec = tf.exp( mean_pred_gpflow - var_pred_gpflow ) # Mode
		mean_vec = tf.exp( mean_pred_gpflow ) # Median

		# The noise sigma_n plays an important role

		fpred_quan_plus = tf.exp( mean_pred_gpflow  + tf.sqrt(2.0*var_pred_gpflow) * tf.cast(tf.math.erfinv(2.*0.95 - 1.),dtype=tf.float64) )
		fpred_quan_minus = tf.exp( mean_pred_gpflow  + tf.sqrt(2.0*var_pred_gpflow) * tf.cast(tf.math.erfinv(2.*0.05 - 1.),dtype=tf.float64) )

		# pdb.set_trace()
		Ytrain = tf.exp(Ytrain)

		# mean_vec = mean_pred_gpflow
		
		# std_pred_gpflow = np.sqrt(var_pred_gpflow)
		# fpred_quan_plus = mean_pred_gpflow + 2.*std_pred_gpflow
		# fpred_quan_minus = mean_pred_gpflow - 2.*std_pred_gpflow


	else:

		mean_vec = mean_pred_gpflow
		
		std_pred_gpflow = np.sqrt(var_pred_gpflow)
		fpred_quan_plus = mean_pred_gpflow + 2.*std_pred_gpflow
		fpred_quan_minus = mean_pred_gpflow - 2.*std_pred_gpflow


	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(14,10),sharex=True)
	hdl_splots[0].plot(xxpred,mean_vec)
	hdl_splots[0].fill(tf.concat([xxpred, xxpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[0].set_xlabel("x")
	hdl_splots[0].set_xlim(xxpred[0,0],xxpred[-1,0])
	hdl_splots[0].plot(Xtrain,Ytrain,marker="o",color="black",linestyle="None")


	plt.show(block=True)


if __name__ == "__main__":

	main()


