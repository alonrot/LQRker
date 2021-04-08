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
from lqrker.models.gp_lqr import GPLQR
from lqrker.losses.loss_elbo_qAB import LossElboLQR_MatrixNormalWishart

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

from lqrker.utils.generate_linear_systems import GenerateLinearSystems

@hydra.main(config_path="../experiments/",config_name="config.yaml")
def main(cfg: dict) -> None:
	"""

	LQR - Inifinite horizon case
	No process noise, i.e., v_k = 0
	E[x0] = 0

	Use GPflow and a tailored kernel
	"""

	# tf.config.run_functions_eagerly(False)


	my_seed = 1
	np.random.seed(my_seed)
	tf.random.set_seed(my_seed)

	X,Y,A,B = generate_dataset(cfg)

	# Split dataset:
	Xtrain, Ytrain, Xtest, Ytest = split_dataset(X,Y,
												perc_training=cfg.validation.perc_training,
												Ncut=cfg.validation.Ncut)
	transform_moments_back = True
	if transform_moments_back:
		Ytrain = tf.math.log(Ytrain)
		

	if isinstance(cfg.dataset.dim,str):
		dim = eval(cfg.dataset.dim)
	else:
		dim = cfg.dataset.dim

	Npred = 50
	xlim = eval(cfg.dataset.xlims)
	xpred = 10**tf.reshape(tf.linspace(xlim[0],xlim[1],Npred),(-1,1))

	gp_lqr = GPLQR(cfg,dim)
	gp_lqr.update_model(Xtrain,Ytrain)

	f_mean, f_var =	gp_lqr.get_predictive_moments(xpred)
	f_std = tf.sqrt(f_var)


	if transform_moments_back:

		# We transform the moments of p(f* | D) back to the moments of Y:
		mean_pred = tf.exp( f_mean + 0.5 * f_var ) # Mean
		# mean_vec = tf.exp( f_mean - f_var ) # Mode
		# mean_vec = tf.exp( f_mean ) # Median

		fpred_quan_plus = tf.exp( f_mean  + tf.sqrt(2.0*f_var) * tf.cast(tf.math.erfinv(2.*0.95 - 1.),dtype=tf.float64) )
		fpred_quan_minus = tf.exp( f_mean  + tf.sqrt(2.0*f_var) * tf.cast(tf.math.erfinv(2.*0.05 - 1.),dtype=tf.float64) )

		# pdb.set_trace()
		Ytrain = tf.exp(Ytrain)

	else:
		
		mean_pred = f_mean
		fpred_quan_minus = f_mean - 2.*f_std
		fpred_quan_plus = f_mean + 2.*f_std


	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(14,10),sharex=True)
	hdl_splots[0].plot(xpred,mean_pred)
	hdl_splots[0].fill(tf.concat([xpred, xpred[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[0].set_title("logGP with LQR kernel. We do GP regression on f")
	hdl_splots[0].set_xlabel("x")
	hdl_splots[0].set_xlim(xpred[0,0],xpred[-1,0])
	hdl_splots[0].plot(Xtrain,Ytrain,marker="o",color="black",linestyle="None")

	plt.show(block=True)

	"""

	1) Double-check the equations for ELBO
	2) Eventually, implement the kernel with non-zero process noise and with finite time horizon!! (LQG!!) Look at Henning
		2.1) It's quite hard to obtain it for N < infty and non-zero process noise. Instead, we caould aim for just non-zero process noise.
	3) Look at the BO acquisition function: How does the GMM enter there?? Look at FITBO.
	"""



if __name__ == "__main__":

	main()
