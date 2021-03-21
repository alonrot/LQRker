import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures
import hydra
import numpy as np

from lqrker.experiments.generate_dataset import generate_dataset
from lqrker.experiments.validate_model import split_dataset

from lqrker.models.lqr_kernel_gpflow import LQRkernel
from lqrker.models.lqr_kernel_trans_gpflow import LQRkernelTransformed

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

@hydra.main(config_path="../experiments/",config_name="config.yaml")
def kernel_analysis(cfg):
	"""
	Since the LQR kernel is non-stationary, we need a 2D plot.

	We assume that the input dimensionalty is 1D and simply compute all the
	entries of the Gram matrix for a 1D input vector, and plot the values in a 2D
	plot.
	"""

	Ndiv = 51
	activate_log_process = True
	# activate_log_process = False

	if activate_log_process:
		ker = LQRkernelTransformed(cfg.RRTPLQRfeatures)
	else:
		ker = LQRkernel(cfg.RRTPLQRfeatures)

	# Input vector:
	xlim = eval(cfg.dataset.xlims)
	xpred = 10**tf.linspace(xlim[0],xlim[1],Ndiv)
	xpred = tf.reshape(xpred,(-1,1))

	# xpredXX, xpredYY = np.meshgrid(*[xpred]*2)

	Kxpred = ker.K(xpred)

	# pdb.set_trace()
	Kxpredmin = np.amin(Kxpred)
	Kxpredmax = np.amax(Kxpred)

	print("Kxpredmin:",Kxpredmin)
	print("Kxpredmax:",Kxpredmax)

	# pdb.set_trace()

	Kxpred_tf = tf.convert_to_tensor(Kxpred, dtype=tf.float64)
	eigvals = tf.eigvals(Kxpred_tf)
	print("eigvals:",eigvals)
	try:
		tf.linalg.cholesky(Kxpred_tf)
	except:
		print("Cholesky decomposition failed!")
	Kxpred_tf_fixed = RRTPLQRfeatures.fix_eigvals(Kxpred_tf)
	Kxpred_tf_fixed_chol = tf.linalg.cholesky(Kxpred_tf_fixed)

	hdl_fig, hdl_splots = plt.subplots(1,2,figsize=(14,10),sharex=True)
	hdl_splots[0].imshow(Kxpred,interpolation="None",origin="lower")
	hdl_splots[0].set_label("Kernel without fix")
	hdl_splots[1].imshow(Kxpred_tf_fixed.numpy(),interpolation="None",origin="lower")
	hdl_splots[1].set_label("Kernel fixed")
	plt.show(block=True)


if __name__ == "__main__":

	kernel_analysis()


