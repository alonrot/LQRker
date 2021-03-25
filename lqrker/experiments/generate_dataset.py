import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt

from lqrker.objectives.lqr_cost_chi2 import LQRCostChiSquared
from lqrker.losses.loss_collection import LossStudentT, LossGaussian

import gpflow
import hydra
import time

import pickle
import numpy as np

import os

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)


@hydra.main(config_path=".",config_name="config.yaml")
def generate_dataset(cfg: dict):

	# Get parameters:
	if isinstance(cfg.dataset.dim,str):
		dim = eval(cfg.dataset.dim)
	else:
		dim = cfg.dataset.dim
	noise_eval_std = cfg.dataset.noise_eval_std
	nu = cfg.dataset.nu
	xlim = eval(cfg.dataset.xlims)
	Nevals = cfg.dataset.Nevals

	lqr_cost = LQRCostChiSquared(dim_in=dim,cfg=cfg.RRTPLQRfeatures,Nsys=1)
	# if cfg.dataset.which_cost == "LQRCostChiSquared":
	# elif cfg.dataset.which_cost == "LQRCostStudent":
	# 	lqr_cost = LQRCostStudent(dim_in=dim,sigma_n=noise_eval_std,nu=nu,cfg=cfg.RRTPLQRfeatures,Nsys=1)
	# else:
	# 	raise ValueError("Cost misspecified")

	if cfg.dataset.massive.use:

		# Generate training data:
		Nobj_functions = cfg.dataset.massive.Nobj_functions
		for ii in range(Nobj_functions):
			
			X = xlim[0] + (xlim[1] - xlim[0])*tf.math.sobol_sample(dim=dim, num_results=Nevals, skip=2000, dtype=tf.dtypes.float32, name=None) # Samples in unit hypercube
			X = 10**X
			# X = 10**tf.random.uniform(shape=(Nevals,dim),minval=xlim[0],maxval=xlim[1])
			Y = lqr_cost.evaluate(X,with_noise=cfg.dataset.with_noise,verbo=True)

			if cfg.dataset.massive.save:
				XY_dataset = dict(X=X,Y=Y)

				# Get path and file name:
				path = cfg.dataset.massive.path
				ext = cfg.dataset.massive.ext
				file_name = cfg.dataset.massive.file_name + "_{0:d}.{1:s}".format(ii,ext)
				path2file = os.path.join(path,file_name)
				fid = open(path2file, "wb")

				# Place data into file:
				logger.info("Saving {0:s} || Dataset {1:d} / {2:d}".format(path2file,ii+1,Nobj_functions))
				pickle.dump(XY_dataset,fid)

			del X, Y

	else: # Generate single cost

		X = xlim[0] + (xlim[1] - xlim[0])*tf.math.sobol_sample(dim=dim, num_results=Nevals, skip=2000, dtype=tf.dtypes.float32, name=None) # Samples in unit hypercube
		X = 10**X
		# X = 10**tf.random.uniform(shape=(Nevals,dim),minval=xlim[0],maxval=xlim[1])
		Y = lqr_cost.evaluate(X,with_noise=cfg.dataset.with_noise,verbo=True)

		if cfg.dataset.return_sampled_system:
			A = lqr_cost.A_samples
			B = lqr_cost.B_samples
			return X,Y,A,B
		else:
			return X,Y



if __name__ == "__main__":

	generate_dataset()
