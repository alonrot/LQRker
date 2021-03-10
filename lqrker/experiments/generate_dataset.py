import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures

from lqrker.objectives.lqr_cost_student import LQRCostStudent
from lqrker.losses import LossStudentT, LossGaussian

import gpflow
import hydra
import time

import pickle
import numpy as np

import os

@hydra.main(config_path=".",config_name="config.yaml")
def generate_dataset(cfg: dict):

	# Get parameters:
	dim = eval(cfg.dataset.dim)
	noise_eval_std = cfg.dataset.noise_eval_std
	nu = cfg.dataset.nu
	xlim = eval(cfg.dataset.xlims)
	Nevals = cfg.dataset.Nevals

	if cfg.dataset.generate.use:

		# Generate training data:
		Nobj_functions = cfg.dataset.generate.Nobj_functions
		for ii in range(Nobj_functions):
			lqr_cost_student = LQRCostStudent(dim_in=dim,sigma_n=noise_eval_std,nu=nu,cfg=cfg.RRTPLQRfeatures,Nsys=1)
			X = 10**tf.random.uniform(shape=(Nevals,dim),minval=xlim[0],maxval=xlim[1])
			Y = lqr_cost_student.evaluate(X,add_noise=True,verbo=True)

			if cfg.dataset.generate.save:
				XY_dataset = dict(X=X,Y=Y)

				# Get path and file name:
				path = cfg.dataset.generate.path
				ext = cfg.dataset.generate.ext
				file_name = cfg.dataset.generate.file_name + "_{0:d}.{1:s}".format(ii,ext)
				path2file = os.path.join(path,file_name)
				fid = open(path2file, "wb")

				# Place data into file:
				print("Saving {0:s} || Dataset {1:d} / {2:d}".format(path2file,ii+1,Nobj_functions))
				pickle.dump(XY_dataset,fid)

			del X, Y, lqr_cost_student

	else:

		lqr_cost_student = LQRCostStudent(dim_in=dim,sigma_n=noise_eval_std,nu=nu,cfg=cfg.RRTPLQRfeatures,Nsys=1)
		X = 10**tf.random.uniform(shape=(Nevals,dim),minval=xlim[0],maxval=xlim[1])
		Y = lqr_cost_student.evaluate(X,add_noise=True,verbo=True)

		return X,Y



if __name__ == "__main__":

	generate_dataset()
