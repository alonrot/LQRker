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

from generate_dataset import generate_dataset

import numpy as np

@hydra.main(config_path=".",config_name="config.yaml")
def debug_basic_test(cfg: dict) -> None:

	X,Y = generate_dataset(cfg)

	rrtp_lqr = RRTPLQRfeatures(dim=eval(cfg.dataset.dim),cfg=cfg.RRTPLQRfeatures)

	# rrtp_lqr.get_features_mat() takes:
	# 0.499546 [sec] on average per point with 200 features
	# 49.954554 [sec] in total with 200 features and 100 points

	rrtp_lqr.update_model(X,Y)
	rrtp_lqr.train_model()


@hydra.main(config_path=".",config_name="config.yaml")
def debug_exponentially_increasing_cost(cfg: dict) -> None:
	"""

	As the dimensionality goes up, the trace of the Ricatti matrix P inceases exponentially.
	Herein, we analyze (a) the evolution of the log10() of the trace of P w.r.t the dimensionality AND
	(b) the evolution of the log10() of the evolution of the maximum negative eigenvalue of PhiX^T.PhiX.

	a) The trace of P has an impact on the features computation of the class RRTPLQRfeatures()

	b) PhiX^T.PhiX is called in self.get_MLII_loss() and self._update_features() from ReducedRankStudentTProcessBase()
	Because we compule chol(PhiX^T.PhiX + Diag_matrix), the eigenvalues of Diag_matrix must be at least as large
	as the maximum negative eigenvalue of PhiX^T.PhiX.

	Action: We divided the cost obtained at LQRCostStudent.evaluate() by an empirical law 10**(0.15*dim)

	"""


	Ndims_debug = 10
	import numpy as np

	mean_DBG_P_trace_list_vec = np.zeros(Ndims_debug)
	std_DBG_P_trace_list_vec = np.zeros(Ndims_debug)
	DBG_eigvals_neg_max_vec = np.zeros(Ndims_debug)
	dims_vec = np.zeros(Ndims_debug)
	for kk in range(1,Ndims_debug+1):

		cfg.dataset.dim = int(kk*2)
		cfg.RRTPLQRfeatures.empirical_weights.Q_emp = "np.eye("+str(kk)+")"
		cfg.RRTPLQRfeatures.empirical_weights.R_emp = "np.eye("+str(kk)+")"

		X,Y = generate_dataset(cfg)

		rrtp_lqr = RRTPLQRfeatures(dim=cfg.dataset.dim,cfg=cfg.RRTPLQRfeatures)

		# See the trace:
		aaa = rrtp_lqr.get_features_mat(X)
		DBG_P_trace_list = rrtp_lqr.lqr_cost_student.lqr_data.solve_lqr.DBG_P_trace_list

		# See the minimum negative eigenvalue:
		rrtp_lqr.update_model(X,Y)
		# pdb.set_trace()
		eigvals_real = tf.math.real(rrtp_lqr.DBG_eigvals)
		eigvals_real_neg = eigvals_real[eigvals_real < 0.0]
		eigvals_max = tf.reduce_max(eigvals_real_neg)
		DBG_eigvals_neg_max_vec[kk-1] = tf.experimental.numpy.log10(-eigvals_max)

		
		DBG_P_trace_list = np.array(DBG_P_trace_list)
		DBG_P_trace_list = np.log10(DBG_P_trace_list)
		mean_DBG_P_trace_list_vec[kk-1] = np.mean(DBG_P_trace_list)
		std_DBG_P_trace_list_vec[kk-1] = np.std(DBG_P_trace_list)
		dims_vec[kk-1] = int(kk*2)


	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(14,10),sharex=True)
	# hdl_fig.suptitle("Reduced-rank Student-t process")
	hdl_splots[0].errorbar(x=dims_vec,y=mean_DBG_P_trace_list_vec,yerr=std_DBG_P_trace_list_vec,marker="o",linestyle="--")
	hdl_splots[0].set_ylabel("log10(trace(P))")
	hdl_splots[1].plot(dims_vec,DBG_eigvals_neg_max_vec,marker="o",linestyle="--")
	hdl_splots[1].set_title("-(Maximum negative eignevalue of PhiX.T@PhiX)")
	hdl_splots[1].set_ylabel("log10()")
	hdl_splots[1].set_xlabel("dims")

	plt.show(block=True)

	rrtp_lqr.update_model(X,Y)



if __name__ == "__main__":

	debug_basic_test()
	# debug_exponentially_increasing_cost()




	