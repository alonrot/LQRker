import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures

from lqrker.objectives.lqr_cost_student import LQRCostStudent
from lqrker.losses import LossStudentT, LossGaussian

import gpflow
import pickle
import hydra
import numpy as np

from lqrker.utils.parsing import get_logger
logger = get_logger(__name__)

from scipy import linalg as la
from scipy import stats as sts
import pdb
import numpy as np
import control

from lqrker.experiments.generate_dataset import generate_dataset
from lqrker.experiments.validate_model import split_dataset

class LQRkernelNoiselessProcess(gpflow.kernels.Kernel):
	def __init__(self,cfg, A=None, B=None):
		super().__init__(active_dims=[0])

		# Get parameters:
		nu = -10 # This parameter is irrelevanat here
		
		# nu = cfg.hyperpars.nu
		# Nsys = cfg.hyperpars.weights_features.Nfeat # Use as many systems as number of features

		# self.dim_in = eval(cfg.dataset.dim)
		self.dim_in = eval(cfg.empirical_weights.Q_emp).shape[0]
		self.lqr_cost_student = LQRCostStudent(	dim_in=self.dim_in,sigma_n=0.0,nu=nu,\
												cfg=cfg,Nsys=1,A=A,B=B)

		self.A = self.lqr_cost_student.A_samples
		self.B = self.lqr_cost_student.B_samples

		self.dim_state = self.lqr_cost_student.dim_state

		self.Q_emp = eval(cfg.empirical_weights.Q_emp)
		self.R_emp = eval(cfg.empirical_weights.R_emp)

		if cfg.initial_state_distribution.Sigma0 == "identity":
			self.Sigma0 = np.eye(self.dim_state)
		else:
			raise ValueError

	def _get_laypunov_sol(self,theta):
		"""

		theta: [dim,]
		"""

		if self.dim_in > 1:
			Q_des = tf.linalg.diag(X[0:self.dim_state])
			R_des = tf.linalg.diag(X[self.dim_state::])
		else:
			Q_des = tf.linalg.diag(theta)
			R_des = tf.constant([[1]])


		A = self.A[0,:,:]
		B = self.B[0,:,:]

		# Compute controller:
		# P, eig, K = control.dare(A, B, Q_des, R_des)
		_,_, K = control.dare(A, B, Q_des, R_des)

		# Closed loop system:
		A_tilde = A - B @ K
		Q_tilde = self.Q_emp + K.T @ (self.R_emp @ K)

		P = la.solve_discrete_lyapunov(A_tilde,Q_tilde)

		return P

	def _LQR_kernel(self,SP1,SP2=None):

		if SP2 is None:
			ker_val = np.trace(SP1)**2 + 2.0*np.linalg.matrix_power(SP1,2)
		else:
			ker_val = np.trace(SP1) * np.trace(SP2) + 2.0*np.trace(SP1 @ SP2)

		return ker_val

	def K(self,X,X2=None):
		"""

		Gram Matrix

		X: [Npoints, dim_in]
		X2: [Npoints2, dim_in]

		return:
		Kmat: [Npoints,Npoints2]

		TODO: Vectorize the call to _get_laypunov_sol() and _LQR_kernel()

		"""



		if X2 is None:
			X2 = X
		
		Kmat = np.zeros((X.shape[0],X2.shape[0]))
		for ii in range(X.shape[0]):
			for jj in range(X2.shape[0]):

				P_X_ii = self._get_laypunov_sol(X[ii,:])
				P_X_jj = self._get_laypunov_sol(X2[jj,:])

				P_X_ii = self.Sigma0 @ P_X_ii
				P_X_jj = self.Sigma0 @ P_X_jj

				Kmat[ii,jj] = self._LQR_kernel(P_X_ii,P_X_jj)

		# print("@K(): Kmat", Kmat)

		return Kmat

	def K_diag(self,X):
		"""
		It’s simply the diagonal of the K function, in the case where X2 is None.
		It must return a one-dimensional vector.

		X: [Npoints, dim_in]

		return:
		Kmat: [Npoints,]
		"""

		Kmat_diag = np.zeros(X.shape[0])
		for ii in range(X.shape[0]):
			P_X_ii = self._get_laypunov_sol(X[ii,:])
			P_X_ii = self.Sigma0 @ P_X_ii
			Kmat_diag[ii] = self._LQR_kernel(P_X_ii)

		# print("@K(): Kmat_diag", Kmat_diag)

		return Kmat_diag


class LQRMean(gpflow.mean_functions.MeanFunction):

	def __init__(self, cfg, A=None, B=None):

		# Get parameters:
		nu = -10 # This parameter is irrelevanat here
		
		# nu = cfg.hyperpars.nu
		# Nsys = cfg.hyperpars.weights_features.Nfeat # Use as many systems as number of features

		# self.dim_in = eval(cfg.dataset.dim)
		# pdb.set_trace()
		self.dim_in = eval(cfg.empirical_weights.Q_emp).shape[0]
		self.lqr_cost_student = LQRCostStudent(	dim_in=self.dim_in,sigma_n=0.0,nu=nu,\
												cfg=cfg,Nsys=1,A=A,B=B)

		self.A = self.lqr_cost_student.A_samples
		self.B = self.lqr_cost_student.B_samples

		self.dim_state = self.lqr_cost_student.dim_state

		self.Q_emp = eval(cfg.empirical_weights.Q_emp)
		self.R_emp = eval(cfg.empirical_weights.R_emp)

		if cfg.initial_state_distribution.Sigma0 == "identity":
			self.Sigma0 = np.eye(self.dim_state)
		else:
			raise ValueError

	def _get_laypunov_sol(self,theta):
		"""

		theta: [dim,]
		"""

		if self.dim_in > 1:
			Q_des = tf.linalg.diag(X[0:self.dim_state])
			R_des = tf.linalg.diag(X[self.dim_state::])
		else:
			Q_des = tf.linalg.diag(theta)
			R_des = tf.constant([[1]])


		A = self.A[0,:,:]
		B = self.B[0,:,:]

		# Compute controller:
		# P, eig, K = control.dare(A, B, Q_des, R_des)
		_,_, K = control.dare(A, B, Q_des, R_des)

		# Closed loop system:
		A_tilde = A - B @ K
		Q_tilde = self.Q_emp + K.T @ (self.R_emp @ K)

		P = la.solve_discrete_lyapunov(A_tilde,Q_tilde)

		return P

	def _mean_fun(self,SP):

		return np.trace(SP)

	def __call__(self,X):
		"""

		X: [Npoints, dim]

		return: [Npoints,1]
		"""

		mean_vec = np.zeros((X.shape[0],1))
		for ii in range(X.shape[0]):
			P_X_ii = self._get_laypunov_sol(X[ii,:])
			P_X_ii = self.Sigma0 @ P_X_ii
			mean_vec[ii,0] = self._mean_fun(P_X_ii)

		# print("@K(): mean_vec", mean_vec)

		return mean_vec

class logLQRkernelNoiselessProcess(gpflow.kernels.Kernel):
	def __init__(self,cfg: dict,A=None, B=None):
		super().__init__(active_dims=[0])

		self.lqr_ker = LQRkernelNoiselessProcess(cfg,A,B)
		self.lqr_mean = LQRMean(cfg, A, B)

	def K(self,X,X2=None):
		"""

		Gram Matrix

		X: [Npoints, dim_in]
		X2: [Npoints2, dim_in]

		return:
		Kmat: [Npoints,Npoints2]

		"""

		add_noise = False
		if X2 is None:
			X2 = X
			add_noise = True

		mX = self.lqr_mean(X)
		mX2 = self.lqr_mean(X2)

		KXX2 = self.lqr_ker.K(X,X2)

		KXX2 = tf.convert_to_tensor(KXX2,dtype=tf.float64)

		mX_times_mX2 = mX @ tf.transpose(mX2) # outer product mX.mX^T

		Kmat = tf.math.log( KXX2 / mX_times_mX2 + 1.0)

		if add_noise:
			# pdb.set_trace()
			Kmat += 1e-1*tf.eye(Kmat.shape[0],dtype=tf.float64)
			# pass

		# pdb.set_trace()

		return Kmat

	def K_diag(self,X):
		"""
		It’s simply the diagonal of the K function, in the case where X2 is None.
		It must return a one-dimensional vector.

		X: [Npoints, dim_in]

		return:
		Kmat: [Npoints,]
		"""

		KX_vec = self.lqr_ker.K_diag(X)
		mX = self.lqr_mean(X)

		Kvec = tf.math.log(KX_vec / tf.squeeze(mX)**2 + 1.0)

		# pdb.set_trace()

		return Kvec


class logLQRMean(gpflow.mean_functions.MeanFunction):
	def __init__(self,cfg: dict,A=None, B=None):
		self.lqr_ker = LQRkernelNoiselessProcess(cfg,A,B)
		self.lqr_mean = LQRMean(cfg, A, B)

	def __call__(self,X):

		mX = self.lqr_mean(X)

		Kvec = self.lqr_ker.K_diag(X)

		mean_vec = tf.math.log(mX) - 0.5 * tf.reshape(Kvec,(-1,1))

		# pdb.set_trace()

		return mean_vec


def kernel_analysis(cfg,Ndiv,activate_log_process=False):
	"""
	Since the LQR kernel is non-stationary, we need a 2D plot.

	We assume that the input dimensionalty is 1D and simply compute all the
	entries of the Gram matrix for a 1D input vector, and plot the values in a 2D
	plot.
	"""

	if activate_log_process:
		ker = logLQRkernelNoiselessProcess(cfg.RRTPLQRfeatures)
	else:
		ker = LQRkernelNoiselessProcess(cfg.RRTPLQRfeatures)

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
	Kxpred_tf_fixed = RRTPLQRfeatures.fix_eigvals(Kxpred_tf)
	Kxpred_tf_fixed_chol = tf.linalg.cholesky(Kxpred_tf_fixed)

	hdl_fig, hdl_splots = plt.subplots(1,2,figsize=(14,10),sharex=True)
	hdl_splots[0].imshow(Kxpred,interpolation="None",origin="lower")
	hdl_splots[1].imshow(Kxpred_tf_fixed.numpy(),interpolation="None",origin="lower")
	plt.show(block=True)


@hydra.main(config_path="../experiments/",config_name="config.yaml")
def main(cfg: dict) -> None:
	"""

	Inifinite horizon case
	No process noise, i.e., v_k = 0
	E[x0] = 0

	Use GPflow and a tailored kernel
	"""

	# kernel_analysis(cfg,Ndiv=51)
	
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

	# # TRain with the log of the data:
	# Ytrain = tf.math.log(Ytrain)
	# Ytest = tf.math.log(Ytest)
	# pdb.set_trace()

	xlim = eval(cfg.dataset.xlims)

	Npred = 60
	xpred = 10**tf.reshape(tf.linspace(xlim[0],xlim[1],Npred),(-1,1))

	# Regression with gpflow:
	if activate_log_process:
		lqr_ker = logLQRkernelNoiselessProcess(cfg=cfg.RRTPLQRfeatures,A=A,B=B)
		lqr_mean = logLQRMean(cfg=cfg.RRTPLQRfeatures,A=A,B=B)
	else:
		lqr_ker = LQRkernelNoiselessProcess(cfg.RRTPLQRfeatures,A=A,B=B)
		lqr_mean = LQRMean(cfg=cfg.RRTPLQRfeatures,A=A,B=B)

	# pdb.set_trace()

	XX = tf.cast(Xtrain,dtype=tf.float64)
	YY = tf.cast(tf.reshape(Ytrain,(-1,1)),dtype=tf.float64)
	mod = gpflow.models.GPR(data=(XX,YY), kernel=lqr_ker, mean_function=lqr_mean)
	# sigma_n = cfg.RRTPLQRfeatures.hyperpars.sigma_n.init
	sigma_n = 0.1
	mod.likelihood.variance.assign(sigma_n**2)
	# mod.kernel.lengthscales.assign(10)
	# mod.kernel.variance.assign(5.0)
	xxpred = tf.cast(xpred,dtype=tf.float64)
	# opt = gpflow.optimizers.Scipy()
	# opt_logs = opt.minimize(mod.training_loss, mod.trainable_variables, options=dict(maxiter=300))
	gpflow.utilities.print_summary(mod)
	mean_pred_gpflow, var_pred_gpflow = mod.predict_f(xxpred)

	if activate_log_process:

		# mean_vec = tf.exp( mean_pred_gpflow + 0.5 * var_pred_gpflow )
		# mean_vec = tf.exp( mean_pred_gpflow - var_pred_gpflow ) # Mode
		mean_vec = tf.exp( mean_pred_gpflow ) # Median

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

	# # Get variance:
	# Ndiv = 51
	# # theta_vec = np.linspace(0.01,2.0,Ndiv)
	# theta_vec = 10**np.linspace(-2.0,0.1,Ndiv)

	# # Kernel:
	# variance_vec = np.zeros(Ndiv)
	# mean_vec = np.zeros(Ndiv)
	# for k in range(Ndiv):
	# 	variance_vec[k] = LQRkernel(theta_vec[k],theta_vec[k],Sigma0,A,B,Q_emp,R_emp)
	# 	mean_vec[k] = LQRexp(theta_vec[k],Sigma0,A,B,Q_emp,R_emp)

	

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


