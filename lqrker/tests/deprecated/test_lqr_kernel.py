import tensorflow as tf
import pdb
import math
import matplotlib.pyplot as plt
from lqrker.models.rrtp import RRTPLQRfeatures

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

def get_laypunov_sol(A,B,Q_emp,R_emp,theta):

	# Get design weights:
	Q_des = np.array([[theta]])
	R_des = 0.5*np.eye(1)

	# Compute controller:
	# P, eig, K = control.dare(A, B, Q_des, R_des)
	_,_, K = control.dare(A, B, Q_des, R_des)

	# Closed loop system:
	A_tilde = A - B @ K
	Q_tilde = Q_emp + K.T @ (R_emp @ K)

	# print("np.linalg.eigvals(A_tilde):",np.linalg.eigvals(A_tilde))

	P = la.solve_discrete_lyapunov(A_tilde,Q_tilde)

	return P

def LQRexp(theta,Sigma0,A,B,Q_emp,R_emp):

	P = get_laypunov_sol(A,B,Q_emp,R_emp,theta)

	return np.trace(Sigma0 @ P)


# @hydra.main(config_path=".",config_name="config.yaml")
def LQRkernel(theta1,theta2,Sigma0,A,B,Q_emp,R_emp):

	#Infinite horizon case
	P1 = get_laypunov_sol(A,B,Q_emp,R_emp,theta1)
	P2 = get_laypunov_sol(A,B,Q_emp,R_emp,theta1)


	ker_val = np.trace(Sigma0 @ P1) * np.trace(Sigma0 @ P2) + 2.0*np.trace((Sigma0 @ P1) @ (Sigma0 @ P2))

	return ker_val


def main():
	"""

	Inifinite horizon case
	No process noise, i.e., v_k = 0
	"""

	Q_emp = np.eye(1)
	R_emp = 0.5*np.eye(1)

	dim_state = Q_emp.shape[0]
	dim_control = R_emp.shape[1]

	A = np.random.uniform(size=Q_emp.shape,low=-2.0,high=2.0)
	B = np.random.uniform(size=(dim_state,dim_control),low=-2.0,high=2.0)

	# Initial condition:
	mu0 = np.zeros((dim_state,1))
	Sigma0 = np.eye(dim_state)
	# Sigma0_L = np.random.rand(dim_state, dim_state)
	# Sigma0 = Sigma0_L @ Sigma0_L.T
	# x0 = np.random.randn(A.shape[0],1)

	# # Noise covariance:
	# V = np.eye(dim_state)

	# Get variance:
	Ndiv = 51
	# theta_vec = np.linspace(0.01,2.0,Ndiv)
	theta_vec = 10**np.linspace(-2.0,0.1,Ndiv)

	# Kernel:
	variance_vec = np.zeros(Ndiv)
	mean_vec = np.zeros(Ndiv)
	for k in range(Ndiv):
		variance_vec[k] = LQRkernel(theta_vec[k],theta_vec[k],Sigma0,A,B,Q_emp,R_emp)
		mean_vec[k] = LQRexp(theta_vec[k],Sigma0,A,B,Q_emp,R_emp)

	std_vec = np.sqrt(variance_vec)

	hdl_fig, hdl_splots = plt.subplots(2,1,figsize=(14,10),sharex=True)
	hdl_splots[0].plot(theta_vec,mean_vec)
	fpred_quan_plus = mean_vec + std_vec
	fpred_quan_minus = mean_vec - std_vec
	hdl_splots[0].fill(tf.concat([theta_vec, theta_vec[::-1]],axis=0),tf.concat([fpred_quan_minus,(fpred_quan_plus)[::-1]],axis=0),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[0].set_xlabel("prior")


	plt.show(block=True)


if __name__ == "__main__":

	main()


