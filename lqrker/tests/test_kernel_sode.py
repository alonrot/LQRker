import numpy as np
import pdb
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import control
import scipy.linalg as la


def compute_kernel(Pinf,A,B,L,H,plot=False):

	# Compute the corresponding kernel for only the first state:
	Ac_exp = np.exp(Ac)
	# phi = lambda tau: la.matrix_power()
	phi = lambda tau: la.expm(Ac * tau)
	# phi = lambda tau: Ac_exp**

	H_Pinf = np.matmul(H.T,Pinf)
	kernel_tau_pos = lambda tau: np.matmul(H_Pinf , np.matmul( phi(tau).T , H ) )

	Pinf_H = np.matmul(Pinf,H)
	kernel_tau_neg = lambda tau: np.matmul(H.T , np.matmul( phi(tau) , Pinf_H ) )

	kernel = lambda tau: kernel_tau_pos(tau) if tau > 0 else kernel_tau_neg(-tau)

	kernel_scaled = lambda tau: kernel(tau) / kernel(0.0)
	
	# pdb.set_trace()
	
	# Plotting:
	if plot: 
		Ndiv = 201
		tau_vec = np.linspace(-8.0,8.0,Ndiv)
		ker_vec = np.zeros(Ndiv)
		for k in range(Ndiv):
			ker_vec[k] = kernel_scaled(tau_vec[k])
			# if tau_vec[k] > 0.0:
			# 	ker_vec[k] = kernel_tau_pos(tau_vec[k])
			# else:
			# 	ker_vec[k] = kernel_tau_neg(-tau_vec[k])

		hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(9,9))
		hdl_plot.plot(tau_vec,ker_vec)

		plt.show(block=True)

	return kernel_scaled

def generate_data(cov_x0,A,B,L,Q_des,R_des,Nsimu,q,dt):

	# Sample x0 from x0 ~ N(0,cov_x0)
	dim = cov_x0.shape[0]
	# x0 = np.random.multivariate_normal(np.zeros(dim), cov_x0, size=1)
	# x0 = np.reshape(x0,(dim,1))
	x0 = np.array([[2.0],[2.0]])

	# Pre-sample noise:
	w_vec = np.random.multivariate_normal(np.zeros(1), q*np.ones((1,1)), size=Nsimu)

	# pdb.set_trace()

	# Discretize:
	Ad = A + np.eye(dim)*dt
	Bd = B * dt
	Ld = L * dt

	# Compute controller:
	P, eig, K = control.dare(Ad, Bd, Q_des, R_des) # Assumes u(t) = -Kx(t)
	Ac = A - np.matmul(Bd,K)

	# Forward simulate:
	x_simu = np.zeros((dim,Nsimu))
	x_simu[:,0] = x0[:,0]
	for ii in range(1,Nsimu):
		# pdb.set_trace()
		x_next = np.matmul(Ac,x_simu[:,ii-1].reshape(dim,1)) + Ld * w_vec[ii,0]
		x_simu[:,ii] = np.asarray(x_next).reshape(-1)
	t_simu = np.linspace(0.0,dt*(Nsimu-1),Nsimu)

	return x_simu,t_simu

def get_posterior(kernel,x_simu,t_simu,t_pred):

	def my_kernel(tau):
		ker_val = kernel(tau)
		# if ker_val < 0.0:
		# 	return 1e-10
		return ker_val

	# The kernel is designed only for the first state:
	x_simu_s1 = x_simu[0,:][:,None]

	# Kernel Gram matrix:
	Nsimu = len(t_simu)
	Kgram = np.zeros((Nsimu,Nsimu))
	for ii in range(Nsimu):
		for jj in range(Nsimu):
			Kgram[ii,jj] = my_kernel( t_simu[jj] - t_simu[ii] )

	Kgram_inv = la.inv(Kgram + 1e-6*np.eye(Nsimu))
	# Kgram_inv = Kgram

	Npred = len(t_pred)
	kpred = np.zeros((Nsimu,Npred))
	for ii in range(Nsimu):
		for jj in range(Npred):
			kpred[ii,jj] = my_kernel( t_pred[jj] - t_simu[ii] )


	# Posterior:
	mean_post = np.matmul(kpred.T , np.matmul(Kgram_inv,x_simu_s1))
	var_post = np.ones((Npred,1))*my_kernel(0.0) - np.matmul(kpred.T , np.matmul(Kgram_inv,kpred))
	# print(np.diag(np.matmul(kpred.T , np.matmul(Kgram_inv,kpred))))

	std_post = np.sqrt(np.diag(var_post))

	pdb.set_trace()
	return np.asarray(mean_post).reshape(-1), std_post




if __name__ == "__main__":

	# Define a 2D continuous LTI state-space stystem using the canonical controlable form:
	A = np.array([[0.0, 1.0],[-1.0,-1.5]])
	B = np.array([[0],[1]])
	L = np.array([[0],[1]])
	H = np.array([[1],[0]]) # We observe the first state exactly y = H.T x

	# Compute the corresponding LQR controller:
	Q_des = np.diag([1.0,1.0])
	R_des = np.array([1.0])
	P, eig, K = control.care(A, B, Q_des, R_des) # Assumes u(t) = -Kx(t)

	# Define the continuous closed-loop system:
	Ac = A - np.matmul(B,K)

	# Compute the corresponding stationary covariance, assuming that
	# x'(t) = Ax(t) + Bu(t) + Lw(t), where w(t) \sim N(0,q)
	q = 0.5
	Q_syl = q * np.matmul(L,L.T)
	# Pinf = la.solve_sylvester(Ac, Ac.T, Q_syl)
	Pinf = control.lyap(Ac,Q_syl)

	kernel = compute_kernel(Pinf,A,B,L,H,plot=False)

	dt = 0.01
	Nsimu = 100
	x_simu, t_simu = generate_data(Pinf,A,B,L,Q_des,R_des,Nsimu,q,dt)

	# Use data to predict forward:
	Npred = 100
	t_pred = np.linspace(t_simu[-1],t_simu[-1] + (Npred-1)*(10*dt),Npred)

	# Squared exponential kernel:
	def SEker(tau): # Scalar case
		return 2.0*np.exp(-np.abs(tau)**2/(0.2)**2)



	mean_post, std_post = get_posterior(SEker,x_simu,t_simu,t_pred)

	# Plot data and prediction:
	hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(9,9))
	x_simu_s1 = x_simu[0,:]
	hdl_plot.plot(t_simu,x_simu_s1)
	hdl_plot.plot(t_pred,mean_post)
	fpred_quan_plus = mean_post + 2.0*std_post
	fpred_quan_minus = mean_post - 2.0*std_post
	# pdb.set_trace()
	hdl_plot.fill(np.concatenate([t_pred, t_pred[::-1]]),np.concatenate([fpred_quan_minus,(fpred_quan_plus)[::-1]]),alpha=.2, fc="blue", ec='None')
	plt.show(block=True)



	

