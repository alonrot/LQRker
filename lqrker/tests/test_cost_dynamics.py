import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.special import gamma

# Example from
# ============
# J. Hartikainen and S. Särkkä,
# "KALMAN FILTERING AND SMOOTHING SOLUTIONS TO TEMPORAL GAUSSIAN PROCESS REGRESSION MODELS"
# 2010
# Sec. 4.1, for p = 2

def spectral_density(omega_vec, lambda_val, nu, sigma):

	S_vec = sigma**2 * ((2*np.sqrt(np.pi)*gamma(nu+0.5)) / (gamma(nu))) * lambda_val**(2*nu)/((lambda_val**2 + omega_vec**2)**(nu+0.5))

	return S_vec


def cost(x,Qxx,Qx):

	if x.ndim == 1:
		x = np.reshape(x,(-1,1))

	cost_val = np.abs(np.matmul(Qx , x)) +  np.matmul(x.T , np.matmul( Qxx , x ) )

	# pdb.set_trace()
	return cost_val.squeeze()


def main():

	# Use Matern p = 2:
	p = 2
	nu = p + 0.5
	# ls = 2.5 # Marginally stable for DeltaT = 0.01
	ls = 3.0
	lambda_val = np.sqrt(2*nu)/ls

	# Continuous-time system matrices:
	dim = 3
	A = np.zeros((dim,dim))
	A[0:-1,1::] = np.eye(dim-1)
	A[-1,:] = np.array([-lambda_val**3 , -3*lambda_val**2 , -3*lambda_val])
	L = np.zeros((dim,1))
	L[-1] = 1
	q = 2.0

	# Discretize:
	DeltaT = 0.01
	Ad = A + np.eye(dim)*DeltaT
	Ld = L * np.sqrt(DeltaT*q)

	print("eigvals(Ad) = ",np.linalg.eigvals(Ad))

	# Define cost:
	Qxx = np.eye(dim)
	Qxx[0,0] = 10.0
	Qx = np.array([[1.0,5.0,2.0]])

	# Simulate:
	# x0 = np.random.normal(loc=0.0,scale=1.0,size=(dim,))
	x0 = np.ones((dim,))
	x0[1] = -1.0
	x0[2] = 0.5

	Nrep = 20
	Nsteps = 200
	x_all = np.zeros((Nrep,dim,Nsteps))
	x_all[:,:,0] = x0
	cost_vec = np.zeros((Nrep,Nsteps))
	for ii in range(Nrep):
		cost_vec[ii,0] = cost(x_all[ii,:,0],Qxx,Qx)
	
	for jj in range(Nrep):
		for ii in range(1,Nsteps):
			# pdb.set_trace()
			x_curr = np.reshape(x_all[jj,:,ii-1],(-1,1))
			x_new = np.matmul(Ad,x_curr) + Ld * np.random.normal(loc=0.0,scale=1.0,size=(dim,1))
			x_all[jj,:,ii] = x_new.squeeze()
			cost_vec[jj,ii] = cost(x_all[jj,:,ii],Qxx,Qx)


	t_vec = np.linspace(0.0,(Nsteps-1)*DeltaT,Nsteps)
	# pdb.set_trace()

	# Compute mean and std:
	x_all_mean = np.mean(x_all,0)
	x_all_std = np.std(x_all,0)
	cost_mean = np.mean(np.log(cost_vec),0)
	cost_std = np.std(np.log(cost_vec),0)
	# pdb.set_trace()

	hdl_fig, hdl_splots = plt.subplots(5,1,figsize=(14,10),sharex=True)
	hdl_fig.suptitle("Evolution of 3D system according to Matern kernel")
	
	# Plot states:
	for kk in range(dim):
		hdl_splots[kk].plot(t_vec,x_all_mean[kk,:])
		fpred_quan_plus = x_all_mean[kk,:] + x_all_std[kk,:]
		fpred_quan_minus = x_all_mean[kk,:] - x_all_std[kk,:]
		hdl_splots[kk].fill(np.concatenate([t_vec, t_vec[::-1]]),np.concatenate([fpred_quan_minus,(fpred_quan_plus)[::-1]]),\
			alpha=.2, fc="blue", ec='None')
		hdl_splots[kk].set_xlim([t_vec[0],t_vec[-1]])
		hdl_splots[kk].set_ylabel("x{0:d}(k)".format(kk+1))
		hdl_splots[kk].set_title("Evolution of state 1")

	# Cost:
	hdl_splots[3].plot(t_vec,np.exp(cost_mean))
	fpred_quan_plus = np.exp(cost_mean + cost_std)
	fpred_quan_minus = np.exp(cost_mean - cost_std)
	hdl_splots[3].fill(np.concatenate([t_vec, t_vec[::-1]]),np.concatenate([fpred_quan_minus,(fpred_quan_plus)[::-1]]),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[3].set_xlim([t_vec[0],t_vec[-1]])
	hdl_splots[3].set_ylabel("c(x(k))")
	hdl_splots[3].set_title("Evolution of instantaneous cost")


	# Compute cumulated cost:
	cost_cum = np.zeros((Nrep,Nsteps))
	for ii in range(Nsteps):
		cost_cum[:,ii] = np.sum( cost_vec[:,ii::] , 1 )

	cost_cum_mean = np.mean(np.log(cost_cum),0)
	cost_cum_std = np.std(np.log(cost_cum),0)

	# Cost cumulated:
	hdl_splots[4].plot(t_vec,np.exp(cost_cum_mean))
	fpred_quan_plus = np.exp(cost_cum_mean + cost_cum_std)
	fpred_quan_minus = np.exp(cost_cum_mean - cost_cum_std)
	hdl_splots[4].fill(np.concatenate([t_vec, t_vec[::-1]]),np.concatenate([fpred_quan_minus,(fpred_quan_plus)[::-1]]),\
		alpha=.2, fc="blue", ec='None')
	hdl_splots[4].set_xlim([t_vec[0],t_vec[-1]])
	hdl_splots[4].set_ylabel("C(x(k))")
	hdl_splots[4].set_title("Evolution of cost-to-go")
	hdl_splots[4].set_xlabel("time [sec]")

	hdl_fig, hdl_splots = plt.subplots(1,1,figsize=(6,6))
	omega_vec = np.linspace(-3.0,3.0,101)
	S_vec = spectral_density(omega_vec, lambda_val, nu, sigma=1.0)
	hdl_splots.plot(omega_vec,S_vec)
	hdl_splots.set_xlim([omega_vec[0],omega_vec[-1]])
	hdl_splots.set_xlabel("w [1/sec]")
	hdl_splots.set_ylabel("S(w) [1/sec]")
	hdl_splots.set_title("Spectral density of Matern p = 2")

	plt.show(block=True)





if __name__ == "__main__":

	main()