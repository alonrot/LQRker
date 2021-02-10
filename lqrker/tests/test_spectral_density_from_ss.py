import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.special import gamma


def get_density(w_vec,A,L,q):

	# # Notatino according to Rasmusssen Sec. A.3
	# U, d, V = scipy.linalg.svd(A)

	# W = np.diag(d)

	# # Compute (iwI + Ac)^{-1} = (1/iw)I + (1/w**2)*Vt
	# Vt = 

	dim = A.shape[0]
	Ndiv = len(w_vec)
	S_vec = np.zeros((dim,Ndiv))

	for ii in range(Ndiv):

		Aplus = A + np.eye(dim) * w_vec[ii] * 1j
		Aplus_inv = np.linalg.inv(Aplus)

		Aminus = A - np.eye(dim) * w_vec[ii] * 1j
		Aminus_inv = np.linalg.inv(Aminus)

		LQL = q*np.matmul(L,L.T)

		S_mat = np.matmul( np.matmul(Aplus_inv , LQL) , Aminus_inv.T)

		# pdb.set_trace()
		S_vec[:,ii] = np.real(np.diag(S_mat))

	return S_vec


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


	# Compute density:
	Ndiv = 101
	w_vec = np.linspace(-5.0,5.0,Ndiv)
	S_vec = get_density(w_vec,A,L,q)


	hdl_fig, hdl_splots = plt.subplots(3,1,figsize=(14,10),sharex=True)
	hdl_fig.suptitle("Spectral density of each state X(w) = [X1(w), X2(w), X3(w)]")

	for ii in range(dim):
		hdl_splots[ii].plot(w_vec,S_vec[ii,:])
		hdl_splots[ii].set_xlim([w_vec[0],w_vec[-1]])
		hdl_splots[ii].set_xlabel("w")
		hdl_splots[ii].set_ylabel("S{0:d}(w)".format(ii+1))

	plt.show(block=True)



if __name__ == "__main__":

	main()


	# # Discretize:
	# DeltaT = 0.01
	# Ad = A + np.eye(dim)*DeltaT
	# Ld = L * np.sqrt(DeltaT*q)

	# print("eigvals(Ad) = ",np.linalg.eigvals(Ad))




